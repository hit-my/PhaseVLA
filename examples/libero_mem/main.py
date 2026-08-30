from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from openpi_client import image_tools

try:
    import metrics as _metrics
except ModuleNotFoundError:
    _METRICS_SPEC = importlib.util.spec_from_file_location("libero_mem_metrics", Path(__file__).with_name("metrics.py"))
    _metrics = importlib.util.module_from_spec(_METRICS_SPEC)
    assert _METRICS_SPEC.loader is not None
    _METRICS_SPEC.loader.exec_module(_metrics)



LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_INFERENCE_IMAGE_SIZE = 224
LIBERO_SUITE_MAX_STEPS = {
    "libero_10": 520,
    "libero_mem": 600,
}
LIBERO_MEM_SETTLE_STEPS = 20
LIBERO_MEM_BOTTLE_JOINT = "wine_bottle_1_joint0"
LIBERO_MEM_PLATE_JOINT_PREFIX = "plate_"
LIBERO_MEM_BOTTLE_MAX_TILT_DEGREES = 20.0
LIBERO_MEM_BOTTLE_PLACEMENT_HEIGHT = 0.0135


@dataclasses.dataclass(frozen=True)
class ObjectStabilizationPlan:
    object_joint: str
    support_joint: str
    support_xy_offset: np.ndarray
    upright_quaternion: np.ndarray


def _joint_names(model: Any) -> list[str]:
    names = []
    for index in range(int(model.njnt)):
        name = model.joint_id2name(index)
        if name is not None:
            names.append(str(name))
    return names


def _upright_quaternion(quaternion: Any) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    yaw_components = np.asarray([quaternion[0], quaternion[3]], dtype=np.float64)
    norm = float(np.linalg.norm(yaw_components))
    if norm == 0.0:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray([yaw_components[0] / norm, 0.0, 0.0, yaw_components[1] / norm], dtype=np.float64)


def _object_tilt_degrees(quaternion: Any) -> float:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    w, x, y, z = quaternion
    del w, z
    vertical_component = 1.0 - 2.0 * (x * x + y * y)
    return float(np.degrees(np.arccos(np.clip(abs(vertical_component), -1.0, 1.0))))


def _capture_libero_mem_stabilization_plan(env: Any) -> ObjectStabilizationPlan | None:
    sim = getattr(env, "sim", None)
    if sim is None:
        raise TypeError("LIBERO-Mem object stabilization requires env.sim")
    joint_names = _joint_names(sim.model)
    if LIBERO_MEM_BOTTLE_JOINT not in joint_names:
        return None
    plate_joints = [
        name
        for name in joint_names
        if name.startswith(LIBERO_MEM_PLATE_JOINT_PREFIX) and name.endswith("_joint0")
    ]
    if not plate_joints:
        raise ValueError("LIBERO-Mem scene contains a wine bottle but no plate free joint")
    bottle_qpos = np.asarray(sim.data.get_joint_qpos(LIBERO_MEM_BOTTLE_JOINT), dtype=np.float64).copy()
    support_joint = min(
        plate_joints,
        key=lambda name: float(
            np.linalg.norm(bottle_qpos[:2] - np.asarray(sim.data.get_joint_qpos(name), dtype=np.float64)[:2])
        ),
    )
    support_qpos = np.asarray(sim.data.get_joint_qpos(support_joint), dtype=np.float64)
    return ObjectStabilizationPlan(
        object_joint=LIBERO_MEM_BOTTLE_JOINT,
        support_joint=support_joint,
        support_xy_offset=(bottle_qpos[:2] - support_qpos[:2]).copy(),
        upright_quaternion=_upright_quaternion(bottle_qpos[3:7]),
    )


def _stabilize_libero_mem_object(env: Any, plan: ObjectStabilizationPlan | None) -> bool:
    if plan is None:
        return False
    sim = env.sim
    object_qpos = np.asarray(sim.data.get_joint_qpos(plan.object_joint), dtype=np.float64)
    if _object_tilt_degrees(object_qpos[3:7]) <= LIBERO_MEM_BOTTLE_MAX_TILT_DEGREES:
        return False
    support_qpos = np.asarray(sim.data.get_joint_qpos(plan.support_joint), dtype=np.float64)
    stabilized_qpos = np.concatenate(
        (
            support_qpos[:2] + plan.support_xy_offset,
            np.asarray([support_qpos[2] + LIBERO_MEM_BOTTLE_PLACEMENT_HEIGHT]),
            plan.upright_quaternion,
        )
    )
    sim.data.set_joint_qpos(plan.object_joint, stabilized_qpos)
    sim.data.set_joint_qvel(plan.object_joint, np.zeros(6, dtype=np.float64))
    sim.forward()
    return True


@dataclasses.dataclass(frozen=True)
class PlainEnvSnapshot:
    observation: Any
    success: bool
    satisfied_subgoals: list[Any]
    overshot: bool
    atomic_predicates: dict[str, bool]


class PlainLiberoEnvAdapter:
    """Adapter for LIBERO suites without LIBERO-Mem progress APIs."""

    def __init__(self, env: Any):
        self._env = env

    def reset(self) -> PlainEnvSnapshot:
        return self.snapshot(self._env.reset(), success=False)

    def step(self, action: Any) -> PlainEnvSnapshot:
        observation, reward, done, info = self._env.step(action)
        success = bool(done)
        if isinstance(info, dict):
            success = success or bool(info.get("success", False))
        return self.snapshot(observation, success=success)

    def snapshot(self, observation: Any, *, success: bool = False) -> PlainEnvSnapshot:
        return PlainEnvSnapshot(
            observation=observation,
            success=bool(success),
            satisfied_subgoals=[],
            overshot=False,
            atomic_predicates={},
        )


class EpisodeSetupAdapter:
    """Resets a LIBERO task to a fixed init state and stabilizes fragile objects."""

    def __init__(
        self,
        base_adapter: Any,
        *,
        env: Any,
        init_state: Any,
        num_steps_wait: int,
        stabilize_libero_mem_objects: bool = False,
    ):
        self._base_adapter = base_adapter
        self._env = env
        self._init_state = init_state
        self._num_steps_wait = int(num_steps_wait)
        self._stabilize_libero_mem_objects = bool(stabilize_libero_mem_objects)

    def reset(self) -> Any:
        if not hasattr(self._env, "set_init_state"):
            raise TypeError(
                "LIBERO evaluation environment must provide set_init_state(init_state); "
                f"got {type(self._env).__module__}.{type(self._env).__qualname__}"
            )
        self._base_adapter.reset()
        observation = self._env.set_init_state(self._init_state)
        stabilization_plan = (
            _capture_libero_mem_stabilization_plan(self._env) if self._stabilize_libero_mem_objects else None
        )
        for _ in range(self._num_steps_wait):
            observation, _, _, _ = self._env.step(LIBERO_DUMMY_ACTION)
        if _stabilize_libero_mem_object(self._env, stabilization_plan):
            for _ in range(self._num_steps_wait):
                observation, _, _, _ = self._env.step(LIBERO_DUMMY_ACTION)
        return self._snapshot(observation, success=False)

    def step(self, action: Any) -> Any:
        return self._base_adapter.step(action)

    def advance(self, action: Any) -> Any:
        if hasattr(self._base_adapter, "advance"):
            return self._base_adapter.advance(action)
        return self._base_adapter.step(action)

    def _snapshot(self, observation: Any, *, success: bool) -> Any:
        if hasattr(self._base_adapter, "snapshot"):
            return self._base_adapter.snapshot(observation, success=success)
        return PlainEnvSnapshot(
            observation=observation,
            success=bool(success),
            satisfied_subgoals=[],
            overshot=False,
            atomic_predicates={},
        )


@dataclasses.dataclass(frozen=True)
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    task_suite_name: str = "libero_mem"
    replan_steps: int = 5
    max_steps: int | None = None
    stable_frames: int = 6
    train_seed: int = 0
    rollout_seed: int = 0
    num_trials_per_task: int = 1
    num_steps_wait: int = LIBERO_MEM_SETTLE_STEPS
    stabilize_libero_mem_objects: bool = True
    task_ids: tuple[int, ...] | None = None
    results_path: str = "data/libero_mem/rollouts.jsonl"
    checkpoint_path: str | None = None
    video_out_path: str | None = None
    handoff_ratio: float = 0.2
    history_condition: str = "futuremamba"


def build_episode_log(
    *,
    config: dict[str, Any],
    checkpoint_checksum: str,
    task: str,
    task_family: str,
    memory_length: int,
    train_seed: int,
    rollout_seed: int,
    episode: int,
    metrics: _metrics.EpisodeMetrics,
    subgoal_events: list[dict[str, Any]],
    handoff_ratio: float,
    history_condition: str,
    timing: dict[str, Any],
    video_path: str | None,
) -> dict[str, Any]:
    return {
        "task": str(task),
        "task_family": str(task_family),
        "memory_length": int(memory_length),
        "train_seed": int(train_seed),
        "rollout_seed": int(rollout_seed),
        "episode": int(episode),
        "success": bool(metrics.success),
        "completed_subgoals": int(metrics.completed_subgoals),
        "total_subgoals": int(metrics.total_subgoals),
        "redundant_chunks": int(metrics.redundant_chunks),
        "decidable_chunks": int(metrics.decidable_chunks),
        "overshot": bool(metrics.overshot),
        "steps": int(metrics.steps),
        "subgoal_events": [dict(event) for event in subgoal_events],
        "handoff_ratio": float(handoff_ratio),
        "history_condition": str(history_condition),
        "checkpoint_checksum": str(checkpoint_checksum),
        "config": dict(config),
        "timing": dict(timing),
        "video_path": video_path,
    }


def write_episode_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    _metrics.append_jsonl(path, record)


def run_single_episode(
    *,
    adapter: Any,
    client: Any,
    task: str,
    task_family: str,
    memory_length: int,
    goals: Any,
    config: dict[str, Any],
    checkpoint_checksum: str,
    train_seed: int,
    rollout_seed: int,
    episode: int,
    handoff_ratio: float,
    history_condition: str,
    max_steps: int,
    replan_steps: int,
    results_path: str | Path,
    stable_frames: int = 6,
    video_path: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    monitor = _metrics.SymbolicEventMonitor(goals=goals, stable_frames=stable_frames)
    last_executed_chunk: list[Any] = []
    subgoal_events: list[dict[str, Any]] = []
    snapshot = None
    steps = 0
    success = False
    overshot = False
    query_count = 0
    try:
        if hasattr(client, "reset"):
            client.reset()
        snapshot = adapter.reset()
        previous_satisfied = list(getattr(snapshot, "satisfied_subgoals", []) or [])
        while steps < max_steps:
            query_index = query_count
            actions = _query_client(
                client,
                observation=getattr(snapshot, "observation", None),
                task=task,
                executed_prefix=last_executed_chunk,
                query_index=query_index,
            )
            query_count += 1
            if len(actions) < replan_steps:
                raise ValueError(
                    f"replan_steps={replan_steps} requires at least {replan_steps} actions, got {len(actions)}"
                )
            current_executed_chunk: list[Any] = []
            for action in actions[:replan_steps]:
                if steps >= max_steps:
                    break
                before = previous_satisfied
                snapshot = _advance_adapter(adapter, action)
                steps += 1
                current_executed_chunk.append(action)
                after = list(getattr(snapshot, "satisfied_subgoals", []) or [])
                subgoal_events.extend(
                    monitor.observe(
                        frame=steps,
                        query_index=query_index,
                        atomic_predicates=getattr(snapshot, "atomic_predicates", {}) or {},
                        satisfied_before=before,
                        satisfied_after=after,
                    )
                )
                previous_satisfied = after
                success = bool(getattr(snapshot, "success", False))
                overshot = bool(getattr(snapshot, "overshot", False))
                if success:
                    break
            last_executed_chunk = current_executed_chunk
            if success:
                break
    finally:
        final_snapshot = snapshot
        episode_metrics = monitor.finish(success=success, overshot=overshot, steps=steps)
        if final_snapshot is not None:
            episode_metrics = dataclasses.replace(
                episode_metrics,
                success=bool(getattr(final_snapshot, "success", episode_metrics.success)),
                overshot=bool(getattr(final_snapshot, "overshot", episode_metrics.overshot)),
            )
        record = build_episode_log(
            config=config,
            checkpoint_checksum=checkpoint_checksum,
            task=task,
            task_family=task_family,
            memory_length=memory_length,
            train_seed=train_seed,
            rollout_seed=rollout_seed,
            episode=episode,
            metrics=episode_metrics,
            subgoal_events=subgoal_events,
            handoff_ratio=handoff_ratio,
            history_condition=history_condition,
            timing={"episode_sec": time.perf_counter() - started},
            video_path=video_path,
        )
        write_episode_jsonl(results_path, record)
    return record


def eval_libero_mem(args: Args, *, episode_runner=run_single_episode) -> list[dict[str, Any]]:
    _imports = _load_real_dependencies()
    env_adapter_module = _imports["env_adapter"]
    websocket_client_policy = _imports["websocket_client_policy"]
    benchmark = _imports["benchmark"]
    get_libero_path = _imports["get_libero_path"]
    offscreen_env = _imports["OffScreenRenderEnv"]

    max_steps = _select_max_steps(args.task_suite_name, args.max_steps)
    run_config = dataclasses.asdict(args)
    run_config["max_steps"] = max_steps

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    task_ids = _select_task_ids(args.task_ids, task_suite.n_tasks)
    records: list[dict[str, Any]] = []
    for task_id in task_ids:
        task_object = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        task_text = str(getattr(task_object, "language", task_object))
        task_family = str(getattr(task_object, "task_family", args.task_suite_name))
        memory_length = int(getattr(task_object, "memory_length", 0))
        is_plain_libero = args.task_suite_name == "libero_10"
        goals = {"Sequence": []} if is_plain_libero else getattr(task_object, "goals", {"Sequence": []})
        task_bddl = Path(get_libero_path("bddl_files")) / task_object.problem_folder / task_object.bddl_file
        env = offscreen_env(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
        if hasattr(env, "seed"):
            env.seed(args.rollout_seed)
        if not hasattr(env, "set_init_state"):
            raise TypeError(
                "OffScreenRenderEnv must expose set_init_state; "
                f"got {type(env).__module__}.{type(env).__qualname__}"
            )
        base_adapter = PlainLiberoEnvAdapter(env) if is_plain_libero else env_adapter_module.LiberoMemEnvAdapter(env, task_text=task_text)
        for episode_index in range(args.num_trials_per_task):
            init_state = _trial_init_state(initial_states, episode_index, task_id=task_id)
            adapter = EpisodeSetupAdapter(
                base_adapter,
                env=env,
                init_state=init_state,
                num_steps_wait=args.num_steps_wait,
                stabilize_libero_mem_objects=(
                    args.task_suite_name == "libero_mem" and args.stabilize_libero_mem_objects
                ),
            )
            records.append(
                episode_runner(
                    adapter=adapter,
                    client=client,
                    task=task_text,
                    task_family=task_family,
                    memory_length=memory_length,
                    goals=goals,
                    config=run_config,
                    checkpoint_checksum=checkpoint_checksum(args.checkpoint_path),
                    train_seed=args.train_seed,
                    rollout_seed=args.rollout_seed,
                    episode=episode_index,
                    handoff_ratio=args.handoff_ratio,
                    history_condition=args.history_condition,
                    max_steps=max_steps,
                    replan_steps=args.replan_steps,
                    results_path=args.results_path,
                    stable_frames=args.stable_frames,
                    video_path=None,
                )
            )
    return records


def _select_task_ids(task_ids: tuple[int, ...] | list[int] | None, task_count: int) -> list[int]:
    if task_ids is None:
        return list(range(task_count))
    selected = [int(task_id) for task_id in task_ids]
    invalid = [task_id for task_id in selected if task_id < 0 or task_id >= task_count]
    if invalid:
        raise ValueError(f"task_ids contains out-of-range ids {invalid} for suite with {task_count} tasks")
    return selected

def _select_max_steps(task_suite_name: str, max_steps: int | None) -> int:
    if max_steps is not None:
        return int(max_steps)
    try:
        return LIBERO_SUITE_MAX_STEPS[task_suite_name]
    except KeyError as error:
        raise ValueError(f"Unknown task suite: {task_suite_name}") from error


def _trial_init_state(initial_states: Any, episode_index: int, *, task_id: int) -> Any:
    try:
        return initial_states[episode_index]
    except IndexError as error:
        raise ValueError(
            f"task {task_id} has no init state for trial {episode_index}; "
            f"received {len(initial_states)} initial states"
        ) from error



def checkpoint_checksum(path: str | Path | None) -> str:
    if path is None:
        return "unavailable"
    checkpoint_path = Path(path)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"

def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(float(den), 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


def _libero_infer_payload(*, observation: Any, task: str, executed_prefix: list[Any]) -> dict[str, Any]:
    if not isinstance(observation, dict):
        raise TypeError("LIBERO infer observation must be a dict")

    image = np.ascontiguousarray(observation["agentview_image"][::-1, :])
    wrist_image = np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, :])
    image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(image, LIBERO_INFERENCE_IMAGE_SIZE, LIBERO_INFERENCE_IMAGE_SIZE)
    )
    wrist_image = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(wrist_image, LIBERO_INFERENCE_IMAGE_SIZE, LIBERO_INFERENCE_IMAGE_SIZE)
    )

    state = np.concatenate(
        (
            observation["robot0_eef_pos"],
            _quat2axisangle(observation["robot0_eef_quat"]),
            observation["robot0_gripper_qpos"],
        )
    )

    return {
        "observation/image": image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
        "prompt": str(task),
        "executed_actions": np.asarray(executed_prefix, dtype=np.float32).reshape((-1, 7)),
    }


def _query_client(client: Any, *, observation: Any, task: str, executed_prefix: list[Any], query_index: int) -> list[Any]:
    if hasattr(client, "query"):
        return list(
            client.query(
                observation=observation,
                task=task,
                executed_prefix=list(executed_prefix),
                query_index=query_index,
            )
        )
    if hasattr(client, "infer"):
        response = client.infer(
            _libero_infer_payload(observation=observation, task=task, executed_prefix=list(executed_prefix))
        )
        return list(response["actions"])
    raise TypeError("client must provide query(...) or infer(...)")


def _advance_adapter(adapter: Any, action: Any) -> Any:
    if hasattr(adapter, "advance"):
        return adapter.advance(action)
    if hasattr(adapter, "step"):
        return adapter.step(action)
    raise TypeError("adapter must provide advance(action) or step(action)")


def _load_real_dependencies() -> dict[str, Any]:
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from openpi_client import websocket_client_policy

    try:
        from . import env_adapter
    except ImportError:
        spec = importlib.util.spec_from_file_location("libero_mem_env_adapter", Path(__file__).with_name("env_adapter.py"))
        env_adapter = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(env_adapter)
    return {
        "benchmark": benchmark,
        "get_libero_path": get_libero_path,
        "OffScreenRenderEnv": OffScreenRenderEnv,
        "websocket_client_policy": websocket_client_policy,
        "env_adapter": env_adapter,
    }


if __name__ == "__main__":
    import tyro

    tyro.cli(eval_libero_mem)
