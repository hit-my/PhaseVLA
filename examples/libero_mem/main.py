from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import time
from pathlib import Path
from typing import Any

try:
    import metrics as _metrics
except ModuleNotFoundError:
    _METRICS_SPEC = importlib.util.spec_from_file_location("libero_mem_metrics", Path(__file__).with_name("metrics.py"))
    _metrics = importlib.util.module_from_spec(_METRICS_SPEC)
    assert _METRICS_SPEC.loader is not None
    _METRICS_SPEC.loader.exec_module(_metrics)



LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]


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
    """Resets a LIBERO task to a fixed init state and waits before metric steps."""

    def __init__(self, base_adapter: Any, *, env: Any, init_state: Any, num_steps_wait: int):
        self._base_adapter = base_adapter
        self._env = env
        self._init_state = init_state
        self._num_steps_wait = int(num_steps_wait)

    def reset(self) -> Any:
        snapshot = self._base_adapter.reset()
        observation = getattr(snapshot, "observation", None)
        if hasattr(self._env, "set_init_state"):
            observation = self._env.set_init_state(self._init_state)
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
    max_steps: int = 520
    stable_frames: int = 6
    train_seed: int = 0
    rollout_seed: int = 0
    num_trials_per_task: int = 1
    num_steps_wait: int = 10
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
        base_adapter = PlainLiberoEnvAdapter(env) if is_plain_libero else env_adapter_module.LiberoMemEnvAdapter(env, task_text=task_text)
        for episode_index in range(args.num_trials_per_task):
            init_state = _trial_init_state(initial_states, episode_index, task_id=task_id)
            adapter = EpisodeSetupAdapter(
                base_adapter,
                env=env,
                init_state=init_state,
                num_steps_wait=args.num_steps_wait,
            )
            records.append(
                episode_runner(
                    adapter=adapter,
                    client=client,
                    task=task_text,
                    task_family=task_family,
                    memory_length=memory_length,
                    goals=goals,
                    config=dataclasses.asdict(args),
                    checkpoint_checksum=checkpoint_checksum(args.checkpoint_path),
                    train_seed=args.train_seed,
                    rollout_seed=args.rollout_seed,
                    episode=episode_index,
                    handoff_ratio=args.handoff_ratio,
                    history_condition=args.history_condition,
                    max_steps=args.max_steps,
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
            {
                "observation": observation,
                "prompt": task,
                "executed_prefix": list(executed_prefix),
                "query_index": query_index,
            }
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
