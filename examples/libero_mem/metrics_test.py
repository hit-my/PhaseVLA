from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
import numpy as np
import sys


_MODULE_DIR = pathlib.Path(__file__).parent

_METRICS_SPEC = importlib.util.spec_from_file_location("libero_mem_metrics", _MODULE_DIR / "metrics.py")
metrics = importlib.util.module_from_spec(_METRICS_SPEC)
sys.modules[_METRICS_SPEC.name] = metrics
_METRICS_SPEC.loader.exec_module(metrics)

_MAIN_SPEC = importlib.util.spec_from_file_location("libero_mem_main", _MODULE_DIR / "main.py")
main = importlib.util.module_from_spec(_MAIN_SPEC)
sys.modules[_MAIN_SPEC.name] = main
_MAIN_SPEC.loader.exec_module(main)


@dataclasses.dataclass(frozen=True)
class FakeSnapshot:
    success: bool
    satisfied_subgoals: list[str]
    overshot: bool
    atomic_predicates: dict[str, bool]
    observation: dict[str, int] | None = None


class FakeAdapter:
    def __init__(self, snapshots: list[FakeSnapshot]):
        self._snapshots = list(snapshots)
        self.reset_calls = 0
        self.advance_calls = 0
        self.step_actions = []

    def reset(self):
        self.reset_calls += 1
        return self._snapshots[0]

    def advance(self, action):
        self.advance_calls += 1
        self.step_actions.append(action)
        return self._snapshots[self.advance_calls]


class FakeClient:
    def __init__(self, chunks: list[list[str]]):
        self._chunks = list(chunks)
        self.reset_calls = 0
        self.query_calls = []

    def reset(self):
        self.reset_calls += 1

    def query(self, *, observation, task, executed_prefix, query_index):
        self.query_calls.append(
            {
                "observation": observation,
                "task": task,
                "executed_prefix": list(executed_prefix),
                "query_index": query_index,
            }
        )
        return list(self._chunks[query_index])


class ThreeQueryClient(FakeClient):
    pass

class InferOnlyClient:
    def __init__(self, chunks):
        self._chunks = [np.asarray(chunk, dtype=np.float32) for chunk in chunks]
        self.reset_calls = 0
        self.infer_calls = []

    def reset(self):
        self.reset_calls += 1

    def infer(self, payload):
        self.infer_calls.append(dict(payload))
        return {"actions": self._chunks[len(self.infer_calls) - 1]}


def libero_observation(frame):
    image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    wrist_image = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3) + np.uint8(1)
    return {
        "agentview_image": image + np.uint8(frame),
        "robot0_eye_in_hand_image": wrist_image + np.uint8(frame),
        "robot0_eef_pos": np.asarray([1.0, 2.0, 3.0], dtype=np.float32) + frame,
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.asarray([0.25], dtype=np.float32) + frame,
    }


def assert_infer_payload(payload, *, observation, prompt, executed_actions):
    assert set(payload) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state",
        "prompt",
        "executed_actions",
    }
    assert payload["prompt"] == prompt
    assert payload["observation/image"].shape == (224, 224, 3)
    assert payload["observation/image"].dtype == np.uint8
    np.testing.assert_array_equal(
        payload["observation/image"],
        main.image_tools.convert_to_uint8(
            main.image_tools.resize_with_pad(np.ascontiguousarray(observation["agentview_image"][::-1, ::-1]), 224, 224)
        ),
    )
    assert payload["observation/wrist_image"].shape == (224, 224, 3)
    assert payload["observation/wrist_image"].dtype == np.uint8
    np.testing.assert_array_equal(
        payload["observation/wrist_image"],
        main.image_tools.convert_to_uint8(
            main.image_tools.resize_with_pad(
                np.ascontiguousarray(observation["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224
            )
        ),
    )
    np.testing.assert_allclose(
        payload["observation/state"],
        np.asarray([*observation["robot0_eef_pos"], 0.0, 0.0, 0.0, *observation["robot0_gripper_qpos"]]),
    )
    assert payload["executed_actions"].dtype == np.float32
    assert payload["executed_actions"].shape == executed_actions.shape
    np.testing.assert_allclose(payload["executed_actions"], executed_actions)


class FakeLibero10Env:
    def __init__(self):
        self.reset_calls = 0
        self.step_actions = []
        self.success_after = 1

    def reset(self):
        self.reset_calls += 1
        return {"frame": "reset"}

    def step(self, action):
        self.step_actions.append(action)
        success = len(self.step_actions) >= self.success_after and action != "wait"
        return {"frame": len(self.step_actions)}, 1.0 if success else 0.0, success, {"success": success}


class FakeInitStateEnv:
    def __init__(self):
        self.seed_calls = []
        self.reset_calls = 0
        self.set_init_state_calls = []
        self.step_actions = []

    def seed(self, seed):
        self.seed_calls.append(seed)

    def reset(self):
        self.reset_calls += 1
        return {"phase": "reset"}

    def set_init_state(self, state):
        self.set_init_state_calls.append(state)
        return {"phase": "init", "state": state}

    def step(self, action):
        self.step_actions.append(action)
        success = action == "act"
        return {"phase": "step", "action": action}, 1.0 if success else 0.0, success, {"success": success}


class FakeTask:
    problem_folder = "folder"
    bddl_file = "task.bddl"
    task_family = "family"
    memory_length = 0
    goals = {"Sequence": ["done"]}

    def __init__(self, language):
        self.language = language


class FakeTaskSuite:
    n_tasks = 2

    def __init__(self):
        self.init_state_calls = []

    def get_task(self, task_id):
        return FakeTask(f"task-{task_id}")

    def get_task_init_states(self, task_id):
        self.init_state_calls.append(task_id)
        return [f"state-{task_id}-0", f"state-{task_id}-1"]


class FakeBenchmark:
    def __init__(self, suite):
        self.suite = suite

    def get_benchmark_dict(self):
        return {"libero_10": lambda: self.suite, "libero_mem": lambda: self.suite}


class FakeWebsocketModule:
    class WebsocketClientPolicy:
        def __init__(self, host, port):
            self.host = host
            self.port = port
            self.reset_calls = 0

        def reset(self):
            self.reset_calls += 1


class RecordingMemAdapter:
    constructed = []

    def __init__(self, env, *, task_text):
        RecordingMemAdapter.constructed.append((env, task_text))
        self.env = env

    def reset(self):
        observation = self.env.reset()
        return FakeSnapshot(False, [], False, {}, observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return FakeSnapshot(bool(info.get("success", done)), ["done"] if info.get("success", done) else [], False, {"done": bool(info.get("success", done))}, observation)


class FakeEnvAdapterModule:
    LiberoMemEnvAdapter = RecordingMemAdapter


def install_fake_dependencies(monkeypatch, *, suite, envs):
    env_iter = iter(envs)

    def fake_offscreen_env(**kwargs):
        env = next(env_iter)
        env.kwargs = kwargs
        return env

    monkeypatch.setattr(
        main,
        "_load_real_dependencies",
        lambda: {
            "env_adapter": FakeEnvAdapterModule,
            "websocket_client_policy": FakeWebsocketModule,
            "benchmark": FakeBenchmark(suite),
            "get_libero_path": lambda name: "/tmp/bddl",
            "OffScreenRenderEnv": fake_offscreen_env,
        },
    )


def successful_one_step_runner(**kwargs):
    adapter = kwargs["adapter"]
    client = kwargs["client"]
    if hasattr(client, "reset"):
        client.reset()
    snapshot = adapter.reset()
    snapshot = adapter.step("act")
    record = main.build_episode_log(
        config=kwargs["config"],
        checkpoint_checksum=kwargs["checkpoint_checksum"],
        task=kwargs["task"],
        task_family=kwargs["task_family"],
        memory_length=kwargs["memory_length"],
        train_seed=kwargs["train_seed"],
        rollout_seed=kwargs["rollout_seed"],
        episode=kwargs["episode"],
        metrics=metrics.EpisodeMetrics(
            success=bool(snapshot.success),
            completed_subgoals=len(snapshot.satisfied_subgoals),
            total_subgoals=len(snapshot.satisfied_subgoals),
            redundant_chunks=0,
            decidable_chunks=0,
            overshot=bool(snapshot.overshot),
            steps=1,
        ),
        subgoal_events=[],
        handoff_ratio=kwargs["handoff_ratio"],
        history_condition=kwargs["history_condition"],
        timing={"episode_sec": 0.0},
        video_path=kwargs["video_path"],
    )
    main.write_episode_jsonl(kwargs["results_path"], record)
    return record


def test_run_single_episode_sends_only_last_executed_chunk_and_independent_query_count(tmp_path):
    goals = {"Sequence": ["a", "b", "c"]}
    adapter = FakeAdapter(
        [
            FakeSnapshot(False, [], False, {}, {"frame": 0}),
            FakeSnapshot(False, [], False, {}, {"frame": 1}),
            FakeSnapshot(False, ["a"], False, {"a": True}, {"frame": 2}),
            FakeSnapshot(False, ["a"], False, {"b": True}, {"frame": 3}),
            FakeSnapshot(False, ["a", "b"], False, {"b": True}, {"frame": 4}),
            FakeSnapshot(True, ["a", "b", "c"], False, {"c": True}, {"frame": 5}),
        ]
    )
    client = ThreeQueryClient([["a0", "a1"], ["b0", "b1"], ["c0", "c1"]])

    main.run_single_episode(
        adapter=adapter,
        client=client,
        task="three chunks",
        task_family="spatial",
        memory_length=3,
        goals=goals,
        config={"task_suite_name": "libero_mem", "replan_steps": 2},
        checkpoint_checksum="sha256:def",
        train_seed=5,
        rollout_seed=13,
        episode=0,
        handoff_ratio=0.2,
        history_condition="futuremamba",
        max_steps=5,
        replan_steps=2,
        results_path=tmp_path / "episodes.jsonl",
        stable_frames=1,
        video_path=None,
    )

    assert [call["query_index"] for call in client.query_calls] == [0, 1, 2]
    assert [call["executed_prefix"] for call in client.query_calls] == [[], ["a0", "a1"], ["b0", "b1"]]

def test_infer_only_client_receives_flat_libero_payloads_with_previous_executed_actions(tmp_path):
    action_a0 = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    action_a1 = np.asarray([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], dtype=np.float32)
    action_b0 = np.asarray([2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7], dtype=np.float32)
    action_b1 = np.asarray([3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7], dtype=np.float32)
    action_c0 = np.asarray([4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7], dtype=np.float32)
    action_c1 = np.asarray([5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7], dtype=np.float32)
    observations = [libero_observation(frame) for frame in range(6)]
    adapter = FakeAdapter(
        [
            FakeSnapshot(False, [], False, {}, observations[0]),
            FakeSnapshot(False, [], False, {}, observations[1]),
            FakeSnapshot(False, ["a"], False, {"a": True}, observations[2]),
            FakeSnapshot(False, ["a"], False, {"b": True}, observations[3]),
            FakeSnapshot(False, ["a", "b"], False, {"b": True}, observations[4]),
            FakeSnapshot(True, ["a", "b", "c"], False, {"c": True}, observations[5]),
        ]
    )
    client = InferOnlyClient([[action_a0, action_a1], [action_b0, action_b1], [action_c0, action_c1]])

    main.run_single_episode(
        adapter=adapter,
        client=client,
        task="three chunks",
        task_family="spatial",
        memory_length=3,
        goals={"Sequence": ["a", "b", "c"]},
        config={"task_suite_name": "libero_mem", "replan_steps": 2},
        checkpoint_checksum="sha256:def",
        train_seed=5,
        rollout_seed=13,
        episode=0,
        handoff_ratio=0.2,
        history_condition="futuremamba",
        max_steps=5,
        replan_steps=2,
        results_path=tmp_path / "episodes.jsonl",
        stable_frames=1,
        video_path=None,
    )

    assert client.reset_calls == 1
    assert len(client.infer_calls) == 3
    assert_infer_payload(
        client.infer_calls[0],
        observation=observations[0],
        prompt="three chunks",
        executed_actions=np.empty((0, 7), dtype=np.float32),
    )
    assert_infer_payload(
        client.infer_calls[1],
        observation=observations[2],
        prompt="three chunks",
        executed_actions=np.asarray([action_a0, action_a1], dtype=np.float32),
    )
    assert_infer_payload(
        client.infer_calls[2],
        observation=observations[4],
        prompt="three chunks",
        executed_actions=np.asarray([action_b0, action_b1], dtype=np.float32),
    )


def test_eval_uses_suite_specific_max_steps_when_args_keep_default(monkeypatch, tmp_path):
    records = []

    def recording_runner(**kwargs):
        records.append({"task": kwargs["task"], "max_steps": kwargs["max_steps"]})
        return {"task": kwargs["task"], "max_steps": kwargs["max_steps"]}

    libero_10_suite = FakeTaskSuite()
    install_fake_dependencies(monkeypatch, suite=libero_10_suite, envs=[FakeLibero10Env()])
    main.eval_libero_mem(
        main.Args(
            task_suite_name="libero_10",
            task_ids=[0],
            num_trials_per_task=1,
            num_steps_wait=0,
            results_path=str(tmp_path / "libero_10.jsonl"),
        ),
        episode_runner=recording_runner,
    )

    libero_mem_suite = FakeTaskSuite()
    install_fake_dependencies(monkeypatch, suite=libero_mem_suite, envs=[FakeInitStateEnv()])
    main.eval_libero_mem(
        main.Args(
            task_suite_name="libero_mem",
            task_ids=[0],
            num_trials_per_task=1,
            num_steps_wait=0,
            results_path=str(tmp_path / "libero_mem.jsonl"),
        ),
        episode_runner=recording_runner,
    )

    assert records == [
        {"task": "task-0", "max_steps": 520},
        {"task": "task-0", "max_steps": 600},
    ]

def test_libero_10_uses_plain_adapter_without_progress_api_and_empty_events(monkeypatch, tmp_path):
    suite = FakeTaskSuite()
    env = FakeLibero10Env()
    install_fake_dependencies(monkeypatch, suite=suite, envs=[env])
    RecordingMemAdapter.constructed = []

    records = main.eval_libero_mem(
        main.Args(
            task_suite_name="libero_10",
            num_trials_per_task=1,
            task_ids=[0],
            results_path=str(tmp_path / "rollouts.jsonl"),
            num_steps_wait=0,
        ),
        episode_runner=successful_one_step_runner,
    )

    assert RecordingMemAdapter.constructed == []
    assert records[0]["subgoal_events"] == []
    assert records[0]["redundant_chunks"] == 0
    assert records[0]["decidable_chunks"] == 0


def test_eval_uses_task_init_states_and_waits_before_eval_steps(monkeypatch, tmp_path):
    suite = FakeTaskSuite()
    env = FakeInitStateEnv()
    install_fake_dependencies(monkeypatch, suite=suite, envs=[env, FakeInitStateEnv()])
    RecordingMemAdapter.constructed = []

    records = main.eval_libero_mem(
        main.Args(
            task_suite_name="libero_mem",
            task_ids=[1],
            num_trials_per_task=2,
            num_steps_wait=2,
            rollout_seed=99,
            results_path=str(tmp_path / "rollouts.jsonl"),
        ),
        episode_runner=successful_one_step_runner,
    )

    assert suite.init_state_calls == [1]
    assert env.seed_calls == [99]
    assert env.reset_calls == 2
    assert env.set_init_state_calls == ["state-1-0", "state-1-1"]
    assert env.step_actions == [main.LIBERO_DUMMY_ACTION, main.LIBERO_DUMMY_ACTION, "act", main.LIBERO_DUMMY_ACTION, main.LIBERO_DUMMY_ACTION, "act"]
    assert [record["episode"] for record in records] == [0, 1]
    assert all(record["steps"] == 1 for record in records)


def test_eval_filters_task_ids_and_rejects_out_of_range(monkeypatch, tmp_path):
    suite = FakeTaskSuite()
    install_fake_dependencies(monkeypatch, suite=suite, envs=[FakeLibero10Env()])

    records = main.eval_libero_mem(
        main.Args(
            task_suite_name="libero_10",
            task_ids=[1],
            num_trials_per_task=1,
            num_steps_wait=0,
            results_path=str(tmp_path / "selected.jsonl"),
        ),
        episode_runner=successful_one_step_runner,
    )
    assert [record["task"] for record in records] == ["task-1"]

    try:
        main.eval_libero_mem(
            main.Args(
                task_suite_name="libero_10",
                task_ids=[2],
                results_path=str(tmp_path / "bad.jsonl"),
            ),
            episode_runner=successful_one_step_runner,
        )
    except ValueError as error:
        assert "task_ids" in str(error)
        assert "2" in str(error)
    else:
        raise AssertionError("out-of-range task_ids must raise ValueError")


def test_aggregate_jsonl_reports_required_breakdowns_integrity_bootstrap_and_retention(tmp_path):
    rows = [
        {"task": "task-a", "task_family": "spatial", "memory_length": 1, "train_seed": 7, "rollout_seed": 0, "episode": 0, "success": True, "steps": 3, "overshot": False, "redundant_chunks": 0, "decidable_chunks": 1, "subgoal_events": []},
        {"task": "task-a", "task_family": "spatial", "memory_length": 1, "train_seed": 7, "rollout_seed": 1, "episode": 1, "success": False, "steps": 5, "overshot": True, "redundant_chunks": 1, "decidable_chunks": 1, "subgoal_events": [{"reason": "redundant"}]},
        {"task": "task-b", "task_family": "semantic", "memory_length": 3, "train_seed": 7, "rollout_seed": 0, "episode": 0, "success": True, "steps": 4, "overshot": False, "redundant_chunks": 0, "decidable_chunks": 1, "subgoal_events": []},
    ]
    baseline_rows = [
        {"task": "task-a", "task_suite_name": "libero_10", "train_seed": 7, "success": False},
        {"task": "task-b", "task_suite_name": "libero_10", "train_seed": 7, "success": True},
    ]
    path = tmp_path / "rollouts.jsonl"
    baseline_path = tmp_path / "baseline.jsonl"
    metrics.write_jsonl(path, rows)
    metrics.write_jsonl(baseline_path, baseline_rows)

    report = metrics.aggregate_jsonl(path, baseline=baseline_path, expected_trials_per_task=2)

    assert [bucket["key"] for bucket in report["by_task"]] == ["task-a", "task-b"]
    assert [bucket["key"] for bucket in report["by_memory_length"]] == ["1", "3"]
    assert report["temporal_scaling"] == report["by_memory_length"]
    assert report["trial_seed_integrity"] == {
        "expected_trials_per_task": 2,
        "complete": False,
        "missing": [{"task": "task-b", "missing_trial_keys": ["1"]}],
        "duplicates": [],
    }
    assert len(report["overall"]["success_bootstrap95"]) == 2
    assert report["capability_retention_abs"] == {"task-a|7": 0.5, "task-b|7": 0.0}


def test_symbolic_monitor_handles_repeated_signature_occurrences_in_order():
    monitor = metrics.SymbolicEventMonitor(goals={"Sequence": ["touch(block)", "touch(block)"]}, stable_frames=1)

    first = monitor.observe(
        frame=1,
        query_index=0,
        atomic_predicates={"touch(block)": True},
        satisfied_before=[],
        satisfied_after=["touch(block)"],
    )
    still_needed = monitor.observe(
        frame=2,
        query_index=1,
        atomic_predicates={"touch(block)": False},
        satisfied_before=["touch(block)"],
        satisfied_after=["touch(block)"],
    )
    second = monitor.observe(
        frame=3,
        query_index=1,
        atomic_predicates={"touch(block)": True},
        satisfied_before=["touch(block)"],
        satisfied_after=["touch(block)", "touch(block)"],
    )

    assert still_needed == []
    assert [(row["signature"], row["reason"]) for row in first + second] == [
        ("touch(block)", "progress"),
        ("touch(block)", "progress"),
    ]
    episode = monitor.finish(success=True, overshot=False, steps=3)
    assert episode.completed_subgoals == 2
    assert episode.total_subgoals == 2
    assert episode.redundant_chunks == 0
    assert episode.decidable_chunks == 2



def test_symbolic_monitor_counts_sequence_only_in_order_and_marks_redundant_after_completion():
    monitor = metrics.SymbolicEventMonitor(
        goals={"Sequence": ["open(cabinet)", "put(cup,cabinet)"]},
        stable_frames=2,
    )

    rows = []
    rows.extend(
        monitor.observe(
            frame=0,
            query_index=0,
            atomic_predicates={"put(cup,cabinet)": True},
            satisfied_before=[],
            satisfied_after=[],
        )
    )
    rows.extend(
        monitor.observe(
            frame=1,
            query_index=0,
            atomic_predicates={"put(cup,cabinet)": True},
            satisfied_before=[],
            satisfied_after=[],
        )
    )
    rows.extend(
        monitor.observe(
            frame=2,
            query_index=1,
            atomic_predicates={"put(cup,cabinet)": False, "open(cabinet)": True},
            satisfied_before=[],
            satisfied_after=[],
        )
    )
    rows.extend(
        monitor.observe(
            frame=3,
            query_index=1,
            atomic_predicates={"put(cup,cabinet)": False, "open(cabinet)": True},
            satisfied_before=[],
            satisfied_after=["open(cabinet)"],
        )
    )
    rows.extend(
        monitor.observe(
            frame=4,
            query_index=2,
            atomic_predicates={"open(cabinet)": True, "put(cup,cabinet)": True},
            satisfied_before=["open(cabinet)"],
            satisfied_after=["open(cabinet)"],
        )
    )
    rows.extend(
        monitor.observe(
            frame=5,
            query_index=2,
            atomic_predicates={"open(cabinet)": True, "put(cup,cabinet)": True},
            satisfied_before=["open(cabinet)"],
            satisfied_after=["open(cabinet)", "put(cup,cabinet)"],
        )
    )
    rows.extend(
        monitor.observe(
            frame=6,
            query_index=3,
            atomic_predicates={"open(cabinet)": False, "put(cup,cabinet)": True},
            satisfied_before=["open(cabinet)", "put(cup,cabinet)"],
            satisfied_after=["open(cabinet)", "put(cup,cabinet)"],
        )
    )
    rows.extend(
        monitor.observe(
            frame=7,
            query_index=3,
            atomic_predicates={"open(cabinet)": True, "put(cup,cabinet)": True},
            satisfied_before=["open(cabinet)", "put(cup,cabinet)"],
            satisfied_after=["open(cabinet)", "put(cup,cabinet)"],
        )
    )
    rows.extend(
        monitor.observe(
            frame=8,
            query_index=3,
            atomic_predicates={"open(cabinet)": True, "put(cup,cabinet)": True},
            satisfied_before=["open(cabinet)", "put(cup,cabinet)"],
            satisfied_after=["open(cabinet)", "put(cup,cabinet)"],
        )
    )

    assert [(row["signature"], row["reason"]) for row in rows] == [
        ("put(cup,cabinet)", "out_of_order"),
        ("open(cabinet)", "progress"),
        ("put(cup,cabinet)", "progress"),
        ("open(cabinet)", "redundant"),
    ]
    episode = monitor.finish(success=True, overshot=True, steps=9)
    assert episode == metrics.EpisodeMetrics(
        success=True,
        completed_subgoals=2,
        total_subgoals=2,
        redundant_chunks=1,
        decidable_chunks=3,
        overshot=True,
        steps=9,
    )


def test_symbolic_monitor_or_keeps_only_branch_compatible_with_completed_prefix_and_no_repeated_true_events():
    monitor = metrics.SymbolicEventMonitor(
        goals={"Sequence": [{"Or": [["open(drawer)", "place(block,drawer)"], ["open(cabinet)", "place(block,cabinet)"]]}]},
        stable_frames=2,
    )

    emitted = []
    emitted += monitor.observe(
        frame=0,
        query_index=0,
        atomic_predicates={"open(cabinet)": True},
        satisfied_before=[],
        satisfied_after=[],
    )
    emitted += monitor.observe(
        frame=1,
        query_index=0,
        atomic_predicates={"open(cabinet)": True},
        satisfied_before=[],
        satisfied_after=["open(cabinet)"],
    )
    for frame in range(2, 6):
        emitted += monitor.observe(
            frame=frame,
            query_index=0,
            atomic_predicates={"open(cabinet)": True},
            satisfied_before=["open(cabinet)"],
            satisfied_after=["open(cabinet)"],
        )
    emitted += monitor.observe(
        frame=6,
        query_index=1,
        atomic_predicates={"open(cabinet)": True, "place(block,drawer)": True},
        satisfied_before=["open(cabinet)"],
        satisfied_after=["open(cabinet)"],
    )
    emitted += monitor.observe(
        frame=7,
        query_index=1,
        atomic_predicates={"open(cabinet)": True, "place(block,drawer)": True},
        satisfied_before=["open(cabinet)"],
        satisfied_after=["open(cabinet)"],
    )
    emitted += monitor.observe(
        frame=8,
        query_index=2,
        atomic_predicates={"open(cabinet)": True, "place(block,cabinet)": True},
        satisfied_before=["open(cabinet)"],
        satisfied_after=["open(cabinet)"],
    )
    emitted += monitor.observe(
        frame=9,
        query_index=2,
        atomic_predicates={"open(cabinet)": True, "place(block,cabinet)": True},
        satisfied_before=["open(cabinet)"],
        satisfied_after=["open(cabinet)", "place(block,cabinet)"],
    )

    assert [(row["signature"], row["reason"]) for row in emitted] == [
        ("open(cabinet)", "progress"),
        ("place(block,drawer)", "incompatible_branch"),
        ("place(block,cabinet)", "progress"),
    ]
    episode = monitor.finish(success=False, overshot=False, steps=10)
    assert episode.completed_subgoals == 2
    assert episode.total_subgoals == 2
    assert episode.redundant_chunks == 0
    assert episode.decidable_chunks == 2


def test_wilson_interval_is_finite_at_extremes():
    zero = metrics.wilson_interval(successes=0, total=12)
    full = metrics.wilson_interval(successes=12, total=12)

    assert zero[0] == 0.0
    assert 0.0 < zero[1] < 1.0
    assert 0.0 < full[0] < 1.0
    assert full[1] == 1.0


def test_aggregate_jsonl_groups_by_task_family_memory_length_and_train_seed(tmp_path):
    rows = [
        {
            "task": "task-a",
            "task_family": "spatial",
            "memory_length": 1,
            "train_seed": 11,
            "success": True,
            "subgoal_events": [{"reason": "progress"}],
            "redundant_chunks": 0,
            "decidable_chunks": 1,
            "steps": 3,
            "overshot": False,
        },
        {
            "task": "task-a",
            "task_family": "spatial",
            "memory_length": 1,
            "train_seed": 22,
            "success": False,
            "subgoal_events": [{"reason": "redundant"}],
            "redundant_chunks": 1,
            "decidable_chunks": 1,
            "steps": 4,
            "overshot": True,
        },
        {
            "task": "task-b",
            "task_family": "semantic",
            "memory_length": 3,
            "train_seed": 11,
            "success": True,
            "subgoal_events": [{"reason": "progress"}],
            "redundant_chunks": 0,
            "decidable_chunks": 1,
            "steps": 5,
            "overshot": False,
        },
    ]
    path = tmp_path / "rollouts.jsonl"
    metrics.write_jsonl(path, rows)

    report = metrics.aggregate_jsonl(path)

    assert report["overall"]["trials"] == 3
    assert report["overall"]["successes"] == 2
    assert report["overall"]["train_seed_success_rates"] == {"11": 1.0, "22": 0.0}
    assert report["overall"]["train_seed_mean_success_rate"] == 0.5
    assert {bucket["key"] for bucket in report["by_task_family_memory_length"]} == {
        "spatial|1",
        "semantic|3",
    }
    spatial = next(bucket for bucket in report["by_task_family_memory_length"] if bucket["key"] == "spatial|1")
    assert spatial["trials"] == 2
    assert spatial["train_seed_success_rates"] == {"11": 1.0, "22": 0.0}
    assert report["redundant_rate_from_events"] == 1 / 3


def test_episode_log_contains_recomputable_structured_event_trace(tmp_path):
    event = {
        "signature": "open(cabinet)",
        "frame": 5,
        "query": 1,
        "progress_before": ["open(cabinet)", "place(cup,cabinet)"],
        "progress_after": ["open(cabinet)", "place(cup,cabinet)"],
        "reason": "redundant",
    }
    record = main.build_episode_log(
        config={"task_suite_name": "libero_mem", "replan_steps": 5},
        checkpoint_checksum="sha256:abc",
        task="pick cup",
        task_family="spatial",
        memory_length=2,
        train_seed=3,
        rollout_seed=7,
        episode=4,
        metrics=metrics.EpisodeMetrics(
            success=False,
            completed_subgoals=2,
            total_subgoals=2,
            redundant_chunks=1,
            decidable_chunks=1,
            overshot=False,
            steps=6,
        ),
        subgoal_events=[event],
        handoff_ratio=0.2,
        history_condition="futuremamba",
        timing={"episode_sec": 1.25},
        video_path="videos/ep4.mp4",
    )
    path = tmp_path / "one.jsonl"
    main.write_episode_jsonl(path, record)

    loaded = metrics.read_jsonl(path)

    assert loaded == [record]
    assert loaded[0]["subgoal_events"][0] == event
    assert metrics.redundant_rate_from_events(loaded) == 1.0


def test_fake_single_episode_runner_resets_and_replans_with_executed_prefix(tmp_path):
    goals = {"Sequence": ["open(cabinet)", "place(cup,cabinet)"]}
    adapter = FakeAdapter(
        [
            FakeSnapshot(False, [], False, {"open(cabinet)": False, "place(cup,cabinet)": False}, {"frame": 0}),
            FakeSnapshot(False, [], False, {"open(cabinet)": True, "place(cup,cabinet)": False}, {"frame": 1}),
            FakeSnapshot(False, ["open(cabinet)"], False, {"open(cabinet)": True, "place(cup,cabinet)": False}, {"frame": 2}),
            FakeSnapshot(False, ["open(cabinet)"], False, {"open(cabinet)": True, "place(cup,cabinet)": True}, {"frame": 3}),
            FakeSnapshot(True, ["open(cabinet)", "place(cup,cabinet)"], False, {"open(cabinet)": True, "place(cup,cabinet)": True}, {"frame": 4}),
        ]
    )
    client = FakeClient([["a0", "a1"], ["b0", "b1"]])
    path = tmp_path / "episodes.jsonl"

    record = main.run_single_episode(
        adapter=adapter,
        client=client,
        task="pick cup",
        task_family="spatial",
        memory_length=2,
        goals=goals,
        config={"task_suite_name": "libero_mem", "replan_steps": 2},
        checkpoint_checksum="sha256:def",
        train_seed=5,
        rollout_seed=13,
        episode=0,
        handoff_ratio=0.2,
        history_condition="futuremamba",
        max_steps=4,
        replan_steps=2,
        results_path=path,
        stable_frames=2,
        video_path=None,
    )

    assert adapter.reset_calls == 1
    assert client.reset_calls == 1
    assert adapter.advance_calls == 4
    assert adapter.step_actions == ["a0", "a1", "b0", "b1"]
    assert [call["executed_prefix"] for call in client.query_calls] == [[], ["a0", "a1"]]
    assert record["success"] is True
    assert record["redundant_chunks"] == 0
    assert len(record["subgoal_events"]) == 2
    assert metrics.read_jsonl(path) == [record]
