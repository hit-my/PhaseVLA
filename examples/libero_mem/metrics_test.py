from __future__ import annotations

import dataclasses
import importlib.util
import json
import pathlib
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
