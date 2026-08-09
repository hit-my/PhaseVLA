from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import sys

import pytest

_MODULE_DIR = pathlib.Path(__file__).parent

_BUILD_SPEC = importlib.util.spec_from_file_location("libero_mem_build_history_pairs", _MODULE_DIR / "build_history_pairs.py")
build_history_pairs = importlib.util.module_from_spec(_BUILD_SPEC)
sys.modules[_BUILD_SPEC.name] = build_history_pairs
_BUILD_SPEC.loader.exec_module(build_history_pairs)

_EVAL_SPEC = importlib.util.spec_from_file_location("libero_mem_eval_history_pairs", _MODULE_DIR / "eval_history_pairs.py")
eval_history_pairs = importlib.util.module_from_spec(_EVAL_SPEC)
sys.modules[_EVAL_SPEC.name] = eval_history_pairs
_EVAL_SPEC.loader.exec_module(eval_history_pairs)


class SyntheticHistoryReplayer:
    def __init__(self):
        self.calls = []

    def __call__(self, history_indices, *, branch):
        self.calls.append((tuple(history_indices), branch))
        return {"branch": branch, "history": list(history_indices), "hidden": f"state:{branch}"}


class RecordingPolicy:
    def __init__(self):
        self.calls = []

    def act(self, observation, *, state, noise, task, progress_state=None):
        self.calls.append(
            {
                "observation": copy.deepcopy(observation),
                "state": copy.deepcopy(state),
                "noise": noise,
                "task": task,
                "progress_state": copy.deepcopy(progress_state),
            }
        )
        state_branch = state.get("branch") if isinstance(state, dict) else None
        if state_branch is None:
            state_branch = "zero"
        return {"branch": state_branch, "vector": [len(str(state_branch)), noise["seed"] % 10]}


class RecordingEnvironment:
    def __init__(self):
        self.restores = []

    def restore(self, *, observation, physical_state, evaluator_progress_state, noise):
        self.restores.append(
            {
                "observation": copy.deepcopy(observation),
                "physical_state": copy.deepcopy(physical_state),
                "evaluator_progress_state": copy.deepcopy(evaluator_progress_state),
                "noise": copy.deepcopy(noise),
            }
        )


class BranchScorer:
    def __call__(self, action, *, target_branch, task, progress_label):
        return {
            "predicted_branch": action["branch"],
            "target_branch": target_branch,
            "correct": action["branch"] == target_branch,
            "progress_label": progress_label,
        }


def _candidate(
    episode_id,
    query_id,
    *,
    progress_label,
    next_predicate,
    token,
    history_indices,
    canonical_state=None,
    current_observation=None,
    feasible_predicates=None,
):
    return build_history_pairs.SyntheticCandidate(
        task_id="open-drawer",
        episode_id=episode_id,
        query_id=query_id,
        progress_label=progress_label,
        source_episode_ids=[episode_id],
        retrieval_token=token,
        next_subgoal_predicate=next_predicate,
        target_branch=next_predicate,
        history_indices=history_indices,
        canonical_physical_state=canonical_state or {"drawer": "closed", "cube": "reachable"},
        current_observation=current_observation or {"rgb": [[1, 2], [3, 4]], "robot": [0.1, 0.2]},
        evaluator_progress_state={"label": progress_label, "hidden_counter": len(history_indices)},
        feasible_predicates=feasible_predicates or {"pull(drawer)", "push(drawer)", "lift(cube)"},
    )


def test_build_pairs_greedy_selects_only_valid_non_overlapping_history_contrast():
    candidates = [
        _candidate(1, "a", progress_label="before", next_predicate="pull(drawer)", token=[1.0, 0.0], history_indices=[0, 1]),
        _candidate(2, "b", progress_label="after", next_predicate="push(drawer)", token=[0.96, 0.0], history_indices=[2, 3]),
        _candidate(3, "c", progress_label="before", next_predicate="lift(cube)", token=[1.0, 0.0], history_indices=[4, 5]),
        _candidate(1, "d", progress_label="after", next_predicate="push(drawer)", token=[0.98, 0.0], history_indices=[6, 7]),
    ]

    pairs = build_history_pairs.build_history_pairs(candidates, similarity_threshold=0.95, noise_seed=123)

    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.a.query_id == "a"
    assert pair.b.query_id == "b"
    assert pair.similarity == pytest.approx(1.0)
    assert pair.manifest["a"]["history_indices"] == [0, 1]
    assert pair.manifest["b"]["history_indices"] == [2, 3]
    assert pair.manifest["threshold"] == 0.95
    assert pair.manifest["noise_seed"] == 123


def test_manifest_validation_rejects_missing_required_fields():
    pair = build_history_pairs.build_history_pairs(
        [
            _candidate(1, "a", progress_label="before", next_predicate="pull(drawer)", token=[1.0], history_indices=[0]),
            _candidate(2, "b", progress_label="after", next_predicate="push(drawer)", token=[1.0], history_indices=[1]),
        ],
        similarity_threshold=0.9,
        noise_seed=7,
    )[0]
    manifest = copy.deepcopy(pair.manifest)
    del manifest["a"]["evaluator_progress_checksum"]

    with pytest.raises(ValueError, match="evaluator_progress_checksum"):
        build_history_pairs.validate_pair_manifest(manifest)


def test_checksums_require_same_current_observation_and_canonical_physical_state_but_distinct_progress_state():
    pair = build_history_pairs.build_history_pairs(
        [
            _candidate(1, "a", progress_label="before", next_predicate="pull(drawer)", token=[1.0], history_indices=[0]),
            _candidate(2, "b", progress_label="after", next_predicate="push(drawer)", token=[1.0], history_indices=[1]),
        ],
        similarity_threshold=0.9,
        noise_seed=42,
    )[0]

    assert pair.manifest["canonical_physical_state_checksum"] == build_history_pairs.stable_checksum(
        pair.canonical_physical_state
    )
    assert pair.manifest["current_observation_checksum"] == build_history_pairs.stable_checksum(pair.current_observation)
    assert pair.manifest["a"]["evaluator_progress_checksum"] != pair.manifest["b"]["evaluator_progress_checksum"]
    assert pair.manifest["a"]["evaluator_progress_state"] == {"label": "before", "hidden_counter": 1}
    assert "evaluator_progress_state" not in pair.model_inputs["a"]
    assert "evaluator_progress_state" not in pair.model_inputs["b"]


def test_evaluator_runs_all_history_state_interventions_with_fixed_inputs_noise_and_semantic_scorer():
    pair = build_history_pairs.build_history_pairs(
        [
            _candidate(1, "a", progress_label="before", next_predicate="pull(drawer)", token=[1.0], history_indices=[10, 11, 12]),
            _candidate(2, "b", progress_label="after", next_predicate="push(drawer)", token=[1.0], history_indices=[20, 21, 22]),
        ],
        similarity_threshold=0.9,
        noise_seed=99,
    )[0]
    env = RecordingEnvironment()
    policy = RecordingPolicy()
    replayer = SyntheticHistoryReplayer()

    rows = eval_history_pairs.evaluate_history_pair(
        pair,
        policy=policy,
        environment=env,
        history_replayer=replayer,
        branch_scorer=BranchScorer(),
        truncated_k=2,
        shuffle_seed=5,
    )

    conditions = {(row["trial_branch"], row["condition"]) for row in rows}
    assert conditions == {
        ("a", "frozen_baseline"),
        ("a", "correct"),
        ("a", "zero"),
        ("a", "truncated_2"),
        ("a", "shuffled"),
        ("a", "swapped"),
        ("b", "frozen_baseline"),
        ("b", "correct"),
        ("b", "zero"),
        ("b", "truncated_2"),
        ("b", "shuffled"),
        ("b", "swapped"),
    }
    assert {row["observation_checksum"] for row in rows} == {pair.manifest["current_observation_checksum"]}
    assert {row["physical_state_checksum"] for row in rows} == {pair.manifest["canonical_physical_state_checksum"]}
    assert {row["noise_checksum"] for row in rows} == {build_history_pairs.stable_checksum({"seed": 99})}
    assert {restore["observation"] == pair.current_observation for restore in env.restores} == {True}
    assert {restore["physical_state"] == pair.canonical_physical_state for restore in env.restores} == {True}
    assert {call["progress_state"] is None for call in policy.calls} == {True}
    assert all("branch_correct" in row for row in rows)
    assert all("action_prefix_distance" in row for row in rows)

    swapped = [row for row in rows if row["condition"] == "swapped"]
    assert {row["evaluator_progress_label"] for row in swapped} == {"before", "after"}
    assert {row["target_branch"] for row in swapped} == {"pull(drawer)", "push(drawer)"}
    assert any(row["condition"] == "correct" and row["branch_correct"] for row in rows)
    assert any(row["condition"] == "swapped" and not row["branch_correct"] for row in rows)


def test_write_jsonl_preserves_recomputable_branch_accuracy(tmp_path):
    rows = [
        {"pair_id": "p0", "condition": "correct", "branch_correct": True},
        {"pair_id": "p1", "condition": "correct", "branch_correct": False},
    ]
    output_path = tmp_path / "history_pairs.jsonl"

    eval_history_pairs.write_jsonl(output_path, rows)
    loaded = [json.loads(line) for line in output_path.read_text().splitlines()]

    assert loaded == rows
    assert eval_history_pairs.branch_accuracy(loaded, condition="correct") == 0.5
