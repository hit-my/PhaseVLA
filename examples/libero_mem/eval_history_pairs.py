from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


_BUILD_PATH = Path(__file__).with_name("build_history_pairs.py")
_BUILD_SPEC = importlib.util.spec_from_file_location("libero_mem_build_history_pairs_for_eval", _BUILD_PATH)
build_history_pairs = importlib.util.module_from_spec(_BUILD_SPEC)
sys.modules[_BUILD_SPEC.name] = build_history_pairs
assert _BUILD_SPEC.loader is not None
_BUILD_SPEC.loader.exec_module(build_history_pairs)
HistoryPair = build_history_pairs.HistoryPair
SyntheticCandidate = build_history_pairs.SyntheticCandidate
stable_checksum = build_history_pairs.stable_checksum


CONDITIONS = ("frozen_baseline", "correct", "zero", "truncated", "shuffled", "swapped")


def evaluate_history_pair(
    pair: HistoryPair,
    *,
    policy: Any,
    environment: Any,
    history_replayer: Any,
    branch_scorer: Any,
    truncated_k: int,
    shuffle_seed: int = 0,
) -> list[dict[str, Any]]:
    """Evaluate a pair under injected state interventions.

    The policy only receives the canonical current observation and an explicit
    policy-state snapshot. Hidden progress state is restored on the evaluator
    environment, never passed into policy input.
    """

    build_history_pairs.validate_pair_manifest(pair.manifest)
    noise = {"seed": int(pair.noise_seed)}
    snapshots = _snapshots_for_pair(pair, history_replayer=history_replayer, truncated_k=truncated_k, shuffle_seed=shuffle_seed)
    rows: list[dict[str, Any]] = []
    for trial_branch in ("a", "b"):
        candidate = getattr(pair, trial_branch)
        trial_manifest = pair.manifest[trial_branch]
        evaluator_progress_state = copy.deepcopy(candidate.evaluator_progress_state)
        for condition, state in _states_for_trial(trial_branch, snapshots=snapshots, truncated_k=truncated_k):
            environment.restore(
                observation=copy.deepcopy(pair.current_observation),
                physical_state=copy.deepcopy(pair.canonical_physical_state),
                evaluator_progress_state=copy.deepcopy(evaluator_progress_state),
                noise=copy.deepcopy(noise),
            )
            action = _policy_action(
                policy,
                observation=copy.deepcopy(pair.current_observation),
                state=copy.deepcopy(state),
                noise=copy.deepcopy(noise),
                task=pair.manifest["task_id"],
            )
            semantic = dict(
                branch_scorer(
                    action,
                    target_branch=trial_manifest["expert_branch"],
                    task=pair.manifest["task_id"],
                    progress_label=trial_manifest["progress_label"],
                )
            )
            row = {
                "pair_id": pair.pair_id,
                "trial_branch": trial_branch,
                "condition": condition,
                "episode_id": trial_manifest["episode_id"],
                "query_id": trial_manifest["query_id"],
                "target_branch": trial_manifest["expert_branch"],
                "target_predicate": trial_manifest["target_predicate"],
                "evaluator_progress_label": trial_manifest["progress_label"],
                "observation_checksum": stable_checksum(pair.current_observation),
                "physical_state_checksum": stable_checksum(pair.canonical_physical_state),
                "evaluator_progress_checksum": stable_checksum(evaluator_progress_state),
                "noise_checksum": stable_checksum(noise),
                "state_checksum": stable_checksum(state),
                "branch_correct": bool(semantic.get("correct", False)),
                "semantic_branch": semantic,
                "action_prefix_distance": action_prefix_distance(action, trial_manifest["expert_branch"]),
            }
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def branch_accuracy(rows: list[dict[str, Any]], *, condition: str | None = None) -> float:
    filtered = [row for row in rows if condition is None or row.get("condition") == condition]
    if not filtered:
        raise ValueError("branch accuracy is undefined for zero rows")
    return sum(1 for row in filtered if row.get("branch_correct")) / len(filtered)


def load_policy(policy_path: str | Path | None = None) -> Any:
    """Late-import the real FutureMamba policy only when the real CLI path asks for it."""

    try:
        from openpi.policies.future_mamba_policy import FutureMambaPolicy  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - exercised only with real dependencies.
        raise RuntimeError(
            "FutureMambaPolicy is unavailable. Install/provide the real FutureMamba policy, or call "
            "evaluate_history_pair with an injected synthetic policy. No fallback policy is fabricated."
        ) from exc
    if policy_path is None:
        return FutureMambaPolicy()
    if hasattr(FutureMambaPolicy, "load"):
        return FutureMambaPolicy.load(policy_path)
    return FutureMambaPolicy(policy_path)


def _snapshots_for_pair(pair: HistoryPair, *, history_replayer: Any, truncated_k: int, shuffle_seed: int) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for branch_name in ("a", "b"):
        candidate = getattr(pair, branch_name)
        full_history = list(candidate.history_indices)
        snapshots[f"{branch_name}:correct"] = history_replayer(full_history, branch=candidate.target_branch)
        snapshots[f"{branch_name}:truncated"] = history_replayer(full_history[-truncated_k:], branch=candidate.target_branch)
        snapshots[f"{branch_name}:shuffled"] = history_replayer(
            _deterministic_shuffle(full_history, seed=shuffle_seed), branch=candidate.target_branch
        )
    return snapshots


def _states_for_trial(
    trial_branch: str, *, snapshots: dict[str, dict[str, Any]], truncated_k: int
) -> list[tuple[str, Any]]:
    other_branch = "b" if trial_branch == "a" else "a"
    return [
        ("frozen_baseline", None),
        ("correct", snapshots[f"{trial_branch}:correct"]),
        ("zero", {}),
        (f"truncated_{truncated_k}", snapshots[f"{trial_branch}:truncated"]),
        ("shuffled", snapshots[f"{trial_branch}:shuffled"]),
        ("swapped", snapshots[f"{other_branch}:correct"]),
    ]


def _policy_action(policy: Any, *, observation: Any, state: Any, noise: dict[str, int], task: str) -> Any:
    if hasattr(policy, "act"):
        return policy.act(observation, state=state, noise=noise, task=task, progress_state=None)
    return policy(observation, state=state, noise=noise, task=task, progress_state=None)


def _deterministic_shuffle(values: list[int], *, seed: int) -> list[int]:
    decorated = [
        (stable_checksum({"seed": int(seed), "position": index, "value": value}), index, value)
        for index, value in enumerate(values)
    ]
    return [value for _checksum, _index, value in sorted(decorated)]


def action_prefix_distance(action: Any, target_branch: str) -> float:
    predicted = ""
    if isinstance(action, dict):
        predicted = str(action.get("branch", ""))
    else:
        predicted = str(action)
    target = str(target_branch)
    common = 0
    for left, right in zip(predicted, target, strict=False):
        if left != right:
            break
        common += 1
    return float(max(len(predicted), len(target)) - common)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LIBERO-Mem history-pair interventions.")
    parser.add_argument("--manifest-jsonl", type=Path)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.manifest_jsonl is None:
        raise SystemExit(
            "No manifest supplied. Pass --manifest-jsonl and inject a real environment/history replayer from Python; "
            "the CLI will not fabricate FutureMambaPolicy, MuJoCo, or LIBERO state."
        )
    raise SystemExit(
        "Real LIBERO evaluation requires injected environment, history_replayer, and semantic branch_scorer. "
        "Use evaluate_history_pair from Python after constructing those real components."
    )


if __name__ == "__main__":
    main()
