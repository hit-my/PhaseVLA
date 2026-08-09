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
    _validate_pair_state_checksums(pair)
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
            semantic = _validate_branch_semantics(
                branch_scorer(
                    action,
                    target_branch=trial_manifest["expert_branch"],
                    task=pair.manifest["task_id"],
                    progress_label=trial_manifest["progress_label"],
                ),
                target_branch=trial_manifest["expert_branch"],
            )
            observed_branch = _observed_branch_from_semantics(semantic)
            branch_correct = observed_branch == str(trial_manifest["expert_branch"])
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
                "observed_branch": observed_branch,
                "event_trace": copy.deepcopy(semantic["event_trace"]),
                "branch_correct": branch_correct,
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
    return sum(1 for row in filtered if branch_correct_from_result(row)) / len(filtered)


def branch_correct_from_result(row: dict[str, Any]) -> bool:
    target_branch = row.get("target_branch")
    if target_branch is None:
        raise ValueError("result row missing target_branch")
    return _observed_branch_from_result(row) == str(target_branch)


def _observed_branch_from_result(row: dict[str, Any]) -> str:
    if "event_trace" not in row:
        raise ValueError("result row missing event_trace")
    observed_from_trace = _observed_branch_from_event_trace(row["event_trace"])
    observed_branch = row.get("observed_branch")
    if observed_branch is not None and str(observed_branch) != observed_from_trace:
        raise ValueError("result row observed_branch conflicts with event_trace")
    return observed_from_trace


def _validate_branch_semantics(raw_semantic: Any, *, target_branch: str) -> dict[str, Any]:
    if not isinstance(raw_semantic, dict):
        raise ValueError("branch_scorer must return an object")
    semantic = dict(raw_semantic)
    if "event_trace" not in semantic:
        raise ValueError("branch_scorer result missing event_trace")
    observed_from_trace = _observed_branch_from_event_trace(semantic["event_trace"])
    observed_branch = semantic.get("observed_branch", semantic.get("entered_branch", observed_from_trace))
    if str(observed_branch) != observed_from_trace:
        raise ValueError("branch_scorer observed branch conflicts with event_trace")
    if "target_branch" in semantic and str(semantic["target_branch"]) != str(target_branch):
        raise ValueError("branch_scorer target_branch conflicts with trial target")
    semantic["observed_branch"] = observed_from_trace
    semantic["target_branch"] = str(target_branch)
    return semantic


def _observed_branch_from_semantics(semantic: dict[str, Any]) -> str:
    return str(semantic["observed_branch"])


def _observed_branch_from_event_trace(event_trace: Any) -> str:
    if not isinstance(event_trace, list) or not event_trace:
        raise ValueError("event_trace must be a non-empty list")
    for event in event_trace:
        if not isinstance(event, dict):
            continue
        if "branch" in event:
            return str(event["branch"])
        if "observed_branch" in event:
            return str(event["observed_branch"])
        if "entered_branch" in event:
            return str(event["entered_branch"])
        if "target_predicate" in event and event.get("satisfied") is True:
            return str(event["target_predicate"])
    raise ValueError("event_trace does not contain an entered branch label")


def _validate_pair_state_checksums(pair: HistoryPair) -> None:
    if stable_checksum(pair.canonical_physical_state) != pair.manifest["canonical_physical_state_checksum"]:
        raise ValueError("canonical_physical_state_checksum does not match pair state")
    if stable_checksum(pair.current_observation) != pair.manifest["current_observation_checksum"]:
        raise ValueError("current_observation_checksum does not match pair observation")
    for branch_name in ("a", "b"):
        candidate = getattr(pair, branch_name)
        expected_checksum = pair.manifest[branch_name]["evaluator_progress_checksum"]
        if stable_checksum(candidate.evaluator_progress_state) != expected_checksum:
            raise ValueError(f"{branch_name} evaluator_progress_checksum does not match pair state")


def load_policy(policy_path: str | Path | None = None) -> Any:
    """Late-import the real FutureMamba policy only when the real CLI path asks for it."""

    try:
        from openpi.policies.futuremamba_policy import FutureMambaPolicy  # type: ignore[attr-defined]
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


def evaluate_history_pairs(
    pairs: list[HistoryPair],
    *,
    policy: Any,
    environment_factory: Any,
    history_replayer: Any,
    branch_scorer: Any,
    truncated_k: int,
    shuffle_seed: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        environment = environment_factory(pair) if callable(environment_factory) else environment_factory
        rows.extend(
            evaluate_history_pair(
                pair,
                policy=policy,
                environment=environment,
                history_replayer=history_replayer,
                branch_scorer=branch_scorer,
                truncated_k=truncated_k,
                shuffle_seed=shuffle_seed,
            )
        )
    return rows


def _load_real_environment_factory() -> Any:
    raise RuntimeError(
        "Real LIBERO environment construction is unavailable from this standalone CLI. "
        "Call main(..., environment_factory=...) after constructing the real environment factory."
    )


def _load_real_history_replayer() -> Any:
    raise RuntimeError(
        "Real history replayer is unavailable from this standalone CLI. "
        "Call main(..., history_replayer=...) after constructing the real replayer."
    )


def _load_real_branch_scorer() -> Any:
    raise RuntimeError(
        "Real semantic branch_scorer is unavailable from this standalone CLI. "
        "Call main(..., branch_scorer=...) after constructing the real scorer."
    )


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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LIBERO-Mem history-pair interventions.")
    parser.add_argument("--pairs", "--manifest-jsonl", dest="pairs", type=Path, required=True)
    parser.add_argument("--results", "--output-jsonl", dest="results", type=Path, required=True)
    parser.add_argument("--policy-path", type=Path)
    parser.add_argument("--truncated-k", type=int, default=2)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    policy: Any | None = None,
    environment_factory: Any | None = None,
    history_replayer: Any | None = None,
    branch_scorer: Any | None = None,
) -> list[dict[str, Any]]:
    args = _parse_args(argv)
    pairs = build_history_pairs.load_history_pairs(args.pairs)
    resolved_policy = policy if policy is not None else load_policy(args.policy_path)
    resolved_environment_factory = (
        environment_factory if environment_factory is not None else _load_real_environment_factory()
    )
    resolved_history_replayer = history_replayer if history_replayer is not None else _load_real_history_replayer()
    resolved_branch_scorer = branch_scorer if branch_scorer is not None else _load_real_branch_scorer()
    rows = evaluate_history_pairs(
        pairs,
        policy=resolved_policy,
        environment_factory=resolved_environment_factory,
        history_replayer=resolved_history_replayer,
        branch_scorer=resolved_branch_scorer,
        truncated_k=args.truncated_k,
        shuffle_seed=args.shuffle_seed,
    )
    write_jsonl(args.results, rows)
    return rows


if __name__ == "__main__":
    main()
