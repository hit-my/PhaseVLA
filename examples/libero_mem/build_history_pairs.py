from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True)
class SyntheticCandidate:
    task_id: str
    episode_id: int | str
    query_id: int | str
    progress_label: str
    source_episode_ids: list[int | str]
    retrieval_token: list[float]
    next_subgoal_predicate: str
    target_branch: str
    history_indices: list[int]
    canonical_physical_state: Any
    current_observation: Any
    evaluator_progress_state: Any
    feasible_predicates: set[str] | list[str] | tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class HistoryPair:
    pair_id: str
    a: SyntheticCandidate
    b: SyntheticCandidate
    similarity: float
    threshold: float
    noise_seed: int
    canonical_physical_state: Any
    current_observation: Any
    manifest: dict[str, Any]
    model_inputs: dict[str, dict[str, Any]]


_REQUIRED_TOP_LEVEL_FIELDS = {
    "pair_id",
    "task_id",
    "a",
    "b",
    "canonical_physical_state",
    "canonical_physical_state_checksum",
    "current_observation",
    "current_observation_checksum",
    "threshold",
    "noise_seed",
    "similarity",
}
_REQUIRED_BRANCH_FIELDS = {
    "episode_id",
    "query_id",
    "history_indices",
    "progress_label",
    "evaluator_progress_checksum",
    "evaluator_progress_state",
    "target_predicate",
    "expert_branch",
    "source_episode_ids",
}


def build_history_pairs(
    candidates: list[SyntheticCandidate], *, similarity_threshold: float, noise_seed: int
) -> list[HistoryPair]:
    """Build greedily de-duplicated causal history pairs from local candidates.

    Retrieval-token cosine similarity is used only for pair retrieval. The final
    manifest carries both the canonical physical/observation checksums that are
    visible to the policy and the evaluator-only progress-state checksums used
    to restore hidden task progress during evaluation.
    """

    scored_pairs: list[tuple[float, int, int, SyntheticCandidate, SyntheticCandidate]] = []
    for left_index, left in enumerate(candidates):
        for right_index in range(left_index + 1, len(candidates)):
            right = candidates[right_index]
            similarity = cosine_similarity(left.retrieval_token, right.retrieval_token)
            if not _eligible_pair(left, right, similarity=similarity, threshold=similarity_threshold):
                continue
            scored_pairs.append((similarity, left_index, right_index, left, right))

    scored_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_candidate_keys: set[tuple[str, str]] = set()
    used_source_episodes: set[tuple[str, str]] = set()
    pairs: list[HistoryPair] = []
    for similarity, _left_index, _right_index, left, right in scored_pairs:
        left_key = _candidate_key(left)
        right_key = _candidate_key(right)
        source_keys = _source_keys(left) | _source_keys(right)
        if left_key in used_candidate_keys or right_key in used_candidate_keys:
            continue
        if used_source_episodes & source_keys:
            continue
        pair = make_history_pair(
            left,
            right,
            similarity=similarity,
            threshold=similarity_threshold,
            noise_seed=noise_seed,
            pair_index=len(pairs),
        )
        pairs.append(pair)
        used_candidate_keys.update({left_key, right_key})
        used_source_episodes.update(source_keys)
    return pairs


def make_history_pair(
    left: SyntheticCandidate,
    right: SyntheticCandidate,
    *,
    similarity: float,
    threshold: float,
    noise_seed: int,
    pair_index: int = 0,
) -> HistoryPair:
    if stable_checksum(left.canonical_physical_state) != stable_checksum(right.canonical_physical_state):
        raise ValueError("paired candidates must share a canonical physical state")
    if stable_checksum(left.current_observation) != stable_checksum(right.current_observation):
        raise ValueError("paired candidates must share a unique current observation")
    pair_id = _pair_id(left, right, pair_index=pair_index)
    canonical_physical_state = _canonical_json(left.canonical_physical_state)
    current_observation = _canonical_json(left.current_observation)
    canonical_physical_state_checksum = stable_checksum(canonical_physical_state)
    current_observation_checksum = stable_checksum(current_observation)
    manifest = {
        "pair_id": pair_id,
        "task_id": str(left.task_id),
        "similarity": float(similarity),
        "threshold": float(threshold),
        "noise_seed": int(noise_seed),
        "canonical_physical_state": canonical_physical_state,
        "canonical_physical_state_checksum": canonical_physical_state_checksum,
        "current_observation": current_observation,
        "current_observation_checksum": current_observation_checksum,
        "a": _manifest_branch(left),
        "b": _manifest_branch(right),
    }
    validate_pair_manifest(manifest)
    return HistoryPair(
        pair_id=pair_id,
        a=left,
        b=right,
        similarity=float(similarity),
        threshold=float(threshold),
        noise_seed=int(noise_seed),
        canonical_physical_state=canonical_physical_state,
        current_observation=current_observation,
        manifest=manifest,
        model_inputs={"a": _model_input(left), "b": _model_input(right)},
    )


def validate_pair_manifest(manifest: dict[str, Any]) -> None:
    missing_top = sorted(field for field in _REQUIRED_TOP_LEVEL_FIELDS if field not in manifest)
    if missing_top:
        raise ValueError(f"pair manifest missing required field(s): {', '.join(missing_top)}")

    for value_field, checksum_field in (
        ("canonical_physical_state", "canonical_physical_state_checksum"),
        ("current_observation", "current_observation_checksum"),
    ):
        expected_checksum = stable_checksum(manifest[value_field])
        if manifest[checksum_field] != expected_checksum:
            raise ValueError(f"pair manifest {checksum_field} does not match persisted {value_field}")

    for branch_name in ("a", "b"):
        branch = manifest.get(branch_name)
        if not isinstance(branch, dict):
            raise ValueError(f"pair manifest field {branch_name!r} must be an object")
        missing_branch = sorted(field for field in _REQUIRED_BRANCH_FIELDS if field not in branch)
        if missing_branch:
            raise ValueError(
                f"pair manifest branch {branch_name!r} missing required field(s): {', '.join(missing_branch)}"
            )
        if not branch["history_indices"]:
            raise ValueError(f"pair manifest branch {branch_name!r} has empty history_indices")
        if branch["evaluator_progress_state"] is None:
            raise ValueError(f"pair manifest branch {branch_name!r} missing evaluator progress state")
        expected_progress_checksum = stable_checksum(branch["evaluator_progress_state"])
        if branch["evaluator_progress_checksum"] != expected_progress_checksum:
            raise ValueError(
                f"pair manifest branch {branch_name!r} evaluator_progress_checksum does not match "
                "persisted evaluator_progress_state"
            )
    if manifest["a"]["progress_label"] == manifest["b"]["progress_label"]:
        raise ValueError("pair manifest requires contrasting progress labels")
    if manifest["a"]["target_predicate"] == manifest["b"]["target_predicate"]:
        raise ValueError("pair manifest requires contrasting target predicates")
    a_sources = set(manifest["a"]["source_episode_ids"])
    b_sources = set(manifest["b"]["source_episode_ids"])
    if a_sources & b_sources:
        raise ValueError("pair manifest source episodes must be disjoint")


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            validate_pair_manifest(row)
        except ValueError as exc:
            raise ValueError(f"invalid manifest row {line_number}: {exc}") from exc
        rows.append(row)
    return rows


def load_history_pairs(path: str | Path) -> list[HistoryPair]:
    return [_history_pair_from_manifest(manifest) for manifest in load_manifest(path)]


def limit_pairs_per_task(pairs: list[HistoryPair], max_pairs_per_task: int | None) -> list[HistoryPair]:
    if max_pairs_per_task is None:
        return pairs
    if max_pairs_per_task < 1:
        raise ValueError("max_pairs_per_task must be positive")
    counts: dict[str, int] = {}
    limited: list[HistoryPair] = []
    for pair in pairs:
        task_id = str(pair.manifest["task_id"])
        if counts.get(task_id, 0) >= max_pairs_per_task:
            continue
        limited.append(pair)
        counts[task_id] = counts.get(task_id, 0) + 1
    return limited


def _history_pair_from_manifest(manifest: dict[str, Any]) -> HistoryPair:
    validate_pair_manifest(manifest)
    left = _candidate_from_manifest_branch(manifest, "a")
    right = _candidate_from_manifest_branch(manifest, "b")
    return HistoryPair(
        pair_id=str(manifest["pair_id"]),
        a=left,
        b=right,
        similarity=float(manifest["similarity"]),
        threshold=float(manifest["threshold"]),
        noise_seed=int(manifest["noise_seed"]),
        canonical_physical_state=_canonical_json(manifest["canonical_physical_state"]),
        current_observation=_canonical_json(manifest["current_observation"]),
        manifest=manifest,
        model_inputs={"a": _model_input(left), "b": _model_input(right)},
    )


def _candidate_from_manifest_branch(manifest: dict[str, Any], branch_name: str) -> SyntheticCandidate:
    branch = manifest[branch_name]
    feasible_predicates = [manifest["a"]["target_predicate"], manifest["b"]["target_predicate"]]
    return SyntheticCandidate(
        task_id=str(manifest["task_id"]),
        episode_id=branch["episode_id"],
        query_id=branch["query_id"],
        progress_label=branch["progress_label"],
        source_episode_ids=list(branch["source_episode_ids"]),
        retrieval_token=list(branch.get("retrieval_token", [])),
        next_subgoal_predicate=branch["target_predicate"],
        target_branch=branch["expert_branch"],
        history_indices=list(branch["history_indices"]),
        canonical_physical_state=_canonical_json(manifest["canonical_physical_state"]),
        current_observation=_canonical_json(manifest["current_observation"]),
        evaluator_progress_state=_canonical_json(branch["evaluator_progress_state"]),
        feasible_predicates=branch.get("feasible_predicates", feasible_predicates),
    )


def write_manifest_jsonl(path: str | Path, pairs: list[HistoryPair]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for pair in pairs:
            file.write(json.dumps(pair.manifest, sort_keys=True, separators=(",", ":")) + "\n")


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"retrieval tokens must have the same length, got {len(left)} and {len(right)}")
    left_norm = math.sqrt(sum(float(value) * float(value) for value in left))
    right_norm = math.sqrt(sum(float(value) * float(value) for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


def stable_checksum(value: Any) -> str:
    payload = json.dumps(_canonical_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _eligible_pair(left: SyntheticCandidate, right: SyntheticCandidate, *, similarity: float, threshold: float) -> bool:
    if left.task_id != right.task_id:
        return False
    if left.progress_label == right.progress_label:
        return False
    if set(left.source_episode_ids) & set(right.source_episode_ids):
        return False
    if similarity < threshold:
        return False
    if left.next_subgoal_predicate == right.next_subgoal_predicate:
        return False
    if stable_checksum(left.canonical_physical_state) != stable_checksum(right.canonical_physical_state):
        return False
    if stable_checksum(left.current_observation) != stable_checksum(right.current_observation):
        return False
    feasible = set(left.feasible_predicates) & set(right.feasible_predicates)
    return left.next_subgoal_predicate in feasible and right.next_subgoal_predicate in feasible


def _manifest_branch(candidate: SyntheticCandidate) -> dict[str, Any]:
    return {
        "episode_id": candidate.episode_id,
        "query_id": candidate.query_id,
        "source_episode_ids": list(candidate.source_episode_ids),
        "history_indices": list(candidate.history_indices),
        "progress_label": candidate.progress_label,
        "evaluator_progress_state": _canonical_json(candidate.evaluator_progress_state),
        "evaluator_progress_checksum": stable_checksum(candidate.evaluator_progress_state),
        "target_predicate": candidate.next_subgoal_predicate,
        "expert_branch": candidate.target_branch,
    }


def _model_input(candidate: SyntheticCandidate) -> dict[str, Any]:
    return {
        "task_id": candidate.task_id,
        "episode_id": candidate.episode_id,
        "query_id": candidate.query_id,
        "history_indices": list(candidate.history_indices),
        "current_observation": candidate.current_observation,
        "canonical_physical_state_checksum": stable_checksum(candidate.canonical_physical_state),
        "current_observation_checksum": stable_checksum(candidate.current_observation),
    }


def _candidate_key(candidate: SyntheticCandidate) -> tuple[str, str]:
    return str(candidate.episode_id), str(candidate.query_id)


def _source_keys(candidate: SyntheticCandidate) -> set[tuple[str, str]]:
    return {(str(candidate.task_id), str(episode_id)) for episode_id in candidate.source_episode_ids}


def _pair_id(left: SyntheticCandidate, right: SyntheticCandidate, *, pair_index: int) -> str:
    return stable_checksum(
        {
            "pair_index": pair_index,
            "task_id": left.task_id,
            "a": {"episode_id": left.episode_id, "query_id": left.query_id},
            "b": {"episode_id": right.episode_id, "query_id": right.query_id},
        }
    )[:16]


def _canonical_json(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _canonical_json(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _canonical_json(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item) for item in value]
    if isinstance(value, set):
        return [_canonical_json(item) for item in sorted(value, key=str)]
    if hasattr(value, "tolist"):
        return _canonical_json(value.tolist())
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _load_synthetic_candidates(path: str | Path) -> list[SyntheticCandidate]:
    candidates: list[SyntheticCandidate] = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        try:
            candidates.append(_coerce_candidate(raw))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid candidate row {line_number}: {exc}") from exc
    return candidates


def _load_candidates_for_cli(args: argparse.Namespace, candidate_provider: Any | None) -> list[SyntheticCandidate]:
    if args.synthetic_candidates_jsonl is not None:
        return _load_synthetic_candidates(args.synthetic_candidates_jsonl)
    if args.repo_id is None:
        raise ValueError("No candidate source supplied. Pass --synthetic-candidates-jsonl or --repo-id.")
    provider = candidate_provider if candidate_provider is not None else _load_real_candidate_provider()
    if callable(provider):
        raw_candidates = provider(repo_id=args.repo_id)
    elif hasattr(provider, "load_candidates"):
        raw_candidates = provider.load_candidates(repo_id=args.repo_id)
    else:
        raise TypeError("candidate_provider must be callable or define load_candidates(repo_id=...)")
    return [_coerce_candidate(candidate) for candidate in raw_candidates]


def _coerce_candidate(raw: Any) -> SyntheticCandidate:
    if isinstance(raw, SyntheticCandidate):
        return raw
    if isinstance(raw, dict):
        return SyntheticCandidate(**raw)
    raise TypeError(f"unsupported candidate object {type(raw).__name__}")


def _load_real_candidate_provider() -> Any:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - exercised only with real dependencies.
        raise RuntimeError(
            "LeRobotDataset is unavailable. Install/provide LeRobot dependencies, pass --synthetic-candidates-jsonl, "
            "or call main(..., candidate_provider=...) with an injected provider."
        ) from exc
    return _LeRobotCandidateProvider(LeRobotDataset)


class _LeRobotCandidateProvider:
    def __init__(self, dataset_cls: Any):
        self._dataset_cls = dataset_cls

    def __call__(self, *, repo_id: str) -> list[SyntheticCandidate]:
        dataset = self._dataset_cls(repo_id=repo_id)
        candidates: list[SyntheticCandidate] = []
        for row_index, row in enumerate(dataset):
            payload = _candidate_payload_from_dataset_row(row)
            if payload is None:
                continue
            try:
                candidates.append(_coerce_candidate(payload))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid history-pair candidate at dataset row {row_index}: {exc}") from exc
        if not candidates:
            raise RuntimeError(
                f"dataset {repo_id!r} did not expose SyntheticCandidate-compatible history-pair candidate records"
            )
        return candidates


def _candidate_payload_from_dataset_row(row: Any) -> dict[str, Any] | SyntheticCandidate | None:
    if isinstance(row, SyntheticCandidate):
        return row
    if not isinstance(row, dict):
        return None
    required = set(SyntheticCandidate.__dataclass_fields__)
    if required <= set(row):
        return row
    for key in ("history_pair_candidate", "candidate", "metadata"):
        value = row.get(key)
        if isinstance(value, dict) and required <= set(value):
            return value
    return None


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local LIBERO-Mem history-pair causal experiment manifests.")
    parser.add_argument("--synthetic-candidates-jsonl", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--max-pairs-per-task", type=int)
    parser.add_argument("--output", "--output-jsonl", dest="output_jsonl", type=Path, required=True)
    parser.add_argument("--similarity-threshold", type=float, default=0.95)
    parser.add_argument("--noise-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, *, candidate_provider: Any | None = None) -> list[HistoryPair]:
    args = _parse_args(argv)
    pairs = build_history_pairs(
        _load_candidates_for_cli(args, candidate_provider),
        similarity_threshold=args.similarity_threshold,
        noise_seed=args.noise_seed,
    )
    pairs = limit_pairs_per_task(pairs, args.max_pairs_per_task)
    write_manifest_jsonl(args.output_jsonl, pairs)
    return pairs


if __name__ == "__main__":
    main()
