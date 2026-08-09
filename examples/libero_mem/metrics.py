from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


_PROGRESS_REASONS = {"progress"}
_REDUNDANT_REASONS = {"redundant"}
_DECIDABLE_REASONS = {"progress", "redundant"}


@dataclasses.dataclass(frozen=True)
class EpisodeMetrics:
    success: bool
    completed_subgoals: int
    total_subgoals: int
    redundant_chunks: int
    decidable_chunks: int
    overshot: bool
    steps: int


class SymbolicEventMonitor:
    """Tracks stable rising symbolic predicate events against Sequence/Or goals."""

    def __init__(self, goals: Mapping[str, Any] | Sequence[Any], *, stable_frames: int = 6):
        if stable_frames <= 0:
            raise ValueError(f"stable_frames must be positive, got {stable_frames}")
        self._stable_frames = int(stable_frames)
        self._branches = _expand_goal_branches(goals)
        if not self._branches:
            self._branches = [()]
        self._all_signatures = {signature for branch in self._branches for signature in branch}
        self._true_streaks: dict[str, int] = {}
        self._seen_true_events: set[str] = set()
        self._events: list[dict[str, Any]] = []
        self._completed: set[str] = set()

    @property
    def events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._events]

    def observe(
        self,
        *,
        frame: int,
        query_index: int,
        atomic_predicates: Mapping[str, Any] | None,
        satisfied_before: Iterable[Any],
        satisfied_after: Iterable[Any],
    ) -> list[dict[str, Any]]:
        before = _string_list(satisfied_before)
        after = _string_list(satisfied_after)
        self._completed.update(after)
        emitted: list[dict[str, Any]] = []
        predicates = {str(key): bool(value) for key, value in (atomic_predicates or {}).items()}
        for signature in sorted(set(predicates) | set(self._true_streaks)):
            if predicates.get(signature, False):
                self._true_streaks[signature] = self._true_streaks.get(signature, 0) + 1
            else:
                self._true_streaks[signature] = 0
                self._seen_true_events.discard(signature)
                continue
            if self._true_streaks[signature] < self._stable_frames or signature in self._seen_true_events:
                continue
            self._seen_true_events.add(signature)
            reason = self._classify(signature, before=before, after=after)
            event = {
                "signature": signature,
                "frame": int(frame),
                "query": int(query_index),
                "progress_before": before,
                "progress_after": after,
                "reason": reason,
            }
            self._events.append(event)
            emitted.append(dict(event))
        return emitted

    def finish(self, *, success: bool, overshot: bool, steps: int) -> EpisodeMetrics:
        completed_subgoals = len(_goal_compatible_completed(self._completed, self._branches))
        redundant_chunks = sum(1 for event in self._events if event["reason"] in _REDUNDANT_REASONS)
        decidable_chunks = sum(1 for event in self._events if event["reason"] in _DECIDABLE_REASONS)
        return EpisodeMetrics(
            success=bool(success),
            completed_subgoals=completed_subgoals,
            total_subgoals=max((len(branch) for branch in self._branches), default=0),
            redundant_chunks=redundant_chunks,
            decidable_chunks=decidable_chunks,
            overshot=bool(overshot),
            steps=int(steps),
        )

    def _classify(self, signature: str, *, before: list[str], after: list[str]) -> str:
        if signature not in self._all_signatures:
            return "undecidable"
        before_set = set(before)
        after_set = set(after)
        compatible_branches = _compatible_branches(before_set, self._branches)
        if not compatible_branches:
            return "undecidable"
        if signature in before_set:
            return "redundant"
        if not any(signature in branch for branch in compatible_branches):
            return "incompatible_branch"
        expected = {branch[len(before_set)] for branch in compatible_branches if len(before_set) < len(branch)}
        if signature not in expected:
            return "out_of_order"
        return "progress" if signature in after_set - before_set else "undecidable"


def wilson_interval(*, successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total < 0:
        raise ValueError(f"total must be non-negative, got {total}")
    if successes < 0 or successes > total:
        raise ValueError(f"successes must be in [0, total], got {successes} of {total}")
    if total == 0:
        return (0.0, 0.0)
    phat = successes / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    center = (phat + z2 / (2.0 * total)) / denominator
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * total)) / total) / denominator
    return (max(0.0, center - margin), min(1.0, center + margin))


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(_jsonable(row), sort_keys=True, separators=(",", ":")))
            file.write("\n")


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(_jsonable(row), sort_keys=True, separators=(",", ":")))
        file.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    input_path = Path(path)
    if not input_path.exists():
        return rows
    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{input_path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def redundant_rate_from_events(rows: Iterable[Mapping[str, Any]]) -> float:
    total = 0
    redundant = 0
    for row in rows:
        for event in row.get("subgoal_events", []) or []:
            reason = event.get("reason") if isinstance(event, Mapping) else None
            if reason in _DECIDABLE_REASONS:
                total += 1
                if reason in _REDUNDANT_REASONS:
                    redundant += 1
    return redundant / total if total else 0.0


def aggregate_jsonl(path: str | Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    return {
        "overall": _summarize_rows(rows),
        "by_task_family_memory_length": [
            {"key": key, **_summarize_rows(bucket_rows)}
            for key, bucket_rows in sorted(_group_rows(rows).items(), key=lambda item: item[0])
        ],
        "redundant_rate_from_events": redundant_rate_from_events(rows),
    }


def _group_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        key = f"{row.get('task_family')}|{row.get('memory_length')}"
        grouped.setdefault(key, []).append(row)
    return grouped


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trials = len(rows)
    successes = sum(1 for row in rows if bool(row.get("success", False)))
    seed_rates = _train_seed_success_rates(rows)
    return {
        "trials": trials,
        "successes": successes,
        "success_rate": successes / trials if trials else 0.0,
        "success_wilson95": wilson_interval(successes=successes, total=trials),
        "train_seed_success_rates": seed_rates,
        "train_seed_mean_success_rate": sum(seed_rates.values()) / len(seed_rates) if seed_rates else 0.0,
        "mean_steps": sum(int(row.get("steps", 0)) for row in rows) / trials if trials else 0.0,
        "overshot_rate": sum(1 for row in rows if bool(row.get("overshot", False))) / trials if trials else 0.0,
        "redundant_rate": _ratio_from_row_counts(rows, numerator="redundant_chunks", denominator="decidable_chunks"),
    }


def _train_seed_success_rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    by_seed: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(str(row.get("train_seed")), []).append(row)
    return {
        seed: sum(1 for row in seed_rows if bool(row.get("success", False))) / len(seed_rows)
        for seed, seed_rows in sorted(by_seed.items(), key=lambda item: item[0])
    }


def _ratio_from_row_counts(rows: Sequence[Mapping[str, Any]], *, numerator: str, denominator: str) -> float:
    total_denominator = sum(int(row.get(denominator, 0)) for row in rows)
    if total_denominator == 0:
        return 0.0
    return sum(int(row.get(numerator, 0)) for row in rows) / total_denominator


def _expand_goal_branches(goals: Mapping[str, Any] | Sequence[Any]) -> list[tuple[str, ...]]:
    if isinstance(goals, Mapping):
        if "Sequence" in goals:
            return _expand_sequence(goals["Sequence"])
        if "Or" in goals:
            branches: list[tuple[str, ...]] = []
            for option in goals["Or"]:
                branches.extend(_expand_goal_branches(option))
            return branches
        return [(str(key),) for key in goals]
    if isinstance(goals, (str, bytes)):
        return [(str(goals),)]
    return _expand_sequence(goals)


def _expand_sequence(items: Iterable[Any]) -> list[tuple[str, ...]]:
    branches: list[tuple[str, ...]] = [()]
    for item in items:
        item_branches = _expand_goal_branches(item)
        branches = [prefix + branch for prefix in branches for branch in item_branches]
    return branches


def _compatible_branches(completed: set[str], branches: Sequence[tuple[str, ...]]) -> list[tuple[str, ...]]:
    compatible: list[tuple[str, ...]] = []
    for branch in branches:
        prefix = branch[: len(completed)]
        if set(prefix) == completed:
            compatible.append(branch)
    return compatible


def _goal_compatible_completed(completed: set[str], branches: Sequence[tuple[str, ...]]) -> set[str]:
    compatible = _compatible_branches(completed, branches)
    if compatible:
        return set(compatible[0][: len(completed)])
    longest: tuple[str, ...] = ()
    for branch in branches:
        prefix: list[str] = []
        for signature in branch:
            if signature not in completed:
                break
            prefix.append(signature)
        if len(prefix) > len(longest):
            longest = tuple(prefix)
    return set(longest)


def _string_list(values: Iterable[Any] | None) -> list[str]:
    return [str(value) for value in values or []]


def _jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
