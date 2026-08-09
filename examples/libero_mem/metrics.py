from __future__ import annotations

import dataclasses
import json
import math
import random
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
        self._seen_true_events: set[tuple[str, int]] = set()
        self._run_emitted_signatures: set[str] = set()
        self._events: list[dict[str, Any]] = []
        self._completed: list[str] = []

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
        before = _goal_compatible_completed(_string_list(satisfied_before), self._branches)
        after = _goal_compatible_completed(_string_list(satisfied_after), self._branches)
        if len(after) >= len(self._completed):
            self._completed = after
        emitted: list[dict[str, Any]] = []
        predicates = {str(key): bool(value) for key, value in (atomic_predicates or {}).items()}
        for signature in sorted(set(predicates) | set(self._true_streaks)):
            if predicates.get(signature, False):
                self._true_streaks[signature] = self._true_streaks.get(signature, 0) + 1
            else:
                self._true_streaks[signature] = 0
                self._seen_true_events = {key for key in self._seen_true_events if key[0] != signature}
                self._run_emitted_signatures.discard(signature)
                continue
            if self._true_streaks[signature] < self._stable_frames:
                continue
            reason = self._classify(signature, before=before, after=after)
            event_key = (signature, _signature_count(before, signature))
            if event_key in self._seen_true_events:
                continue
            if signature in self._run_emitted_signatures and reason != "progress":
                continue
            self._seen_true_events.add(event_key)
            self._run_emitted_signatures.add(signature)
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
        compatible_branches = _compatible_branches(before, self._branches)
        if not compatible_branches:
            return "undecidable"
        remaining_branches = [branch[len(before) :] for branch in compatible_branches]
        if not any(signature in branch for branch in compatible_branches):
            return "incompatible_branch"
        if not any(signature in remaining for remaining in remaining_branches):
            return "redundant"
        expected = {remaining[0] for remaining in remaining_branches if remaining}
        if signature not in expected:
            return "out_of_order"
        progressed = len(after) > len(before) and after[: len(before)] == before and after[len(before)] == signature
        return "progress" if progressed else "undecidable"


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


def aggregate_jsonl(
    path: str | Path,
    *,
    baseline: str | Path | Iterable[Mapping[str, Any]] | None = None,
    expected_trials_per_task: int | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(path)
    by_memory_length = _bucket_summaries(rows, lambda row: str(row.get("memory_length")))
    report = {
        "overall": _summarize_rows(rows),
        "by_task": _bucket_summaries(rows, lambda row: str(row.get("task"))),
        "by_memory_length": by_memory_length,
        "temporal_scaling": by_memory_length,
        "by_task_family_memory_length": _bucket_summaries(
            rows,
            lambda row: f"{row.get('task_family')}|{row.get('memory_length')}",
        ),
        "trial_seed_integrity": _trial_seed_integrity(rows, expected_trials_per_task=expected_trials_per_task),
        "redundant_rate_from_events": redundant_rate_from_events(rows),
    }
    if baseline is not None:
        report["capability_retention_abs"] = _capability_retention_abs(rows, _coerce_rows(baseline))
    return report


def _bucket_summaries(
    rows: Sequence[Mapping[str, Any]], key_fn: Any
) -> list[dict[str, Any]]:
    return [
        {"key": key, **_summarize_rows(bucket_rows)}
        for key, bucket_rows in sorted(_group_rows(rows, key_fn).items(), key=lambda item: item[0])
    ]


def _group_rows(rows: Iterable[Mapping[str, Any]], key_fn: Any) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(key_fn(row)), []).append(row)
    return grouped


def _summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    trials = len(rows)
    successes = sum(1 for row in rows if bool(row.get("success", False)))
    seed_rates = _train_seed_success_rates(rows)
    success_values = [1.0 if bool(row.get("success", False)) else 0.0 for row in rows]
    return {
        "trials": trials,
        "successes": successes,
        "success_rate": successes / trials if trials else 0.0,
        "success_wilson95": wilson_interval(successes=successes, total=trials),
        "success_bootstrap95": _bootstrap_interval(success_values),
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


def _bootstrap_interval(values: Sequence[float], *, samples: int = 1000, seed: int = 0) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    generator = random.Random(seed)
    size = len(values)
    means = sorted(sum(values[generator.randrange(size)] for _ in range(size)) / size for _ in range(samples))
    lower_index = int(0.025 * (samples - 1))
    upper_index = int(0.975 * (samples - 1))
    return (means[lower_index], means[upper_index])


def _trial_seed_integrity(
    rows: Sequence[Mapping[str, Any]], *, expected_trials_per_task: int | None
) -> dict[str, Any]:
    seen: dict[str, set[str]] = {}
    duplicate_counts: dict[tuple[str, str, str, str], int] = {}
    for row in rows:
        task = str(row.get("task"))
        episode = str(row.get("episode"))
        seen.setdefault(task, set()).add(episode)
        key = (task, str(row.get("train_seed")), str(row.get("rollout_seed")), episode)
        duplicate_counts[key] = duplicate_counts.get(key, 0) + 1
    duplicates = [
        {"task": task, "train_seed": train_seed, "rollout_seed": rollout_seed, "episode": episode, "count": count}
        for (task, train_seed, rollout_seed, episode), count in sorted(duplicate_counts.items())
        if count > 1
    ]
    missing: list[dict[str, Any]] = []
    if expected_trials_per_task is not None:
        expected = {str(index) for index in range(expected_trials_per_task)}
        missing = [
            {"task": task, "missing_trial_keys": sorted(expected - observed, key=int)}
            for task, observed in sorted(seen.items(), key=lambda item: item[0])
            if expected - observed
        ]
    return {
        "expected_trials_per_task": expected_trials_per_task,
        "complete": not missing and not duplicates,
        "missing": missing,
        "duplicates": duplicates,
    }


def _capability_retention_abs(
    rows: Sequence[Mapping[str, Any]], baseline_rows: Sequence[Mapping[str, Any]]
) -> dict[str, float]:
    current = _task_seed_success_rates(rows)
    baseline = _task_seed_success_rates(
        [row for row in baseline_rows if _task_suite_name(row) in (None, "libero_10")]
    )
    return {
        key: current_rate - baseline[key]
        for key, current_rate in sorted(current.items(), key=lambda item: item[0])
        if key in baseline
    }


def _task_seed_success_rates(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    grouped = _group_rows(rows, lambda row: f"{row.get('task')}|{row.get('train_seed')}")
    return {key: _mean_success(bucket_rows) for key, bucket_rows in sorted(grouped.items(), key=lambda item: item[0])}


def _mean_success(rows: Sequence[Mapping[str, Any]]) -> float:
    return sum(1 for row in rows if bool(row.get("success", False))) / len(rows) if rows else 0.0


def _coerce_rows(value: str | Path | Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, Path)):
        return read_jsonl(value)
    return list(value)


def _task_suite_name(row: Mapping[str, Any]) -> str | None:
    if "task_suite_name" in row:
        return str(row["task_suite_name"])
    config = row.get("config")
    if isinstance(config, Mapping) and "task_suite_name" in config:
        return str(config["task_suite_name"])
    return None


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


def _compatible_branches(completed: Sequence[str], branches: Sequence[tuple[str, ...]]) -> list[tuple[str, ...]]:
    prefix = tuple(completed)
    return [branch for branch in branches if branch[: len(prefix)] == prefix]


def _goal_compatible_completed(completed: Sequence[str], branches: Sequence[tuple[str, ...]]) -> list[str]:
    compatible = _compatible_branches(completed, branches)
    if compatible:
        return list(completed)
    longest: tuple[str, ...] = ()
    for branch in branches:
        prefix: list[str] = []
        for observed, expected in zip(completed, branch, strict=False):
            if observed != expected:
                break
            prefix.append(observed)
        if len(prefix) > len(longest):
            longest = tuple(prefix)
    return list(longest)


def _signature_count(values: Sequence[str], signature: str) -> int:
    return sum(1 for value in values if value == signature)


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
