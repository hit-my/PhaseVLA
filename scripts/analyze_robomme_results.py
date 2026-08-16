#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
ANALYZER_NAME = "openpi.scripts.analyze_robomme_results"

# Frozen from the locked RoboMME policy-learning evaluation task list in
# third_party/robomme_policy_learning/examples/robomme/utils.py and grouped by
# the official RoboMME four-suite taxonomy documented on robomme.github.io.
ROBOMME_TASKS = (
    "BinFill",
    "StopCube",
    "PickXtimes",
    "SwingXtimes",
    "ButtonUnmask",
    "VideoUnmask",
    "VideoUnmaskSwap",
    "ButtonUnmaskSwap",
    "PickHighlight",
    "VideoRepick",
    "VideoPlaceButton",
    "VideoPlaceOrder",
    "MoveCube",
    "InsertPeg",
    "PatternLock",
    "RouteStick",
)

ROBOMME_CATEGORY_TASKS = {
    "Counting": ("BinFill", "StopCube", "PickXtimes", "SwingXtimes"),
    "Permanence": ("ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap"),
    "Reference": ("PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"),
    "Imitation": ("MoveCube", "InsertPeg", "PatternLock", "RouteStick"),
}
ROBOMME_TASK_CATEGORIES = {
    task_name: category
    for category, task_names in ROBOMME_CATEGORY_TASKS.items()
    for task_name in task_names
}

REQUIRED_RESULT_FIELDS = (
    "experiment_id",
    "episode_id",
    "success",
    "checkpoint",
    "config",
    "provenance",
)
FORMAL_PROFILE_TYPE = "futuremamba_task15_profile"
FORMAL_PROFILE_FIELDS = (
    "profile_type",
    "method_id",
    "train_seed",
    "config_name",
    "bundle_identity",
    "runtime",
    "source_commits",
    "parameters",
    "memory_state_bytes",
    "latency_ms",
    "inference_peak_memory_bytes",
    "training_peak_memory_bytes",
    "episode_average_query_ms",
    "episode_timing",
    "flops",
    "training_memory_artifact",
    "measurement_counts",
)
ERROR_OUTCOMES = frozenset({"unknown", "error"})
FAILURE_OUTCOMES = frozenset({"failure", "fail", "failed", "timeout", "false"})
SUCCESS_OUTCOMES = frozenset({"success", "true"})
TASK_ALIASES = ("task", "task_name")
CATEGORY_ALIASES = ("category", "task_category")
METHOD_ALIASES = ("method_id", "method")

_T95 = {
    1: 12.706204736432095,
    2: 4.302652729696142,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.570581835636314,
    6: 2.446911848791681,
    7: 2.3646242510102993,
    8: 2.306004135204166,
    9: 2.2621571627409915,
    10: 2.2281388519649385,
    11: 2.200985160091638,
    12: 2.1788128296634177,
    13: 2.160368656461013,
    14: 2.1447866879169273,
    15: 2.131449545559323,
    16: 2.1199052992210112,
    17: 2.1098155778331806,
    18: 2.10092204024096,
    19: 2.093024054408263,
    20: 2.0859634472658364,
    21: 2.079613844727662,
    22: 2.0738730679040147,
    23: 2.0686576104190406,
    24: 2.0638985616280205,
    25: 2.059538552753294,
    26: 2.055529438642871,
    27: 2.0518305164802833,
    28: 2.048407141795244,
    29: 2.045229642132703,
}
_NORMAL_975 = 1.959963984540054


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _optional_string(row: Mapping[str, Any], field: str) -> str | None:
    value = row.get(field)
    if value is None:
        return None
    return _require_nonempty_string(value, field)


def _first_present(row: Mapping[str, Any], aliases: Sequence[str], label: str) -> Any:
    for field in aliases:
        if field in row:
            return row[field]
    raise ValueError(f"manifest experiment missing required field {label}")


def _result_task(row: Mapping[str, Any]) -> Any:
    for field in ("task_name", "task"):
        if field in row:
            return row[field]
    raise ValueError("result record missing required field task_name")


def _task_order_key(task_name: str) -> int:
    try:
        return ROBOMME_TASKS.index(task_name)
    except ValueError:
        return len(ROBOMME_TASKS)


def _manifest_experiments(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = manifest.get("experiments")
    if not isinstance(raw, list) or not raw:
        raise ValueError("manifest experiments must be a non-empty list")
    return [_require_mapping(row, f"manifest experiments[{index}]") for index, row in enumerate(raw)]


def _identity_key(experiment: Mapping[str, Any], episode_identity: Mapping[str, Any]) -> str:
    identity = experiment.get("identity")
    if identity is None:
        identity = {
            "method_id": episode_identity["method_id"],
            "train_seed": episode_identity["train_seed"],
            "split": episode_identity["split"],
            "task_name": episode_identity["task_name"],
            "episode_id": episode_identity["episode_id"],
            "checkpoint": experiment.get("checkpoint"),
            "config": experiment.get("config"),
            "provenance": experiment.get("provenance"),
        }
    return _canonical_json(identity)


def _flat_experiment_identity(experiment: Mapping[str, Any], index: int) -> dict[str, Any]:
    experiment_id = _require_nonempty_string(
        experiment.get("experiment_id"), f"manifest experiments[{index}] experiment_id"
    )
    method_id = _require_nonempty_string(_first_present(experiment, METHOD_ALIASES, "method_id"), f"{experiment_id} method_id")
    train_seed = _require_int(experiment.get("train_seed"), f"{experiment_id} train_seed")
    task_name = _require_nonempty_string(_first_present(experiment, TASK_ALIASES, "task"), f"{experiment_id} task")
    if task_name not in ROBOMME_TASK_CATEGORIES:
        raise ValueError(f"manifest experiment {experiment_id!r} contains unknown RoboMME task {task_name!r}")
    category = _require_nonempty_string(_first_present(experiment, CATEGORY_ALIASES, "category"), f"{experiment_id} category")
    expected_category = ROBOMME_TASK_CATEGORIES[task_name]
    if category != expected_category:
        raise ValueError(
            f"category mismatch for experiment_id {experiment_id!r}: got {category!r}, expected {expected_category!r}"
        )
    split = _require_nonempty_string(experiment.get("split"), f"{experiment_id} split")
    episode_id = _require_int(experiment.get("episode_id"), f"{experiment_id} episode_id")
    _require_mapping(experiment.get("checkpoint"), f"{experiment_id} checkpoint")
    _require_mapping(experiment.get("config"), f"{experiment_id} config")
    _require_mapping(experiment.get("provenance"), f"{experiment_id} provenance")
    return {
        "experiment_id": experiment_id,
        "method_id": method_id,
        "method": _optional_string(experiment, "method"),
        "variant": _optional_string(experiment, "variant"),
        "axis": _optional_string(experiment, "axis"),
        "train_seed": train_seed,
        "task_name": task_name,
        "category": category,
        "split": split,
        "episode_id": episode_id,
    }


def _validate_manifest(manifest: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    experiments = _manifest_experiments(manifest)
    seen_ids: dict[str, int] = {}
    seen_identities: dict[str, str] = {}
    seen_episode_identities: dict[tuple[str, int, str, str, int], str] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for index, experiment in enumerate(experiments):
        if "episodes" in experiment:
            raise ValueError(
                "manifest experiments must be flat per-episode rows; found unsupported episodes array "
                f"in manifest experiments[{index}]"
            )
        identity = _flat_experiment_identity(experiment, index)
        experiment_id = str(identity["experiment_id"])
        if experiment_id in seen_ids:
            raise ValueError(
                f"duplicate experiment_id {experiment_id!r} at manifest experiments[{seen_ids[experiment_id]}] "
                f"and experiments[{index}]"
            )
        seen_ids[experiment_id] = index
        manifest_identity = _identity_key(experiment, identity)
        if manifest_identity in seen_identities:
            raise ValueError(
                f"duplicate manifest identity for experiment_id {experiment_id!r} and "
                f"{seen_identities[manifest_identity]!r}"
            )
        seen_identities[manifest_identity] = experiment_id
        episode_identity = (
            str(identity["method_id"]),
            int(identity["train_seed"]),
            str(identity["split"]),
            str(identity["task_name"]),
            int(identity["episode_id"]),
        )
        if episode_identity in seen_episode_identities:
            raise ValueError(
                f"duplicate manifest episode identity for experiment_id {experiment_id!r} and "
                f"{seen_episode_identities[episode_identity]!r}"
            )
        seen_episode_identities[episode_identity] = experiment_id
        normalized = dict(experiment)
        normalized["_identity"] = identity
        by_id[experiment_id] = normalized
    return list(by_id.values()), by_id


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"file not found: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in {path}: {error}") from error


def _load_result_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError as error:
            raise ValueError(f"file not found: {path}") from error
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL record in {path}:{line_number}: {error}") from error
            rows.append(_require_mapping(value, f"result record {path}:{line_number}"))
        return rows

    value = _read_json(path)
    if isinstance(value, list):
        return [_require_mapping(row, f"result record {path}[{index}]") for index, row in enumerate(value)]
    if isinstance(value, dict):
        for field in ("results", "episodes", "records"):
            rows = value.get(field)
            if isinstance(rows, list):
                return [
                    _require_mapping(row, f"result record {path}.{field}[{index}]")
                    for index, row in enumerate(rows)
                ]
        if "experiment_id" in value and "episode_id" in value:
            return [dict(value)]
    raise ValueError(f"result file {path} must be a JSON array/object or JSONL records")


def load_results(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_load_result_file(path))
    return rows


def _normalize_success(row: Mapping[str, Any], experiment_id: str, task_name: str, episode_id: int) -> bool:
    if "status" in row and str(row["status"]).lower() == "error":
        raise ValueError(
            f"error status for experiment_id={experiment_id!r} task_name={task_name!r} episode_id={episode_id}"
        )
    if "error" in row and row["error"] not in (None, ""):
        raise ValueError(
            f"error result for experiment_id={experiment_id!r} task_name={task_name!r} episode_id={episode_id}: "
            f"{row['error']}"
        )
    success = row.get("success")
    if not isinstance(success, bool):
        raise ValueError(
            f"success must be boolean for experiment_id={experiment_id!r} task_name={task_name!r} "
            f"episode_id={episode_id}"
        )
    outcome = row.get("outcome")
    if outcome is None:
        return success
    if not isinstance(outcome, str):
        raise ValueError(
            f"outcome must be a string for experiment_id={experiment_id!r} task_name={task_name!r} "
            f"episode_id={episode_id}"
        )
    normalized = outcome.lower()
    if normalized in SUCCESS_OUTCOMES:
        if success is not True:
            raise ValueError(
                f"success disagrees with outcome for experiment_id={experiment_id!r} task_name={task_name!r} "
                f"episode_id={episode_id}"
            )
        return True
    if normalized in FAILURE_OUTCOMES:
        if success is not False:
            raise ValueError(
                f"success disagrees with outcome for experiment_id={experiment_id!r} task_name={task_name!r} "
                f"episode_id={episode_id}"
            )
        return False
    if normalized in ERROR_OUTCOMES:
        raise ValueError(
            f"unknown outcome/error for experiment_id={experiment_id!r} task_name={task_name!r} "
            f"episode_id={episode_id}"
        )
    raise ValueError(
        f"unknown outcome for experiment_id={experiment_id!r} task_name={task_name!r} episode_id={episode_id}: "
        f"{outcome!r}"
    )


def _validate_result_identity(row: Mapping[str, Any], experiment: Mapping[str, Any]) -> None:
    identity = experiment["_identity"]
    experiment_id = str(identity["experiment_id"])
    task_name = str(identity["task_name"])
    episode_id = int(identity["episode_id"])
    for field in ("checkpoint", "config", "provenance"):
        if field not in row:
            raise ValueError(
                f"result missing {field} for experiment_id={experiment_id!r} task_name={task_name!r} "
                f"episode_id={episode_id}"
            )
        if row[field] != experiment[field]:
            raise ValueError(
                f"result {field} mismatch for experiment_id={experiment_id!r} task_name={task_name!r} "
                f"episode_id={episode_id}"
            )
    if "method_id" in row and row["method_id"] != identity["method_id"]:
        raise ValueError(f"result method_id mismatch for experiment_id={experiment_id!r}")
    if "train_seed" in row and row["train_seed"] != identity["train_seed"]:
        raise ValueError(f"result train_seed mismatch for experiment_id={experiment_id!r}")
    result_task = _result_task(row)
    if result_task != task_name:
        raise ValueError(
            f"result task mismatch for experiment_id={experiment_id!r}: got {result_task!r}, expected {task_name!r}"
        )
    if row.get("episode_id") != episode_id:
        raise ValueError(
            f"result episode_id mismatch for experiment_id={experiment_id!r}: got {row.get('episode_id')!r}, expected {episode_id!r}"
        )


def _validate_results(
    manifest_experiments: Sequence[Mapping[str, Any]],
    experiments_by_id: Mapping[str, Mapping[str, Any]],
    records: Iterable[Mapping[str, Any]],
) -> dict[str, bool]:
    expected_ids = {str(experiment["_identity"]["experiment_id"]) for experiment in manifest_experiments}
    observed: dict[str, bool] = {}
    for index, row in enumerate(records):
        record = _require_mapping(row, f"result record {index}")
        missing = [field for field in REQUIRED_RESULT_FIELDS if field not in record]
        if missing:
            raise ValueError(f"result record {index} missing required field(s): {', '.join(missing)}")
        experiment_id = _require_nonempty_string(record.get("experiment_id"), f"result record {index} experiment_id")
        if experiment_id not in experiments_by_id:
            raise ValueError(f"unknown experiment_id {experiment_id!r} in result record {index}")
        if experiment_id in observed:
            raise ValueError(f"duplicate result for experiment_id={experiment_id!r}")
        experiment = experiments_by_id[experiment_id]
        identity = experiment["_identity"]
        _validate_result_identity(record, experiment)
        observed[experiment_id] = _normalize_success(
            record,
            experiment_id,
            str(identity["task_name"]),
            int(identity["episode_id"]),
        )

    missing_ids = sorted(expected_ids - set(observed))
    if missing_ids:
        missing_id = missing_ids[0]
        identity = experiments_by_id[missing_id]["_identity"]
        raise ValueError(
            f"missing result for experiment_id={missing_id!r} task_name={identity['task_name']!r} "
            f"episode_id={identity['episode_id']}; missing_count={len(missing_ids)}"
        )
    return observed


def _rate(successes: Sequence[bool]) -> dict[str, Any]:
    n = len(successes)
    success_count = sum(1 for value in successes if value)
    return {
        "n": n,
        "success_count": success_count,
        "success_rate": success_count / n if n else None,
    }


def _sample_std(values: Sequence[float]) -> float:
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def summarize_success_rates(values: Sequence[float]) -> dict[str, Any]:
    rates = [float(value) for value in values]
    if not rates:
        raise ValueError("cannot summarize zero train seeds")
    seed_count = len(rates)
    mean = sum(rates) / seed_count
    result: dict[str, Any] = {
        "seed_count": seed_count,
        "mean_success_rate": mean,
    }
    if seed_count == 1:
        result["sample_std_success_rate"] = None
        result["ci95"] = {
            "status": "insufficient",
            "confidence": 0.95,
            "reason": "requires_at_least_two_train_seeds",
        }
        return result

    sample_std = _sample_std(rates)
    df = seed_count - 1
    if seed_count < 30:
        method = "student_t"
        critical_value = _T95[df]
    else:
        method = "normal"
        critical_value = _NORMAL_975
    half_width = critical_value * sample_std / math.sqrt(seed_count)
    result["sample_std_success_rate"] = sample_std
    result["ci95"] = {
        "status": "ok",
        "confidence": 0.95,
        "method": method,
        "critical_value": critical_value,
        "half_width": half_width,
        "low": mean - half_width,
        "high": mean + half_width,
        "df": df if method == "student_t" else None,
    }
    return result


def _group_metadata(rows: Sequence[Mapping[str, Any]], field: str) -> str | None:
    values = {row["_identity"].get(field) for row in rows if row["_identity"].get(field) is not None}
    if not values:
        return None
    if len(values) > 1:
        method_id = rows[0]["_identity"]["method_id"]
        train_seed = rows[0]["_identity"]["train_seed"]
        raise ValueError(f"inconsistent {field} for method_id={method_id!r} train_seed={train_seed!r}")
    return next(iter(values))


def _summarize_seed(rows: Sequence[Mapping[str, Any]], observed: Mapping[str, bool]) -> dict[str, Any]:
    first_identity = rows[0]["_identity"]
    episodes_by_task: dict[str, list[bool]] = {task_name: [] for task_name in ROBOMME_TASKS}
    for row in rows:
        identity = row["_identity"]
        episodes_by_task[str(identity["task_name"])].append(observed[str(identity["experiment_id"])])

    present_tasks = [task_name for task_name in ROBOMME_TASKS if episodes_by_task[task_name]]
    per_task = {task_name: _rate(episodes_by_task[task_name]) for task_name in present_tasks}
    categories: dict[str, dict[str, Any]] = {}
    for category, task_names in ROBOMME_CATEGORY_TASKS.items():
        successes = [success for task_name in task_names for success in episodes_by_task[task_name]]
        if successes:
            categories[category] = _rate(successes)
    overall_successes = [success for task_name in ROBOMME_TASKS for success in episodes_by_task[task_name]]
    experiment_ids = sorted(str(row["_identity"]["experiment_id"]) for row in rows)
    return {
        "method_id": first_identity["method_id"],
        "train_seed": first_identity["train_seed"],
        "split": _group_metadata(rows, "split"),
        "experiment_count": len(experiment_ids),
        "experiment_ids": experiment_ids,
        "per_task": per_task,
        "categories": categories,
        "overall": _rate(overall_successes),
    }


def _aggregate_metric(seed_summaries: Sequence[Mapping[str, Any]], section: str, key: str) -> dict[str, Any]:
    rates = [float(seed[section][key]["success_rate"]) for seed in seed_summaries if key in seed[section]]
    raw_n = [int(seed[section][key]["n"]) for seed in seed_summaries if key in seed[section]]
    raw_success = [int(seed[section][key]["success_count"]) for seed in seed_summaries if key in seed[section]]
    summary = summarize_success_rates(rates)
    summary["raw_n_by_seed"] = raw_n
    summary["raw_success_count_by_seed"] = raw_success
    summary["total_n"] = sum(raw_n)
    summary["total_success_count"] = sum(raw_success)
    return summary


def _aggregate_overall(seed_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rates = [float(seed["overall"]["success_rate"]) for seed in seed_summaries]
    raw_n = [int(seed["overall"]["n"]) for seed in seed_summaries]
    raw_success = [int(seed["overall"]["success_count"]) for seed in seed_summaries]
    summary = summarize_success_rates(rates)
    summary["raw_n_by_seed"] = raw_n
    summary["raw_success_count_by_seed"] = raw_success
    summary["total_n"] = sum(raw_n)
    summary["total_success_count"] = sum(raw_success)
    return summary


def _aggregate_method(seed_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_task: dict[str, Any] = {}
    for task_name in ROBOMME_TASKS:
        if any(task_name in seed["per_task"] for seed in seed_summaries):
            per_task[task_name] = _aggregate_metric(seed_summaries, "per_task", task_name)
    categories: dict[str, Any] = {}
    for category in ROBOMME_CATEGORY_TASKS:
        if any(category in seed["categories"] for seed in seed_summaries):
            categories[category] = _aggregate_metric(seed_summaries, "categories", category)
    return {"per_task": per_task, "categories": categories, "overall": _aggregate_overall(seed_summaries)}


def _profile_positive_int(value: Any, label: str) -> int:
    result = _require_int(value, label)
    if result <= 0:
        raise ValueError(f"{label} must be positive")
    return result


def _profile_nonnegative_int(value: Any, label: str) -> int:
    result = _require_int(value, label)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _profile_positive_number(value: Any, label: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{label} must be a positive finite number")
    return value


def _profile_metric_pair(value: Any, label: str) -> dict[str, Any]:
    metric = _require_mapping(value, label)
    _profile_positive_number(metric.get("median"), f"{label}.median")
    _profile_positive_number(metric.get("p95"), f"{label}.p95")
    return metric


def _validate_formal_profile(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    label = f"profile row {index}"
    missing = [field for field in FORMAL_PROFILE_FIELDS if field not in row]
    if missing:
        raise ValueError(f"{label} missing required formal field {missing[0]}")
    if row.get("profile_type") != FORMAL_PROFILE_TYPE:
        raise ValueError(f"{label} profile_type must be {FORMAL_PROFILE_TYPE!r}")
    if _require_int(row.get("schema_version"), f"{label} schema_version") != 1:
        raise ValueError(f"{label} schema_version must be 1")
    _require_nonempty_string(row.get("config_name"), f"{label} config_name")

    bundle = _require_mapping(row.get("bundle_identity"), f"{label} bundle_identity")
    for field in ("bundle_path", "bundle_metadata_sha256", "base_checkpoint_checksum"):
        _require_nonempty_string(bundle.get(field), f"{label} bundle_identity.{field}")
    runtime = _require_mapping(row.get("runtime"), f"{label} runtime")
    for field in ("device", "torch_version"):
        _require_nonempty_string(runtime.get(field), f"{label} runtime.{field}")
    commits = _require_mapping(row.get("source_commits"), f"{label} source_commits")
    for field in ("mamba", "robomme_policy", "robomme_benchmark"):
        _require_nonempty_string(commits.get(field), f"{label} source_commits.{field}")

    parameters = _require_mapping(row.get("parameters"), f"{label} parameters")
    _profile_positive_int(parameters.get("total"), f"{label} parameters.total")
    _profile_nonnegative_int(parameters.get("trainable"), f"{label} parameters.trainable")
    _profile_positive_int(parameters.get("plugin"), f"{label} parameters.plugin")
    _profile_positive_number(parameters.get("plugin_ratio"), f"{label} parameters.plugin_ratio")
    memory = _require_mapping(row.get("memory_state_bytes"), f"{label} memory_state_bytes")
    if not isinstance(memory.get("layers"), list) or not memory["layers"]:
        raise ValueError(f"{label} memory_state_bytes.layers must be a non-empty list")
    _profile_positive_int(memory.get("total"), f"{label} memory_state_bytes.total")

    latency = _require_mapping(row.get("latency_ms"), f"{label} latency_ms")
    _profile_metric_pair(latency.get("memory_step"), f"{label} latency_ms.memory_step")
    _profile_metric_pair(latency.get("action_chunk_20"), f"{label} latency_ms.action_chunk_20")
    _profile_positive_int(row.get("inference_peak_memory_bytes"), f"{label} inference_peak_memory_bytes")
    _profile_positive_int(row.get("training_peak_memory_bytes"), f"{label} training_peak_memory_bytes")
    _profile_positive_number(row.get("episode_average_query_ms"), f"{label} episode_average_query_ms")

    episode = _require_mapping(row.get("episode_timing"), f"{label} episode_timing")
    _profile_positive_int(episode.get("query_count"), f"{label} episode_timing.query_count")
    _profile_positive_int(episode.get("episode_count"), f"{label} episode_timing.episode_count")
    if episode.get("measurement_source") != "futuremamba_policy_timing":
        raise ValueError(f"{label} episode_timing.measurement_source must be 'futuremamba_policy_timing'")
    _require_nonempty_string(episode.get("artifact_sha256"), f"{label} episode_timing.artifact_sha256")

    flops = _require_mapping(row.get("flops"), f"{label} flops")
    for field in ("base_flops", "plugin_flops", "futuremamba_total_flops"):
        _profile_positive_int(flops.get(field), f"{label} flops.{field}")
    _profile_positive_number(flops.get("relative_plugin_over_base"), f"{label} flops.relative_plugin_over_base")
    if flops.get("measurement_source") not in {"tool_analysis", "measured"}:
        raise ValueError(f"{label} flops.measurement_source must be tool_analysis or measured")
    for field in ("tool", "artifact_sha256"):
        _require_nonempty_string(flops.get(field), f"{label} flops.{field}")

    training = _require_mapping(row.get("training_memory_artifact"), f"{label} training_memory_artifact")
    if training.get("measurement_source") not in {
        "torch.cuda.max_memory_allocated",
        "torch.cuda.max_memory_reserved",
    }:
        raise ValueError(f"{label} training_memory_artifact.measurement_source must be a PyTorch CUDA allocator peak")
    for field in ("producer", "artifact_sha256"):
        _require_nonempty_string(training.get(field), f"{label} training_memory_artifact.{field}")
    counts = _require_mapping(row.get("measurement_counts"), f"{label} measurement_counts")
    _profile_nonnegative_int(counts.get("warmup"), f"{label} measurement_counts.warmup")
    _profile_positive_int(counts.get("measured"), f"{label} measurement_counts.measured")
    if _profile_positive_int(counts.get("batch_size"), f"{label} measurement_counts.batch_size") != 1:
        raise ValueError(f"{label} measurement_counts.batch_size must be 1")
    return dict(row)


def _profile_rows(profile: Mapping[str, Any] | list[Any] | None) -> dict[tuple[str, int], dict[str, Any]]:
    if profile is None:
        return {}
    raw_rows = profile.get("profiles") if isinstance(profile, dict) else profile
    if not isinstance(raw_rows, list):
        raise ValueError("profile JSON must contain a profiles list")
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    for index, raw_row in enumerate(raw_rows):
        row = _require_mapping(raw_row, f"profile row {index}")
        method_id = _require_nonempty_string(row.get("method_id"), f"profile row {index} method_id")
        train_seed = _require_int(row.get("train_seed"), f"profile row {index} train_seed")
        key = (method_id, train_seed)
        if key in rows:
            raise ValueError(f"duplicate profile identity method_id={method_id!r} train_seed={train_seed!r}")
        rows[key] = _validate_formal_profile(row, index)
    return rows


def _attach_profiles(methods: dict[str, Any], profile: Mapping[str, Any] | list[Any] | None) -> None:
    profiles = _profile_rows(profile)
    if not profiles:
        return
    expected = {
        (method_summary["method_id"], int(seed))
        for method_summary in methods.values()
        for seed in method_summary["train_seeds"]
    }
    unknown = sorted(set(profiles) - expected)
    if unknown:
        method_id, train_seed = unknown[0]
        raise ValueError(f"profile contains unknown identity method_id={method_id!r} train_seed={train_seed!r}")
    for (method_id, train_seed), formal_profile in profiles.items():
        methods[method_id]["train_seeds"][str(train_seed)]["profile"] = formal_profile


def _validate_heatmap_inputs(methods: Mapping[str, Any]) -> None:
    if "pi05_baseline" not in methods:
        raise ValueError("heatmap delta requires pi05_baseline method")
    for method_id, method_summary in methods.items():
        per_task = method_summary["aggregate"]["per_task"]
        missing = [task_name for task_name in ROBOMME_TASKS if task_name not in per_task]
        if missing:
            raise ValueError(
                f"method_id {method_id!r} lacks complete 16-task coverage for heatmap; missing {missing[0]!r}"
            )
        for seed, seed_summary in method_summary["train_seeds"].items():
            seed_missing = [task_name for task_name in ROBOMME_TASKS if task_name not in seed_summary["per_task"]]
            if seed_missing:
                raise ValueError(
                    f"method_id {method_id!r} train_seed {seed!r} lacks complete 16-task coverage; "
                    f"missing {seed_missing[0]!r}"
                )


def _heatmap(methods: Mapping[str, Any]) -> dict[str, Any]:
    _validate_heatmap_inputs(methods)
    baseline = methods["pi05_baseline"]["aggregate"]["per_task"]
    cells: list[dict[str, Any]] = []
    for method_id in sorted(methods):
        per_task = methods[method_id]["aggregate"]["per_task"]
        for task_name in ROBOMME_TASKS:
            task_summary = per_task[task_name]
            baseline_rate = float(baseline[task_name]["mean_success_rate"])
            rate = float(task_summary["mean_success_rate"])
            cells.append(
                {
                    "method_id": method_id,
                    "method": methods[method_id].get("method"),
                    "variant": methods[method_id].get("variant"),
                    "axis": methods[method_id].get("axis"),
                    "task_name": task_name,
                    "category": ROBOMME_TASK_CATEGORIES[task_name],
                    "success_rate_mean": rate,
                    "seed_count": task_summary["seed_count"],
                    "baseline_method_id": "pi05_baseline",
                    "baseline_success_rate_mean": baseline_rate,
                    "delta_success_rate_vs_pi05_baseline": rate - baseline_rate,
                }
            )
    return {
        "baseline_method_id": "pi05_baseline",
        "task_order": list(ROBOMME_TASKS),
        "category_order": list(ROBOMME_CATEGORY_TASKS),
        "cells": cells,
    }


def _schema_block() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "analyzer": ANALYZER_NAME,
        "manifest_grain": "one flat experiment row per preregistered episode",
        "result_grain": "one row per preregistered experiment_id",
        "success_contract": "success must be boolean; unknown/error rows fail closed",
        "ci95": {
            "confidence": 0.95,
            "unit": "train_seed_success_rate",
            "single_seed_policy": "insufficient",
            "std": "sample_std_ddof_1",
            "student_t_threshold": "2_to_29_train_seeds",
            "normal_threshold": "30_or_more_train_seeds",
        },
        "profile_grain": "formal futuremamba_task15_profile at method_id/train_seed",
        "profile_required_fields": list(FORMAL_PROFILE_FIELDS),
    }


def _method_group_metadata(rows: Sequence[Mapping[str, Any]], field: str) -> str | None:
    values = {row["_identity"].get(field) for row in rows if row["_identity"].get(field) is not None}
    if not values:
        return None
    if len(values) > 1:
        method_id = rows[0]["_identity"]["method_id"]
        raise ValueError(f"inconsistent {field} for method_id={method_id!r}")
    return next(iter(values))


def _merge_method_metadata(method_summary: dict[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    for field in ("method", "variant", "axis"):
        value = _method_group_metadata(rows, field)
        if method_summary[field] is None:
            method_summary[field] = value
        elif value is not None and method_summary[field] != value:
            method_id = method_summary["method_id"]
            raise ValueError(f"inconsistent {field} for method_id={method_id!r}")


def analyze_results(
    manifest: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    profile: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    manifest_obj = _require_mapping(dict(manifest), "manifest")
    manifest_experiments, experiments_by_id = _validate_manifest(manifest_obj)
    observed = _validate_results(manifest_experiments, experiments_by_id, records)

    rows_by_method_seed: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for experiment in manifest_experiments:
        identity = experiment["_identity"]
        rows_by_method_seed.setdefault((str(identity["method_id"]), int(identity["train_seed"])), []).append(experiment)

    methods: dict[str, Any] = {}
    for (method_id, train_seed), rows in sorted(rows_by_method_seed.items()):
        method_summary = methods.setdefault(
            method_id,
            {"method_id": method_id, "method": None, "variant": None, "axis": None, "train_seeds": {}},
        )
        _merge_method_metadata(method_summary, rows)
        seed_key = str(train_seed)
        if seed_key in method_summary["train_seeds"]:
            raise ValueError(f"duplicate train_seed {seed_key!r} for method_id {method_id!r}")
        method_summary["train_seeds"][seed_key] = _summarize_seed(rows, observed)

    for method_summary in methods.values():
        seed_summaries = [
            method_summary["train_seeds"][seed]
            for seed in sorted(method_summary["train_seeds"], key=lambda value: int(value))
        ]
        method_summary["aggregate"] = _aggregate_method(seed_summaries)

    profile_mapping = profile if profile is None or isinstance(profile, (dict, list)) else list(profile)
    _attach_profiles(methods, profile_mapping)

    return {
        "schema": _schema_block(),
        "provenance": {
            "manifest_schema_version": manifest_obj.get("schema_version"),
            "manifest_provenance": manifest_obj.get("provenance", {}),
            "official_task_source": "third_party/robomme_policy_learning/examples/robomme/utils.py:TASK_NAME_LIST",
            "official_taxonomy_source": "https://robomme.github.io/",
        },
        "tasks": {
            "order": list(ROBOMME_TASKS),
            "categories": {category: list(tasks) for category, tasks in ROBOMME_CATEGORY_TASKS.items()},
        },
        "methods": methods,
        "heatmap": _heatmap(methods),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly aggregate preregistered RoboMME episode results into deterministic JSON."
    )
    parser.add_argument("--manifest", required=True, type=Path, help="Preregistered flat experiment manifest JSON.")
    parser.add_argument(
        "--results",
        required=True,
        type=Path,
        nargs="+",
        help="One or more episode result JSON/JSONL files.",
    )
    parser.add_argument("--profile", type=Path, help="Optional measured profile JSON at method_id/train_seed grain.")
    parser.add_argument("--output", type=Path, help="Output summary JSON path; stdout when omitted.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = _require_mapping(_read_json(args.manifest), "manifest")
        results = load_results(args.results)
        profile = _read_json(args.profile) if args.profile is not None else None
        summary = analyze_results(manifest, results, profile)
        payload = _stable_json(summary)
        if args.output is None:
            sys.stdout.write(payload)
        else:
            args.output.write_text(payload, encoding="utf-8")
    except ValueError as error:
        print(f"analyze_robomme_results: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
