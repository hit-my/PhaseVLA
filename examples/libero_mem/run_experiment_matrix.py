from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
DEFAULT_TRAIN_SEEDS = (0, 1, 2)
DEFAULT_ROLLOUT_SEEDS = (10_001, 10_002, 10_003)
DEFAULT_ROLLOUT_TRIALS_PER_TASK = 50
DEFAULT_TASK_SUITE = "libero_mem"
DEFAULT_BASE_CONFIG: dict[str, Any] = {
    "model": "futuremamba",
    "base_policy": "task_adapted_pi05",
    "pi05": True,
    "discrete_state_input": True,
    "memory_backend": "mamba",
    "memory_input": "token_action",
    "conditioning_pool": "last_valid",
    "decoder_mode": "handoff",
    "coupling": "hard",
    "handoff_ratio": 0.2,
    "num_denoise_steps": 10,
    "progress_depth_fraction": "1/4",
    "progress_depth": None,
    "use_prefix_cache": True,
    "reset_memory_every_query": False,
    "history_intervention": "normal",
    "oracle_progress": False,
    "handoff_loss_weight": 1.0,
    "boundary_loss_weight": 0.1,
    "frame_stack_window": 4,
    "bptt_window_queries": None,
    "parameter_matched": None,
}
PARAMETER_MATCH_BACKENDS = frozenset({"gru", "lstm", "frame_stack"})
IDENTITY_FIELDS = (
    "table",
    "name",
    "train_seed",
    "rollout_seeds",
    "rollout_trials_per_task",
    "task_suite",
    "config",
    "parameter_match",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row.get(field) for field in IDENTITY_FIELDS if field in row}


def stable_experiment_id(row: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_canonical_json(_canonical_identity(row)).encode("utf-8")).hexdigest()[:12]
    table = str(row.get("table", "experiment")).replace("_", "-")
    name = str(row.get("name", "unnamed")).replace("_", "-")
    seed = row.get("train_seed")
    suffix = f"-seed{seed}" if seed is not None else ""
    return f"{table}-{name}{suffix}-{digest}"


def _with_config(**overrides: Any) -> dict[str, Any]:
    config = dict(DEFAULT_BASE_CONFIG)
    config.update(overrides)
    return config


def _row(
    name: str,
    *,
    table: str = "ablation",
    train_seed: int | None = None,
    rollout_seeds: Sequence[int] = DEFAULT_ROLLOUT_SEEDS,
    rollout_trials_per_task: int = DEFAULT_ROLLOUT_TRIALS_PER_TASK,
    task_suite: str = DEFAULT_TASK_SUITE,
    config: Mapping[str, Any] | None = None,
    parameter_match: Mapping[str, int] | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "table": table,
        "name": name,
        "train_seed": train_seed,
        "rollout_seeds": list(rollout_seeds),
        "rollout_trials_per_task": int(rollout_trials_per_task),
        "task_suite": task_suite,
        "config": dict(config or DEFAULT_BASE_CONFIG),
    }
    if parameter_match is not None:
        row["parameter_match"] = dict(parameter_match)
    if notes is not None:
        row["notes"] = notes
    row = validate_parameter_match(row)
    row["experiment_id"] = stable_experiment_id(row)
    return row


def validate_parameter_match(row: Mapping[str, Any], *, tolerance: float = 0.05) -> dict[str, Any]:
    result = dict(row)
    config = dict(result.get("config", {}))
    match = result.get("parameter_match")
    if match is None:
        backend = config.get("memory_backend")
        if backend in PARAMETER_MATCH_BACKENDS:
            config["parameter_matched"] = False
            result["parameter_match_error"] = None
        else:
            config["parameter_matched"] = None
            result["parameter_match_error"] = None
        result["config"] = config
        return result

    reference = int(match["reference"])
    candidate = int(match["candidate"])
    if reference <= 0:
        raise ValueError(f"parameter_match reference must be positive for {result.get('name')!r}")
    error = abs(candidate - reference) / reference
    matched = error <= tolerance
    claimed = result.get("parameter_matched", config.get("parameter_matched"))
    if claimed is True and not matched:
        raise ValueError(
            f"parameter_matched=true is invalid for {result.get('name')!r}: "
            f"relative error {error:.6f} exceeds tolerance {tolerance:.6f}"
        )
    config["parameter_matched"] = matched
    result["parameter_matched"] = matched
    result["config"] = config
    result["parameter_match_error"] = error
    return result


def _main_rows(
    *,
    train_seeds: Sequence[int],
    rollout_seeds: Sequence[int],
    rollout_trials_per_task: int,
) -> list[dict[str, Any]]:
    return [
        _row(
            "main_futuremamba",
            table="main",
            train_seed=seed,
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
            config=_with_config(preregistered_primary=True),
            notes="Primary FutureMamba table row: train seed varies; rollout seeds/trials stay fixed.",
        )
        for seed in train_seeds
    ]


def _baseline_rows(rollout_seeds: Sequence[int], rollout_trials_per_task: int) -> list[dict[str, Any]]:
    return [
        _row(
            "frozen_task_adapted_pi05",
            table="baseline",
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
            config=_with_config(model="pi05", memory_backend="none", handoff_ratio=0.0, expected_pi05_equivalence=True),
            notes="Frozen task-adapted pi0.5 without FutureMamba plugin updates.",
        ),
        _row(
            "rho_0_equivalence",
            table="baseline",
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
            config=_with_config(handoff_ratio=0.0, expected_pi05_equivalence=True),
            notes="Numerical equivalence guard for K=0 with the same observation and noise.",
        ),
        _row(
            "recent_frame_stack",
            table="baseline",
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
            config=_with_config(memory_backend="frame_stack", frame_stack_window=4),
            parameter_match={"reference": 100_000, "candidate": 103_000},
        ),
        _row(
            "gru_memory",
            table="baseline",
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
            config=_with_config(memory_backend="gru"),
            parameter_match={"reference": 100_000, "candidate": 99_000},
        ),
        _row(
            "lstm_memory",
            table="baseline",
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
            config=_with_config(memory_backend="lstm"),
            parameter_match={"reference": 100_000, "candidate": 104_500},
        ),
    ]


def _mechanism_rows(rollout_seeds: Sequence[int], rollout_trials_per_task: int) -> list[dict[str, Any]]:
    rows = [
        _row(
            "progress_shared_prefix_no_memory",
            config=_with_config(memory_backend="none", memory_input="token_only"),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
        _row(
            "memory_without_prefix",
            config=_with_config(use_prefix_cache=False),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
        _row(
            "action_expert_memory_full_horizon",
            config=_with_config(decoder_mode="action_memory_full", handoff_ratio=1.0),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
        _row(
            "reset_every_query",
            config=_with_config(reset_memory_every_query=True),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
        _row(
            "shuffled_history",
            config=_with_config(history_intervention="shuffled"),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
        _row(
            "truncated_history",
            config=_with_config(history_intervention="truncated", history_truncation_queries=2),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
        _row(
            "zero_history",
            config=_with_config(history_intervention="zero"),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
        _row(
            "full_horizon_progress",
            config=_with_config(handoff_ratio=1.0),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
        _row(
            "oracle_progress",
            config=_with_config(oracle_progress=True, oracle_inputs=("progress", "object_mask")),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
        _row(
            "handoff_loss_off",
            config=_with_config(handoff_loss_weight=0.0),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
        _row(
            "boundary_loss_off",
            config=_with_config(boundary_loss_weight=0.0),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
    ]
    rows.extend(
        _row(
            f"rho_{str(rho).replace('.', '_')}",
            config=_with_config(handoff_ratio=rho),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        )
        for rho in (0.1, 0.2, 0.3, 0.5, 1.0)
    )
    return rows


def _representation_rows(rollout_seeds: Sequence[int], rollout_trials_per_task: int) -> list[dict[str, Any]]:
    return [
        _row(
            f"pool_{pool}",
            config=_with_config(conditioning_pool=pool),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        )
        for pool in ("last_valid", "attention", "tokens4", "tokens8")
    ] + [
        _row(
            "memory_token_only",
            config=_with_config(memory_input="token_only"),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
        _row(
            "memory_token_action",
            config=_with_config(memory_input="token_action"),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
    ]


def _capacity_rows(rollout_seeds: Sequence[int], rollout_trials_per_task: int) -> list[dict[str, Any]]:
    return [
        _row(
            "depth_1_8",
            config=_with_config(progress_depth_fraction="1/8"),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
        _row(
            "depth_1_4",
            config=_with_config(progress_depth_fraction="1/4"),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
        _row(
            "depth_1_2",
            config=_with_config(progress_depth_fraction="1/2"),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
        _row(
            "bptt_full",
            config=_with_config(bptt_window_queries=None),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
        _row(
            "bptt_truncated",
            config=_with_config(bptt_window_queries=8),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        ),
    ]


def _coupling_rows(rollout_seeds: Sequence[int], rollout_trials_per_task: int) -> list[dict[str, Any]]:
    return [
        _row(
            f"coupling_{coupling}",
            config=_with_config(coupling=coupling),
            rollout_seeds=rollout_seeds,
            rollout_trials_per_task=rollout_trials_per_task,
        )
        for coupling in ("hard", "convex", "residual")
    ]


def build_matrix(
    *,
    train_seeds: Sequence[int] = DEFAULT_TRAIN_SEEDS,
    rollout_seeds: Sequence[int] = DEFAULT_ROLLOUT_SEEDS,
    rollout_trials_per_task: int = DEFAULT_ROLLOUT_TRIALS_PER_TASK,
) -> list[dict[str, Any]]:
    rows = [
        *_main_rows(train_seeds=train_seeds, rollout_seeds=rollout_seeds, rollout_trials_per_task=rollout_trials_per_task),
        *_baseline_rows(rollout_seeds, rollout_trials_per_task),
        *_mechanism_rows(rollout_seeds, rollout_trials_per_task),
        *_representation_rows(rollout_seeds, rollout_trials_per_task),
        *_capacity_rows(rollout_seeds, rollout_trials_per_task),
        *_coupling_rows(rollout_seeds, rollout_trials_per_task),
    ]
    return deduplicate_experiments(rows)


def deduplicate_experiments(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source in rows:
        row = validate_parameter_match(source)
        row["experiment_id"] = stable_experiment_id(row)
        key = stable_experiment_id(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            tmp_file.write(text)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def write_experiment_config(path: str | Path, row: Mapping[str, Any]) -> Path:
    config_row = validate_parameter_match(row)
    config_row["experiment_id"] = stable_experiment_id(config_row)
    payload = _canonical_json(config_row) + "\n"
    target = Path(path)
    _atomic_write_text(target, payload)
    return target


def write_matrix_manifest(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    experiments = deduplicate_experiments(rows)
    rollout_seed_sets = {tuple(row["rollout_seeds"]) for row in experiments}
    trial_counts = {int(row["rollout_trials_per_task"]) for row in experiments}
    if len(rollout_seed_sets) > 1:
        raise ValueError(f"matrix rows must use fixed rollout seeds, got {sorted(rollout_seed_sets)}")
    if len(trial_counts) > 1:
        raise ValueError(f"matrix rows must use fixed rollout trial counts, got {sorted(trial_counts)}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "train_seeds": sorted({row["train_seed"] for row in experiments if row.get("train_seed") is not None}),
        "rollout_seeds": list(next(iter(rollout_seed_sets), ())),
        "rollout_trials_per_task": next(iter(trial_counts), 0),
        "experiments": experiments,
    }
    target = Path(path)
    _atomic_write_text(target, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return target


def _default_runner(command: Sequence[str]) -> int:
    return subprocess.run(command, check=False).returncode


def launch_matrix(
    rows: Iterable[Mapping[str, Any]],
    *,
    output_dir: str | Path,
    train_command: Sequence[str],
    rollout_command: Sequence[str],
    runner: Callable[[Sequence[str]], int] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    runner = _default_runner if runner is None else runner
    launched = 0
    failures: list[dict[str, Any]] = []
    commands: list[list[str]] = []
    for row in deduplicate_experiments(rows):
        exp_dir = Path(output_dir) / row["experiment_id"]
        config_path = write_experiment_config(exp_dir / "config.json", row)
        train = [*train_command, "--config-json", str(config_path)]
        rollout = [*rollout_command, "--config-json", str(config_path)]
        commands.extend([train, rollout])
        if dry_run:
            launched += 1
            continue
        for phase, command in (("train", train), ("rollout", rollout)):
            code = runner(command)
            if code != 0:
                failures.append({"experiment_id": row["experiment_id"], "phase": phase, "returncode": code})
                break
        else:
            launched += 1
    return {"launched": launched, "failures": failures, "commands": commands, "dry_run": dry_run}


def _parse_ints(raw: str) -> tuple[int, ...]:
    if not raw:
        return ()
    return tuple(int(part) for part in raw.split(",") if part)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write and optionally launch the preregistered FutureMamba matrix.")
    parser.add_argument("--output", type=Path, default=Path("experiment_matrix.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/futuremamba_matrix"))
    parser.add_argument("--train-seeds", default=",".join(str(seed) for seed in DEFAULT_TRAIN_SEEDS))
    parser.add_argument("--rollout-seeds", default=",".join(str(seed) for seed in DEFAULT_ROLLOUT_SEEDS))
    parser.add_argument("--rollout-trials-per-task", type=int, default=DEFAULT_ROLLOUT_TRIALS_PER_TASK)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--train-command", nargs="+", default=["python", "scripts/train_futuremamba.py"])
    parser.add_argument("--rollout-command", nargs="+", default=["python", "examples/libero_mem/main.py"])
    args = parser.parse_args(argv)

    rows = build_matrix(
        train_seeds=_parse_ints(args.train_seeds),
        rollout_seeds=_parse_ints(args.rollout_seeds),
        rollout_trials_per_task=args.rollout_trials_per_task,
    )
    manifest = write_matrix_manifest(args.output, rows)
    result = {"manifest": str(manifest), "experiments": len(rows)}
    if args.launch:
        result["launch"] = launch_matrix(
            rows,
            output_dir=args.output_dir,
            train_command=args.train_command,
            rollout_command=args.rollout_command,
            dry_run=False,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
