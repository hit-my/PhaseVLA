from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = Path(__file__).with_name("analyze_robomme_results.py")
spec = importlib.util.spec_from_file_location("analyze_robomme_results", SCRIPT)
assert spec is not None and spec.loader is not None
analyzer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = analyzer
spec.loader.exec_module(analyzer)


OFFICIAL_TASKS = (
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

CATEGORY_TASKS = {
    "Counting": ("BinFill", "StopCube", "PickXtimes", "SwingXtimes"),
    "Permanence": ("ButtonUnmask", "VideoUnmask", "VideoUnmaskSwap", "ButtonUnmaskSwap"),
    "Reference": ("PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"),
    "Imitation": ("MoveCube", "InsertPeg", "PatternLock", "RouteStick"),
}


def _flat_experiment(
    method_id: str,
    train_seed: int,
    task: str,
    episode_id: int,
    *,
    split: str = "test",
    experiment_id: str | None = None,
    category: str | None = None,
    identity: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "experiment_id": experiment_id
        or f"{method_id}-seed{train_seed}-{split}-{task}-ep{episode_id}",
        "method_id": method_id,
        "method": "π0.5 baseline" if method_id == "pi05_baseline" else "FutureMamba",
        "variant": "frozen" if method_id == "pi05_baseline" else "mamba2-uniform6",
        "axis": "baseline" if method_id == "pi05_baseline" else "primary",
        "train_seed": train_seed,
        "task": task,
        "category": category or analyzer.ROBOMME_TASK_CATEGORIES[task],
        "split": split,
        "episode_id": episode_id,
        "checkpoint": {"uri": f"runs/ckpts/{method_id}/{train_seed}", "sha256": f"ckpt-{method_id}-{train_seed}"},
        "config": {"policy": method_id, "memory_backend": "none" if method_id == "pi05_baseline" else "mamba2"},
        "provenance": {
            "robomme_policy_commit": "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b",
            "robomme_benchmark_commit": "856bc3a189d4172f3f47dbee4424d585f8d78db3",
            "conversion_manifest_sha256": f"manifest-{method_id}-{train_seed}",
        },
        "identity": identity
        or {
            "method_id": method_id,
            "train_seed": train_seed,
            "split": split,
            "task": task,
            "episode_id": episode_id,
        },
    }


def _manifest(*experiments: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "provenance": {"matrix": "unit-test", "locked_official_sources": True},
        "experiments": list(experiments),
    }


def _full_manifest(*, episodes_per_task: int = 2) -> dict[str, object]:
    experiments = []
    for method_id, train_seeds in (("pi05_baseline", (0,)), ("futuremamba", (0, 1))):
        for train_seed in train_seeds:
            for task in OFFICIAL_TASKS:
                for episode_id in range(episodes_per_task):
                    experiments.append(_flat_experiment(method_id, train_seed, task, episode_id))
    return _manifest(*experiments)


def _records(
    manifest: dict[str, object],
    success_by_cell: dict[tuple[str, int, str, int], bool] | None = None,
):
    success_by_cell = success_by_cell or {}
    records = []
    for experiment in manifest["experiments"]:
        success = success_by_cell.get(
            (
                experiment["method_id"],
                experiment["train_seed"],
                experiment["task"],
                experiment["episode_id"],
            ),
            True,
        )
        records.append(
            {
                "experiment_id": experiment["experiment_id"],
                "method_id": experiment["method_id"],
                "train_seed": experiment["train_seed"],
                "task_name": experiment["task"],
                "episode_id": experiment["episode_id"],
                "checkpoint": copy.deepcopy(experiment["checkpoint"]),
                "config": copy.deepcopy(experiment["config"]),
                "provenance": copy.deepcopy(experiment["provenance"]),
                "success": success,
                "outcome": "success" if success else "failure",
            }
        )
    return records


def test_official_robomme_task_taxonomy_is_frozen_to_locked_sources():
    assert analyzer.ROBOMME_TASKS == OFFICIAL_TASKS
    assert analyzer.ROBOMME_CATEGORY_TASKS == CATEGORY_TASKS
    assert analyzer.ROBOMME_TASK_CATEGORIES == {
        task_name: category
        for category, task_names in CATEGORY_TASKS.items()
        for task_name in task_names
    }


def test_flat_matrix_cross_contract_groups_episode_rows_by_method_id_and_seed():
    manifest = _full_manifest()
    assert "episodes" not in manifest["experiments"][0]
    records = _records(
        manifest,
        {
            ("futuremamba", 0, "BinFill", 1): False,
            ("futuremamba", 1, "BinFill", 0): False,
            ("futuremamba", 1, "BinFill", 1): False,
            ("futuremamba", 0, "StopCube", 0): False,
            ("futuremamba", 0, "StopCube", 1): False,
        },
    )

    summary = analyzer.analyze_results(manifest, records)

    future = summary["methods"]["futuremamba"]
    assert future["method_id"] == "futuremamba"
    assert future["method"] == "FutureMamba"
    assert future["variant"] == "mamba2-uniform6"
    assert future["axis"] == "primary"
    seed0 = future["train_seeds"]["0"]
    assert seed0["experiment_count"] == 32
    assert seed0["experiment_ids"][0] == "futuremamba-seed0-test-BinFill-ep0"
    assert seed0["per_task"]["BinFill"] == {"n": 2, "success_count": 1, "success_rate": 0.5}
    assert seed0["per_task"]["StopCube"] == {"n": 2, "success_count": 0, "success_rate": 0.0}
    assert seed0["categories"]["Counting"] == {"n": 8, "success_count": 5, "success_rate": 0.625}
    assert seed0["overall"] == {"n": 32, "success_count": 29, "success_rate": 0.90625}

    binfill_aggregate = future["aggregate"]["per_task"]["BinFill"]
    assert binfill_aggregate["seed_count"] == 2
    assert binfill_aggregate["raw_n_by_seed"] == [2, 2]
    assert binfill_aggregate["raw_success_count_by_seed"] == [1, 0]
    assert binfill_aggregate["mean_success_rate"] == pytest.approx(0.25)
    assert binfill_aggregate["sample_std_success_rate"] == pytest.approx(0.35355339059)
    assert binfill_aggregate["ci95"]["status"] == "ok"
    assert binfill_aggregate["ci95"]["method"] == "student_t"
    assert binfill_aggregate["ci95"]["df"] == 1

    cells = {
        (cell["method_id"], cell["task_name"]): cell for cell in summary["heatmap"]["cells"]
    }
    assert len(cells) == 2 * 16
    assert cells[("pi05_baseline", "BinFill")]["delta_success_rate_vs_pi05_baseline"] == 0.0
    assert cells[("futuremamba", "BinFill")]["success_rate_mean"] == pytest.approx(0.25)
    assert cells[("futuremamba", "BinFill")]["baseline_success_rate_mean"] == 1.0
    assert cells[("futuremamba", "BinFill")]["delta_success_rate_vs_pi05_baseline"] == pytest.approx(-0.75)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda manifest: manifest["experiments"].append(copy.deepcopy(manifest["experiments"][0])),
            "duplicate experiment_id",
        ),
        (
            lambda manifest: manifest["experiments"].append(
                {**copy.deepcopy(manifest["experiments"][0]), "experiment_id": "different-id"}
            ),
            "duplicate manifest identity",
        ),
        (
            lambda manifest: manifest["experiments"].append(
                {
                    **copy.deepcopy(manifest["experiments"][0]),
                    "experiment_id": "same-logical-episode",
                    "identity": {"custom": "still-distinct"},
                }
            ),
            "duplicate manifest episode identity",
        ),
    ],
)
def test_flat_manifest_requires_unique_experiment_ids_identities_and_episode_identities(mutate, message: str):
    manifest = _full_manifest()
    mutate(manifest)

    with pytest.raises(ValueError, match=message):
        analyzer.analyze_results(manifest, _records(manifest))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda records: records.pop(), "missing result"),
        (lambda records: records.append(copy.deepcopy(records[0])), "duplicate result"),
        (
            lambda records: records.append({**copy.deepcopy(records[0]), "experiment_id": "unknown-exp"}),
            "unknown experiment_id",
        ),
        (lambda records: records[0].update({"task_name": "StopCube"}), "task mismatch"),
        (lambda records: records[0].update({"episode_id": 99}), "episode_id mismatch"),
    ],
)
def test_results_must_match_each_flat_experiment_id_exactly_once(mutate, message: str):
    manifest = _full_manifest()
    records = _records(manifest)
    mutate(records)

    with pytest.raises(ValueError, match=message):
        analyzer.analyze_results(manifest, records)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update({"success": "yes"}), "success must be boolean"),
        (lambda row: row.update({"success": 1}), "success must be boolean"),
        (lambda row: row.update({"outcome": "unknown", "success": False}), "unknown outcome"),
        (lambda row: row.update({"status": "error", "success": False}), "error status"),
        (lambda row: row.update({"error": "policy disconnected", "success": False}), "error"),
        (lambda row: row.update({"outcome": "success", "success": False}), "success disagrees with outcome"),
        (lambda row: row.update({"outcome": "partial", "success": False}), "unknown outcome"),
    ],
)
def test_results_reject_unknown_error_and_illegal_success_without_filling_failures(mutate, message: str):
    manifest = _full_manifest()
    records = _records(manifest)
    mutate(records[0])

    with pytest.raises(ValueError, match=message):
        analyzer.analyze_results(manifest, records)


@pytest.mark.parametrize("field", ["checkpoint", "config", "provenance"])
def test_results_reject_checkpoint_config_and_provenance_mismatches(field: str):
    manifest = _full_manifest()
    records = _records(manifest)
    records[0][field] = {"tampered": True}

    with pytest.raises(ValueError, match=field):
        analyzer.analyze_results(manifest, records)


def test_manifest_category_must_match_official_four_way_taxonomy():
    manifest = _full_manifest()
    manifest["experiments"][0]["category"] = "Reference"

    with pytest.raises(ValueError, match="category mismatch"):
        analyzer.analyze_results(manifest, _records(manifest))


def test_single_seed_ci_is_insufficient_and_normal_ci_choice_is_schema_visible():
    one_seed = analyzer.summarize_success_rates([0.25])
    assert one_seed["seed_count"] == 1
    assert one_seed["sample_std_success_rate"] is None
    assert one_seed["ci95"] == {
        "status": "insufficient",
        "confidence": 0.95,
        "reason": "requires_at_least_two_train_seeds",
    }

    thirty_seeds = analyzer.summarize_success_rates([0.0, 1.0] * 15)
    assert thirty_seeds["seed_count"] == 30
    assert thirty_seeds["ci95"]["status"] == "ok"
    assert thirty_seeds["ci95"]["method"] == "normal"
    assert thirty_seeds["ci95"]["critical_value"] == pytest.approx(1.959963984540054)


def test_heatmap_requires_pi05_baseline_and_complete_official_task_coverage_only_after_grouping():
    no_baseline = _manifest(
        *[
            _flat_experiment("futuremamba", 0, task, 0)
            for task in OFFICIAL_TASKS
        ]
    )
    with pytest.raises(ValueError, match="pi05_baseline"):
        analyzer.analyze_results(no_baseline, _records(no_baseline))

    incomplete_seed = _full_manifest()
    incomplete_seed["experiments"] = [
        row
        for row in incomplete_seed["experiments"]
        if not (
            row["method_id"] == "futuremamba"
            and row["train_seed"] == 1
            and row["task"] == "RouteStick"
        )
    ]
    with pytest.raises(ValueError, match="complete 16-task coverage"):
        analyzer.analyze_results(incomplete_seed, _records(incomplete_seed))


def _formal_profile(method_id: str, train_seed: int) -> dict[str, object]:
    return {
        "schema_version": 1,
        "profile_type": "futuremamba_task15_profile",
        "method_id": method_id,
        "train_seed": train_seed,
        "config_name": "futuremamba_robomme_mamba2",
        "bundle_identity": {
            "bundle_path": f"/ckpts/{method_id}/{train_seed}",
            "bundle_metadata_sha256": f"bundle-{method_id}-{train_seed}",
            "base_checkpoint_checksum": "base-sha256",
        },
        "runtime": {"device": "cuda:0", "torch_version": "2.9.1"},
        "source_commits": {"mamba": "mamba-commit", "robomme_policy": "policy-commit", "robomme_benchmark": "benchmark-commit"},
        "parameters": {"total": 1_000_000, "trainable": 80_000, "plugin": 80_000, "plugin_ratio": 0.08},
        "memory_state_bytes": {"layers": [{"layer": 0, "bytes": 128}], "total": 128},
        "latency_ms": {
            "memory_step": {"median": 2.5, "p95": 3.0},
            "action_chunk_20": {"median": 12.25, "p95": 14.5},
        },
        "inference_peak_memory_bytes": 2_048_000,
        "training_peak_memory_bytes": 4_096_000,
        "episode_average_query_ms": 8.5,
        "episode_timing": {
            "query_count": 30,
            "episode_count": 2,
            "measurement_source": "futuremamba_policy_timing",
            "artifact_sha256": f"episode-{method_id}-{train_seed}",
        },
        "flops": {
            "base_flops": 100_000_000,
            "plugin_flops": 23_456_789,
            "futuremamba_total_flops": 123_456_789,
            "relative_plugin_over_base": 0.23456789,
            "measurement_source": "tool_analysis",
            "tool": "fvcore",
            "artifact_sha256": f"flops-{method_id}-{train_seed}",
        },
        "training_memory_artifact": {
            "measurement_source": "torch.cuda.max_memory_allocated",
            "producer": "scripts/train_futuremamba_pytorch.py",
            "artifact_sha256": f"training-{method_id}-{train_seed}",
        },
        "measurement_counts": {"warmup": 3, "measured": 20, "batch_size": 1},
    }


def _profiles(manifest: dict[str, object]) -> dict[str, object]:
    identities = sorted(
        {
            (row["method_id"], row["train_seed"])
            for row in manifest["experiments"]
            if row["method_id"] == "futuremamba"
        },
        key=lambda item: (item[0], item[1]),
    )
    return {"profiles": [_formal_profile(method_id, train_seed) for method_id, train_seed in identities]}


def test_profile_json_consumes_only_formal_per_query_reports_and_rejects_legacy_numeric_rows():
    manifest = _full_manifest()
    records = _records(manifest)
    profile = _profiles(manifest)

    summary = analyzer.analyze_results(manifest, records, profile)
    attached = summary["methods"]["futuremamba"]["train_seeds"]["0"]["profile"]
    assert attached["profile_type"] == "futuremamba_task15_profile"
    assert attached["episode_average_query_ms"] == 8.5
    assert attached["episode_timing"]["measurement_source"] == "futuremamba_policy_timing"
    assert attached["episode_timing"]["artifact_sha256"] == "episode-futuremamba-0"
    assert attached["flops"]["futuremamba_total_flops"] == 123_456_789
    assert "profile" not in summary["methods"]["pi05_baseline"]["train_seeds"]["0"]

    duplicate = copy.deepcopy(profile)
    duplicate["profiles"].append(copy.deepcopy(duplicate["profiles"][0]))
    with pytest.raises(ValueError, match="duplicate profile identity"):
        analyzer.analyze_results(manifest, records, duplicate)

    missing = copy.deepcopy(profile)
    del missing["profiles"][0]["flops"]["artifact_sha256"]
    with pytest.raises(ValueError, match="flops.*artifact_sha256"):
        analyzer.analyze_results(manifest, records, missing)

    forged = copy.deepcopy(profile)
    forged["profiles"][0]["episode_timing"]["measurement_source"] = "manual_average"
    with pytest.raises(ValueError, match="futuremamba_policy_timing"):
        analyzer.analyze_results(manifest, records, forged)

    legacy = {
        "profiles": [
            {
                "method_id": "futuremamba",
                "train_seed": 0,
                "total_parameters": 1_000_000,
                "trainable_parameters": 80_000,
                "plugin_parameters": 80_000,
                "ms_per_query": 8.5,
                "chunk_latency_ms": 12.25,
                "peak_vram_bytes": 2_048_000,
                "flops": 123_456_789,
            }
        ]
    }
    with pytest.raises(ValueError, match="profile_type"):
        analyzer.analyze_results(manifest, records, legacy)


def test_cli_writes_deterministic_json_and_reports_specific_errors(tmp_path: Path, capsys):
    manifest = _full_manifest()
    records = _records(manifest)
    manifest_path = tmp_path / "manifest.json"
    results_path = tmp_path / "results.jsonl"
    output_path = tmp_path / "summary.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    results_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")

    assert analyzer.main(["--manifest", str(manifest_path), "--results", str(results_path), "--output", str(output_path)]) == 0
    first = output_path.read_text(encoding="utf-8")
    assert analyzer.main(["--manifest", str(manifest_path), "--results", str(results_path), "--output", str(output_path)]) == 0
    assert output_path.read_text(encoding="utf-8") == first
    assert json.loads(first)["schema"]["ci95"]["single_seed_policy"] == "insufficient"

    bad_results_path = tmp_path / "bad_results.jsonl"
    bad_results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records[:-1]),
        encoding="utf-8",
    )
    assert analyzer.main(["--manifest", str(manifest_path), "--results", str(bad_results_path)]) == 2
    stderr = capsys.readouterr().err
    assert "missing result" in stderr
    assert "RouteStick" in stderr
