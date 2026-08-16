from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).with_name("run_robomme_experiment_matrix.py")
SPEC = importlib.util.spec_from_file_location("run_robomme_experiment_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
matrix = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = matrix
SPEC.loader.exec_module(matrix)


EXPECTED_TASK_SUITES = {
    "Counting": ("BinFill", "PickXtimes", "SwingXtimes", "StopCube"),
    "Permanence": ("VideoUnmask", "VideoUnmaskSwap", "ButtonUnmask", "ButtonUnmaskSwap"),
    "Reference": ("PickHighlight", "VideoRepick", "VideoPlaceButton", "VideoPlaceOrder"),
    "Imitation": ("MoveCube", "InsertPeg", "PatternLock", "RouteStick"),
}
EXPECTED_RUNTIME_TASK_ORDER = (
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


def _passed_gate(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "passed",
                "gates": [
                    {"name": name, "status": "passed", "evidence": {"source": name}}
                    for name in matrix.MAMBA3_REQUIRED_GATES
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _failed_gate(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "unsupported_on_current_stack",
                "gates": [
                    {
                        "name": name,
                        "status": "failed" if name == "device_256_steps" else "not_run",
                        "evidence": {"source": name},
                    }
                    for name in matrix.MAMBA3_REQUIRED_GATES
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _checkpoint_entry(tmp_path: Path, spec, seed: int | None = None) -> dict[str, object]:
    leaf = tmp_path / "ckpts" / spec.checkpoint_key
    if seed is not None:
        leaf = leaf / f"seed{seed}"
    leaf = leaf / "79999"
    leaf.mkdir(parents=True, exist_ok=True)
    return {
        "checkpoint": str(leaf),
        "checkpoint_id": 79999,
        "config": spec.config_name,
        "backend": spec.backend,
        "policy_name": spec.policy_name,
        "provenance": {"source": spec.checkpoint_key, "train_seed": seed if seed is not None else spec.train_seeds[0]},
    }


def _write_checkpoint_mapping(tmp_path: Path, *, include_mamba3: bool = False) -> Path:
    checkpoints: dict[str, object] = {}
    for spec in matrix.method_specs(include_mamba3=include_mamba3):
        if len(spec.train_seeds) == 1:
            checkpoints[spec.checkpoint_key] = _checkpoint_entry(tmp_path, spec)
        else:
            checkpoints[spec.checkpoint_key] = {
                str(seed): _checkpoint_entry(tmp_path, spec, seed) for seed in spec.train_seeds
            }
    path = tmp_path / "checkpoint_mapping.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "provenance": {"created_by": "unit-test"},
                "checkpoints": checkpoints,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def test_official_robomme_tasks_categories_and_split_counts_are_frozen():
    assert matrix.OFFICIAL_TASK_SUITES == EXPECTED_TASK_SUITES
    assert matrix.OFFICIAL_TASKS == EXPECTED_RUNTIME_TASK_ORDER
    assert matrix.OFFICIAL_SPLIT_EPISODE_COUNTS == {"train": 100, "validation": 50, "test": 50}
    assert matrix.ROBOMME_POLICY_LEARNING_COMMIT == "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
    assert matrix.ROBOMME_BENCHMARK_COMMIT == "856bc3a189d4172f3f47dbee4424d585f8d78db3"


def test_phase_episode_plans_match_preregistered_matrix_sizes():
    minimal = matrix.phase_episode_plan("minimal")
    assert [(selection.task, selection.split, len(selection.episode_ids)) for selection in minimal] == [
        ("PickXtimes", "validation", 10),
        ("BinFill", "validation", 10),
    ]
    assert all(selection.episode_ids == tuple(range(10)) for selection in minimal)

    counting = matrix.phase_episode_plan("counting")
    assert {selection.task for selection in counting} == set(EXPECTED_TASK_SUITES["Counting"])
    assert all(selection.split == "validation" for selection in counting)
    assert sum(len(selection.episode_ids) for selection in counting) == 4 * 50

    full_val = matrix.phase_episode_plan("full_val")
    assert [selection.task for selection in full_val] == list(EXPECTED_RUNTIME_TASK_ORDER)
    assert sum(len(selection.episode_ids) for selection in full_val) == 16 * 50

    final_test = matrix.phase_episode_plan("final_test")
    assert all(selection.split == "test" for selection in final_test)
    assert sum(len(selection.episode_ids) for selection in final_test) == 16 * 50


def test_final_test_manifest_has_16_tasks_50_test_episodes_and_three_main_seeds(tmp_path: Path):
    mapping = _write_checkpoint_mapping(tmp_path)
    manifest = matrix.build_manifest(checkpoint_mapping_path=mapping, stage="final_test")

    main = [row for row in manifest["experiments"] if row["method_id"] == "futuremamba_mamba2"]
    assert len(main) == 16 * 50 * 3
    assert {row["task"] for row in main} == set(EXPECTED_RUNTIME_TASK_ORDER)
    assert {row["episode_id"] for row in main} == set(range(50))
    assert {row["split"] for row in main} == {"test"}
    assert {row["train_seed"] for row in main} == {0, 42, 7}


def test_validation_manifest_uses_official_val_dataset_not_test(tmp_path: Path):
    mapping = _write_checkpoint_mapping(tmp_path)
    manifest = matrix.build_manifest(checkpoint_mapping_path=mapping, stage="minimal")

    record = next(row for row in manifest["experiments"] if row["method_id"] == "futuremamba_mamba2")
    eval_argv = record["eval"]["argv"]
    assert eval_argv[eval_argv.index("--split") + 1] == "validation"
    assert eval_argv[eval_argv.index("--dataset") + 1] == "val"
    assert record["eval"]["params"]["split"] == "validation"

    replay = record["eval"]["params"]["official_single_episode_launcher"]
    assert replay["argv"][replay["argv"].index("--dataset") + 1] == "val"
    assert "validation" not in replay["argv"]
    assert replay["params"]["dataset"] == "val"


def test_method_specs_cover_required_baselines_and_ablation_axes_without_mamba3_by_default():
    specs = matrix.method_specs(include_mamba3=False)
    method_ids = {spec.method_id for spec in specs}
    assert {
        "pi05_baseline",
        "past-actions",
        "MemER",
        "perceptual-framesamp-modul",
        "perceptual-framesamp-expert",
        "recurrent-rmt-expert",
        "futuremamba_mamba2",
    } <= method_ids
    assert "futuremamba_mamba3_siso" not in method_ids

    future_specs = [spec for spec in specs if spec.method == "futuremamba"]
    axis_values: dict[str, set[object]] = {}
    for spec in future_specs:
        for key, value in spec.axes.items():
            axis_values.setdefault(key, set()).add(value)

    assert {"Uniform", "First", "Sensitivity", "Random"} <= axis_values["layer_selection"]
    assert {4, 6, 9} <= axis_values["progress_depth"]
    assert {0.0, 0.2, 0.4, 0.6, 1.0} <= axis_values["handoff_ratio"]
    assert {"full", "none", "shuffled"} <= axis_values["memory"]
    assert {"Action Expert", "random"} <= axis_values["initialization"]
    assert {"flow-only", "flow+terminal"} <= axis_values["loss"]
    assert {"memory", "no-memory"} <= axis_values["progress_expert"]
    assert {spec.backend for spec in future_specs} == {"mamba2"}
    assert next(spec for spec in specs if spec.method_id == "futuremamba_mamba2").train_seeds == (0, 42, 7)


def test_mamba3_gate_must_be_explicitly_passed_and_keeps_backend_identity(tmp_path: Path):
    assert matrix.gate_allows_mamba3(None) is False
    assert matrix.gate_allows_mamba3(_failed_gate(tmp_path / "failed_gate.json")) is False

    passed = _passed_gate(tmp_path / "passed_gate.json")
    assert matrix.gate_allows_mamba3(passed) is True
    assert "futuremamba_mamba3_siso" in {
        spec.method_id for spec in matrix.method_specs(include_mamba3=True)
    }

    mapping = _write_checkpoint_mapping(tmp_path, include_mamba3=True)
    manifest = matrix.build_manifest(
        checkpoint_mapping_path=mapping,
        stage="minimal",
        mamba3_gate_path=passed,
    )
    mamba3_records = [
        row for row in manifest["experiments"] if row["method_id"] == "futuremamba_mamba3_siso"
    ]
    assert mamba3_records
    assert {row["backend"] for row in mamba3_records} == {"mamba3_siso"}
    assert all(row["config"]["name"] == "futuremamba_robomme_mamba3_siso" for row in mamba3_records)


def test_illegal_mamba3_gate_is_rejected(tmp_path: Path):
    gate = tmp_path / "bad_gate.json"
    gate.write_text(
        json.dumps({"schema_version": 1, "status": "passed", "gates": []}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Mamba-3 gate"):
        matrix.gate_allows_mamba3(gate)


def test_checkpoint_mapping_fails_closed_for_missing_or_incomplete_identity(tmp_path: Path):
    mapping = _write_checkpoint_mapping(tmp_path)
    data = json.loads(mapping.read_text(encoding="utf-8"))
    del data["checkpoints"]["futuremamba_depth_4"]
    mapping.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="missing checkpoint mapping.*futuremamba_depth_4"):
        matrix.build_manifest(checkpoint_mapping_path=mapping, stage="minimal")

    mapping = _write_checkpoint_mapping(tmp_path)
    data = json.loads(mapping.read_text(encoding="utf-8"))
    data["checkpoints"]["futuremamba_depth_4"]["checkpoint"] = str(tmp_path / "missing" / "79999")
    mapping.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="does not exist"):
        matrix.build_manifest(checkpoint_mapping_path=mapping, stage="minimal")

    mapping = _write_checkpoint_mapping(tmp_path)
    data = json.loads(mapping.read_text(encoding="utf-8"))
    data["checkpoints"]["futuremamba_depth_4"]["provenance"] = {}
    mapping.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance"):
        matrix.build_manifest(checkpoint_mapping_path=mapping, stage="minimal")

    mapping = _write_checkpoint_mapping(tmp_path)
    data = json.loads(mapping.read_text(encoding="utf-8"))
    data["checkpoints"]["futuremamba_depth_4"]["backend"] = "mamba3_siso"
    mapping.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="backend"):
        matrix.build_manifest(checkpoint_mapping_path=mapping, stage="minimal")


def test_multi_seed_checkpoint_mapping_requires_per_seed_identity(tmp_path: Path):
    mapping = _write_checkpoint_mapping(tmp_path)
    data = json.loads(mapping.read_text(encoding="utf-8"))
    data["checkpoints"]["futuremamba_mamba2"] = data["checkpoints"]["futuremamba_mamba2"]["0"]
    mapping.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="per-seed entries"):
        matrix.build_manifest(checkpoint_mapping_path=mapping, stage="minimal")


def test_checkpoint_mapping_rejects_mismatched_provenance_train_seed(tmp_path: Path):
    mapping = _write_checkpoint_mapping(tmp_path)
    data = json.loads(mapping.read_text(encoding="utf-8"))
    data["checkpoints"]["futuremamba_mamba2"]["0"]["provenance"]["train_seed"] = 42
    mapping.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ValueError, match="does not match requested seed 0"):
        matrix.build_manifest(checkpoint_mapping_path=mapping, stage="minimal")


def test_main_futuremamba_records_have_three_training_seeds_and_complete_identity(tmp_path: Path):
    mapping = _write_checkpoint_mapping(tmp_path)
    manifest = matrix.build_manifest(checkpoint_mapping_path=mapping, stage="minimal")
    records = manifest["experiments"]
    required = {
        "method",
        "variant",
        "task",
        "category",
        "split",
        "episode_id",
        "train_seed",
        "eval_seed",
        "checkpoint",
        "config",
        "backend",
        "experiment_id",
        "provenance",
        "server",
        "eval",
    }
    assert all(required <= row.keys() for row in records)
    assert all(row["checkpoint"]["path"] for row in records)
    assert all(row["config"]["name"] for row in records)
    assert all(row["provenance"] for row in records)

    main = [row for row in records if row["method_id"] == "futuremamba_mamba2"]
    assert {row["train_seed"] for row in main} == {0, 42, 7}
    assert len(main) == 2 * 10 * 3

    experiment_ids = [row["experiment_id"] for row in records]
    assert len(experiment_ids) == len(set(experiment_ids))
    assert all(row["split"] == "validation" for row in records)
    assert {row["episode_id"] for row in main if row["task"] == "PickXtimes"} == set(range(10))


def test_records_carry_structured_server_and_single_episode_launcher_argv(tmp_path: Path):
    mapping = _write_checkpoint_mapping(tmp_path)
    manifest = matrix.build_manifest(checkpoint_mapping_path=mapping, stage="minimal", port=8123)

    baseline = next(row for row in manifest["experiments"] if row["method_id"] == "pi05_baseline")
    assert isinstance(baseline["server"]["argv"], list)
    assert isinstance(baseline["eval"]["argv"], list)
    assert "scripts/run_robomme_single_episode.py" in baseline["eval"]["argv"]
    assert "--task-name" in baseline["eval"]["argv"]
    assert "--episode-id" in baseline["eval"]["argv"]
    assert "--port" in baseline["eval"]["argv"]
    assert "8123" in baseline["eval"]["argv"]
    assert "--use-history" not in baseline["eval"]["argv"]

    memer = next(row for row in manifest["experiments"] if row["method_id"] == "MemER")
    assert "--use-history" in memer["eval"]["argv"]
    assert "--use-memer" in memer["eval"]["argv"]
    assert "--subgoal-type" in memer["eval"]["argv"]

    future = next(row for row in manifest["experiments"] if row["method_id"] == "futuremamba_mamba2")
    assert future["server"]["argv"][:3] == ["uv", "run", "scripts/serve_policy.py"]
    assert future["eval"]["argv"][:3] == ["python", "scripts/run_robomme_single_episode.py", "--task-name"]


    identity_json = future["eval"]["argv"][future["eval"]["argv"].index("--result-identity-json") + 1]
    assert json.loads(identity_json) == {
        "experiment_id": future["experiment_id"],
        "method_id": future["method_id"],
        "train_seed": future["train_seed"],
        "task_name": future["task"],
        "episode_id": future["episode_id"],
        "split": future["split"],
        "checkpoint": future["checkpoint"],
        "config": future["config"],
        "provenance": future["provenance"],
    }

def test_cli_writes_stably_sorted_json_manifest(tmp_path: Path):
    mapping = _write_checkpoint_mapping(tmp_path)
    output = tmp_path / "manifest.json"

    result = matrix.main(
        [
            "--checkpoint-mapping",
            str(mapping),
            "--output",
            str(output),
            "--stage",
            "minimal",
        ]
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert result == payload
    assert output.read_text(encoding="utf-8") == json.dumps(payload, indent=2, sort_keys=True) + "\n"
    assert payload["schema"] == "openpi.robomme_experiment_matrix"
    assert payload["schema_version"] == 1
    assert payload["provenance"]["task_source"].endswith("examples/robomme/utils.py:TASK_NAME_LIST")
