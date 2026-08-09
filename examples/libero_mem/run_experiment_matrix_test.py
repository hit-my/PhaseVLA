from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

_MODULE_DIR = pathlib.Path(__file__).parent
_MATRIX_SPEC = importlib.util.spec_from_file_location(
    "libero_mem_run_experiment_matrix", _MODULE_DIR / "run_experiment_matrix.py"
)
run_experiment_matrix = importlib.util.module_from_spec(_MATRIX_SPEC)
sys.modules[_MATRIX_SPEC.name] = run_experiment_matrix
_MATRIX_SPEC.loader.exec_module(run_experiment_matrix)

_PROFILE_SPEC = importlib.util.spec_from_file_location(
    "futuremamba_profile", _MODULE_DIR.parent.parent / "scripts" / "profile_futuremamba.py"
)
profile_futuremamba = importlib.util.module_from_spec(_PROFILE_SPEC)
sys.modules[_PROFILE_SPEC.name] = profile_futuremamba
_PROFILE_SPEC.loader.exec_module(profile_futuremamba)


def _names(rows):
    return {row["name"] for row in rows}


def test_preregistered_matrix_contains_required_baselines_mechanisms_and_fixed_rollout_design():
    rows = run_experiment_matrix.build_matrix()
    names = _names(rows)

    required = {
        "main_futuremamba",
        "frozen_task_adapted_pi05",
        "rho_0_equivalence",
        "recent_frame_stack",
        "gru_memory",
        "lstm_memory",
        "progress_shared_prefix_no_memory",
        "memory_without_prefix",
        "action_expert_memory_full_horizon",
        "reset_every_query",
        "shuffled_history",
        "truncated_history",
        "zero_history",
        "full_horizon_progress",
        "oracle_progress",
        "handoff_loss_off",
        "boundary_loss_off",
        "pool_last_valid",
        "pool_attention",
        "pool_tokens4",
        "pool_tokens8",
        "depth_1_8",
        "depth_1_4",
        "depth_1_2",
        "bptt_full",
        "bptt_truncated",
        "coupling_hard",
        "coupling_convex",
        "coupling_residual",
    }
    assert required <= names

    main_rows = [row for row in rows if row["table"] == "main"]
    assert {row["train_seed"] for row in main_rows} == {0, 1, 2}
    assert len(main_rows) == 3
    assert {tuple(row["rollout_seeds"]) for row in main_rows} == {(10_001, 10_002, 10_003)}
    assert {row["rollout_trials_per_task"] for row in main_rows} == {50}

    rho_zero = next(row for row in rows if row["name"] == "rho_0_equivalence")
    assert rho_zero["config"]["handoff_ratio"] == 0.0
    assert rho_zero["config"]["expected_pi05_equivalence"] is True

    full_horizon = next(row for row in rows if row["name"] == "action_expert_memory_full_horizon")
    assert full_horizon["config"]["decoder_mode"] == "action_memory_full"
    assert full_horizon["config"]["handoff_ratio"] == 1.0

    assert next(row for row in rows if row["name"] == "memory_without_prefix")["config"]["use_prefix_cache"] is False
    assert next(row for row in rows if row["name"] == "progress_shared_prefix_no_memory")["config"]["memory_backend"] == "none"


def test_experiment_ids_are_stable_unique_and_deduplicate_semantically_identical_rows():
    rows = run_experiment_matrix.build_matrix()
    ids = [row["experiment_id"] for row in rows]
    assert ids == [run_experiment_matrix.stable_experiment_id(row) for row in rows]
    assert len(ids) == len(set(ids))

    duplicate = dict(rows[0])
    duplicate["experiment_id"] = "will-be-recomputed"
    duplicate["notes"] = "metadata must not affect identity"
    deduped = run_experiment_matrix.deduplicate_experiments([rows[0], duplicate])
    assert deduped == [rows[0]]


def test_parameter_match_validation_marks_error_and_rejects_false_claims():
    exact = {"name": "exact", "config": {"memory_backend": "gru"}, "parameter_match": {"reference": 100, "candidate": 104}}
    assert run_experiment_matrix.validate_parameter_match(exact)["parameter_matched"] is True
    assert run_experiment_matrix.validate_parameter_match(exact)["parameter_match_error"] == pytest.approx(0.04)

    mismatch = {"name": "mismatch", "config": {"memory_backend": "lstm"}, "parameter_match": {"reference": 100, "candidate": 106}}
    assert run_experiment_matrix.validate_parameter_match(mismatch)["parameter_matched"] is False

    false_claim = {
        "name": "bad",
        "config": {"memory_backend": "frame_stack"},
        "parameter_match": {"reference": 100, "candidate": 106},
        "parameter_matched": True,
    }
    with pytest.raises(ValueError, match="parameter_matched"):
        run_experiment_matrix.validate_parameter_match(false_claim)


def test_write_config_manifest_is_atomic_complete_json_and_dry_run_launcher(tmp_path):
    rows = run_experiment_matrix.build_matrix(train_seeds=(7,), rollout_seeds=(31,), rollout_trials_per_task=2)
    commands = []

    manifest_path = run_experiment_matrix.write_matrix_manifest(tmp_path / "matrix.json", rows)
    loaded = json.loads(manifest_path.read_text())
    assert loaded["schema_version"] == 1
    assert loaded["rollout_seeds"] == [31]
    assert loaded["rollout_trials_per_task"] == 2
    assert len(loaded["experiments"]) == len(rows)
    assert all("experiment_id" in row and "config" in row for row in loaded["experiments"])
    assert not list(tmp_path.glob("*.tmp"))

    result = run_experiment_matrix.launch_matrix(
        rows[:2],
        output_dir=tmp_path,
        train_command=("python", "scripts/train_futuremamba.py"),
        rollout_command=("python", "examples/libero_mem/main.py"),
        runner=lambda cmd: commands.append(tuple(cmd)) or 0,
        dry_run=False,
    )

    assert result["launched"] == 2
    assert result["failures"] == []
    for row in rows[:2]:
        config_path = tmp_path / row["experiment_id"] / "config.json"
        assert config_path.exists()
        written = json.loads(config_path.read_text())
        assert written["experiment_id"] == row["experiment_id"]
        assert written["config"] == row["config"]
    assert all("--config-json" in command for command in commands)


def test_checkpoint_root_selects_largest_numeric_step_params_and_rejects_latest_symlink(tmp_path):
    root = tmp_path / "ckpts"
    (root / "9" / "params").mkdir(parents=True)
    (root / "10" / "params").mkdir(parents=True)
    (root / "latest" / "params").mkdir(parents=True)
    (root / "notes").write_text("ignore")

    assert profile_futuremamba.resolve_checkpoint_params(root) == root / "10" / "params"

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="numeric step"):
        profile_futuremamba.resolve_checkpoint_params(empty)


def test_fake_profile_smoke_outputs_json_with_required_metrics_and_no_fabricated_values(tmp_path):
    class FakeTimer:
        def __init__(self):
            self.values = iter([1.0, 1.001, 2.0, 2.002, 3.0, 3.004, 4.0, 4.008])

        def __call__(self):
            return next(self.values)

    class FakeModel:
        backend_parameter_errors = {"gru": 0.03, "lstm": 0.04, "frame_stack": 0.02}
        progress_flops = 12345

        def __init__(self):
            self.calls = 0

        def parameter_counts(self):
            return {"total": 1000, "trainable": 80, "plugin": 80}

        def initial_memory_state(self, batch_size):
            return {"state": bytearray(batch_size * 16)}

        def sample_actions_with_memory(self, rng, observation, memory_state, executed_actions, executed_action_mask, *, num_steps):
            assert num_steps == 10
            self.calls += 1
            return [[0.0]], memory_state, {"backend_parameter_error": 0.0}

    report = profile_futuremamba.profile_model(
        FakeModel(),
        timer=FakeTimer(),
        warmup_queries=1,
        measured_queries=3,
        gpu_memory_reader=lambda: 456,
    )

    assert report["batch_size"] == 1
    assert report["solver_steps"] == 10
    assert report["total_parameters"] == 1000
    assert report["trainable_parameters"] == 80
    assert report["plugin_parameter_ratio"] == 0.08
    assert report["lightweight_claim"] is True
    assert report["progress_flops"] == 12345
    assert report["state_bytes"] >= 16
    assert report["latency_p50_ms"] == pytest.approx(4.0)
    assert report["latency_p95_ms"] == pytest.approx(7.6)
    assert report["gpu_peak_bytes"] == 456
    assert report["backend_parameter_errors"] == {"gru": 0.03, "lstm": 0.04, "frame_stack": 0.02}
    assert json.loads(profile_futuremamba.to_json(report))["lightweight_claim"] is True


def test_real_profile_loader_fails_at_dependency_boundary_without_fake_values(monkeypatch):
    def fail_import(name):
        raise ModuleNotFoundError("missing real dependency")

    monkeypatch.setattr(profile_futuremamba.importlib, "import_module", fail_import)
    with pytest.raises(RuntimeError, match="Unable to load real FutureMamba dependencies"):
        profile_futuremamba.load_real_model("futuremamba_libero_mem", checkpoint_root=None)
