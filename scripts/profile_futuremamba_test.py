from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

SCRIPT = Path(__file__).with_name("profile_futuremamba.py")
spec = importlib.util.spec_from_file_location("profile_futuremamba", SCRIPT)
assert spec is not None and spec.loader is not None
profile = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = profile
spec.loader.exec_module(profile)


def _metadata() -> dict:
    return {
        "schema_version": 3,
        "base_checkpoint_uri": "file:///ckpts/pi05",
        "base_checkpoint_checksum": "base-sha256",
        "base_assets_checksum": "assets-sha256",
        "robomme_policy_commit": "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b",
        "robomme_benchmark_commit": "856bc3a189d4172f3f47dbee4424d585f8d78db3",
        "mamba_repo_commit": "77069de5cdb55cbe98b670889c80df211e031039",
        "memory_backend": "mamba2",
        "memory_state_schema_version": 1,
        "memory_config": {"d_model": 4, "depth": 2},
        "progress_depth": 6,
        "progress_layer_mapping": [0, 1, 2, 3, 4, 5],
        "handoff_ratio": 0.2,
        "num_denoise_steps": 10,
        "prediction_horizon": 20,
        "execution_horizon": 16,
        "kernel_mode": "fallback",
    }


def _bound_fields(metadata: dict, metadata_sha: str) -> dict:
    return {
        "bundle_metadata_sha256": metadata_sha,
        "base_checkpoint_checksum": metadata["base_checkpoint_checksum"],
        "base_assets_checksum": metadata["base_assets_checksum"],
        "robomme_policy_commit": metadata["robomme_policy_commit"],
        "robomme_benchmark_commit": metadata["robomme_benchmark_commit"],
        "mamba_repo_commit": metadata["mamba_repo_commit"],
        "memory_backend": metadata["memory_backend"],
        "memory_state_schema_version": metadata["memory_state_schema_version"],
    }


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _flops_payload(metadata: dict, metadata_sha: str) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "futuremamba_flop_analysis",
        "measurement_source": "tool_analysis",
        "tool": "fvcore",
        "input_contract": {"batch_size": 1, "action_chunk_steps": 20},
        "base_flops": 1_000_000,
        "plugin_flops": 250_000,
        **_bound_fields(metadata, metadata_sha),
    }


def _training_memory_payload(metadata: dict, metadata_sha: str) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "futuremamba_training_memory",
        "measurement_source": "torch.cuda.max_memory_allocated",
        "current_training_run": True,
        "peak_memory_bytes": 9_876_543,
        "producer": "scripts/train_futuremamba_pytorch.py",
        **_bound_fields(metadata, metadata_sha),
    }


def _episode_payload(metadata: dict, metadata_sha: str) -> dict:
    return {
        "schema_version": 1,
        "artifact_type": "futuremamba_episode_timing",
        "episode_count": 2,
        "episodes": [
            {
                "episode_id": "ep0",
                "queries": [
                    {"measurement_source": "futuremamba_policy_timing", "policy_timing": {"infer_ms": 11.0}},
                    {"measurement_source": "futuremamba_policy_timing", "policy_timing": {"infer_ms": 13.0}},
                ],
            },
            {
                "episode_id": "ep1",
                "queries": [
                    {"measurement_source": "futuremamba_policy_timing", "policy_timing": {"infer_ms": 15.0}}
                ],
            },
        ],
        **_bound_fields(metadata, metadata_sha),
    }


def _runtime_info() -> dict:
    return {
        "device": "cpu",
        "device_type": "cpu",
        "gpu_name": "not_cuda_device",
        "compute_capability": "not_cuda",
        "torch_version": "2.9.1",
        "triton_version": "3.5.1",
        "cuda_version": "not_available",
    }


class _InjectedState:
    def __init__(self, layers):
        self.backend_id = "mamba2"
        self.state_schema_version = 1
        self.batch_size = 1
        self.layers = layers

    def clone(self):
        return _InjectedState(tuple(tuple(tensor.detach().clone() for tensor in layer) for layer in self.layers))


class _InjectedAdapter:
    device = torch.device("cpu")
    num_steps = 10
    action_chunk_steps = 20

    def __init__(self):
        self.memory_calls = 0
        self.action_calls = 0

    def parameter_counts(self) -> dict[str, int]:
        return {"total": 100, "trainable": 10, "plugin": 10}

    def initial_memory_state(self):
        return _InjectedState(
            (
                (torch.zeros((1, 2), dtype=torch.float32),),
                (torch.zeros((1, 3), dtype=torch.float64),),
            )
        )

    def run_memory_step(self, memory_state):
        self.memory_calls += 1
        return _InjectedState(tuple(tuple(tensor + 1 for tensor in layer) for layer in memory_state.layers))

    def run_action_chunk(self, observation, memory_state):
        assert observation == {"model_ready": True}
        self.action_calls += 1
        actions = torch.zeros((1, 20, 32), dtype=torch.float32)
        return actions, self.run_memory_step(memory_state), {"handoff_steps": 2}


class _Timer:
    def __init__(self, durations_ms):
        self._values = []
        now = 0.0
        for duration_ms in durations_ms:
            self._values.extend([now, now + duration_ms / 1000.0])
            now += 1.0
        self._index = 0

    def __call__(self):
        value = self._values[self._index]
        self._index += 1
        return value


class _Synchronizer:
    def __init__(self, events: list[str]):
        self.calls = 0
        self.events = events

    def __call__(self):
        self.calls += 1
        self.events.append("sync")


class _PeakMemory:
    def __init__(self, value: int, events: list[str]):
        self.value = value
        self.events = events
        self.reset_calls = 0
        self.read_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1
        self.events.append("reset")

    def read(self) -> int:
        self.read_calls += 1
        self.events.append("read")
        return self.value


def _assert_no_none(value):
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_none(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_none(item)
    else:
        assert value is not None


def test_profile_report_contains_task15_fields_synchronized_percentiles_and_no_null_metrics(tmp_path: Path):
    metadata = _metadata()
    metadata_sha = profile.stable_json_sha256(metadata)
    flops = profile.load_flop_artifact(
        _write_json(tmp_path / "flops.json", _flops_payload(metadata, metadata_sha)), metadata, metadata_sha
    )
    training = profile.load_training_memory_artifact(
        _write_json(tmp_path / "training_memory.json", _training_memory_payload(metadata, metadata_sha)),
        metadata,
        metadata_sha,
    )
    episode = profile.load_episode_artifact(
        _write_json(tmp_path / "episode.json", _episode_payload(metadata, metadata_sha)), metadata, metadata_sha
    )
    method_id = "futuremamba_mamba2"
    train_seed = 0
    adapter = _InjectedAdapter()
    events: list[str] = []
    synchronizer = _Synchronizer(events)
    peak_memory = _PeakMemory(456_789, events)

    report = profile.profile_model(
        adapter,
        {"model_ready": True},
        config_name="futuremamba_robomme_mamba2",
        bundle_path=tmp_path / "bundle",
        bundle_metadata=metadata,
        bundle_metadata_sha256=metadata_sha,
        episode_metrics=episode,
        flop_metrics=flops,
        training_memory_metrics=training,
        runtime_info=_runtime_info(),
        method_id=method_id,
        train_seed=train_seed,
        warmup=1,
        measured=3,
        timer=_Timer([1.0, 2.0, 4.0, 8.0, 1.0, 10.0, 20.0, 30.0]),
        synchronizer=synchronizer,
        peak_memory=peak_memory,
    )

    assert report["schema_version"] == 1
    assert report["profile_type"] == "futuremamba_task15_profile"
    assert report["method_id"] == method_id
    assert report["train_seed"] == train_seed
    assert report["parameters"] == {"total": 100, "trainable": 10, "plugin": 10, "plugin_ratio": 0.1}
    assert report["memory_state_bytes"]["layers"] == [{"layer": 0, "bytes": 8}, {"layer": 1, "bytes": 24}]
    assert report["memory_state_bytes"]["total"] == 32
    assert report["latency_ms"]["memory_step"] == {"median": pytest.approx(4.0), "p95": pytest.approx(7.6)}
    assert report["latency_ms"]["action_chunk_20"] == {"median": pytest.approx(20.0), "p95": pytest.approx(29.0)}
    assert synchronizer.calls == 18
    assert events[:2] == ["sync", "reset"]
    assert events[-2:] == ["sync", "read"]
    assert peak_memory.reset_calls == 1
    assert peak_memory.read_calls == 1
    assert report["inference_peak_memory_bytes"] == 456_789
    assert report["training_peak_memory_bytes"] == 9_876_543
    assert report["episode_average_query_ms"] == pytest.approx(13.0)
    assert report["episode_timing"]["query_count"] == 3
    assert report["flops"] == {
        "base_flops": 1_000_000,
        "plugin_flops": 250_000,
        "futuremamba_total_flops": 1_250_000,
        "relative_plugin_over_base": 0.25,
        "measurement_source": "tool_analysis",
        "tool": "fvcore",
        "artifact_sha256": flops["artifact_sha256"],
    }
    assert report["source_commits"] == {
        "mamba": metadata["mamba_repo_commit"],
        "robomme_policy": metadata["robomme_policy_commit"],
        "robomme_benchmark": metadata["robomme_benchmark_commit"],
    }
    _assert_no_none(report)
    assert json.loads(profile.to_json(report))["schema_version"] == 1


def test_flop_training_and_episode_artifacts_reject_missing_or_mismatched_provenance(tmp_path: Path):
    metadata = _metadata()
    metadata_sha = profile.stable_json_sha256(metadata)

    bad_flops = _flops_payload(metadata, metadata_sha)
    bad_flops["bundle_metadata_sha256"] = "wrong"
    with pytest.raises(ValueError, match="bundle_metadata_sha256"):
        profile.load_flop_artifact(_write_json(tmp_path / "bad_flops.json", bad_flops), metadata, metadata_sha)

    missing_plugin = _flops_payload(metadata, metadata_sha)
    del missing_plugin["plugin_flops"]
    with pytest.raises(ValueError, match="plugin_flops"):
        profile.load_flop_artifact(_write_json(tmp_path / "missing_plugin.json", missing_plugin), metadata, metadata_sha)

    stale_training = _training_memory_payload(metadata, metadata_sha)
    stale_training["current_training_run"] = False
    with pytest.raises(ValueError, match="current_training_run"):
        profile.load_training_memory_artifact(
            _write_json(tmp_path / "stale_training.json", stale_training), metadata, metadata_sha
        )

    missing_episode_values = _episode_payload(metadata, metadata_sha)
    missing_episode_values["episodes"] = [{"episode_id": "ep0", "queries": [{"policy_timing": {}}]}]
    with pytest.raises(ValueError, match="measurement_source"):
        profile.load_episode_artifact(
            _write_json(tmp_path / "missing_episode_values.json", missing_episode_values), metadata, metadata_sha
        )

    array_episode_values = _episode_payload(metadata, metadata_sha)
    array_episode_values["episodes"][0] = {"episode_id": "ep0", "query_infer_ms": [1.0]}
    with pytest.raises(ValueError, match="per query"):
        profile.load_episode_artifact(
            _write_json(tmp_path / "array_episode_values.json", array_episode_values), metadata, metadata_sha
        )


def test_required_formal_artifacts_fail_closed_before_report_generation(tmp_path: Path):
    metadata = _metadata()
    metadata_sha = profile.stable_json_sha256(metadata)
    flops = _write_json(tmp_path / "flops.json", _flops_payload(metadata, metadata_sha))
    training = _write_json(tmp_path / "training.json", _training_memory_payload(metadata, metadata_sha))

    with pytest.raises(ValueError, match="episode_artifact"):
        profile.load_required_artifacts(
            flop_artifact=flops,
            training_memory_artifact=training,
            episode_artifact=None,
            bundle_metadata=metadata,
            bundle_metadata_sha256=metadata_sha,
        )


def test_loader_uses_production_strict_futuremamba_bundle_loader(monkeypatch, tmp_path: Path):
    calls = {}

    class FutureMambaPytorchConfig:
        marker = True

    config = FutureMambaPytorchConfig()
    train_config = SimpleNamespace(model=config)
    loaded_model = object()
    metadata = _metadata()

    def fake_import(name):
        if name == "openpi.training.config":
            return SimpleNamespace(get_config=lambda config_name: train_config)
        if name == "openpi.models_pytorch.futuremamba_config":
            return SimpleNamespace(FutureMambaPytorchConfig=FutureMambaPytorchConfig)
        if name == "openpi.policies.policy_config":
            def strict_loader(model_config, bundle_dir, device):
                calls["args"] = (model_config, Path(bundle_dir), device)
                return loaded_model, metadata, tmp_path / "base"
            return SimpleNamespace(_load_futuremamba_bundle=strict_loader)
        raise AssertionError(f"unexpected import {name}")

    monkeypatch.setattr(profile.importlib, "import_module", fake_import)

    bundle = profile.load_futuremamba_bundle("futuremamba_robomme_mamba2", tmp_path / "bundle", torch.device("cpu"))

    assert bundle.model is loaded_model
    assert bundle.metadata == metadata
    assert bundle.base_root == tmp_path / "base"
    assert calls["args"] == (config, tmp_path / "bundle", torch.device("cpu"))
