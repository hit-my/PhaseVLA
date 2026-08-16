from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import safetensors.torch
import torch
from torch import nn

from openpi.models_pytorch.futuremamba_config import FutureMambaPytorchConfig
from openpi.policies import policy_config


class _TinyBundleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2)
        self.futuremamba = nn.Linear(2, 2)

    def freeze_base(self):
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def base_checksum(self):
        digest = hashlib.sha256()
        for name, tensor in self.base.state_dict().items():
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def _write_bundle(tmp_path: Path, monkeypatch, *, bad_checksum: bool = False):
    base = tmp_path / "base"
    base.mkdir()
    source = _TinyBundleModel()
    safetensors.torch.save_file(source.base.state_dict(), base / "model.safetensors")
    model = _TinyBundleModel()
    monkeypatch.setattr(FutureMambaPytorchConfig, "create_pytorch", lambda self: model)
    plugin = {f"futuremamba.{name}": tensor for name, tensor in source.futuremamba.state_dict().items()}
    safetensors.torch.save_file(plugin, tmp_path / "plugin.safetensors")
    metadata = {
        "schema_version": 3,
        "base_checkpoint_uri": str(base),
        "base_checkpoint_checksum": "bad" if bad_checksum else source.base_checksum(),
        "base_assets_checksum": None,
        "robomme_policy_commit": "policy-commit",
        "robomme_benchmark_commit": "benchmark-commit",
        "robomme_dataset_checksum": "dataset-checksum",
        "robomme_task_suite": "counting",
        "train_seed": 42,
        "mamba_repo_commit": "77069de5cdb55cbe98b670889c80df211e031039",
        "memory_backend": "mamba2",
        "memory_state_schema_version": 1,
        "memory_config": dataclasses.asdict(FutureMambaPytorchConfig().memory),
        "progress_depth": 6,
        "progress_layer_mapping": [0, 3, 7, 10, 14, 17],
        "handoff_ratio": 0.2,
        "num_denoise_steps": 10,
        "prediction_horizon": 50,
        "execution_horizon": 16,
        "action_expert_gradient_checkpointing": False,
        "terminal_loss_batch_fraction": 1.0,
        "terminal_loss_queries_per_episode": None,
        "frozen_prefix_microbatch_size": 2,
        "loss_weights": {"terminal": 1.0, "handoff": 0.0, "boundary": 0.0},
        "training_dtype": "bfloat16",
        "state_dtypes": {},
        "kernel_mode": "fallback",
        "torch_version": torch.__version__,
        "triton_version": None,
        "cuda_version": torch.version.cuda,
        "gpu_name": None,
        "compute_capability": None,
        "step": 2,
        "data_iterator_step": 2,
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return source, model, metadata


def test_explicit_bundle_loader_strictly_loads_base_and_plugin(tmp_path: Path, monkeypatch):
    source, model, metadata = _write_bundle(tmp_path, monkeypatch)
    config = FutureMambaPytorchConfig(base_checkpoint_uri=metadata["base_checkpoint_uri"])

    loaded, loaded_metadata, base_root = policy_config._load_futuremamba_bundle(config, tmp_path, torch.device("cpu"))

    assert loaded is model
    assert loaded_metadata == metadata
    assert base_root == Path(metadata["base_checkpoint_uri"])
    for name, tensor in loaded.base.state_dict().items():
        torch.testing.assert_close(tensor, source.base.state_dict()[name])
    for name, tensor in loaded.futuremamba.state_dict().items():
        torch.testing.assert_close(tensor, source.futuremamba.state_dict()[name])
    assert all(not parameter.requires_grad for parameter in loaded.base.parameters())


def test_bundle_loader_rejects_base_checksum_mismatch(tmp_path: Path, monkeypatch):
    _, _, metadata = _write_bundle(tmp_path, monkeypatch, bad_checksum=True)
    config = FutureMambaPytorchConfig(base_checkpoint_uri=metadata["base_checkpoint_uri"])

    with pytest.raises(ValueError, match="base_checkpoint_checksum"):
        policy_config._load_futuremamba_bundle(config, tmp_path, torch.device("cpu"))


def test_bundle_loader_rejects_missing_plugin_key(tmp_path: Path, monkeypatch):
    _, _, metadata = _write_bundle(tmp_path, monkeypatch)
    safetensors.torch.save_file({}, tmp_path / "plugin.safetensors")
    config = FutureMambaPytorchConfig(base_checkpoint_uri=metadata["base_checkpoint_uri"])

    with pytest.raises(ValueError, match="strict|missing"):
        policy_config._load_futuremamba_bundle(config, tmp_path, torch.device("cpu"))


def test_plugin_bundle_is_not_misdetected_as_plain_pi0(tmp_path: Path, monkeypatch):
    config = FutureMambaPytorchConfig()
    train_config = SimpleNamespace(
        model=config,
        data=SimpleNamespace(create=lambda assets, model: SimpleNamespace(
            asset_id=None,
            data_transforms=SimpleNamespace(inputs=(), outputs=()),
            model_transforms=SimpleNamespace(inputs=(), outputs=()),
            use_quantile_norm=False,
        )),
        assets_dirs=tmp_path,
        policy_metadata={"kind": "futuremamba"},
    )
    (tmp_path / "plugin.safetensors").write_bytes(b"plugin")
    (tmp_path / "metadata.json").write_text("{}", encoding="utf-8")
    sentinel_model = object()
    bundle_metadata = {"kind": "futuremamba", "memory_backend": "mamba2", "step": 2}
    monkeypatch.setattr(
        policy_config,
        "_load_futuremamba_bundle",
        lambda *args: (sentinel_model, bundle_metadata, tmp_path),
    )
    monkeypatch.setattr(policy_config, "_futuremamba_policy", SimpleNamespace(FutureMambaPolicy=lambda *args, **kwargs: (args, kwargs)))
    monkeypatch.setattr(policy_config.download, "maybe_download", lambda value: value)

    created = policy_config.create_trained_policy(train_config, tmp_path, pytorch_device="cpu", norm_stats={})

    assert created[0][0] is sentinel_model
    assert created[1]["pytorch_device"] == "cpu"
    assert created[1]["metadata"] == bundle_metadata


def test_partial_bundle_is_rejected_instead_of_falling_back(tmp_path: Path, monkeypatch):
    (tmp_path / "plugin.safetensors").write_bytes(b"plugin")
    monkeypatch.setattr(policy_config.download, "maybe_download", lambda value: value)
    train_config = SimpleNamespace(model=FutureMambaPytorchConfig())

    with pytest.raises(ValueError, match="metadata.json|incomplete"):
        policy_config.create_trained_policy(train_config, tmp_path, pytorch_device="cpu", norm_stats={})
