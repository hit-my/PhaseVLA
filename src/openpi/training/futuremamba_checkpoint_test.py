from __future__ import annotations

import json
from pathlib import Path

import pytest
import safetensors.torch
import torch
from torch import nn

from openpi.training import futuremamba_checkpoint as checkpoint


class _TinyFutureMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2)
        self.futuremamba = nn.Linear(2, 2)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self._initial_base_checksum = self.base_checksum()

    def base_checksum(self) -> str:
        import hashlib

        digest = hashlib.sha256()
        for name, tensor in self.base.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()


def _metadata(model: _TinyFutureMamba, **overrides):
    metadata = {
        "schema_version": 3,
        "base_checkpoint_uri": "file:///base",
        "base_checkpoint_checksum": model.base_checksum(),
        "base_assets_checksum": "assets",
        "robomme_policy_commit": "policy-commit",
        "robomme_benchmark_commit": "benchmark-commit",
        "robomme_dataset_checksum": "dataset-checksum",
        "robomme_task_suite": "counting",
        "train_seed": 7,
        "mamba_repo_commit": "77069de5cdb55cbe98b670889c80df211e031039",
        "memory_backend": "mamba2",
        "memory_state_schema_version": 1,
        "memory_config": {"d_model": 2},
        "progress_depth": 1,
        "progress_layer_mapping": [0],
        "handoff_ratio": 0.2,
        "num_denoise_steps": 10,
        "prediction_horizon": 20,
        "execution_horizon": 16,
        "action_expert_gradient_checkpointing": False,
        "terminal_loss_batch_fraction": 1.0,
        "terminal_loss_queries_per_episode": None,
        "frozen_prefix_microbatch_size": 2,
        "loss_weights": {"terminal": 1.0, "handoff": 0.0, "boundary": 0.0},
        "training_dtype": "float32",
        "state_dtypes": {"ssm": "float32"},
        "kernel_mode": "fallback",
        "torch_version": torch.__version__,
        "triton_version": None,
        "cuda_version": torch.version.cuda,
        "gpu_name": None,
        "compute_capability": None,
    }
    metadata.update(overrides)
    return metadata


def _objects():
    model = _TinyFutureMamba()
    optimizer = torch.optim.AdamW(model.futuremamba.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 / (step + 1))
    return model, optimizer, scheduler


def _one_step(model, optimizer, scheduler):
    optimizer.zero_grad(set_to_none=True)
    loss = model.futuremamba(torch.ones(2, 2)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()


def test_checkpoint_contains_plugin_only_and_complete_runtime_state(tmp_path: Path):
    model, optimizer, scheduler = _objects()
    _one_step(model, optimizer, scheduler)

    checkpoint.save_futuremamba_checkpoint(
        tmp_path / "2", model, optimizer, scheduler, step=2, metadata=_metadata(model), data_iterator_step=2
    )

    assert {path.name for path in (tmp_path / "2").iterdir()} == {
        "plugin.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pt",
        "metadata.json",
    }
    keys = set(safetensors.torch.load_file(tmp_path / "2" / "plugin.safetensors"))
    assert keys
    assert all(key.startswith("futuremamba.") for key in keys)
    saved_metadata = json.loads((tmp_path / "2" / "metadata.json").read_text())
    assert saved_metadata["step"] == 2
    assert saved_metadata["data_iterator_step"] == 2


@pytest.mark.parametrize("field", ["memory_backend", "memory_state_schema_version", "base_checkpoint_checksum", "train_seed"])
def test_restore_rejects_identity_mismatch_by_field_name(tmp_path: Path, field: str):
    model, optimizer, scheduler = _objects()
    checkpoint.save_futuremamba_checkpoint(
        tmp_path / "2", model, optimizer, scheduler, step=2, metadata=_metadata(model), data_iterator_step=2
    )
    expected = _metadata(model)
    expected[field] = 999 if field in {"memory_state_schema_version", "train_seed"} else "wrong"

    with pytest.raises(ValueError, match=field):
        checkpoint.load_futuremamba_checkpoint(
            tmp_path / "2", model, optimizer, scheduler, expected_metadata=expected
        )


def test_restore_is_strict_and_rejects_missing_plugin_key(tmp_path: Path):
    model, optimizer, scheduler = _objects()
    checkpoint.save_futuremamba_checkpoint(
        tmp_path / "2", model, optimizer, scheduler, step=2, metadata=_metadata(model), data_iterator_step=2
    )
    weights = safetensors.torch.load_file(tmp_path / "2" / "plugin.safetensors")
    weights.pop(next(iter(weights)))
    safetensors.torch.save_file(weights, tmp_path / "2" / "plugin.safetensors")

    with pytest.raises(ValueError, match="strict|missing"):
        checkpoint.load_futuremamba_checkpoint(
            tmp_path / "2", model, optimizer, scheduler, expected_metadata=_metadata(model)
        )


def test_save_rejects_changed_frozen_base(tmp_path: Path):
    model, optimizer, scheduler = _objects()
    metadata = _metadata(model)
    with torch.no_grad():
        model.base.weight.add_(1)

    with pytest.raises(ValueError, match="base_checkpoint_checksum"):
        checkpoint.save_futuremamba_checkpoint(
            tmp_path / "2", model, optimizer, scheduler, step=2, metadata=metadata, data_iterator_step=2
        )


def test_restore_recovers_plugin_optimizer_scheduler_rng_and_cursor(tmp_path: Path):
    model, optimizer, scheduler = _objects()
    _one_step(model, optimizer, scheduler)
    saved_plugin = {name: tensor.detach().clone() for name, tensor in model.futuremamba.state_dict().items()}
    torch.manual_seed(321)
    checkpoint.save_futuremamba_checkpoint(
        tmp_path / "2", model, optimizer, scheduler, step=2, metadata=_metadata(model), data_iterator_step=17
    )
    expected_random = torch.rand(4)

    with torch.no_grad():
        for parameter in model.futuremamba.parameters():
            parameter.zero_()
    optimizer.state.clear()
    scheduler.last_epoch = 99
    torch.manual_seed(999)

    restored = checkpoint.load_futuremamba_checkpoint(
        tmp_path / "2", model, optimizer, scheduler, expected_metadata=_metadata(model)
    )

    assert restored.step == 2
    assert restored.data_iterator_step == 17
    assert optimizer.state
    assert scheduler.last_epoch == 1
    for name, tensor in model.futuremamba.state_dict().items():
        torch.testing.assert_close(tensor, saved_plugin[name])
    torch.testing.assert_close(torch.rand(4), expected_random)


def test_metadata_requires_every_identity_field(tmp_path: Path):
    model, optimizer, scheduler = _objects()
    metadata = _metadata(model)
    metadata.pop("mamba_repo_commit")

    with pytest.raises(ValueError, match="mamba_repo_commit"):
        checkpoint.save_futuremamba_checkpoint(
            tmp_path / "2", model, optimizer, scheduler, step=2, metadata=metadata, data_iterator_step=2
        )
