import dataclasses

import pytest

from openpi.models_pytorch.futuremamba_config import FutureMambaPytorchConfig


def test_backend_names_are_explicit():
    assert FutureMambaPytorchConfig(memory_backend="mamba2").memory_backend == "mamba2"
    assert FutureMambaPytorchConfig(memory_backend="mamba3_siso").memory_backend == "mamba3_siso"
    with pytest.raises(ValueError, match="memory_backend"):
        FutureMambaPytorchConfig(memory_backend="mamba")


def test_default_capacity_and_handoff():
    config = FutureMambaPytorchConfig()
    assert dataclasses.asdict(config.memory) == {
        "d_model": 1024,
        "depth": 2,
        "d_state": 128,
        "expand": 2,
        "headdim": 64,
        "ngroups": 1,
        "d_conv": 4,
        "rms_norm": True,
        "residual_in_fp32": True,
        "fused_add_norm": False,
        "use_mem_eff_path": False,
    }
    assert config.progress_depth == 4
    assert config.executed_horizon == 5
    assert config.num_denoise_steps == 10
