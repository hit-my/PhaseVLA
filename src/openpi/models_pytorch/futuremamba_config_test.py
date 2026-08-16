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
    assert config.progress_depth == 6
    assert config.execution_horizon == 16
    assert config.terminal_loss_weight == 1.0
    assert config.num_denoise_steps == 10
    assert config.frozen_prefix_microbatch_size == 2


def test_terminal_memory_options_are_validated_and_checkpointed():
    config = FutureMambaPytorchConfig(
        action_expert_gradient_checkpointing=True,
        terminal_loss_batch_fraction=0.5,
        terminal_loss_queries_per_episode=2,
    )

    metadata = config.checkpoint_metadata()
    assert metadata["action_expert_gradient_checkpointing"] is True
    assert metadata["terminal_loss_batch_fraction"] == 0.5
    assert metadata["terminal_loss_queries_per_episode"] == 2
    assert metadata["frozen_prefix_microbatch_size"] == 2
    assert metadata["schema_version"] == 3
    with pytest.raises(ValueError, match="terminal_loss_batch_fraction"):
        FutureMambaPytorchConfig(terminal_loss_batch_fraction=0.0)
    with pytest.raises(ValueError, match="terminal_loss_batch_fraction"):
        FutureMambaPytorchConfig(terminal_loss_batch_fraction=1.1)
    with pytest.raises(ValueError, match="terminal_loss_queries_per_episode"):
        FutureMambaPytorchConfig(terminal_loss_queries_per_episode=0)
    with pytest.raises(ValueError, match="frozen_prefix_microbatch_size"):
        FutureMambaPytorchConfig(frozen_prefix_microbatch_size=0)


def test_vlm_only_checkpoint_schema_replaces_executed_history():
    config = FutureMambaPytorchConfig()

    assert not hasattr(config, "executed_horizon")
    metadata = config.checkpoint_metadata()
    assert metadata["prediction_horizon"] == config.action_horizon
    assert metadata["execution_horizon"] == config.execution_horizon
    assert "executed_horizon" not in metadata
    assert {
        "robomme_policy_commit",
        "robomme_benchmark_commit",
        "robomme_dataset_checksum",
        "robomme_task_suite",
    } <= metadata.keys()
