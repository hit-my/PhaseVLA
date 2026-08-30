from __future__ import annotations

import dataclasses

import pytest

from openpi.models_pytorch.futuremamba_config import FutureMambaPytorchConfig


def test_action_only_memory_contract_defaults():
    config = FutureMambaPytorchConfig()
    metadata = config.checkpoint_metadata()

    assert config.memory_input_source == "executed_actions"
    assert config.action_history_encoding == "masked_flat_projection"
    assert config.terminal_loss_weight == 0.0
    assert config.handoff_loss_weight == 0.0
    assert config.boundary_loss_weight == 0.0
    assert metadata["schema_version"] == 4
    assert metadata["memory_input_source"] == "executed_actions"
    assert metadata["action_history_encoding"] == "masked_flat_projection"
    assert metadata["memory_update_timing"] == "before_query_from_previous_executed_chunk"
    assert metadata["empty_history_behavior"] == "learned_token_without_state_update"
    assert metadata["uses_vlm_hidden_for_memory"] is False
    assert metadata["uses_prefix_kv_for_progress"] is False


def test_action_only_memory_contract_rejects_other_sources():
    with pytest.raises(ValueError, match="memory_input_source"):
        FutureMambaPytorchConfig(memory_input_source="vlm")
    with pytest.raises(ValueError, match="action_history_encoding"):
        FutureMambaPytorchConfig(action_history_encoding="mean")


def test_action_only_protocol_supports_eval_aligned_horizon():
    config = FutureMambaPytorchConfig(action_horizon=20, execution_horizon=20)
    assert dataclasses.asdict(config)["action_horizon"] == 20
    assert config.execution_horizon == 20
