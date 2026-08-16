import pytest

from openpi.models_pytorch.futuremamba_config import FutureMambaPytorchConfig
from openpi.training import config as training_config


def test_robomme_futuremamba_uses_official_horizons_and_uniform_six():
    config = FutureMambaPytorchConfig(
        action_horizon=20,
        discrete_state_input=False,
        execution_horizon=16,
        progress_depth=6,
        terminal_loss_weight=1.0,
    )

    assert config.action_horizon == 20
    assert config.execution_horizon == 16
    assert config.progress_depth == 6
    assert config.discrete_state_input is False
    assert config.terminal_loss_weight == 1.0


def test_robomme_futuremamba_rejects_execution_horizon_above_prediction():
    with pytest.raises(ValueError, match="execution_horizon"):
        FutureMambaPytorchConfig(action_horizon=20, execution_horizon=21)


def test_robomme_pytorch_baseline_uses_eager_inference():
    config = training_config.get_config("pi05_robomme_pytorch")

    assert config.model.pytorch_compile_mode is None
