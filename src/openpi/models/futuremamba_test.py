import dataclasses

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import pytest

from openpi.models import futuremamba_config as _futuremamba_config
from openpi.models import model as _model
from openpi.models.futuremamba import FutureMamba
from openpi.models.mamba import MambaConfig
from openpi.models.progress_expert import make_layer_mapping


def _flat_param_paths(config: _futuremamba_config.FutureMambaConfig):
    model = nnx.eval_shape(config.create, jax.random.key(0))
    state = nnx.state(model, nnx.Param)
    return {"/".join(str(part) for part in path): value for path, value in state.flat_state().items()}


def _dummy_config(**kwargs) -> _futuremamba_config.FutureMambaConfig:
    return _futuremamba_config.FutureMambaConfig(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        memory=MambaConfig(d_model=64),
        **kwargs,
    )


def test_config_forces_pi05_and_discrete_state_input():
    config = _dummy_config()

    assert config.pi05 is True
    assert config.discrete_state_input is True
    assert config.model_type is _model.ModelType.PI05

    with pytest.raises(ValueError, match="pi05"):
        _futuremamba_config.FutureMambaConfig(pi05=False)
    with pytest.raises(ValueError, match="discrete_state_input"):
        _futuremamba_config.FutureMambaConfig(discrete_state_input=False)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("handoff_ratio", -0.1, "handoff_ratio"),
        ("handoff_ratio", 1.1, "handoff_ratio"),
        ("num_denoise_steps", 0, "num_denoise_steps"),
        ("executed_horizon", 0, "executed_horizon"),
        ("executed_horizon", 51, "executed_horizon"),
        ("handoff_loss_weight", -0.1, "handoff_loss_weight"),
        ("boundary_loss_weight", -0.1, "boundary_loss_weight"),
        ("progress_depth", 0, "progress_depth"),
        ("progress_depth", 5, "progress_depth"),
        ("frame_stack_window", 0, "frame_stack_window"),
    ],
)
def test_config_validation_rejects_invalid_scalars(field, value, match):
    kwargs = {"memory": MambaConfig(d_model=64), "paligemma_variant": "dummy", "action_expert_variant": "dummy", field: value}
    with pytest.raises(ValueError, match=match):
        _futuremamba_config.FutureMambaConfig(**kwargs)


@pytest.mark.parametrize(
    ("indices", "match"),
    [
        ((0, 3), "length"),
        ((0, 2, 1, 3), "strictly increasing"),
        ((0, -1, 2, 3), "range"),
        ((0, 1, 4, 3), "range"),
        ((1, 2, 3, 4), "cover"),
        ((0, 1, 2, 2), "cover"),
    ],
)
def test_config_validation_rejects_invalid_progress_layer_indices(indices, match):
    with pytest.raises(ValueError, match=match):
        _dummy_config(progress_prefix_layer_indices=indices)


def test_config_default_progress_layer_indices_use_even_mapping():
    config = _dummy_config(progress_depth=4)

    assert config.resolved_progress_layer_indices == make_layer_mapping(4, 4)


def test_config_rejects_unimplemented_creation_variants():
    with pytest.raises(NotImplementedError, match="memory_backend"):
        _futuremamba_config.FutureMambaConfig(memory_backend="gru").create(jax.random.key(0))
    with pytest.raises(NotImplementedError, match="decoder_mode"):
        _futuremamba_config.FutureMambaConfig(decoder_mode="action_memory_full").create(jax.random.key(0))


def test_config_rejects_mamba_width_mismatch():
    with pytest.raises(ValueError, match="memory.*d_model"):
        _futuremamba_config.FutureMambaConfig(
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            memory=MambaConfig(d_model=32),
        )


def test_futuremamba_creates_real_plugin_components_under_single_path():
    config = _dummy_config()
    model = nnx.eval_shape(config.create, jax.random.key(0))

    assert isinstance(model, FutureMamba)
    assert hasattr(model, "futuremamba")
    assert hasattr(model.futuremamba, "memory")
    assert hasattr(model.futuremamba, "progress_expert")

    flat_params = _flat_param_paths(config)
    plugin_paths = [path for path in flat_params if path.startswith("futuremamba/")]
    assert plugin_paths
    assert any("vlm_memory_in_proj" in path for path in plugin_paths)
    assert any("executed_action_encoder" in path for path in plugin_paths)
    assert any("memory/" in path for path in plugin_paths)
    assert any("memory_token_proj" in path for path in plugin_paths)
    assert any("progress_expert" in path for path in plugin_paths)
    assert any(path.startswith("PaliGemma/") for path in flat_params)
    assert any(path.startswith("action_in_proj/") for path in flat_params)
    assert any(path.startswith("time_mlp_in/") for path in flat_params)
    assert any(path.startswith("time_mlp_out/") for path in flat_params)
    assert any(path.startswith("action_out_proj/") for path in flat_params)

    non_base_top_levels = {path.split("/", 1)[0] for path in flat_params} - {
        "PaliGemma",
        "action_in_proj",
        "time_mlp_in",
        "time_mlp_out",
        "action_out_proj",
    }
    assert non_base_top_levels == {"futuremamba"}


def test_futuremamba_filters_partition_all_params_exactly():
    config = _dummy_config()
    model = nnx.eval_shape(config.create, jax.random.key(0))
    all_params = nnx.state(model, nnx.Param).flat_state()
    trainable = nnx.state(model, nnx.All(nnx.Param, config.get_trainable_filter())).flat_state()
    frozen = nnx.state(model, nnx.All(nnx.Param, config.get_freeze_filter())).flat_state()

    assert trainable
    assert frozen
    assert all(path[0] == "futuremamba" for path in trainable)
    assert all(path[0] != "futuremamba" for path in frozen)
    assert set(trainable).isdisjoint(frozen)
    assert set(trainable) | set(frozen) == set(all_params)


def test_futuremamba_inherits_pi0_loss_and_sampling_shapes():
    config = _dummy_config()
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)
    act = config.fake_act(batch_size=2)

    loss = model.compute_loss(jax.random.key(1), obs, act)
    sampled = model.sample_actions(jax.random.key(2), obs, num_steps=2)

    assert loss.shape == (2, config.action_horizon)
    assert sampled.shape == (2, config.action_horizon, config.action_dim)


def test_futuremamba_config_dataclass_replace_keeps_forced_flags():
    config = _dummy_config()
    replaced = dataclasses.replace(config, handoff_ratio=0.5)

    assert replaced.pi05 is True
    assert replaced.discrete_state_input is True
    assert replaced.handoff_ratio == 0.5
