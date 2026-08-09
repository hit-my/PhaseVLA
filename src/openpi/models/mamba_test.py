from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models.mamba import MambaConfig
from openpi.models.mamba import MambaLayerState
from openpi.models.mamba import MambaState
from openpi.models.mamba import SelectiveMamba


def test_step_matches_scan():
    model = SelectiveMamba(MambaConfig(d_model=16, d_state=4, d_conv=3, expand=2, depth=2), nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (2, 7, 16))
    scan_y, scan_state = model.scan(x, model.initial_state(batch_size=2))
    state = model.initial_state(batch_size=2)
    ys = []
    for index in range(x.shape[1]):
        y, state = model.step(x[:, index], state)
        ys.append(y)
    np.testing.assert_allclose(scan_y, jnp.stack(ys, axis=1), rtol=2e-5, atol=2e-5)
    assert jax.tree.all(jax.tree.map(lambda a, b: jnp.allclose(a, b), scan_state, state))


def test_initial_state_is_zero_and_repeatable():
    model = SelectiveMamba(MambaConfig(d_model=16, d_state=4, d_conv=3, expand=2, depth=2), nnx.Rngs(0))

    state_a = model.initial_state(batch_size=3, dtype=jnp.bfloat16)
    state_b = model.initial_state(batch_size=3, dtype=jnp.bfloat16)

    assert jax.tree.structure(state_a) == jax.tree.structure(state_b)
    for leaf_a, leaf_b in zip(jax.tree.leaves(state_a), jax.tree.leaves(state_b), strict=True):
        assert leaf_a.dtype == jnp.bfloat16
        np.testing.assert_array_equal(leaf_a, leaf_b)
        np.testing.assert_array_equal(leaf_a, jnp.zeros_like(leaf_a))

def test_bfloat16_state_keeps_dtype_after_step_and_scan():
    model = SelectiveMamba(MambaConfig(d_model=16, d_state=4, d_conv=3, expand=2, depth=2), nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (2, 5, 16))

    _, step_state = model.step(x[:, 0], model.initial_state(batch_size=2, dtype=jnp.bfloat16))
    _, scan_state = model.scan(x, model.initial_state(batch_size=2, dtype=jnp.bfloat16))

    for leaf in jax.tree.leaves(step_state):
        assert leaf.dtype == jnp.bfloat16
    for leaf in jax.tree.leaves(scan_state):
        assert leaf.dtype == jnp.bfloat16


def test_state_tree_and_shapes_do_not_depend_on_sequence_length():
    config = MambaConfig(d_model=12, d_state=5, d_conv=4, expand=2, depth=3)
    model = SelectiveMamba(config, nnx.Rngs(0))
    initial = model.initial_state(batch_size=2)

    assert len(initial.layers) == config.depth
    for layer_state in initial.layers:
        assert layer_state.ssm.shape == (2, config.d_model * config.expand, config.d_state)
        assert layer_state.conv.shape == (2, config.d_conv - 1, config.d_model * config.expand)

    for length in (1, 6):
        x = jax.random.normal(jax.random.key(length), (2, length, config.d_model))
        _, state = model.scan(x, model.initial_state(batch_size=2))
        assert jax.tree.structure(state) == jax.tree.structure(initial)
        assert jax.tree.map(lambda value: value.shape, state) == jax.tree.map(lambda value: value.shape, initial)


def test_future_inputs_do_not_change_past_outputs():
    model = SelectiveMamba(MambaConfig(d_model=8, d_state=4, d_conv=3, expand=2, depth=2), nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(1), (2, 7, 8))
    x_with_different_future = x.at[:, 4:].set(jax.random.normal(jax.random.key(2), (2, 3, 8)) * 100.0)

    y, _ = model.scan(x, model.initial_state(batch_size=2))
    y_with_different_future, _ = model.scan(x_with_different_future, model.initial_state(batch_size=2))

    np.testing.assert_allclose(y[:, :4], y_with_different_future[:, :4], rtol=2e-5, atol=2e-5)


def test_causal_convolution_cache_keeps_oldest_to_newest_projected_tokens():
    config = MambaConfig(d_model=4, d_state=3, d_conv=4, expand=1, depth=1)
    model = SelectiveMamba(config, nnx.Rngs(0))
    state = model.initial_state(batch_size=1)
    projected_tokens = []

    for index in range(5):
        token = (jnp.arange(config.d_model, dtype=jnp.float32) + index * 10)[None, :]
        projected, _ = jnp.split(model.layers[0].in_proj(model.layers[0].norm(token)), 2, axis=-1)
        _, state = model.step(token, state)
        projected_tokens.append(projected)

        kept = projected_tokens[-(config.d_conv - 1) :]
        padding = [jnp.zeros_like(projected)] * (config.d_conv - 1 - len(kept))
        expected = jnp.stack([*padding, *kept], axis=1)
        np.testing.assert_allclose(state.layers[0].conv, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"d_model": 0},
        {"d_model": 4, "d_state": 0},
        {"d_model": 4, "d_conv": 0},
        {"d_model": 4, "expand": 0},
        {"d_model": 4, "depth": 0},
        {"d_model": 4, "dt_rank": 0},
    ],
)
def test_config_rejects_non_positive_dimensions(kwargs):
    with pytest.raises(ValueError):
        MambaConfig(**kwargs)


def test_config_defaults_dt_rank_to_ceiling_d_model_over_16():
    assert MambaConfig(d_model=16).dt_rank == 1
    assert MambaConfig(d_model=17).dt_rank == 2


def test_rejects_input_and_state_shape_mismatch():
    config = MambaConfig(d_model=8, d_state=3, d_conv=3, expand=2, depth=2)
    model = SelectiveMamba(config, nnx.Rngs(0))
    state = model.initial_state(batch_size=2)

    with pytest.raises(ValueError, match="d_model|shape"):
        model.step(jnp.zeros((2, config.d_model + 1)), state)
    with pytest.raises(ValueError, match="batch|state"):
        model.step(jnp.zeros((3, config.d_model)), state)
    with pytest.raises(ValueError, match="rank|shape"):
        model.scan(jnp.zeros((2, config.d_model)), state)
    with pytest.raises(ValueError, match="d_model|shape"):
        model.scan(jnp.zeros((2, 4, config.d_model + 1)), state)

    bad_layer = MambaLayerState(
        ssm=jnp.zeros((2, config.d_model * config.expand, config.d_state + 1)),
        conv=state.layers[0].conv,
    )
    bad_state = MambaState(layers=(bad_layer, state.layers[1]))
    with pytest.raises(ValueError, match="state"):
        model.step(jnp.zeros((2, config.d_model)), bad_state)
