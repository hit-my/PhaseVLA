from flax import traverse_util
from flax.core import freeze
from flax.core import unfreeze
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.gemma as _gemma
from openpi.models.progress_expert import ProgressExpert
from openpi.models.progress_expert import default_progress_depth
from openpi.models.progress_expert import make_layer_mapping


def _tiny_config(depth: int = 8) -> _gemma.Config:
    return _gemma.Config(width=16, depth=depth, mlp_dim=32, num_heads=2, num_kv_heads=1, head_dim=8)


def _inputs(config: _gemma.Config):
    batch_size = 2
    prefix_len = 4
    action_horizon = 3
    action_dim = 5
    key = jax.random.key(1)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    prefix_k = jax.random.normal(
        k1, (config.depth, batch_size, prefix_len, config.num_kv_heads, config.head_dim)
    )
    prefix_v = jax.random.normal(
        k2, (config.depth, batch_size, prefix_len, config.num_kv_heads, config.head_dim)
    )
    prefix_mask = jnp.array([[True, True, False, False], [True, True, True, False]])
    memory_token = jax.random.normal(k3, (batch_size, 1, config.width))
    noisy_actions = jax.random.normal(k4, (batch_size, action_horizon, action_dim))
    timestep = jnp.array([0.25, 0.75], dtype=jnp.float32)
    return (prefix_k, prefix_v), prefix_mask, memory_token, noisy_actions, timestep


def _model(config: _gemma.Config, *, progress_depth: int | None = 2) -> ProgressExpert:
    return ProgressExpert(
        action_expert_config=config,
        action_dim=5,
        action_horizon=3,
        progress_depth=progress_depth,
        embed_dtype="float32",
    )


def _init_variables(model: ProgressExpert, inputs):
    return model.init(jax.random.key(2), *inputs)


def _enable_adarms_residual_gates(variables):
    params = unfreeze(variables["params"])

    def visit(node):
        if isinstance(node, dict):
            dense = node.get("Dense_0")
            if isinstance(dense, dict) and "bias" in dense:
                bias = dense["bias"]
                hidden = bias.shape[0] // 3
                dense["bias"] = bias.at[2 * hidden :].set(1.0)
            for child in node.values():
                visit(child)

    visit(params)
    unfrozen = unfreeze(variables)
    unfrozen["params"] = params
    return freeze(unfrozen)


def _velocity(model: ProgressExpert, variables, inputs, *, use_prefix_cache: bool = True):
    return model.apply(variables, *inputs, use_prefix_cache=use_prefix_cache)


def _with_masked_prefix_changed(prefix_kv, prefix_mask):
    mask = jnp.logical_not(prefix_mask)[None, :, :, None, None]
    return tuple(jnp.where(mask, value + 1000.0, value) for value in prefix_kv)


def _with_all_prefix_changed(prefix_kv):
    return tuple(value + 1000.0 for value in prefix_kv)


def test_forward_shape_and_progress_depth_metadata():
    config = _tiny_config(depth=8)
    inputs = _inputs(config)
    model = _model(config, progress_depth=2)

    variables = _init_variables(model, inputs)
    velocity = _velocity(model, variables, inputs)

    assert velocity.shape == inputs[3].shape
    assert model.layer_mapping == (0, 7)
    assert model.checkpoint_metadata() == {"progress_depth": 2, "layer_mapping": (0, 7)}

    flat_params = traverse_util.flatten_dict(variables["params"], sep="/")
    assert any(path.startswith("progress/layers_0/") for path in flat_params)
    assert any(path.startswith("progress/layers_1/") for path in flat_params)
    assert not any(path.startswith("progress/layers_2/") for path in flat_params)


def test_memory_token_changes_velocity():
    config = _tiny_config()
    inputs = _inputs(config)
    model = _model(config)
    variables = _enable_adarms_residual_gates(_init_variables(model, inputs))

    changed_inputs = (inputs[0], inputs[1], inputs[2] + 50.0, inputs[3], inputs[4])

    original = _velocity(model, variables, inputs)
    changed = _velocity(model, variables, changed_inputs)

    assert not jnp.allclose(original, changed, rtol=1e-5, atol=1e-5)


def test_valid_prefix_cache_changes_velocity():
    config = _tiny_config()
    inputs = _inputs(config)
    model = _model(config)
    variables = _enable_adarms_residual_gates(_init_variables(model, inputs))

    changed_prefix = tuple(value.at[:, :, 0, :, :].add(1000.0) for value in inputs[0])
    changed_inputs = (changed_prefix, inputs[1], inputs[2], inputs[3], inputs[4])

    original = _velocity(model, variables, inputs)
    changed = _velocity(model, variables, changed_inputs)

    assert not jnp.allclose(original, changed, rtol=1e-5, atol=1e-5)


def test_masked_prefix_cache_does_not_change_velocity():
    config = _tiny_config()
    inputs = _inputs(config)
    model = _model(config)
    variables = _enable_adarms_residual_gates(_init_variables(model, inputs))

    changed_inputs = (_with_masked_prefix_changed(inputs[0], inputs[1]), inputs[1], inputs[2], inputs[3], inputs[4])

    original = _velocity(model, variables, inputs)
    changed = _velocity(model, variables, changed_inputs)

    np.testing.assert_allclose(original, changed, rtol=1e-5, atol=1e-5)


def test_disabled_prefix_cache_ignores_prefix_changes():
    config = _tiny_config()
    inputs = _inputs(config)
    model = _model(config)
    variables = _enable_adarms_residual_gates(_init_variables(model, inputs))

    changed_inputs = (_with_all_prefix_changed(inputs[0]), inputs[1], inputs[2], inputs[3], inputs[4])

    original = _velocity(model, variables, inputs, use_prefix_cache=False)
    changed = _velocity(model, variables, changed_inputs, use_prefix_cache=False)

    np.testing.assert_allclose(original, changed, rtol=1e-5, atol=1e-5)


def test_prefix_cache_is_not_modified_in_place():
    config = _tiny_config()
    inputs = _inputs(config)
    model = _model(config)
    variables = _init_variables(model, inputs)
    cache_before = jax.tree.map(lambda value: value.copy(), inputs[0])

    _velocity(model, variables, inputs)

    jax.tree.map(np.testing.assert_array_equal, cache_before, inputs[0])


def test_default_layer_mapping_covers_first_and_last_and_is_strict():
    depth = default_progress_depth(action_depth=18)
    mapping = make_layer_mapping(action_depth=18, progress_depth=depth)

    assert depth == 4
    assert mapping[0] == 0
    assert mapping[-1] == 17
    assert all(left < right for left, right in zip(mapping, mapping[1:]))


def test_parameter_tree_has_no_context_side_parameters():
    config = _tiny_config()
    inputs = _inputs(config)
    model = _model(config)
    variables = _init_variables(model, inputs)

    flat_params = traverse_util.flatten_dict(variables["params"], sep="/")
    assert set(variables["params"]) == {"action_in_proj", "time_mlp_in", "time_mlp_out", "progress", "final_norm", "action_out_proj"}
    forbidden = (
        "context",
        "embedder",
        "prefix_projection",
        "context_projection",
        "context_ffn",
        "context_query",
        "context_output",
    )
    assert not any(any(term in path for term in forbidden) for path in flat_params)


def test_progress_expert_forward_is_jittable():
    config = _tiny_config()
    inputs = _inputs(config)
    model = _model(config)
    variables = _init_variables(model, inputs)

    @jax.jit
    def apply_jit(prefix_kv_cache, prefix_mask, memory_token, noisy_actions, timestep):
        return model.apply(variables, prefix_kv_cache, prefix_mask, memory_token, noisy_actions, timestep)

    velocity = apply_jit(*inputs)

    assert velocity.shape == inputs[3].shape
