import dataclasses

from flax import nnx
from flax import traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import futuremamba as _futuremamba
from openpi.models import futuremamba_config as _futuremamba_config
from openpi.models import model as _model
from openpi.models.futuremamba import FutureMamba, integrate_handoff
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


def test_config_rejects_unimplemented_variants_at_construction():
    with pytest.raises(NotImplementedError, match="memory_backend"):
        _dummy_config(memory_backend="gru")
    with pytest.raises(NotImplementedError, match="decoder_mode"):
        _dummy_config(decoder_mode="action_memory_full")


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


def _episode_obs(config: _futuremamba_config.FutureMambaConfig, *, batch_size: int, num_queries: int):
    return jax.tree.map(
        lambda x: jnp.broadcast_to(x[:, None], (batch_size, num_queries, *x.shape[1:])),
        config.fake_obs(batch_size=batch_size),
    )


def test_memory_pooling_ignores_padding_and_uses_rightmost_valid_token():
    config = _dummy_config()
    model = config.create(jax.random.key(0))
    prefix_out = jnp.arange(2 * 8 * 64, dtype=jnp.float32).reshape(2, 8, 64)
    prefix_mask = jnp.array(
        [
            [True, False, True, False, True, False, False, False],
            [False, True, True, False, False, True, False, False],
        ]
    )
    padded_out = jnp.concatenate([prefix_out, jnp.full((2, 5, 64), 9999.0)], axis=1)
    padded_mask = jnp.pad(prefix_mask, ((0, 0), (0, 5)), constant_values=False)

    expected = prefix_out[jnp.arange(2), jnp.array([4, 5])]
    np.testing.assert_allclose(model._encode_memory_inputs(prefix_out, prefix_mask, "last_valid"), expected)

    for mode in ("last_valid", "attention", "tokens4", "tokens8"):
        pooled = model._encode_memory_inputs(prefix_out, prefix_mask, mode)
        padded = model._encode_memory_inputs(padded_out, padded_mask, mode)
        np.testing.assert_allclose(padded, pooled, rtol=1e-5, atol=1e-5)


def test_vlm_memory_projection_accepts_paligemma_prefix_width(monkeypatch):
    paligemma_config = _futuremamba._gemma.Config(
        width=48,
        depth=4,
        mlp_dim=96,
        num_heads=4,
        num_kv_heads=1,
        head_dim=12,
    )
    action_config = _futuremamba._gemma.Config(
        width=64,
        depth=4,
        mlp_dim=128,
        num_heads=4,
        num_kv_heads=1,
        head_dim=16,
    )

    def fake_get_config(variant):
        return {"vlm_tiny": paligemma_config, "action_tiny": action_config}[variant]

    monkeypatch.setattr(_futuremamba._gemma, "get_config", fake_get_config)
    monkeypatch.setattr(_futuremamba_config._gemma, "get_config", fake_get_config)
    config = _futuremamba_config.FutureMambaConfig(
        paligemma_variant="vlm_tiny",
        action_expert_variant="action_tiny",
        memory=MambaConfig(d_model=action_config.width),
        progress_depth=1,
    )
    plugin = _futuremamba._FutureMambaPlugin(config, rngs=nnx.Rngs(0))

    projected = plugin.vlm_to_memory(jnp.ones((2, 3, paligemma_config.width), dtype=jnp.float32))

    assert projected.shape == (2, 3, config.memory.d_model)


def test_memory_scan_resets_before_step_and_freezes_invalid_queries():
    config = _dummy_config(action_horizon=4, executed_horizon=2)
    model = config.create(jax.random.key(0))
    prefix_inputs = jnp.arange(2 * 3 * 64, dtype=jnp.float32).reshape(2, 3, 64) / 100.0
    executed_actions = jnp.ones((2, 3, 2, config.action_dim), dtype=jnp.float32)
    executed_mask = jnp.ones((2, 3, 2), dtype=jnp.bool_)
    query_mask = jnp.array([[True, True, True], [True, False, True]])
    reset_mask = jnp.array([[True, False, True], [True, False, True]])

    zero_state = model.initial_memory_state(batch_size=2)
    tokens, next_state = model._scan_memory(prefix_inputs, executed_actions, executed_mask, query_mask, reset_mask, zero_state)
    single_tokens, single_state = model._scan_memory(
        prefix_inputs[1:2, 2:3],
        executed_actions[1:2, 2:3],
        executed_mask[1:2, 2:3],
        jnp.ones((1, 1), dtype=jnp.bool_),
        jnp.ones((1, 1), dtype=jnp.bool_),
        model.initial_memory_state(batch_size=1),
    )

    np.testing.assert_allclose(tokens[1, 1], tokens[1, 0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(tokens[1, 2], single_tokens[0, 0], rtol=1e-5, atol=1e-5)
    jax.tree.map(lambda expected, got: np.testing.assert_allclose(got[1:2], expected, rtol=1e-5, atol=1e-5), single_state, next_state)


def test_integrate_handoff_uses_hard_step_split_and_no_extra_resampling():
    calls = {"progress": 0, "action": 0}
    seen = []

    def progress_velocity(x, t):
        calls["progress"] += 1
        seen.append(np.asarray(x).copy())
        return jnp.ones_like(x)

    def action_velocity(x, t):
        calls["action"] += 1
        seen.append(np.asarray(x).copy())
        return 2.0 * jnp.ones_like(x)

    noise = jnp.zeros((1, 2, 3), dtype=jnp.float32)
    result, diagnostics = integrate_handoff(
        noise,
        num_steps=10,
        handoff_steps=2,
        progress_velocity=progress_velocity,
        action_velocity=action_velocity,
    )

    assert calls == {"progress": 2, "action": 8}
    assert diagnostics["progress_calls"] == 2
    assert diagnostics["action_calls"] == 8
    np.testing.assert_allclose(diagnostics["handoff_state"], -0.2 * jnp.ones_like(noise), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result, -1.8 * jnp.ones_like(noise), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(seen[2], diagnostics["handoff_state"], rtol=1e-6, atol=1e-6)


def test_sample_actions_with_memory_matches_parent_at_zero_handoff_and_skips_action_at_full_handoff(monkeypatch):
    config = _dummy_config(action_horizon=4, executed_horizon=2)
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)
    state = model.initial_memory_state(batch_size=2)
    executed_actions = jnp.zeros((2, config.executed_horizon, config.action_dim), dtype=jnp.float32)
    executed_mask = jnp.zeros((2, config.executed_horizon), dtype=jnp.bool_)
    noise = jnp.arange(2 * config.action_horizon * config.action_dim, dtype=jnp.float32).reshape(
        2, config.action_horizon, config.action_dim
    ) / 100.0

    parent = super(FutureMamba, model).sample_actions(jax.random.key(1), obs, num_steps=3, noise=noise)
    actual, next_state, diagnostics = model.sample_actions_with_memory(
        jax.random.key(1), obs, state, executed_actions, executed_mask, num_steps=3, handoff_ratio=0.0, noise=noise
    )

    np.testing.assert_allclose(actual, parent, rtol=1e-5, atol=1e-5)
    assert diagnostics["handoff_steps"] == 0
    assert next_state.layers[0].ssm.shape[0] == 2

    def fail_action_velocity(*args, **kwargs):
        raise AssertionError("action expert must not run when handoff_ratio=1")

    monkeypatch.setattr(model, "action_velocity", fail_action_velocity)
    full, _, full_diagnostics = model.sample_actions_with_memory(
        jax.random.key(2), obs, state, executed_actions, executed_mask, num_steps=2, handoff_ratio=1.0, noise=noise
    )

    assert full.shape == noise.shape
    assert full_diagnostics["action_calls"] == 0


def test_compute_episode_loss_masks_padding_queries_and_returns_fixed_keys(monkeypatch):
    config = _dummy_config(action_horizon=4, executed_horizon=2, handoff_loss_weight=0.0, boundary_loss_weight=0.0)
    model = config.create(jax.random.key(0))
    batch_size = 2
    num_queries = 3
    obs = _episode_obs(config, batch_size=batch_size, num_queries=num_queries)
    actions = jnp.ones((batch_size, num_queries, config.action_horizon, config.action_dim), dtype=jnp.float32)
    action_mask = jnp.ones((batch_size, num_queries, config.action_horizon), dtype=jnp.bool_)
    query_mask = jnp.array([[True, True, False], [True, False, False]])

    batch_fields = {
        "observation": obs,
        "actions": actions,
        "action_mask": action_mask,
        "executed_actions": jnp.zeros((batch_size, num_queries, config.executed_horizon, config.action_dim), dtype=jnp.float32),
        "executed_action_mask": jnp.zeros((batch_size, num_queries, config.executed_horizon), dtype=jnp.bool_),
        "query_mask": query_mask,
        "reset_mask": jnp.array([[True, False, False], [True, False, False]]),
    }
    batch = type("EpisodeBatch", (), batch_fields)()
    padded_batch = type("EpisodeBatch", (), {**batch_fields, "actions": actions.at[~query_mask].set(1_000.0)})()

    def constant_progress(prefix_kv_cache, prefix_mask, memory_token, noisy_actions, timestep, **kwargs):
        del prefix_kv_cache, prefix_mask, memory_token, timestep, kwargs
        return jnp.zeros_like(noisy_actions)

    monkeypatch.setattr(model.futuremamba, "progress_expert", constant_progress)
    losses = model.compute_episode_loss(jax.random.key(0), batch)
    padded_losses = model.compute_episode_loss(jax.random.key(0), padded_batch)

    assert set(losses) == {"loss", "flow_loss", "handoff_loss", "handoff_error", "boundary_loss", "boundary_error"}
    assert losses["loss"].shape == ()
    np.testing.assert_allclose(padded_losses["loss"], losses["loss"], rtol=1e-5, atol=1e-5)

    sampled_time = model._sample_high_noise_time(jax.random.key(1), (1024,), handoff_steps=2, num_steps=10)
    assert jnp.all(sampled_time >= 0.8)


