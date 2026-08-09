import dataclasses

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import futuremamba as _futuremamba
from openpi.models import futuremamba_config as _futuremamba_config
from openpi.models import model as _model
from openpi.models.futuremamba import FutureMamba
from openpi.models.futuremamba import integrate_handoff
from openpi.models.mamba import MambaConfig
from openpi.models.progress_expert import make_layer_mapping


def _flat_param_paths(config: _futuremamba_config.FutureMambaConfig):
    model = nnx.eval_shape(config.create, jax.random.key(0))
    state = nnx.state(model, nnx.Param)
    return {"/".join(str(part) for part in path): value for path, value in state.flat_state().items()}


def _param_shape(param):
    return param.value.shape if hasattr(param, "value") else param.shape


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
    kwargs = {
        "memory": MambaConfig(d_model=64),
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        field: value,
    }
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


def test_config_supports_ablation_backends_and_serializable_metadata():
    for backend in ("mamba", "gru", "lstm", "frame_stack", "none"):
        config = _dummy_config(memory_backend=backend)
        metadata = config.checkpoint_metadata()
        assert metadata["memory_backend"] == backend
        assert metadata["rho"] == config.handoff_ratio
        assert metadata["decoder_mode"] == "handoff"
        assert metadata["coupling"] == "hard"
        assert metadata["memory_parameter_error"] >= 0.0
        assert isinstance(metadata["parameter_matched"], bool)

    full = _dummy_config(decoder_mode="action_memory_full", coupling="residual", bptt_window_queries=2)
    metadata = full.checkpoint_metadata()
    assert metadata["decoder_mode"] == "action_memory_full"
    assert metadata["coupling"] == "residual"
    assert metadata["bptt_window_queries"] == 2
    assert metadata["all_ablation_switches"]["use_prefix_cache"] is True

    unmatched = _dummy_config(memory_backend="frame_stack", frame_stack_window=2_000)
    assert unmatched.memory_backend_parameter_matched is False
    assert unmatched.checkpoint_metadata()["parameter_matched"] is False


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


@pytest.mark.parametrize(
    ("backend", "expected_shapes"),
    [
        (
            "gru",
            {
                "input_proj/kernel": lambda config: (config.memory.d_model, config.memory_backend_hidden_width),
                "input_gates/kernel": lambda config: (
                    config.memory_backend_hidden_width,
                    3 * config.memory_backend_hidden_width,
                ),
                "hidden_gates/kernel": lambda config: (
                    config.memory_backend_hidden_width,
                    3 * config.memory_backend_hidden_width,
                ),
                "output_proj/kernel": lambda config: (config.memory_backend_hidden_width, config.memory.d_model),
            },
        ),
        (
            "lstm",
            {
                "input_proj/kernel": lambda config: (config.memory.d_model, config.memory_backend_hidden_width),
                "input_gates/kernel": lambda config: (
                    config.memory_backend_hidden_width,
                    4 * config.memory_backend_hidden_width,
                ),
                "hidden_gates/kernel": lambda config: (
                    config.memory_backend_hidden_width,
                    4 * config.memory_backend_hidden_width,
                ),
                "output_proj/kernel": lambda config: (config.memory_backend_hidden_width, config.memory.d_model),
            },
        ),
        (
            "frame_stack",
            {
                "stack_proj/kernel": lambda config: (
                    config.frame_stack_window * config.memory.d_model,
                    config.memory_backend_hidden_width,
                ),
                "output_proj/kernel": lambda config: (config.memory_backend_hidden_width, config.memory.d_model),
            },
        ),
    ],
)
def test_futuremamba_instantiates_trainable_non_mamba_memory_backends(backend, expected_shapes):
    config = _dummy_config(memory_backend=backend, frame_stack_window=3)
    flat_params = _flat_param_paths(config)

    backend_paths = {path: value for path, value in flat_params.items() if path.startswith("futuremamba/memory/")}
    assert backend_paths
    assert not any("layers/" in path for path in backend_paths)
    assert not any("A_log" in path or "conv_kernel" in path for path in backend_paths)
    for suffix, expected_shape in expected_shapes.items():
        assert _param_shape(backend_paths[f"futuremamba/memory/{suffix}"]) == expected_shape(config)


def test_futuremamba_none_backend_has_no_backend_parameters():
    flat_params = _flat_param_paths(_dummy_config(memory_backend="none"))

    assert not any(path.startswith("futuremamba/memory/") for path in flat_params)


@pytest.mark.parametrize("backend", ("gru", "lstm", "frame_stack"))
def test_trainable_memory_backend_step_output_depends_on_backend_parameters(backend):
    config = _dummy_config(memory_backend=backend, frame_stack_window=3)
    model = config.create(jax.random.key(0))
    step_input = jnp.ones((1, config.memory.d_model), dtype=jnp.float32)
    state = model.initial_memory_state(batch_size=1)

    before, _ = model._memory_step(step_input, state)
    model.futuremamba.memory.output_proj.bias.value = model.futuremamba.memory.output_proj.bias.value + 1.0
    after, _ = model._memory_step(step_input, state)

    assert not np.allclose(before, after)


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


def _episode_batch(
    config: _futuremamba_config.FutureMambaConfig,
    *,
    batch_size: int = 1,
    num_queries: int = 1,
    actions=None,
    action_mask=None,
    query_mask=None,
    executed_actions=None,
    executed_action_mask=None,
    reset_mask=None,
):
    if actions is None:
        actions = jnp.zeros((batch_size, num_queries, config.action_horizon, config.action_dim), dtype=jnp.float32)
    if action_mask is None:
        action_mask = jnp.ones((batch_size, num_queries, config.action_horizon), dtype=jnp.bool_)
    if query_mask is None:
        query_mask = jnp.ones((batch_size, num_queries), dtype=jnp.bool_)
    if executed_actions is None:
        executed_actions = jnp.zeros(
            (batch_size, num_queries, config.executed_horizon, config.action_dim), dtype=actions.dtype
        )
    if executed_action_mask is None:
        executed_action_mask = jnp.zeros(executed_actions.shape[:-1], dtype=jnp.bool_)
    if reset_mask is None:
        reset_mask = jnp.zeros((batch_size, num_queries), dtype=jnp.bool_)
    return type(
        "EpisodeBatch",
        (),
        {
            "observation": _episode_obs(config, batch_size=batch_size, num_queries=num_queries),
            "actions": actions,
            "action_mask": action_mask,
            "executed_actions": executed_actions,
            "executed_action_mask": executed_action_mask,
            "query_mask": query_mask,
            "reset_mask": reset_mask,
        },
    )()


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
    tokens, next_state = model._scan_memory(
        prefix_inputs, executed_actions, executed_mask, query_mask, reset_mask, zero_state
    )
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
    jax.tree.map(
        lambda expected, got: np.testing.assert_allclose(got[1:2], expected, rtol=1e-5, atol=1e-5),
        single_state,
        next_state,
    )


@pytest.mark.parametrize(
    ("coupling", "expected", "calls"),
    [
        ("hard", -6.0, {"progress": 2, "action": 2}),
        ("convex", -5.0, {"progress": 2, "action": 4}),
        ("residual", -5.75, {"progress": 2, "action": 4}),
    ],
)
def test_integrate_handoff_uses_exact_coupling_formulas_and_reports_solver_stats(coupling, expected, calls):
    actual_calls = {"progress": 0, "action": 0}
    seen = []

    def progress_velocity(x, t):
        actual_calls["progress"] += 1
        seen.append(("progress", np.asarray(x).copy(), np.asarray(t).copy()))
        return 10.0 * jnp.ones_like(x)

    def action_velocity(x, t):
        actual_calls["action"] += 1
        seen.append(("action", np.asarray(x).copy(), np.asarray(t).copy()))
        return 2.0 * jnp.ones_like(x)

    noise = jnp.zeros((1, 1, 1), dtype=jnp.float32)
    result, diagnostics = integrate_handoff(
        noise,
        num_steps=4,
        handoff_steps=2,
        progress_velocity=progress_velocity,
        action_velocity=action_velocity,
        coupling=coupling,
    )

    assert actual_calls == calls
    assert diagnostics["progress_calls"] == calls["progress"]
    assert diagnostics["action_calls"] == calls["action"]
    assert diagnostics["solver_steps"] == 4
    assert diagnostics["solver_dt"] == -0.25
    np.testing.assert_allclose(result, expected * jnp.ones_like(noise), rtol=1e-6, atol=1e-6)
    if coupling == "hard":
        np.testing.assert_allclose(diagnostics["handoff_state"], -5.0 * jnp.ones_like(noise), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(seen[2][1], diagnostics["handoff_state"], rtol=1e-6, atol=1e-6)


def test_sample_actions_with_memory_matches_parent_at_zero_handoff_and_skips_action_at_full_handoff(monkeypatch):
    config = _dummy_config(action_horizon=4, executed_horizon=2)
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=2)
    state = model.initial_memory_state(batch_size=2)
    executed_actions = jnp.zeros((2, config.executed_horizon, config.action_dim), dtype=jnp.float32)
    executed_mask = jnp.zeros((2, config.executed_horizon), dtype=jnp.bool_)
    noise = (
        jnp.arange(2 * config.action_horizon * config.action_dim, dtype=jnp.float32).reshape(
            2, config.action_horizon, config.action_dim
        )
        / 100.0
    )

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
        "executed_actions": jnp.zeros(
            (batch_size, num_queries, config.executed_horizon, config.action_dim), dtype=jnp.float32
        ),
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


@pytest.mark.parametrize(
    ("rho", "expected_steps"),
    [
        (0.0, 0),
        (0.05, 1),
        (0.25, 3),
        (1.0, 10),
    ],
)
def test_handoff_steps_uses_ceiling_and_clamps_ratio(rho, expected_steps):
    config = _dummy_config(handoff_ratio=rho)
    model = config.create(jax.random.key(0))

    assert model._handoff_steps(None, 10) == expected_steps


def test_high_noise_time_uses_beta_tail_rescaled_to_handoff_interval():
    config = _dummy_config()
    model = config.create(jax.random.key(0))

    samples = model._sample_high_noise_time(jax.random.key(1), (32768,), handoff_steps=2, num_steps=10)

    assert jnp.all(samples >= 0.8)
    assert jnp.all(samples <= 1.0)
    np.testing.assert_allclose(jnp.mean(samples), 0.8 + 0.2 * (1.5 / 2.5), atol=0.003)


def test_mean_masked_action_error_weights_actions_queries_and_episodes_equally():
    config = _dummy_config()
    model = config.create(jax.random.key(0))
    error = jnp.array(
        [
            [[1.0, 3.0, 999.0, 999.0], [9.0, 999.0, 999.0, 999.0]],
            [[100.0, 100.0, 100.0, 100.0], [1000.0, 1000.0, 1000.0, 1000.0]],
        ],
        dtype=jnp.float32,
    )
    action_mask = jnp.array(
        [
            [[True, True, False, False], [True, False, False, False]],
            [[True, True, True, True], [True, True, True, True]],
        ]
    )
    query_mask = jnp.array([[True, True], [True, False]])

    actual = model._mean_masked_action_error(error, action_mask, query_mask)

    np.testing.assert_allclose(actual, ((2.0 + 9.0) / 2.0 + 100.0) / 2.0, rtol=1e-6, atol=1e-6)


def test_compute_episode_loss_adds_executed_action_noise_only_for_enabled_token_action(monkeypatch):
    base_config = _dummy_config(
        action_horizon=4,
        executed_horizon=2,
        handoff_loss_weight=0.0,
        boundary_loss_weight=0.0,
        executed_action_noise_std=10.0,
    )
    executed_actions = jnp.zeros((1, 2, base_config.executed_horizon, base_config.action_dim), dtype=jnp.float32)
    executed_action_mask = jnp.array([[[True, False], [True, True]]])
    batch = _episode_batch(
        base_config,
        batch_size=1,
        num_queries=2,
        executed_actions=executed_actions,
        executed_action_mask=executed_action_mask,
    )

    def install_capture(config):
        model = config.create(jax.random.key(0))
        seen = []

        def capture_scan(prefix_inputs, scanned_actions, scanned_action_mask, query_mask, reset_mask, state):
            del scanned_action_mask, query_mask, reset_mask
            seen.append(scanned_actions)
            return prefix_inputs, state

        def zero_progress(prefix_kv_cache, prefix_mask, memory_token, noisy_actions, timestep, **kwargs):
            del prefix_kv_cache, prefix_mask, memory_token, timestep, kwargs
            return jnp.zeros_like(noisy_actions)

        monkeypatch.setattr(model, "_scan_memory", capture_scan)
        monkeypatch.setattr(model.futuremamba, "progress_expert", zero_progress)
        return model, seen

    model, seen = install_capture(base_config)
    no_noise_loss = model.compute_episode_loss(jax.random.key(7), batch, add_executed_action_noise=False)["flow_loss"]
    no_noise_actions = seen[-1]
    noisy_loss = model.compute_episode_loss(jax.random.key(7), batch, add_executed_action_noise=True)["flow_loss"]
    noisy_actions = seen[-1]

    np.testing.assert_allclose(no_noise_loss, noisy_loss, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(no_noise_actions, executed_actions, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(noisy_actions)[np.asarray(~executed_action_mask)],
        np.asarray(executed_actions)[np.asarray(~executed_action_mask)],
        rtol=1e-6,
        atol=1e-6,
    )
    assert not np.allclose(
        np.asarray(noisy_actions)[np.asarray(executed_action_mask)],
        np.asarray(executed_actions)[np.asarray(executed_action_mask)],
    )

    token_only_config = dataclasses.replace(base_config, memory_input="token_only")
    token_only_model, token_only_seen = install_capture(token_only_config)
    token_only_model.compute_episode_loss(jax.random.key(7), batch, add_executed_action_noise=True)
    np.testing.assert_allclose(token_only_seen[-1], executed_actions, rtol=1e-6, atol=1e-6)

    zero_std_config = dataclasses.replace(base_config, executed_action_noise_std=0.0)
    zero_std_model, zero_std_seen = install_capture(zero_std_config)
    zero_std_model.compute_episode_loss(jax.random.key(7), batch, add_executed_action_noise=True)
    np.testing.assert_allclose(zero_std_seen[-1], executed_actions, rtol=1e-6, atol=1e-6)


def test_boundary_loss_stops_gradient_through_handoff_state_and_action_velocity(monkeypatch):
    config = _dummy_config(
        action_horizon=2,
        executed_horizon=1,
        handoff_ratio=0.5,
        num_denoise_steps=2,
        handoff_loss_weight=0.0,
        boundary_loss_weight=1.0,
    )
    model = config.create(jax.random.key(0))
    actions = jnp.zeros((1, 1, config.action_horizon, config.action_dim), dtype=jnp.float32)
    executed_action_mask = jnp.ones((1, 1, config.executed_horizon), dtype=jnp.bool_)
    current_executed = {"value": None}

    def scan_from_executed(prefix_inputs, scanned_actions, scanned_action_mask, query_mask, reset_mask, state):
        del prefix_inputs, scanned_action_mask, query_mask, reset_mask
        memory_tokens = jnp.zeros((1, 1, config.memory.d_model), dtype=scanned_actions.dtype)
        memory_tokens = memory_tokens.at[..., : config.action_dim].set(scanned_actions[:, :, 0, :])
        return memory_tokens, state

    def project_identity(memory_tokens):
        return memory_tokens

    def progress_velocity(prefix_mask, kv_cache, memory_token, x_t, timestep):
        del prefix_mask, kv_cache
        rollout_velocity = jnp.broadcast_to(memory_token[:, :, : config.action_dim], x_t.shape)
        is_rollout = (timestep == 1.0)[:, None, None]
        return jnp.where(is_rollout, rollout_velocity, x_t)

    def action_velocity(observation, x_t, timestep, prefix_mask, kv_cache):
        del observation, x_t, timestep, prefix_mask, kv_cache
        action_from_executed = current_executed["value"].reshape(1, config.executed_horizon, config.action_dim)[:, 0]
        return jnp.broadcast_to(action_from_executed[:, None, :], (1, config.action_horizon, config.action_dim))

    monkeypatch.setattr(model, "_scan_memory", scan_from_executed)
    monkeypatch.setattr(model.futuremamba, "project_memory_token", project_identity)
    monkeypatch.setattr(model, "_progress_velocity", progress_velocity)
    monkeypatch.setattr(model, "action_velocity", action_velocity)

    def boundary_loss(executed_actions):
        current_executed["value"] = executed_actions
        batch = _episode_batch(
            config,
            actions=actions,
            executed_actions=executed_actions,
            executed_action_mask=executed_action_mask,
        )
        return model.compute_episode_loss(jax.random.key(3), batch)["boundary_loss"]

    executed_actions = jnp.ones((1, 1, config.executed_horizon, config.action_dim), dtype=jnp.float32)
    grad = jax.grad(boundary_loss)(executed_actions)

    np.testing.assert_allclose(grad, 0.0, rtol=1e-6, atol=1e-6)


def test_initial_memory_state_is_stable_pytree_for_all_backends():
    for backend in ("mamba", "gru", "lstm", "frame_stack", "none"):
        config = _dummy_config(memory_backend=backend, frame_stack_window=3)
        model = config.create(jax.random.key(0))
        state = model.initial_memory_state(batch_size=2)
        restored = jax.tree.map(lambda x: jnp.array(x), state)
        leaves = jax.tree.leaves(restored)
        assert leaves
        assert all(leaf.shape[0] == 2 for leaf in leaves)

        prefix_inputs = jnp.ones((2, 3, config.memory.d_model), dtype=jnp.float32)
        executed_actions = jnp.zeros((2, 3, config.executed_horizon, config.action_dim), dtype=jnp.float32)
        executed_mask = jnp.zeros((2, 3, config.executed_horizon), dtype=jnp.bool_)
        query_mask = jnp.ones((2, 3), dtype=jnp.bool_)
        reset_mask = jnp.zeros((2, 3), dtype=jnp.bool_)
        tokens, next_state = model._scan_memory(
            prefix_inputs, executed_actions, executed_mask, query_mask, reset_mask, state
        )
        assert tokens.shape == (2, 3, config.memory.d_model)
        assert jax.tree.structure(next_state) == jax.tree.structure(state)


def test_action_memory_full_skips_progress_and_conditions_all_solver_steps(monkeypatch):
    config = _dummy_config(action_horizon=4, executed_horizon=2, decoder_mode="action_memory_full")
    model = config.create(jax.random.key(0))
    obs = config.fake_obs(batch_size=1)
    state = model.initial_memory_state(batch_size=1)
    noise = jnp.zeros((1, config.action_horizon, config.action_dim), dtype=jnp.float32)
    prefix_mask = jnp.ones((1, 1), dtype=jnp.bool_)
    kv_cache = (
        jnp.zeros((4, 1, 1, 1, 16), dtype=jnp.float32),
        jnp.zeros((4, 1, 1, 1, 16), dtype=jnp.float32),
    )
    memory_token = jnp.ones((1, 1, config.memory.d_model), dtype=jnp.float32)
    calls = {"action_memory": 0}

    def prepare_memory_context(*args, **kwargs):
        del args, kwargs
        return prefix_mask, kv_cache, memory_token, state

    def fail_progress(*args, **kwargs):
        raise AssertionError("Progress Expert must not run for action_memory_full")

    def action_memory_velocity(observation, x_t, timestep, prefix_mask_arg, kv_cache_arg, memory_token_arg):
        del observation, timestep, prefix_mask_arg, kv_cache_arg
        calls["action_memory"] += 1
        np.testing.assert_allclose(memory_token_arg, memory_token)
        return jnp.ones_like(x_t)

    monkeypatch.setattr(model, "_prepare_memory_context", prepare_memory_context)
    monkeypatch.setattr(model.futuremamba, "progress_expert", fail_progress)
    monkeypatch.setattr(model, "_action_velocity_with_memory_full", action_memory_velocity)

    actions, next_state, diagnostics = model.sample_actions_with_memory(
        jax.random.key(1),
        obs,
        state,
        jnp.zeros((1, config.executed_horizon, config.action_dim), dtype=jnp.float32),
        jnp.zeros((1, config.executed_horizon), dtype=jnp.bool_),
        num_steps=3,
        noise=noise,
    )

    np.testing.assert_allclose(actions, -jnp.ones_like(noise), rtol=1e-6, atol=1e-6)
    assert next_state is state
    assert calls["action_memory"] == 3
    assert diagnostics["progress_calls"] == 0
    assert diagnostics["action_calls"] == 3


def test_compute_episode_loss_action_memory_full_trains_full_action_path_and_skips_progress_losses(monkeypatch):
    config = _dummy_config(
        action_horizon=4,
        executed_horizon=2,
        decoder_mode="action_memory_full",
        handoff_loss_weight=3.0,
        boundary_loss_weight=5.0,
    )
    model = config.create(jax.random.key(0))
    calls = {"action_memory": 0}

    def fail_progress(*args, **kwargs):
        raise AssertionError("Progress Expert path must not train action_memory_full")

    def action_memory_velocity(observation, x_t, timestep, prefix_mask, kv_cache, memory_token):
        del observation, prefix_mask, kv_cache
        calls["action_memory"] += 1
        assert timestep.shape == (1,)
        assert memory_token.shape[1] == 1
        return jnp.zeros_like(x_t)

    monkeypatch.setattr(model, "_progress_velocity", fail_progress)
    monkeypatch.setattr(model, "_action_velocity_with_memory_full", action_memory_velocity)

    losses = model.compute_episode_loss(jax.random.key(0), _episode_batch(config))

    assert calls["action_memory"] == 1
    assert set(losses) == {"loss", "flow_loss", "handoff_loss", "handoff_error", "boundary_loss", "boundary_error"}
    np.testing.assert_allclose(losses["loss"], losses["flow_loss"], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(losses["handoff_loss"], 0.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(losses["boundary_loss"], 0.0, rtol=1e-6, atol=1e-6)


def test_action_memory_full_appends_trainable_memory_kv_without_mutating_prefix_cache():
    config = _dummy_config(decoder_mode="action_memory_full")
    model = config.create(jax.random.key(0))
    action_config = _futuremamba._gemma.get_config(config.action_expert_variant)
    prefix_k = jnp.zeros(
        (action_config.depth, 2, 3, action_config.num_kv_heads, action_config.head_dim), dtype=jnp.float32
    )
    prefix_v = jnp.zeros_like(prefix_k)
    memory_token = jnp.ones((2, 1, action_config.width), dtype=jnp.float32)

    augmented_k, augmented_v = model._augment_kv_cache_with_memory((prefix_k, prefix_v), memory_token)

    assert augmented_k.shape == (action_config.depth, 2, 4, action_config.num_kv_heads, action_config.head_dim)
    assert augmented_v.shape == augmented_k.shape
    np.testing.assert_allclose(augmented_k[:, :, :3], prefix_k, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(augmented_v[:, :, :3], prefix_v, rtol=1e-6, atol=1e-6)
    assert not np.allclose(np.asarray(augmented_k[:, :, 3]), 0.0)


def test_bptt_window_stops_gradient_at_window_boundary_but_uses_real_burn_in(monkeypatch):
    config = _dummy_config(bptt_window_queries=2)
    model = config.create(jax.random.key(0))

    class AdditiveMemory:
        def initial_state(self, batch_size: int, dtype=jnp.float32):
            return (jnp.zeros((batch_size, config.memory.d_model), dtype=dtype),)

        def step(self, x, state):
            next_value = state[0] + x
            return next_value, (next_value,)

    monkeypatch.setattr(model.futuremamba, "memory", AdditiveMemory())
    monkeypatch.setattr(
        model, "_memory_step_input", lambda prefix_input, executed_actions, executed_action_mask: prefix_input
    )
    prefix_inputs = jnp.arange(4 * config.memory.d_model, dtype=jnp.float32).reshape(1, 4, config.memory.d_model) / 10.0
    executed_actions = jnp.zeros((1, 4, config.executed_horizon, config.action_dim), dtype=jnp.float32)
    executed_mask = jnp.zeros((1, 4, config.executed_horizon), dtype=jnp.bool_)
    query_mask = jnp.ones((1, 4), dtype=jnp.bool_)
    reset_mask = jnp.zeros((1, 4), dtype=jnp.bool_)

    def final_sum(inputs):
        tokens, _ = model._scan_memory(
            inputs,
            executed_actions,
            executed_mask,
            query_mask,
            reset_mask,
            model.initial_memory_state(batch_size=1),
        )
        return jnp.sum(tokens[:, -1])

    tokens, _ = model._scan_memory(
        prefix_inputs,
        executed_actions,
        executed_mask,
        query_mask,
        reset_mask,
        model.initial_memory_state(batch_size=1),
    )
    np.testing.assert_allclose(tokens[:, -1], jnp.sum(prefix_inputs, axis=1), rtol=1e-6, atol=1e-6)

    grad = jax.grad(final_sum)(prefix_inputs)
    np.testing.assert_allclose(grad[:, :2], 0.0, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(grad[:, 2:], 1.0, rtol=1e-6, atol=1e-6)
