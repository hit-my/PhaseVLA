import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
from openpi.models.pi0 import Pi0, make_attn_mask

import openpi.models.pi0_config as _pi0_config


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)



def test_sample_actions_matches_reference_path_with_helpers():
    config = _pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(jax.random.key(0))
    observation = config.fake_obs(batch_size=2)
    noise = jax.random.normal(jax.random.key(1), (2, config.action_horizon, config.action_dim))

    actual = model.sample_actions(jax.random.key(2), observation, num_steps=3, noise=noise)

    preprocessed = _model.preprocess_observation(None, observation, train=False)
    dt = -1.0 / 3
    prefix_out, prefix_mask, _prefix_ar_mask, kv_cache = model.encode_prefix(preprocessed)
    def step(carry):
        x_t, time = carry
        v_t = model.action_velocity(preprocessed, x_t, jnp.broadcast_to(time, 2), prefix_mask, kv_cache)
        return x_t + dt * v_t, time + dt

    def cond(carry):
        _x_t, time = carry
        return time >= -dt / 2

    x_t, _ = jax.lax.while_loop(cond, step, (noise, 1.0))

    assert prefix_out.shape[:2] == prefix_mask.shape
    np.testing.assert_allclose(actual, x_t, rtol=1e-6, atol=1e-6)


def test_sample_actions_encodes_prefix_once(monkeypatch):
    config = _pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(jax.random.key(0))
    observation = config.fake_obs(batch_size=1)
    calls = 0
    original = Pi0.encode_prefix

    def wrapped(self, obs):
        nonlocal calls
        calls += 1
        return original(self, obs)

    monkeypatch.setattr(Pi0, "encode_prefix", wrapped)

    model.sample_actions(jax.random.key(1), observation, num_steps=3)

    assert calls == 1


def test_action_velocity_matches_direct_suffix_forward():
    config = _pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy")
    model = config.create(jax.random.key(0))
    observation = _model.preprocess_observation(None, config.fake_obs(batch_size=2), train=False)
    x_t = jax.random.normal(jax.random.key(1), (2, config.action_horizon, config.action_dim))
    timestep = jnp.array([0.25, 0.75], dtype=jnp.float32)

    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    prefix_positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=prefix_positions)

    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(observation, x_t, timestep)
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = jnp.broadcast_to(prefix_mask[:, None, :], (2, suffix_tokens.shape[1], prefix_tokens.shape[1]))
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    suffix_positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=suffix_positions,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    expected = model.action_out_proj(suffix_out[:, -config.action_horizon :])

    assert prefix_out is None
    actual = model.action_velocity(observation, x_t, timestep, prefix_mask, kv_cache)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)