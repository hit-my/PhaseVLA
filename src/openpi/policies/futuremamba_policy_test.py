import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms as _transforms
from openpi.policies import futuremamba_policy as _futuremamba_policy


@dataclasses.dataclass(frozen=True)
class _MemoryState:
    value: jax.Array


class _FakeStatefulModel:
    def __init__(self, *, action_horizon: int = 4, action_dim: int = 8):
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.calls = []

    def initial_memory_state(self, batch_size: int):
        return _MemoryState(value=jnp.zeros((batch_size, 2), dtype=jnp.float32))

    def sample_actions_with_memory(
        self,
        rng,
        observation,
        memory_state,
        executed_actions,
        executed_action_mask,
        num_steps=None,
        handoff_ratio=None,
        noise=None,
    ):
        self.calls.append(
            {
                "rng": jax.random.key_data(rng),
                "state": np.asarray(observation.state),
                "memory_state": np.asarray(memory_state.value),
                "executed_actions": np.asarray(executed_actions),
                "executed_action_mask": np.asarray(executed_action_mask),
                "num_steps": num_steps,
                "handoff_ratio": handoff_ratio,
                "noise": None if noise is None else np.asarray(noise),
            }
        )
        batch_size = observation.state.shape[0]
        base = memory_state.value[:, :1] + jnp.sum(executed_actions, axis=(1, 2))[:, None]
        actions = jnp.broadcast_to(base[:, None, :], (batch_size, self.action_horizon, self.action_dim))
        actions = actions + jnp.arange(self.action_dim, dtype=jnp.float32)
        if noise is not None:
            actions = actions + noise
        next_state = _MemoryState(value=memory_state.value + 1.0)
        diagnostics = {"handoff_step": jnp.asarray([2], dtype=jnp.int32)}
        return actions, next_state, diagnostics


class _CaptureObservationTransform:
    def __call__(self, data):
        data = dict(data)
        data["state"] = data["state"] + 10.0
        data["image"] = {"base_0_rgb": np.zeros((2, 2, 3), dtype=np.uint8)}
        data["image_mask"] = {"base_0_rgb": np.True_}
        return data


class _ScaleOutputTransform:
    def __call__(self, data):
        data = dict(data)
        data["actions"] = np.asarray(data["actions"]) * 2.0
        return data


def _policy(**kwargs):
    return _futuremamba_policy.FutureMambaPolicy(
        _FakeStatefulModel(),
        transforms=[_CaptureObservationTransform(), _transforms.PadExecutedActions(executed_horizon=3, action_dim=8)],
        output_transforms=[_ScaleOutputTransform()],
        sample_kwargs={"num_steps": 7, "handoff_ratio": 0.5},
        rng=jax.random.key(11),
        metadata={"name": "fake"},
        **kwargs,
    )


def _obs(executed_actions=None):
    if executed_actions is None:
        executed_actions = np.zeros((0, 7), dtype=np.float32)
    return {
        "state": np.array([1.0, 2.0], dtype=np.float32),
        "executed_actions": np.asarray(executed_actions, dtype=np.float32),
    }


def test_infer_updates_memory_and_passes_transformed_executed_actions():
    policy = _policy()
    first = policy.infer(_obs([[1, 2, 3, 4, 5, 6, 7]]))
    second = policy.infer(_obs([[0, 0, 0, 0, 0, 0, 1]]))

    assert first["actions"].shape == (4, 8)
    np.testing.assert_allclose(policy._model.calls[1]["memory_state"], [[1.0, 1.0]])
    np.testing.assert_allclose(policy._model.calls[0]["state"], [[11.0, 12.0]])
    np.testing.assert_allclose(policy._model.calls[0]["executed_actions"][0, 0, :7], [1, 2, 3, 4, 5, 6, 7])
    np.testing.assert_allclose(policy._model.calls[0]["executed_actions"][0, 0, 7:], [0])
    np.testing.assert_array_equal(policy._model.calls[0]["executed_action_mask"], [[True, False, False]])
    assert policy._model.calls[0]["num_steps"] == 7
    assert policy._model.calls[0]["handoff_ratio"] == 0.5
    assert first["handoff_step"] == 2
    assert first["memory_state_bytes"] > 0
    assert "policy_timing" in first


def test_reset_restores_initial_memory_state():
    policy = _policy()
    policy.infer(_obs([[1, 0, 0, 0, 0, 0, 0]]))
    assert not np.allclose(np.asarray(policy.snapshot_state().value), 0.0)

    policy.reset()

    np.testing.assert_allclose(np.asarray(policy.snapshot_state().value), 0.0)


def test_infer_rejects_executed_action_prefix_longer_than_horizon():
    policy = _policy()

    with pytest.raises(ValueError, match="executed_actions.*3"):
        policy.infer(_obs(np.ones((4, 7), dtype=np.float32)))


def test_snapshot_restore_round_trip_replays_state_with_fixed_noise():
    noise = np.full((4, 8), 0.25, dtype=np.float32)
    policy = _policy()
    policy.infer(_obs([[1, 0, 0, 0, 0, 0, 0]]))
    snapshot = policy.snapshot_state()

    expected = policy.infer(_obs([[0, 1, 0, 0, 0, 0, 0]]), noise=noise)
    policy.restore_state(snapshot)
    actual = policy.infer(_obs([[0, 1, 0, 0, 0, 0, 0]]), noise=noise)

    np.testing.assert_allclose(actual["actions"], expected["actions"])


def test_fork_shares_model_and_transforms_but_has_independent_zero_state_and_rng():
    policy = _policy()
    policy.infer(_obs([[1, 0, 0, 0, 0, 0, 0]]))

    fork = policy.fork()

    assert fork is not policy
    assert fork._model is policy._model
    assert fork.metadata is policy.metadata
    np.testing.assert_allclose(np.asarray(fork.snapshot_state().value), 0.0)
    call_count = len(fork._model.calls)
    fork.infer(_obs())
    policy.infer(_obs())
    baseline = _policy()
    baseline.infer(_obs())
    fork_rng = fork._model.calls[call_count]["rng"]
    policy_rng = policy._model.calls[call_count + 1]["rng"]
    baseline_rng = baseline._model.calls[-1]["rng"]
    assert not np.array_equal(np.asarray(fork_rng), np.asarray(policy_rng))
    assert not np.array_equal(np.asarray(fork_rng), np.asarray(baseline_rng))
