from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch
from torch import nn

from openpi import transforms as _transforms
from openpi.models_pytorch.mamba_memory import MemorySnapshot
from openpi.policies import futuremamba_policy as _futuremamba_policy


class _FakeMemoryBackend:
    backend_id = "mamba2"
    state_schema_version = 3
    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype):
        return MemorySnapshot(
            backend_id=self.backend_id,
            state_schema_version=self.state_schema_version,
            batch_size=batch_size,
            layers=((torch.zeros(batch_size, 2, device=device, dtype=dtype),),),
        )


class _FakePlugin(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.memory_backend = _FakeMemoryBackend()


class _FakeStatefulModel(nn.Module):
    def __init__(self, *, action_horizon: int = 4, action_dim: int = 8):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.config = type(
            "Config",
            (),
            {"action_horizon": action_horizon, "action_dim": action_dim},
        )()
        self.futuremamba = _FakePlugin()
        self.calls: list[dict] = []

    def initial_memory_state(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        return MemorySnapshot(
            backend_id="mamba2",
            state_schema_version=3,
            batch_size=batch_size,
            layers=((torch.zeros(batch_size, 2, device=device, dtype=dtype),),),
        )

    def sample_actions_with_memory(
        self,
        observation,
        memory_state,
        num_steps=None,
        handoff_ratio=None,
        noise=None,
    ):
        memory_value = memory_state.layers[0][0]
        self.calls.append(
            {
                "state": observation.state.detach().cpu().clone(),
                "memory_state": memory_value.detach().cpu().clone(),
                "num_steps": num_steps,
                "handoff_ratio": handoff_ratio,
                "noise": None if noise is None else noise.detach().cpu().clone(),
            }
        )
        batch_size = observation.state.shape[0]
        actions = memory_value[:, :1, None].expand(batch_size, self.action_horizon, self.action_dim).clone()
        actions = actions + torch.arange(self.action_dim, dtype=actions.dtype, device=actions.device)
        if noise is not None:
            actions = actions + noise
        next_memory = MemorySnapshot(
            memory_state.backend_id,
            memory_state.state_schema_version,
            memory_state.batch_size,
            ((memory_value + 1.0,),),
        )
        return actions, next_memory, {"handoff_steps": 2}


class _CaptureObservationTransform:
    def __call__(self, data):
        data = dict(data)
        data["state"] = np.asarray(data["state"], dtype=np.float32) + 10.0
        data["image"] = {"base_0_rgb": np.zeros((2, 2, 3), dtype=np.uint8)}
        data["image_mask"] = {"base_0_rgb": np.True_}
        return data


class _ScaleOutputTransform:
    def __call__(self, data):
        data = dict(data)
        data["actions"] = np.asarray(data["actions"]) * 2.0
        return data


def _policy(model=None, **kwargs):
    model = model or _FakeStatefulModel()
    return _futuremamba_policy.FutureMambaPolicy(
        model,
        transforms=[_CaptureObservationTransform()],
        output_transforms=[_ScaleOutputTransform()],
        sample_kwargs={"num_steps": 7, "handoff_ratio": 0.5},
        metadata={"name": "fake"},
        pytorch_device="cpu",
        **kwargs,
    )


def _obs():
    return {"state": np.array([1.0, 2.0], dtype=np.float32)}


def _assert_memory_equal(left: MemorySnapshot, right: MemorySnapshot):
    assert left.backend_id == right.backend_id
    assert left.state_schema_version == right.state_schema_version
    assert left.batch_size == right.batch_size
    for left_layer, right_layer in zip(left.layers, right.layers, strict=True):
        for left_tensor, right_tensor in zip(left_layer, right_layer, strict=True):
            torch.testing.assert_close(left_tensor, right_tensor)

@pytest.mark.parametrize(
    ("requested", "actual", "matches"),
    [
        ("cuda", "cuda:0", True),
        ("cuda:1", "cuda:1", True),
        ("cuda:1", "cuda:0", False),
        ("cpu", "cpu", True),
        ("cuda", "cpu", False),
    ],
)
def test_device_match_resolves_unspecified_accelerator_index(requested, actual, matches):
    assert _futuremamba_policy._devices_match(torch.device(actual), torch.device(requested)) is matches

def test_tree_to_torch_batch_preserves_none_leaves():
    result = _futuremamba_policy._tree_to_torch_batch(
        {"state": np.array([1.0, 2.0], dtype=np.float32), "optional": None},
        torch.device("cpu"),
    )

    assert result["optional"] is None



def test_policy_state_and_infer_are_vlm_only():
    policy = _policy()

    output = policy.infer(_obs())
    snapshot = policy.snapshot_state()

    assert output["actions"].shape == (4, 8)
    assert len(policy._model.calls) == 1
    np.testing.assert_allclose(policy._model.calls[0]["state"].numpy(), [[11.0, 12.0]])
    assert policy._model.calls[0]["num_steps"] == 7
    assert policy._model.calls[0]["handoff_ratio"] == 0.5
    assert output["handoff_step"] == 2
    assert output["memory_state_bytes"] > 0
    assert "policy_timing" in output
    assert not hasattr(snapshot, "executed_actions")
    assert not hasattr(snapshot, "executed_action_mask")
    assert snapshot.query_count == 1




def test_snapshot_is_detached_cpu_clone_with_backend_identity():
    policy = _policy()
    policy.infer(_obs())
    snapshot = policy.snapshot_state()
    snapshot_clone = policy.snapshot_state()

    policy.infer(_obs())

    assert snapshot.backend_id == "mamba2"
    assert snapshot.state_schema_version == 3
    assert snapshot.memory.layers[0][0].device.type == "cpu"
    assert snapshot.memory.layers[0][0].grad_fn is None
    _assert_memory_equal(snapshot.memory, snapshot_clone.memory)
    assert snapshot.memory.layers[0][0].data_ptr() != snapshot_clone.memory.layers[0][0].data_ptr()


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda state: dataclasses.replace(state, backend_id="mamba3_siso"), "backend"),
        (lambda state: dataclasses.replace(state, state_schema_version=99), "schema"),
        (
            lambda state: dataclasses.replace(
                state, memory=dataclasses.replace(state.memory, batch_size=2),
            ),
            "batch",
        ),
        (
            lambda state: dataclasses.replace(
                state, memory=dataclasses.replace(state.memory, layers=state.memory.layers + state.memory.layers),
            ),
            "layer|shape",
        ),
        (
            lambda state: dataclasses.replace(
                state,
                memory=dataclasses.replace(
                    state.memory,
                    layers=((state.memory.layers[0][0][:, :1],),),
                ),
            ),
            "shape",
        ),
        (
            lambda state: dataclasses.replace(
                state,
                memory=dataclasses.replace(
                    state.memory,
                    layers=((state.memory.layers[0][0].to(torch.float64),),),
                ),
            ),
            "dtype",
        ),
    ],
)
def test_restore_rejects_backend_schema_batch_layers_shape_and_dtype(mutate, match):
    policy = _policy()
    snapshot = mutate(policy.snapshot_state())

    with pytest.raises(ValueError, match=match):
        policy.restore_state(snapshot)


def test_restore_clones_storage_and_replays_with_fixed_noise():
    noise = np.full((4, 8), 0.25, dtype=np.float32)
    policy = _policy()
    policy.infer(_obs())
    snapshot = policy.snapshot_state()
    expected = policy.infer(_obs(), noise=noise)

    policy.restore_state(snapshot)
    restored = policy.snapshot_state()
    actual = policy.infer(_obs(), noise=noise)

    assert restored.memory.layers[0][0].data_ptr() != snapshot.memory.layers[0][0].data_ptr()
    np.testing.assert_allclose(actual["actions"], expected["actions"])


def test_reset_clears_memory_and_updates_episode_count():
    policy = _policy()
    initial = policy.snapshot_state()
    policy.infer(_obs())

    policy.reset()
    reset = policy.snapshot_state()

    assert initial.episode_count == 0
    assert reset.episode_count == 1
    assert reset.query_count == 0
    assert reset.memory.layers[0][0].count_nonzero().item() == 0


def test_fork_shares_model_but_owns_independent_state_and_client_id():
    policy = _policy()
    policy.infer(_obs())

    fork = policy.fork()
    fork_initial = fork.snapshot_state()
    fork.infer(_obs())
    parent_after = policy.snapshot_state()

    assert fork is not policy
    assert fork._model is policy._model
    assert fork_initial.query_count == 0
    assert fork_initial.client_id != parent_after.client_id
    assert parent_after.query_count == 1
    assert fork.snapshot_state().query_count == 1


def test_add_buffer_records_boundary_without_advancing_memory_or_query():
    policy = _policy()
    before = policy.snapshot_state()
    payload = {
        "images": np.zeros((3, 1, 2, 2, 3), dtype=np.uint8),
        "state": np.arange(6, dtype=np.float32).reshape(3, 2),
        "add_buffer": True,
        "exec_start_idx": 1,
    }

    diagnostic = policy.add_buffer(payload)
    after = policy.snapshot_state()

    _assert_memory_equal(after.memory, before.memory)
    assert after.query_count == before.query_count
    assert diagnostic["exec_start_idx"] == 1
    assert diagnostic["num_frames"] == 3
    assert len(diagnostic["buffer_checksum"]) == 64
    assert policy.buffer_diagnostics == (diagnostic,)


def test_reset_clears_buffer_diagnostics():
    policy = _policy()
    policy.add_buffer({"images": np.zeros((1, 1, 2, 2, 3), dtype=np.uint8), "state": np.zeros((1, 2)), "add_buffer": True, "exec_start_idx": 0})

    policy.reset()

    assert policy.buffer_diagnostics == ()




def test_infer_requires_handoff_diagnostic():
    model = _FakeStatefulModel()

    def missing(*args, **kwargs):
        actions, state, _ = model.sample_actions_with_memory(*args, **kwargs)
        return actions, state, {}

    policy = _policy(model=model, _sample_actions_with_memory=missing)
    with pytest.raises(ValueError, match="handoff_step"):
        policy.infer(_obs())
