from __future__ import annotations

import numpy as np
import torch
from torch import nn

from openpi.models_pytorch.mamba_memory import MemorySnapshot
from openpi.policies.futuremamba_policy import FutureMambaPolicy


class Backend:
    backend_id = "fake"
    state_schema_version = 1

    def initial_state(self, batch_size, *, device, dtype):
        return MemorySnapshot("fake", 1, batch_size, ((torch.zeros(batch_size, 2, device=device, dtype=dtype),),))


class Plugin(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))
        self.memory_backend = Backend()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.futuremamba = Plugin()
        self.config = type("Config", (), {"action_horizon": 3, "action_dim": 2})()
        self.calls = []

    def initial_memory_state(self, batch_size, device, dtype):
        return self.futuremamba.memory_backend.initial_state(batch_size, device=device, dtype=dtype)

    def sample_actions_with_memory(self, memory_state, executed_actions, executed_action_mask, **kwargs):
        del kwargs
        self.calls.append((executed_actions.detach().clone(), executed_action_mask.detach().clone()))
        updated = int(executed_action_mask.any())
        next_memory = MemorySnapshot(
            "fake", 1, 1, ((memory_state.layers[0][0] + updated,),)
        )
        return torch.zeros(1, 3, 2), next_memory, {
            "handoff_steps": 3,
            "memory_updates": updated,
        }


def test_first_query_is_empty_and_second_query_uses_executed_chunk():
    model = Model()
    policy = FutureMambaPolicy(model, pytorch_device="cpu")

    first = policy.infer({"executed_actions": np.zeros((0, 2), dtype=np.float32)})
    second = policy.infer({"executed_actions": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)})

    assert first["memory_updates"] == 0
    assert second["memory_updates"] == 1
    assert model.calls[0][1].tolist() == [[False, False, False]]
    assert model.calls[1][1].tolist() == [[True, True, False]]
    torch.testing.assert_close(model.calls[1][0][0, :2], torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_explicit_padding_mask_hides_padding_values():
    model = Model()
    policy = FutureMambaPolicy(model, pytorch_device="cpu")

    policy.infer({
        "executed_actions": np.asarray([[1.0, 2.0], [999.0, 999.0], [999.0, 999.0]], dtype=np.float32),
        "executed_action_mask": np.asarray([True, False, False]),
    })

    assert model.calls[0][1].tolist() == [[True, False, False]]
