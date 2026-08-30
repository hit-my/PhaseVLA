from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from openpi.models_pytorch.futuremamba import FutureMambaPluginPytorch, FutureMambaPytorch
from openpi.models_pytorch.mamba_memory import MemorySnapshot
from openpi.training.episode_data_loader import TorchEpisodeBatch


@dataclasses.dataclass(frozen=True)
class TinyMemoryConfig:
    d_model: int = 4


@dataclasses.dataclass(frozen=True)
class TinyConfig:
    action_horizon: int = 3
    action_dim: int = 2
    action_expert_variant: str = "dummy"
    dtype: str = "float32"
    memory: TinyMemoryConfig = dataclasses.field(default_factory=TinyMemoryConfig)
    memory_backend: str = "gru"
    num_denoise_steps: int = 2


class RecordingMemory(nn.Module):
    backend_id = "recording"
    state_schema_version = 2

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def initial_state(self, batch_size, *, device, dtype):
        return MemorySnapshot(self.backend_id, self.state_schema_version, batch_size, ((torch.zeros(batch_size, 4, device=device, dtype=dtype),),))

    def step(self, x, state):
        self.calls.append(x.detach().clone())
        value = state.layers[0][0] + 1
        return x * self.scale, MemorySnapshot(self.backend_id, self.state_schema_version, state.batch_size, ((value,),))


class RecordingProgress(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def forward(self, memory_token, noisy_actions, timestep):
        self.calls.append((memory_token.detach().clone(), noisy_actions.detach().clone(), timestep.detach().clone()))
        return torch.zeros_like(noisy_actions) + self.bias


def make_model():
    config = TinyConfig()
    backend = RecordingMemory()
    progress = RecordingProgress()
    plugin = FutureMambaPluginPytorch(config, memory_backend=backend, progress_expert=progress)
    return FutureMambaPytorch(config, futuremamba=plugin), backend, progress


def test_empty_history_does_not_update_mamba():
    model, backend, _ = make_model()
    state = model.initial_memory_state(1, torch.device("cpu"), torch.float32)
    actions = torch.zeros(1, 3, 2)
    mask = torch.zeros(1, 3, dtype=torch.bool)

    _, next_state, diagnostics = model.sample_actions_with_memory(
        state, actions, mask, noise=torch.ones(1, 3, 2), num_steps=1
    )

    assert backend.calls == []
    assert next_state == state
    assert diagnostics["memory_updates"] == 0


def test_padding_values_are_ignored_but_mask_is_encoded():
    model, backend, _ = make_model()
    state = model.initial_memory_state(1, torch.device("cpu"), torch.float32)
    mask = torch.tensor([[True, False, False]])
    first = torch.tensor([[[1.0, 2.0], [999.0, 999.0], [-999.0, -999.0]]])
    second = torch.tensor([[[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]])

    model.futuremamba.update_from_executed_chunk(first, mask, state)
    encoded_first = backend.calls[-1]
    model.futuremamba.update_from_executed_chunk(second, mask, state)
    encoded_second = backend.calls[-1]

    torch.testing.assert_close(encoded_first, encoded_second)


def test_training_replays_prior_chunks_before_each_query():
    from openpi.models import model as _model

    model, backend, progress = make_model()
    actions = torch.zeros(1, 3, 3, 2)
    executed = torch.zeros_like(actions)
    executed[:, 1, 0] = 1.0
    executed[:, 2, :2] = 2.0
    executed_mask = torch.tensor([[[False, False, False], [True, False, False], [True, True, False]]])
    observation = _model.Observation(
        images={},
        image_masks={},
        state=torch.zeros(1, 3, 1),
    )
    batch = TorchEpisodeBatch(
        observation=observation,
        actions=actions,
        action_mask=torch.ones(1, 3, 3, dtype=torch.bool),
        executed_actions=executed,
        executed_action_mask=executed_mask,
        query_mask=torch.ones(1, 3, dtype=torch.bool),
        reset_mask=torch.tensor([[True, False, False]]),
        episode_index=torch.tensor([0]),
        train_query_mask=torch.ones(1, 3, dtype=torch.bool),
    )

    result = model.compute_episode_loss(batch, noise=torch.ones_like(actions), time=torch.full((1, 3), 0.5))

    assert torch.isfinite(result["loss"])
    assert len(backend.calls) == 2
    assert len(progress.calls) == 3


def test_progress_expert_has_no_prefix_arguments():
    _, _, progress = make_model()
    assert progress.forward.__code__.co_varnames[1:4] == ("memory_token", "noisy_actions", "timestep")
