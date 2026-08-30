from __future__ import annotations

import dataclasses

import torch
from torch import nn

from openpi.models_pytorch.futuremamba import FutureMambaPluginPytorch, FutureMambaPytorch
from openpi.models_pytorch.mamba_memory import MemorySnapshot


@dataclasses.dataclass(frozen=True)
class TinyMemoryConfig:
    d_model: int = 4


@dataclasses.dataclass(frozen=True)
class TinyConfig:
    action_horizon: int = 3
    action_history_chunk_size: int = 20
    action_dim: int = 2
    action_expert_variant: str = "dummy"
    dtype: str = "float32"
    memory: TinyMemoryConfig = dataclasses.field(default_factory=TinyMemoryConfig)
    memory_backend: str = "gru"
    num_denoise_steps: int = 10
    handoff_ratio: float = 0.4
    frozen_prefix_microbatch_size: int = 2

    @property
    def resolved_progress_layer_indices(self):
        return (0,)


class RecordingMemory(nn.Module):
    backend_id = "recording"
    state_schema_version = 1

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def initial_state(self, batch_size, *, device, dtype):
        return MemorySnapshot(
            self.backend_id,
            self.state_schema_version,
            batch_size,
            ((torch.zeros(batch_size, 4, device=device, dtype=dtype),),),
        )

    def step(self, x, state):
        previous = state.layers[0][0]
        self.calls.append((x.detach().clone(), previous.detach().clone()))
        output = previous + x
        return output, MemorySnapshot(
            state.backend_id,
            state.state_schema_version,
            state.batch_size,
            ((output,),),
        )


class RecordingProgress(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.0))
        self.calls = []

    def forward(self, prefix_cache, prefix_mask, memory_token, noisy_actions, timestep):
        self.calls.append(
            {
                "prefix_lengths": prefix_cache.valid_lengths.detach().clone(),
                "prefix_mask": prefix_mask.detach().clone(),
                "memory_token": memory_token.detach().clone(),
                "x": noisy_actions.detach().clone(),
                "time": timestep.detach().clone(),
            }
        )
        return torch.zeros_like(noisy_actions) + self.bias


@dataclasses.dataclass
class Observation:
    state: torch.Tensor


class FrozenBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.0))
        self.action_calls = []

    def sample_noise(self, shape, device):
        return torch.zeros(shape, device=device)

    def _preprocess_observation(self, observation, *, train=False):
        del train
        return [], [], None, None, observation.state

    def extract_prefix_context(self, observation, *, train=False):
        del train
        from openpi.models_pytorch.pi0_pytorch import FrozenPrefix

        batch = observation.state.shape[0]
        mask = torch.ones(batch, 2, dtype=torch.bool, device=observation.state.device)
        cache = ((
            torch.zeros(batch, 1, 2, 4, device=observation.state.device),
            torch.zeros(batch, 1, 2, 4, device=observation.state.device),
        ),)
        return FrozenPrefix(
            hidden=torch.zeros(batch, 2, 4, device=observation.state.device),
            pad_mask=mask,
            kv_cache=cache,
        )

    def action_expert_velocity(self, state, prefix_mask, prefix_cache, x, timestep):
        self.action_calls.append(
            (state.detach().clone(), prefix_mask.detach().clone(), x.detach().clone(), timestep.detach().clone())
        )
        return torch.zeros_like(x)


def make_model():
    config = TinyConfig()
    memory = RecordingMemory()
    progress = RecordingProgress()
    plugin = FutureMambaPluginPytorch(config, memory_backend=memory, progress_expert=progress)
    with torch.no_grad():
        for parameter in plugin.action_chunk_projection.parameters():
            parameter.fill_(0.01)
        plugin.empty_history.zero_()
        plugin.memory_token_projection.weight.fill_(0.1)
        plugin.memory_token_projection.bias.zero_()
    base = FrozenBase()
    return FutureMambaPytorch(config, base=base, futuremamba=plugin), memory, progress, base


def append(model, history, values):
    actions = torch.as_tensor(values, dtype=torch.float32).reshape(1, -1, 2)
    mask = torch.ones(actions.shape[:2], dtype=torch.bool)
    return model.futuremamba.advance_history(history, actions, mask)


def test_frame_21_uses_committed_twenty_then_one_masked_partial_chunk():
    model, memory, _, _ = make_model()
    history = model.initial_history_state(1, torch.device("cpu"), torch.float32)
    values = torch.arange(42, dtype=torch.float32).reshape(21, 2)

    _, history, first = append(model, history, values[:20])
    calls_after_commit = len(memory.calls)
    token, history, second = append(model, history, values[20:21])

    assert first["memory_commits"] == 1
    assert calls_after_commit == 1
    assert second["memory_commits"] == 0
    assert len(memory.calls) == 2
    assert history.committed_chunks.tolist() == [1]
    assert history.pending_mask.sum().item() == 1
    assert history.pending_actions[0, 0].tolist() == values[20].tolist()
    assert token.shape == (1, 1, 64)


def test_partial_padding_values_are_ignored_but_mask_is_encoded():
    model, _, _, _ = make_model()
    first = torch.zeros(1, 20, 2)
    second = first.clone()
    second[:, 1:] = 999.0
    mask = torch.zeros(1, 20, dtype=torch.bool)
    mask[:, 0] = True
    first[:, 0] = torch.tensor([1.0, 2.0])
    second[:, 0] = first[:, 0]

    encoded_first = model.futuremamba.encode_action_chunk(first, mask)
    encoded_second = model.futuremamba.encode_action_chunk(second, mask)

    torch.testing.assert_close(encoded_first, encoded_second)


def test_empty_history_is_true_noop():
    model, memory, _, _ = make_model()
    history = model.initial_history_state(1, torch.device("cpu"), torch.float32)
    actions = torch.zeros(1, 0, 2)
    mask = torch.zeros(1, 0, dtype=torch.bool)

    _, next_history, diagnostics = model.futuremamba.advance_history(history, actions, mask)

    assert not memory.calls
    assert diagnostics == {"memory_commits": 0, "history_actions": 0, "pending_actions": 0}
    assert next_history.memory == history.memory


def test_handoff_point_four_runs_pe_then_ae_on_same_latent():
    model, _, progress, base = make_model()
    history = model.initial_history_state(1, torch.device("cpu"), torch.float32)
    observation = Observation(state=torch.tensor([[3.0, 4.0]]))
    noise = torch.ones(1, 3, 2)
    actions = torch.zeros(1, 0, 2)
    mask = torch.zeros(1, 0, dtype=torch.bool)

    output, _, diagnostics = model.sample_actions_with_memory(
        observation,
        history,
        actions,
        mask,
        noise=noise,
        num_steps=10,
        handoff_ratio=0.4,
    )

    torch.testing.assert_close(output, noise)
    assert diagnostics["progress_calls"] == 4
    assert diagnostics["action_calls"] == 6
    assert diagnostics["handoff_steps"] == 4
    assert len(progress.calls) == 4
    assert len(base.action_calls) == 6
    torch.testing.assert_close(base.action_calls[0][2], progress.calls[-1]["x"])
    assert progress.calls[0]["prefix_lengths"].tolist() == [2]
