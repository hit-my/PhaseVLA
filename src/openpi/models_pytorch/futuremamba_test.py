from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from openpi.models_pytorch.gemma_pytorch import PrefixKVView
from openpi.models_pytorch.mamba_memory import MemorySnapshot


@dataclasses.dataclass(frozen=True)
class _TinyMemoryConfig:
    d_model: int = 4


@dataclasses.dataclass(frozen=True)
class _TinyConfig:
    action_horizon: int = 3
    action_dim: int = 2
    action_expert_variant: str = "dummy"
    paligemma_variant: str = "dummy"
    dtype: str = "float32"
    vlm_width: int = 4
    memory: _TinyMemoryConfig = dataclasses.field(default_factory=_TinyMemoryConfig)
    memory_backend: str = "gru"
    progress_depth: int = 2
    handoff_ratio: float = 0.2
    num_denoise_steps: int = 10
    pytorch_compile_mode: str | None = None
    frozen_prefix_microbatch_size: int = 2

    @property
    def resolved_progress_layer_indices(self) -> tuple[int, ...]:
        return (0, 1)

@dataclasses.dataclass(frozen=True)
class _WideVLMConfig(_TinyConfig):
    vlm_width: int = 8



@dataclasses.dataclass(frozen=True)
class _TinyLossConfig(_TinyConfig):
    terminal_loss_weight: float = 0.0
    handoff_loss_weight: float = 0.0
    boundary_loss_weight: float = 0.0
    action_expert_gradient_checkpointing: bool = False
    terminal_loss_batch_fraction: float = 1.0
    terminal_loss_queries_per_episode: int | None = None


@dataclasses.dataclass
class _Observation:
    state: torch.Tensor


class _TinyBase(nn.Module):
    def __init__(self, config: _TinyConfig | None = None):
        super().__init__()
        self.config = config or _TinyConfig()
        self.weight = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        self.prefix_calls = 0
        self.denoise_calls = 0
        self.last_denoise_inputs = []
    def _preprocess_observation(self, observation, *, train: bool = False):
        del train
        return [], [], None, None, observation.state + 100.0


    def encode_frozen_prefix(self, observation, *, train: bool = False):
        del train
        self.prefix_calls += 1
        from openpi.models_pytorch.pi0_pytorch import FrozenPrefix

        batch = observation.state.shape[0]
        hidden = torch.tensor(
            [
                [[1.0, 1.0, 1.0, 1.0], [9.0, 9.0, 9.0, 9.0], [2.0, 2.0, 2.0, 2.0]],
                [[4.0, 4.0, 4.0, 4.0], [5.0, 5.0, 5.0, 5.0], [99.0, 99.0, 99.0, 99.0]],
            ],
            requires_grad=True,
        )[:batch]
        pad_mask = torch.tensor([[True, False, True], [True, True, False]])[:batch]
        cache = tuple(
            (
                torch.arange(batch * 1 * 3 * 2, dtype=torch.float32).reshape(batch, 1, 3, 2) + layer_idx * 100,
                torch.arange(batch * 1 * 3 * 2, dtype=torch.float32).reshape(batch, 1, 3, 2) + layer_idx * 100 + 50,
            )
            for layer_idx in range(2)
        )
        return FrozenPrefix(hidden=hidden, pad_mask=pad_mask, kv_cache=cache)

    def last_valid_prefix(self, frozen):
        positions = torch.arange(frozen.pad_mask.shape[1]).expand_as(frozen.pad_mask).masked_fill(~frozen.pad_mask, -1)
        index = positions.max(dim=-1).values
        return frozen.hidden[torch.arange(index.numel()), index]

    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        self.denoise_calls += 1
        self.last_denoise_inputs.append((state.detach().clone(), prefix_pad_masks.detach().clone(), past_key_values, x_t.detach().clone(), timestep.detach().clone()))
        return x_t + timestep[:, None, None] + state.sum(dim=-1)[:, None, None] * 0.01

    def sample_noise(self, shape, device):
        return torch.zeros(shape, dtype=torch.float32, device=device)

    def sample_actions(self, device, observation, noise=None, num_steps=10):
        if noise is None:
            noise = self.sample_noise((observation.state.shape[0], self.config.action_horizon, self.config.action_dim), device)
        *_, state = self._preprocess_observation(observation, train=False)
        frozen = self.encode_frozen_prefix(observation, train=False)
        x_t = noise
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            x_t = x_t + dt * self.denoise_step(state, frozen.pad_mask, frozen.kv_cache, x_t, time.expand(observation.state.shape[0]))
            time += dt
        return x_t

    def extract_prefix_context(self, observation, *, train: bool = False):
        return self.encode_frozen_prefix(observation, train=train)

    def action_expert_velocity(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        return self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)


class _RecordingMemoryBackend(nn.Module):
    backend_id = "recording"
    state_schema_version = 7

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.calls = []
        self.scale = nn.Parameter(torch.tensor(1.0))

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype):
        return MemorySnapshot(self.backend_id, self.state_schema_version, batch_size, ((torch.zeros(batch_size, 1, device=device, dtype=dtype),),))

    def step(self, x: torch.Tensor, state: MemorySnapshot):
        self.calls.append((x.detach().clone(), state))
        layer = state.layers[0][0] + 1
        return x * self.scale + 0.5, MemorySnapshot(state.backend_id, state.state_schema_version, state.batch_size, ((layer,),))


class _TraceProgress(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.calls = []
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, prefix_cache, prefix_mask, memory_token, noisy_actions, timestep):
        assert isinstance(prefix_cache, PrefixKVView)
        self.calls.append(
            {
                "prefix_cache": prefix_cache,
                "prefix_mask": prefix_mask.detach().clone(),
                "memory_token": memory_token.detach().clone(),
                "x": noisy_actions.detach().clone(),
                "t": timestep.detach().clone(),
            }
        )
        return torch.full_like(noisy_actions, 2.0) + self.bias


class _ZeroProgress(_TraceProgress):
    def forward(self, prefix_cache, prefix_mask, memory_token, noisy_actions, timestep):
        super().forward(prefix_cache, prefix_mask, memory_token, noisy_actions, timestep)
        return torch.zeros_like(noisy_actions)


def _make_plugin(config=None, memory_backend=None, progress=None):
    from openpi.models_pytorch.futuremamba import FutureMambaPluginPytorch

    config = config or _TinyConfig()
    return FutureMambaPluginPytorch(
        config,
        memory_backend=memory_backend or _RecordingMemoryBackend(config.memory),
        progress_expert=progress or _TraceProgress(config),
    )


def _make_model(config=None, base=None, memory_backend=None, progress=None):
    from openpi.models_pytorch.futuremamba import FutureMambaPytorch

    config = config or _TinyConfig()
    return FutureMambaPytorch(
        config,
        base=base or _TinyBase(config),
        futuremamba=_make_plugin(config=config, memory_backend=memory_backend, progress=progress),
    )


class _ZeroActionBase(_TinyBase):
    def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
        self.denoise_calls += 1
        self.last_denoise_inputs.append((state.detach().clone(), prefix_pad_masks.detach().clone(), past_key_values, x_t.detach().clone(), timestep.detach().clone()))
        return torch.zeros_like(x_t)


class _XScaleProgress(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def forward(self, prefix_cache, prefix_mask, memory_token, noisy_actions, timestep):
        assert isinstance(prefix_cache, PrefixKVView)
        self.calls.append({"x": noisy_actions.detach().clone(), "t": timestep.detach().clone()})
        return noisy_actions * self.scale


class _CarryMemoryBackend(nn.Module):
    backend_id = "carry"
    state_schema_version = 1

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.grad_enabled = []

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype):
        return MemorySnapshot(self.backend_id, self.state_schema_version, batch_size, ((torch.zeros(batch_size, self.config.d_model, device=device, dtype=dtype),),))

    def step(self, x: torch.Tensor, state: MemorySnapshot):
        self.grad_enabled.append(torch.is_grad_enabled())
        previous = state.layers[0][0]
        output = x + previous
        next_state = MemorySnapshot(state.backend_id, state.state_schema_version, state.batch_size, ((previous + x * self.scale,),))
        return output, next_state


class _SecondQueryTokenProgress(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, prefix_cache, prefix_mask, memory_token, noisy_actions, timestep):
        del prefix_cache, prefix_mask, timestep
        self.calls += 1
        if self.calls == 1:
            return torch.zeros_like(noisy_actions)
        return memory_token[:, :, : noisy_actions.shape[-1]].expand_as(noisy_actions)


def _torch_episode_batch(actions, *, action_mask=None, executed_actions=None, executed_action_mask=None, query_mask=None, reset_mask=None, train_query_mask=None, state=None):
    from openpi.training import episode_data_loader as _episode_loader

    batch_size, num_queries, action_horizon, action_dim = actions.shape
    if action_mask is None:
        action_mask = torch.ones(batch_size, num_queries, action_horizon, dtype=torch.bool, device=actions.device)
    if executed_actions is None:
        executed_actions = torch.zeros(batch_size, num_queries, 3, action_dim, dtype=actions.dtype, device=actions.device)
    if executed_action_mask is None:
        executed_action_mask = torch.ones(executed_actions.shape[:3], dtype=torch.bool, device=actions.device)
    if query_mask is None:
        query_mask = torch.ones(batch_size, num_queries, dtype=torch.bool, device=actions.device)
    if reset_mask is None:
        reset_mask = torch.zeros(batch_size, num_queries, dtype=torch.bool, device=actions.device)
        reset_mask[:, 0] = True
    if state is None:
        state = torch.arange(batch_size * num_queries * 2, dtype=actions.dtype, device=actions.device).reshape(batch_size, num_queries, 2)
    return _episode_loader.TorchEpisodeBatch(
        observation=_Observation(state=state),
        actions=actions,
        action_mask=action_mask,
        executed_actions=executed_actions,
        executed_action_mask=executed_action_mask,
        query_mask=query_mask,
        reset_mask=reset_mask,
        episode_index=torch.arange(batch_size, dtype=torch.int64, device=actions.device),
        train_query_mask=train_query_mask,
    )


def test_constructor_does_not_eagerly_hash_base_but_explicit_checksum_works(monkeypatch):
    from openpi.models_pytorch.futuremamba import FutureMambaPytorch

    def fail_if_called(self):
        del self
        raise AssertionError("constructor must not eagerly hash the base model")

    monkeypatch.setattr(FutureMambaPytorch, "base_checksum", fail_if_called)
    config = _TinyConfig()
    model = FutureMambaPytorch(config, base=_TinyBase(config), futuremamba=_make_plugin())

    monkeypatch.undo()
    checksum_before = model.base_checksum()
    model.freeze_base()
    assert model.base_checksum() == checksum_before

def test_base_checksum_supports_bfloat16_state_dict():
    config = _TinyConfig(dtype="bfloat16")
    base = _TinyBase(config).to(dtype=torch.bfloat16)
    model = _make_model(config=config, base=base)

    checksum_before = model.base_checksum()
    model.freeze_base()

    assert len(checksum_before) == 64
    assert model.base_checksum() == checksum_before



def test_wrapper_freezes_base_permanently_and_only_futuremamba_trainable():
    model = _make_model()
    checksum_before = model.base_checksum()

    assert list(dict(model.named_children())) == ["base", "futuremamba"]
    assert model.base.training is False
    assert all(parameter.requires_grad is False for parameter in model.base.parameters())
    assert all(name.startswith("futuremamba.") for name, parameter in model.named_parameters() if parameter.requires_grad)

    model.train(True)

    assert model.training is True
    assert model.futuremamba.training is True
    assert model.base.training is False
    assert all(parameter.requires_grad is False for parameter in model.base.parameters())
    assert model.base_checksum() == checksum_before


def test_public_wrapper_default_construction_uses_exact_backend_without_fallback(monkeypatch):
    import openpi.models_pytorch.futuremamba as futuremamba

    created = {}

    class _FakePI0(_TinyBase):
        def __init__(self, config):
            super().__init__(config)
            created["base"] = config

    class _FakeProgress(_TraceProgress):
        def __init__(self, config):
            super().__init__(config)
            created["progress"] = config

    class _ExplodingMambaBackend(nn.Module):
        def __init__(self, config):
            del config
            raise ImportError("mamba_ssm is required for the official Mamba-2 memory backend")

    monkeypatch.setattr(futuremamba, "PI0Pytorch", _FakePI0)
    monkeypatch.setattr(futuremamba, "ProgressExpertPytorch", _FakeProgress)
    monkeypatch.setattr(futuremamba, "Mamba2MemoryBackend", _ExplodingMambaBackend)

    config = _TinyConfig(memory_backend="mamba2")
    with pytest.raises(ImportError, match="mamba_ssm.*Mamba-2"):
        futuremamba.FutureMambaPytorch(config)

    assert created["base"] is config
    assert created["progress"] is config


def test_wrapper_initializes_progress_from_action_expert_after_base_load():
    from types import SimpleNamespace

    model = _make_model()
    calls = []
    source = SimpleNamespace(
        layers=object(),
        norm=object(),
    )
    model.base.paligemma_with_expert = SimpleNamespace(gemma_expert=SimpleNamespace(model=source))
    model.base.action_in_proj = object()
    model.base.time_mlp_in = object()
    model.base.time_mlp_out = object()
    model.base.action_out_proj = object()

    def initialize(**kwargs):
        calls.append(kwargs)

    model.futuremamba.progress_expert.initialize_from_action_expert = initialize
    model.initialize_progress_from_action_expert()

    assert calls == [
        {
            "action_layers": source.layers,
            "action_norm": source.norm,
            "action_in_proj": model.base.action_in_proj,
            "time_mlp_in": model.base.time_mlp_in,
            "time_mlp_out": model.base.time_mlp_out,
            "action_out_proj": model.base.action_out_proj,
        }
    ]
    assert all(parameter.requires_grad is False for parameter in model.base.parameters())


def test_memory_input_is_current_vlm_token_only():
    plugin = _make_plugin()

    assert not hasattr(plugin, "action_encoder")
    assert not hasattr(plugin, "action_summary_fusion")
    assert not hasattr(plugin, "memory_input_fusion")
    with torch.no_grad():
        plugin.vlm_memory_in_proj.weight.copy_(torch.eye(4))
        plugin.vlm_memory_in_proj.bias.zero_()
    last_vlm = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    state = plugin.initial_memory_state(1, torch.device("cpu"), torch.float32)

    plugin.compute_memory_token(last_vlm, state)

    assert len(plugin.memory_backend.calls) == 1
    torch.testing.assert_close(plugin.memory_backend.calls[0][0], last_vlm)

def test_memory_input_projects_vlm_width_to_memory_width():
    config = _WideVLMConfig()
    plugin = _make_plugin(config=config)
    last_vlm = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    state = plugin.initial_memory_state(1, torch.device("cpu"), torch.float32)

    memory_token, _ = plugin.compute_memory_token(last_vlm, state)

    assert plugin.vlm_memory_in_proj.in_features == 8
    assert plugin.vlm_memory_in_proj.out_features == 4
    assert memory_token.shape == (1, 1, plugin.action_expert_width)


def test_one_sampling_query_advances_memory_exactly_once_without_action_history():
    model = _make_model()
    observation = _Observation(state=torch.tensor([[1.0, 2.0]]))
    state = model.initial_memory_state(1, torch.device("cpu"), torch.float32)

    model.sample_actions_with_memory(
        observation,
        state,
        noise=torch.zeros(1, 3, 2),
        num_steps=10,
        handoff_ratio=0.2,
    )

    assert len(model.futuremamba.memory_backend.calls) == 1




def test_initial_memory_state_delegates_exact_backend_and_preserves_metadata():
    backend = _RecordingMemoryBackend(_TinyConfig().memory)
    plugin = _make_plugin(memory_backend=backend)

    state = plugin.initial_memory_state(2, torch.device("cpu"), torch.float16)
    _, next_state = plugin.compute_memory_token(torch.zeros(2, 4), state)

    assert state.backend_id == "recording"
    assert state.state_schema_version == 7
    assert next_state.backend_id == "recording"
    assert next_state.state_schema_version == 7
    assert next_state.batch_size == 2
    assert next_state.layers[0][0].dtype == torch.float16


@pytest.mark.parametrize(
    "ratio, expected_progress, expected_action, expected_handoff",
    [(0.0, 0, 10, 0), (0.2, 2, 8, 2), (1.0, 10, 0, 10)],
)
def test_hard_handoff_call_counts_for_ratios(ratio, expected_progress, expected_action, expected_handoff):
    model = _make_model()
    observation = _Observation(state=torch.tensor([[1.0, 2.0]]))
    state = model.initial_memory_state(1, torch.device("cpu"), torch.float32)
    noise = torch.zeros(1, 3, 2)

    _, _, diagnostics = model.sample_actions_with_memory(
        observation,
        state,
        noise=noise,
        num_steps=10,
        handoff_ratio=ratio,
    )

    assert diagnostics["progress_calls"] == expected_progress
    assert diagnostics["action_calls"] == expected_action
    assert diagnostics["handoff_steps"] == expected_handoff
    assert len(model.futuremamba.progress_expert.calls) == expected_progress
    assert model.base.denoise_calls == expected_action


def test_ratio_zero_matches_frozen_action_loop_exactly_and_does_not_call_progress():
    progress = _TraceProgress(_TinyConfig())
    model = _make_model(progress=progress)
    observation = _Observation(state=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    state = model.initial_memory_state(2, torch.device("cpu"), torch.float32)
    noise = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 10

    expected = model.base.sample_actions(torch.device("cpu"), observation, noise=noise.clone(), num_steps=10)
    model.base.prefix_calls = 0
    model.base.denoise_calls = 0
    actual, _, diagnostics = model.sample_actions_with_memory(
        observation,
        state,
        noise=noise.clone(),
        num_steps=10,
        handoff_ratio=0.0,
    )

    torch.testing.assert_close(actual, expected)
    assert diagnostics["progress_calls"] == 0
    assert diagnostics["action_calls"] == 10
    assert len(progress.calls) == 0
    assert model.base.prefix_calls == 1

def test_ratio_zero_uses_stable_base_interfaces_for_frozen_prefix_and_action_velocity():
    class InterfaceOnlyBase(_TinyBase):
        def encode_frozen_prefix(self, observation, *, train: bool = False):
            raise AssertionError("sample_actions_with_memory must use extract_prefix_context")

        def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
            raise AssertionError("sample_actions_with_memory must use action_expert_velocity")
        def extract_prefix_context(self, observation, *, train: bool = False):
            return _TinyBase.encode_frozen_prefix(self, observation, train=train)

        def action_expert_velocity(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
            return _TinyBase.denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep)

    progress = _TraceProgress(_TinyConfig())
    base = InterfaceOnlyBase(_TinyConfig())
    model = _make_model(base=base, progress=progress)
    observation = _Observation(state=torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    state = model.initial_memory_state(2, torch.device("cpu"), torch.float32)
    noise = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) / 10
    expected_base = _TinyBase(_TinyConfig())
    expected = expected_base.sample_actions(torch.device("cpu"), observation, noise=noise.clone(), num_steps=10)
    base.prefix_calls = 0
    base.denoise_calls = 0
    actual, _, diagnostics = model.sample_actions_with_memory(
        observation,
        state,
        noise=noise.clone(),
        num_steps=10,
        handoff_ratio=0.0,
    )

    torch.testing.assert_close(actual, expected)
    assert diagnostics["progress_calls"] == 0
    assert diagnostics["action_calls"] == 10
    assert len(progress.calls) == 0
    assert base.prefix_calls == 1
    assert base.denoise_calls == 10


def test_ratio_one_never_calls_action_expert_or_sums_velocity_fields():
    progress = _ZeroProgress(_TinyConfig())
    model = _make_model(progress=progress)
    observation = _Observation(state=torch.tensor([[1.0, 2.0]]))
    state = model.initial_memory_state(1, torch.device("cpu"), torch.float32)
    noise = torch.ones(1, 3, 2)

    actual, _, diagnostics = model.sample_actions_with_memory(
        observation,
        state,
        noise=noise.clone(),
        num_steps=10,
        handoff_ratio=1.0,
    )

    torch.testing.assert_close(actual, noise)
    assert diagnostics["progress_calls"] == 10
    assert diagnostics["action_calls"] == 0
    assert model.base.denoise_calls == 0


def test_action_expert_receives_same_current_xt_at_handoff_and_no_memory_token():
    model = _make_model()
    observation = _Observation(state=torch.tensor([[1.0, 2.0]]))
    state = model.initial_memory_state(1, torch.device("cpu"), torch.float32)
    noise = torch.zeros(1, 3, 2)

    model.sample_actions_with_memory(
        observation,
        state,
        noise=noise,
        num_steps=10,
        handoff_ratio=0.2,
    )

    handoff_xt = model.base.last_denoise_inputs[0][3]
    torch.testing.assert_close(handoff_xt, torch.full_like(noise, -0.4))


def test_sample_actions_with_memory_rejects_invalid_ratio_steps_and_shapes():
    model = _make_model()
    observation = _Observation(state=torch.tensor([[1.0, 2.0]]))
    state = model.initial_memory_state(1, torch.device("cpu"), torch.float32)
    noise = torch.zeros(1, 3, 2)

    with pytest.raises(ValueError, match="handoff_ratio"):
        model.sample_actions_with_memory(observation, state, noise=noise, handoff_ratio=-0.1)
    with pytest.raises(ValueError, match="num_steps"):
        model.sample_actions_with_memory(observation, state, noise=noise, num_steps=0)
    with pytest.raises(ValueError, match="noise"):
        model.sample_actions_with_memory(observation, state, noise=torch.zeros(1, 2, 2))



def test_memory_dependency_diagnostic_only_swaps_memory_token():
    class MemorySensitiveProgress(nn.Module):
        def forward(self, prefix_cache, prefix_mask, memory_token, noisy_actions, timestep):
            del prefix_cache, prefix_mask, timestep
            return memory_token[:, :, : noisy_actions.shape[-1]].expand_as(noisy_actions)

    plugin = _make_plugin(progress=MemorySensitiveProgress())
    frozen = _TinyBase().encode_frozen_prefix(_Observation(state=torch.ones(1, 2)))
    prefix_cache = PrefixKVView.from_cache(
        frozen.kv_cache,
        plugin.progress_layer_indices,
        frozen.pad_mask,
    )
    memory_token = torch.ones(1, 1, plugin.action_expert_width)
    noisy_actions = torch.zeros(1, 3, 2)
    timestep = torch.ones(1)

    diagnostics = plugin.memory_dependency_diagnostics(
        prefix_cache,
        frozen.pad_mask,
        memory_token,
        noisy_actions,
        timestep,
    )

    assert diagnostics["memory_token_rms"] == pytest.approx(1.0)
    assert diagnostics["memory_velocity_delta_rms"] == pytest.approx(1.0)
    assert diagnostics["memory_velocity_cosine_to_zero"] == pytest.approx(0.0)


def test_compute_episode_loss_ignores_padded_queries_for_loss_memory_and_gradients():
    backend = _RecordingMemoryBackend(_TinyConfig().memory)
    progress = _TraceProgress(_TinyConfig())
    model = _make_model(memory_backend=backend, progress=progress)
    actions = torch.ones(1, 2, 3, 2)
    executed = torch.ones(1, 2, 3, 2)
    query_mask = torch.tensor([[True, False]])
    batch = _torch_episode_batch(actions, executed_actions=executed, query_mask=query_mask)
    noise = torch.zeros_like(actions)
    time = torch.full((1, 2), 0.75)

    first = model.compute_episode_loss(batch, noise=noise, time=time)
    first["loss"].backward()
    first_grad = progress.bias.grad.detach().clone()
    first_memory_input = backend.calls[0][0]

    model.zero_grad(set_to_none=True)
    backend.calls.clear()
    progress.calls.clear()
    padded_state = batch.observation.state.clone()
    padded_state[:, 1] = 9999.0
    padded_actions = actions.clone()
    padded_actions[:, 1] = -9999.0
    padded_executed = executed.clone()
    padded_executed[:, 1] = 7777.0
    mutated = _torch_episode_batch(
        padded_actions,
        executed_actions=padded_executed,
        query_mask=query_mask,
        state=padded_state,
    )

    second = model.compute_episode_loss(mutated, noise=noise, time=time)
    second["loss"].backward()

    torch.testing.assert_close(second["loss"], first["loss"])
    torch.testing.assert_close(progress.bias.grad, first_grad)
    assert len(backend.calls) == 1
    torch.testing.assert_close(backend.calls[0][0], first_memory_input)
    assert len(progress.calls) == 1


def test_compute_episode_loss_normalizes_queries_per_episode_before_batch_mean():
    model = _make_model(progress=_ZeroProgress(_TinyConfig()))
    actions = torch.zeros(2, 2, 3, 2)
    actions[0, 0] = 1.0
    actions[0, 1] = 3.0
    actions[1, 0] = 5.0
    actions[1, 1] = 1000.0
    query_mask = torch.tensor([[True, True], [True, False]])
    batch = _torch_episode_batch(actions, query_mask=query_mask)

    outputs = model.compute_episode_loss(batch, noise=torch.zeros_like(actions), time=torch.full((2, 2), 0.5))

    torch.testing.assert_close(outputs["flow_loss"], torch.tensor(15.0))
    torch.testing.assert_close(outputs["loss"], torch.tensor(15.0))
    torch.testing.assert_close(outputs["handoff_loss"], torch.tensor(0.0))
    torch.testing.assert_close(outputs["boundary_loss"], torch.tensor(0.0))


def test_compute_episode_loss_encodes_frozen_prefixes_in_microbatches():
    config = _TinyConfig(frozen_prefix_microbatch_size=2)
    base = _TinyBase(config)
    progress = _ZeroProgress(config)
    model = _make_model(config=config, base=base, progress=progress)
    actions = torch.zeros(1, 4, 3, 2)

    outputs = model.compute_episode_loss(
        _torch_episode_batch(actions),
        noise=torch.zeros_like(actions),
        time=torch.full((1, 4), 0.5),
    )

    torch.testing.assert_close(outputs["flow_loss"], torch.tensor(0.0))
    assert base.prefix_calls == 2
    assert len(progress.calls) == 4




def test_terminal_loss_backpropagates_through_frozen_action_expert_to_progress():
    config = _TinyLossConfig(
        handoff_ratio=0.5,
        num_denoise_steps=2,
        terminal_loss_weight=1.0,
    )
    progress = _XScaleProgress()
    base = _ZeroActionBase(config)
    model = _make_model(config=config, base=base, progress=progress)
    actions = torch.zeros(1, 1, 3, 2)
    batch = _torch_episode_batch(actions)

    outputs = model.compute_episode_loss(batch, noise=torch.ones_like(actions), time=torch.ones(1, 1))
    outputs["terminal_loss"].backward()

    torch.testing.assert_close(outputs["terminal_error"], torch.tensor(0.25))
    torch.testing.assert_close(outputs["terminal_loss"], torch.tensor(0.25))
    torch.testing.assert_close(progress.scale.grad, torch.tensor(-0.5))
    assert base.denoise_calls == 1
    assert all(not parameter.requires_grad for parameter in base.parameters())
    assert all(parameter.grad is None for parameter in base.parameters())


def test_zero_terminal_weight_is_exactly_flow_only():
    config = _TinyLossConfig(
        handoff_ratio=0.5,
        num_denoise_steps=2,
        terminal_loss_weight=0.0,
    )
    model = _make_model(config=config, base=_ZeroActionBase(config), progress=_XScaleProgress())
    actions = torch.zeros(1, 1, 3, 2)

    outputs = model.compute_episode_loss(
        _torch_episode_batch(actions),
        noise=torch.ones_like(actions),
        time=torch.ones(1, 1),
    )

    torch.testing.assert_close(outputs["loss"], outputs["flow_loss"])
    torch.testing.assert_close(outputs["terminal_loss"], torch.tensor(0.0))
    assert model.base.denoise_calls == 0


def test_terminal_loss_ignores_padded_actions():
    config = _TinyLossConfig(
        handoff_ratio=0.5,
        num_denoise_steps=2,
        terminal_loss_weight=1.0,
    )
    action_mask = torch.tensor([[[True, False, False]]])
    actions = torch.zeros(1, 1, 3, 2)
    model = _make_model(config=config, base=_ZeroActionBase(config), progress=_XScaleProgress())
    first = model.compute_episode_loss(
        _torch_episode_batch(actions, action_mask=action_mask),
        noise=torch.ones_like(actions),
        time=torch.ones(1, 1),
    )
    padded_actions = actions.clone()
    padded_actions[:, :, 1:] = 1000.0
    second = model.compute_episode_loss(
        _torch_episode_batch(padded_actions, action_mask=action_mask),
        noise=torch.ones_like(actions),
        time=torch.ones(1, 1),
    )

    torch.testing.assert_close(second["terminal_loss"], first["terminal_loss"])


def test_terminal_loss_batch_fraction_uses_deterministic_episode_prefix():
    config = _TinyLossConfig(
        handoff_ratio=0.5,
        num_denoise_steps=2,
        terminal_loss_weight=1.0,
        terminal_loss_batch_fraction=0.5,
    )
    actions = torch.zeros(2, 1, 3, 2)
    actions[1] = 2.0
    base = _ZeroActionBase(config)
    model = _make_model(config=config, base=base, progress=_XScaleProgress())

    outputs = model.compute_episode_loss(
        _torch_episode_batch(actions),
        noise=torch.ones_like(actions),
        time=torch.ones(2, 1),
    )
    torch.testing.assert_close(outputs["terminal_error"], torch.tensor(0.25))
    assert base.denoise_calls == 1


def test_terminal_loss_samples_uniform_query_subset_per_episode():
    config = _TinyLossConfig(
        handoff_ratio=0.5,
        num_denoise_steps=2,
        terminal_loss_weight=1.0,
        terminal_loss_queries_per_episode=2,
    )
    base = _ZeroActionBase(config)
    model = _make_model(config=config, base=base, progress=_XScaleProgress())
    actions = torch.zeros(1, 4, 3, 2)

    outputs = model.compute_episode_loss(
        _torch_episode_batch(actions),
        noise=torch.ones_like(actions),
        time=torch.ones(1, 4),
    )

    torch.testing.assert_close(outputs["terminal_error"], torch.tensor(0.25))
    assert base.denoise_calls == 2



def test_action_expert_checkpointing_recomputes_frozen_terminal_step():
    class RecomputeBase(_ZeroActionBase):
        def denoise_step(self, state, prefix_pad_masks, past_key_values, x_t, timestep):
            del state, prefix_pad_masks, past_key_values, timestep
            self.denoise_calls += 1
            return torch.sin(x_t)

    config = _TinyLossConfig(
        handoff_ratio=0.5,
        num_denoise_steps=2,
        terminal_loss_weight=1.0,
        action_expert_gradient_checkpointing=True,
    )
    base = RecomputeBase(config)
    model = _make_model(config=config, base=base, progress=_XScaleProgress())
    actions = torch.zeros(1, 1, 3, 2)

    outputs = model.compute_episode_loss(
        _torch_episode_batch(actions),
        noise=torch.ones_like(actions),
        time=torch.ones(1, 1),
    )
    outputs["terminal_loss"].backward()

    assert base.denoise_calls == 2
    assert all(parameter.grad is None for parameter in base.parameters())


def test_boundary_loss_detaches_boundary_state_freezes_base_and_trains_plugin():
    config = _TinyLossConfig(handoff_ratio=0.5, num_denoise_steps=2, boundary_loss_weight=1.0)
    progress = _XScaleProgress()
    model = _make_model(config=config, base=_ZeroActionBase(config), progress=progress)
    actions = torch.zeros(1, 1, 3, 2)
    batch = _torch_episode_batch(actions)

    outputs = model.compute_episode_loss(batch, noise=torch.ones_like(actions), time=torch.ones(1, 1))
    outputs["boundary_loss"].backward()

    torch.testing.assert_close(progress.scale.grad, torch.tensor(0.5))
    assert all(not parameter.requires_grad for parameter in model.base.parameters())
    assert all(parameter.grad is None for parameter in model.base.parameters())
    assert model.base.denoise_calls == 1


def test_two_query_episode_loss_backpropagates_through_previous_memory_update():
    memory = _CarryMemoryBackend(_TinyConfig().memory)
    progress = _SecondQueryTokenProgress()
    model = _make_model(memory_backend=memory, progress=progress)
    for module in (
        model.futuremamba.vlm_memory_in_proj,
        model.futuremamba.memory_token_proj,
    ):
        nn.init.constant_(module.weight, 0.1)
        nn.init.zeros_(module.bias)
    actions = torch.ones(1, 2, 3, 2)
    batch = _torch_episode_batch(actions, query_mask=torch.tensor([[True, True]]))

    outputs = model.compute_episode_loss(batch, noise=torch.zeros_like(actions), time=torch.full((1, 2), 0.5))
    outputs["flow_loss"].backward()

    assert progress.calls == 2
    assert memory.scale.grad is not None
    assert memory.scale.grad.abs() > 0

def test_burn_in_updates_memory_without_loss_or_gradient_across_window_boundary():
    class MemoryTokenProgress(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, prefix_cache, prefix_mask, memory_token, noisy_actions, timestep):
            del prefix_cache, prefix_mask, timestep
            self.calls += 1
            return memory_token[:, :, : noisy_actions.shape[-1]].expand_as(noisy_actions)

    memory = _CarryMemoryBackend(_TinyConfig().memory)
    progress = MemoryTokenProgress()
    model = _make_model(memory_backend=memory, progress=progress)
    for module in (
        model.futuremamba.vlm_memory_in_proj,
        model.futuremamba.memory_token_proj,
    ):
        nn.init.constant_(module.weight, 0.1)
        nn.init.zeros_(module.bias)
    actions = torch.ones(1, 2, 3, 2)
    batch = _torch_episode_batch(
        actions,
        query_mask=torch.tensor([[True, True]]),
        train_query_mask=torch.tensor([[False, True]]),
    )

    outputs = model.compute_episode_loss(batch, noise=torch.zeros_like(actions), time=torch.full((1, 2), 0.5))
    outputs["flow_loss"].backward()

    assert progress.calls == 1
    assert memory.grad_enabled == [False, True]
    assert memory.scale.grad is None

def test_episode_batch_rows_start_from_independent_zero_memory_states():
    backend = _RecordingMemoryBackend(_TinyConfig().memory)
    model = _make_model(memory_backend=backend, progress=_ZeroProgress(_TinyConfig()))
    actions = torch.ones(2, 1, 3, 2)
    batch = _torch_episode_batch(actions, query_mask=torch.tensor([[True], [True]]))

    model.compute_episode_loss(batch, noise=torch.zeros_like(actions), time=torch.full((2, 1), 0.5))

    assert len(backend.calls) == 2
    for _, state in backend.calls:
        torch.testing.assert_close(state.layers[0][0], torch.zeros_like(state.layers[0][0]))


def test_multi_query_rollout_keeps_actions_memory_and_diagnostics_finite():
    model = _make_model()
    observation = _Observation(state=torch.tensor([[1.0, 2.0]]))
    memory_state = model.initial_memory_state(1, torch.device("cpu"), torch.float32)

    for query_index in range(4):
        noise = torch.full((1, 3, 2), 0.1 * (query_index + 1))
        actions, memory_state, diagnostics = model.sample_actions_with_memory(
            observation,
            memory_state,
            noise=noise,
            num_steps=10,
            handoff_ratio=0.2,
        )
        assert torch.isfinite(actions).all()
        assert all(torch.isfinite(tensor).all() for layer in memory_state.layers for tensor in layer)
        assert all(isinstance(value, int) and value >= 0 for value in diagnostics.values())

def test_terminal_rollout_matches_inference_for_identical_noise_and_memory():
    config = _TinyLossConfig(
        handoff_ratio=0.5,
        num_denoise_steps=2,
        terminal_loss_weight=1.0,
    )
    model = _make_model(
        config=config,
        base=_ZeroActionBase(config),
        progress=_XScaleProgress(),
    )
    actions = torch.full((1, 1, 3, 2), 0.25)
    state = torch.tensor([[[1.0, 2.0]]])
    batch = _torch_episode_batch(actions, state=state)
    noise = torch.ones_like(actions)
    memory = model.initial_memory_state(1, torch.device("cpu"), torch.float32)

    sampled, _, diagnostics = model.sample_actions_with_memory(
        _Observation(state=state[:, 0]),
        memory,
        noise=noise[:, 0],
        num_steps=2,
        handoff_ratio=0.5,
    )
    outputs = model.compute_episode_loss(batch, noise=noise, time=torch.ones(1, 1))
    expected_terminal_error = torch.square(sampled - actions[:, 0]).mean()

    torch.testing.assert_close(outputs["terminal_error"], expected_terminal_error)
    assert diagnostics == {"progress_calls": 1, "action_calls": 1, "handoff_steps": 1}

def test_episode_loss_reports_sampled_time_over_training_queries_only():
    model = _make_model(progress=_ZeroProgress(_TinyConfig()))
    actions = torch.zeros(1, 2, 3, 2)
    batch = _torch_episode_batch(
        actions,
        query_mask=torch.tensor([[True, True]]),
        train_query_mask=torch.tensor([[False, True]]),
    )

    outputs = model.compute_episode_loss(
        batch,
        noise=torch.zeros_like(actions),
        time=torch.tensor([[0.81, 0.93]]),
    )

    torch.testing.assert_close(outputs["sample_time_mean"], torch.tensor(0.93))
    torch.testing.assert_close(outputs["sample_time_min"], torch.tensor(0.93))
    torch.testing.assert_close(outputs["sample_time_max"], torch.tensor(0.93))
