from __future__ import annotations

import dataclasses
import math

import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PrefixKVView


def default_progress_depth(action_depth: int) -> int:
    if action_depth <= 0:
        raise ValueError(f"action_depth must be positive, got {action_depth}")
    return min(6, action_depth)


def make_layer_mapping(action_depth: int, progress_depth: int) -> tuple[int, ...]:
    if action_depth <= 0:
        raise ValueError(f"action_depth must be positive, got {action_depth}")
    if progress_depth <= 0:
        raise ValueError("layer mapping must be strictly increasing; progress_depth must be positive")
    if progress_depth > action_depth:
        raise ValueError(f"progress_depth must be <= action_depth ({action_depth}), got {progress_depth}")
    if progress_depth == 1:
        if action_depth == 1:
            return (0,)
        raise ValueError("layer mapping must cover first and last layers when action_depth is greater than 1")
    mapping = tuple(round(index * (action_depth - 1) / (progress_depth - 1)) for index in range(progress_depth))
    if mapping[0] != 0 or mapping[-1] != action_depth - 1 or any(
        left >= right for left, right in zip(mapping[:-1], mapping[1:], strict=True)
    ):
        raise ValueError(f"layer mapping must be strictly increasing and cover endpoints, got {mapping}")
    return mapping


def _dtype_from_config(config) -> torch.dtype:
    precision = getattr(config, "dtype", "float32")
    if precision == "bfloat16":
        return torch.bfloat16
    if precision == "float32":
        return torch.float32
    raise ValueError(f"Unsupported progress expert dtype {precision!r}")


def _gemma_config_to_hf(config: _gemma.Config, depth: int):
    hf_config = CONFIG_MAPPING["gemma"](
        head_dim=config.head_dim,
        hidden_size=config.width,
        intermediate_size=config.mlp_dim,
        num_attention_heads=config.num_heads,
        num_hidden_layers=depth,
        num_key_value_heads=config.num_kv_heads,
        vocab_size=257152,
        attention_bias=False,
        hidden_activation="gelu_pytorch_tanh",
        torch_dtype="float32",
        use_adarms=True,
        adarms_cond_dim=config.width,
    )
    hf_config._attn_implementation = "eager"  # noqa: SLF001
    return hf_config


def _gated_residual(residual: torch.Tensor, update: torch.Tensor, gate: torch.Tensor | None) -> torch.Tensor:
    if gate is None:
        return residual + update
    return residual + update * gate


class _AdaRMSNorm(nn.Module):
    def __init__(self, width: int, eps: float, cond_dim: int) -> None:
        super().__init__()
        self.base_norm = modeling_gemma.GemmaRMSNorm(width, eps=eps)
        self.dense = nn.Linear(cond_dim, width * 3)
        nn.init.zeros_(self.dense.weight)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.base_norm(x)
        if isinstance(normalized, tuple):
            normalized = normalized[0]
        modulation = self.dense(cond).unsqueeze(1)
        scale, shift, gate = torch.chunk(modulation, 3, dim=-1)
        conditioned = normalized.to(torch.float32) * (1.0 + scale.to(torch.float32)) + shift.to(torch.float32)
        return conditioned.to(x.dtype), gate.to(x.dtype)


class _ProgressBlock(nn.Module):
    def __init__(self, hf_config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = modeling_gemma.GemmaAttention(hf_config, layer_idx=layer_idx)
        self.mlp = modeling_gemma.GemmaMLP(hf_config)
        self.input_layernorm = _AdaRMSNorm(hf_config.hidden_size, hf_config.rms_norm_eps, hf_config.adarms_cond_dim)
        self.post_attention_layernorm = _AdaRMSNorm(hf_config.hidden_size, hf_config.rms_norm_eps, hf_config.adarms_cond_dim)

    def forward(
        self,
        action_tokens: torch.Tensor,
        memory_token: torch.Tensor,
        prefix_k: torch.Tensor,
        prefix_v: torch.Tensor,
        attention_mask: torch.Tensor,
        memory_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        action_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        adarms_cond: torch.Tensor,
    ) -> torch.Tensor:
        residual = action_tokens
        hidden_states, gate = self.input_layernorm(action_tokens, adarms_cond)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.self_attn.head_dim)
        query_states = self.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        action_key_states = self.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        action_value_states = self.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        memory_key_states = self.self_attn.k_proj(memory_token).view(
            memory_token.shape[0], memory_token.shape[1], -1, self.self_attn.head_dim
        ).transpose(1, 2)
        memory_value_states = self.self_attn.v_proj(memory_token).view(
            memory_token.shape[0], memory_token.shape[1], -1, self.self_attn.head_dim
        ).transpose(1, 2)
        query_cos, query_sin = action_position_embeddings
        memory_cos, memory_sin = memory_position_embeddings
        query_states, action_key_states = modeling_gemma.apply_rotary_pos_emb(
            query_states, action_key_states, query_cos, query_sin, unsqueeze_dim=1
        )
        memory_key_states = (memory_key_states * memory_cos.unsqueeze(1)) + (
            modeling_gemma.rotate_half(memory_key_states) * memory_sin.unsqueeze(1)
        )
        key_states = torch.cat([prefix_k.detach().to(query_states.dtype), memory_key_states, action_key_states], dim=2)
        value_states = torch.cat([prefix_v.detach().to(query_states.dtype), memory_value_states, action_value_states], dim=2)
        attn_output, _ = modeling_gemma.eager_attention_forward(
            self.self_attn,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0,
            scaling=self.self_attn.scaling,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.self_attn.o_proj(attn_output)
        hidden_states = _gated_residual(residual, attn_output, gate)
        residual = hidden_states
        hidden_states, gate = self.post_attention_layernorm(hidden_states, adarms_cond)
        hidden_states = self.mlp(hidden_states)
        return _gated_residual(residual, hidden_states, gate)


def _sinusoidal_embedding(timestep: torch.Tensor, dimension: int) -> torch.Tensor:
    if dimension % 2 != 0:
        raise ValueError(f"sinusoidal dimension must be even, got {dimension}")
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float32, device=timestep.device)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    scaling_factor = 1.0 / period * 2 * math.pi
    sin = torch.sin(timestep[:, None] * scaling_factor[None, :])
    cos = torch.cos(timestep[:, None] * scaling_factor[None, :])
    return torch.cat([sin, cos], dim=1)


@dataclasses.dataclass(frozen=True)
class _ValidatedInputs:
    batch: int
    action_horizon: int
    dtype: torch.dtype
    prefix_valid_lengths: torch.Tensor


def _copy_progress_block(target: nn.Module, source: nn.Module) -> None:
    _copy_progress_module(target.self_attn, source.self_attn)
    _copy_progress_module(target.mlp, source.mlp)
    _copy_progress_norm(target.input_layernorm, source.input_layernorm)
    _copy_progress_norm(target.post_attention_layernorm, source.post_attention_layernorm)


def _copy_progress_norm(target: nn.Module, source: nn.Module) -> None:
    source_state = source.state_dict()
    aliases: list[tuple[str, str]] = []
    if "weight" in source_state:
        aliases.append(("base_norm.weight", "weight"))
    elif "base_norm.weight" in source_state:
        aliases.append(("base_norm.weight", "base_norm.weight"))
    _copy_progress_module(
        target,
        source,
        aliases=tuple(aliases),
        optional=("base_norm.weight", "dense.weight", "dense.bias"),
    )


def _copy_progress_module(
    target: nn.Module,
    source: nn.Module,
    *,
    aliases: tuple[tuple[str, str], ...] = (),
    optional: tuple[str, ...] = (),
) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    alias_map = dict(aliases)
    with torch.no_grad():
        for target_name, target_tensor in target_state.items():
            source_name = alias_map.get(target_name, target_name)
            if source_name not in source_state:
                if target_name in optional:
                    continue
                raise ValueError(f"cannot initialize Progress Expert parameter {target_name} from Action Expert")
            source_tensor = source_state[source_name]
            if target_tensor.shape != source_tensor.shape:
                raise ValueError(
                    f"Progress Expert parameter {target_name} shape {tuple(target_tensor.shape)} does not match "
                    f"Action Expert {source_name} shape {tuple(source_tensor.shape)}"
                )
            target_tensor.copy_(source_tensor.to(dtype=target_tensor.dtype, device=target_tensor.device))
        target.load_state_dict(target_state, strict=True)
class ProgressExpertPytorch(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.action_horizon = int(config.action_horizon)
        self.action_dim = int(config.action_dim)
        action_config = _gemma.get_config(config.action_expert_variant)
        progress_depth = int(getattr(config, "progress_depth", default_progress_depth(action_config.depth)))
        configured_mapping = getattr(config, "progress_layer_mapping", None)
        self.layer_mapping = (
            tuple(configured_mapping)
            if configured_mapping is not None
            else make_layer_mapping(action_config.depth, progress_depth)
        )
        self.width = action_config.width
        self.num_key_value_heads = action_config.num_kv_heads
        self.head_dim = action_config.head_dim
        hf_config = _gemma_config_to_hf(action_config, progress_depth)
        self.hf_config = hf_config
        self.rotary_emb = modeling_gemma.GemmaRotaryEmbedding(hf_config)
        self.action_in_proj = nn.Linear(self.action_dim, self.width)
        self.time_mlp_in = nn.Linear(self.width, self.width)
        self.time_mlp_out = nn.Linear(self.width, self.width)
        self.layers = nn.ModuleList(_ProgressBlock(hf_config, idx) for idx in range(progress_depth))
        self.norm = _AdaRMSNorm(self.width, hf_config.rms_norm_eps, hf_config.adarms_cond_dim)
        self.action_out_proj = nn.Linear(self.width, self.action_dim)
        self.last_attention_shapes: list[tuple[int, int]] = []
        self.to(dtype=_dtype_from_config(config))

    def initialize_from_action_expert(
        self,
        *,
        action_layers: nn.ModuleList | list[nn.Module] | tuple[nn.Module, ...],
        action_norm: nn.Module,
        action_in_proj: nn.Module,
        time_mlp_in: nn.Module,
        time_mlp_out: nn.Module,
        action_out_proj: nn.Module,
    ) -> None:
        """Copy the selected frozen Action Expert weights into this trainable expert."""
        if len(action_layers) <= max(self.layer_mapping):
            raise ValueError(
                f"action_layers must contain at least {max(self.layer_mapping) + 1} layers, got {len(action_layers)}"
            )
        for target_layer, source_index in zip(self.layers, self.layer_mapping, strict=True):
            _copy_progress_block(target_layer, action_layers[source_index])
        _copy_progress_norm(self.norm, action_norm)
        _copy_progress_module(self.action_in_proj, action_in_proj)
        _copy_progress_module(self.time_mlp_in, time_mlp_in)
        _copy_progress_module(self.time_mlp_out, time_mlp_out)
        _copy_progress_module(self.action_out_proj, action_out_proj)



    def forward(
        self,
        prefix_cache: PrefixKVView,
        prefix_mask: torch.BoolTensor,
        memory_token: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        inputs = self._validate_inputs(prefix_cache, prefix_mask, memory_token, noisy_actions, timestep)
        compute_dtype = self.action_in_proj.weight.dtype
        action_tokens = self.action_in_proj(noisy_actions.to(compute_dtype))
        adarms_cond = self._embed_timestep(timestep.to(device=noisy_actions.device, dtype=torch.float32)).to(compute_dtype)
        memory_token = memory_token.to(compute_dtype)
        prefix_valid_lengths = inputs.prefix_valid_lengths
        memory_token_count = memory_token.shape[1]
        action_positions = prefix_valid_lengths[:, None] + 1 + torch.arange(
            inputs.action_horizon, device=noisy_actions.device
        )[None, :]
        memory_positions = prefix_valid_lengths[:, None].expand(-1, memory_token_count)

        self.last_attention_shapes = []
        for layer_idx, layer in enumerate(self.layers):
            prefix_k, prefix_v = prefix_cache.layer(layer_idx)
            prefix_k = prefix_k.to(device=noisy_actions.device, dtype=compute_dtype)
            prefix_v = prefix_v.to(device=noisy_actions.device, dtype=compute_dtype)
            attention_mask = self._attention_mask(
                prefix_mask.to(noisy_actions.device), inputs.action_horizon, memory_token_count
            )
            action_position_embeddings = self._position_embeddings(action_tokens, action_positions)
            memory_position_embeddings = self._position_embeddings(memory_token, memory_positions)
            self.last_attention_shapes.append((inputs.action_horizon, attention_mask.shape[-1]))
            action_tokens = layer(
                action_tokens,
                memory_token,
                prefix_k,
                prefix_v,
                attention_mask,
                memory_position_embeddings,
                action_position_embeddings,
                adarms_cond,
            )
        action_tokens, _ = self.norm(action_tokens, adarms_cond)
        velocities = self.action_out_proj(action_tokens.to(self.action_out_proj.weight.dtype))
        return velocities.to(inputs.dtype)

    def _embed_timestep(self, timestep: torch.Tensor) -> torch.Tensor:
        time_emb = _sinusoidal_embedding(timestep, self.width)
        hidden = self.time_mlp_in(time_emb.to(self.time_mlp_in.weight.dtype))
        hidden = F.silu(hidden)
        hidden = self.time_mlp_out(hidden)
        return F.silu(hidden)

    def _position_embeddings(
        self, reference: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.rotary_emb(reference, position_ids)

    def _validate_inputs(
        self,
        prefix_cache: PrefixKVView,
        prefix_mask: torch.BoolTensor,
        memory_token: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> _ValidatedInputs:
        if not isinstance(prefix_cache, PrefixKVView):
            raise ValueError("prefix_cache must be a PrefixKVView")
        if prefix_mask.ndim != 2 or prefix_mask.dtype is not torch.bool:
            raise ValueError(f"prefix_mask must have shape [batch, prefix] and bool dtype, got {tuple(prefix_mask.shape)}")
        batch, prefix_len = prefix_mask.shape
        expected_memory_shape = (batch, int(getattr(self.config, "progress_memory_tokens", 1)), self.width)
        if memory_token.shape != expected_memory_shape:
            raise ValueError(f"memory_token must have shape {expected_memory_shape}, got {tuple(memory_token.shape)}")
        if noisy_actions.shape != (batch, self.action_horizon, self.action_dim):
            raise ValueError(
                f"noisy_actions must have shape [{batch}, {self.action_horizon}, {self.action_dim}], got {tuple(noisy_actions.shape)}"
            )
        if timestep.shape != (batch,):
            raise ValueError(f"timestep must have shape [{batch}], got {tuple(timestep.shape)}")
        if len(prefix_cache.layers) != len(self.layers):
            raise ValueError(f"prefix_cache must contain {len(self.layers)} layers, got {len(prefix_cache.layers)}")
        if prefix_cache.valid_lengths.shape != (batch,):
            raise ValueError(
                f"prefix_cache valid lengths must have shape [{batch}], got {tuple(prefix_cache.valid_lengths.shape)}"
            )
        prefix_valid_lengths = prefix_mask.long().sum(dim=-1).to(noisy_actions.device)
        if not torch.equal(prefix_cache.valid_lengths.to(prefix_valid_lengths.device), prefix_valid_lengths):
            raise ValueError("prefix_cache valid lengths do not match prefix_mask")
        max_prefix_len = int(prefix_valid_lengths.max().item())
        for layer_idx, (key, value) in enumerate(prefix_cache.layers):
            expected = (batch, self.num_key_value_heads, max_prefix_len, self.head_dim)
            if key.shape != expected or value.shape != expected:
                raise ValueError(f"prefix_cache layer {layer_idx} must have key/value shape {expected}")
        return _ValidatedInputs(batch, self.action_horizon, noisy_actions.dtype, prefix_valid_lengths)

    def _attention_mask(
        self, prefix_mask: torch.BoolTensor, action_horizon: int, memory_token_count: int
    ) -> torch.Tensor:
        batch = prefix_mask.shape[0]
        prefix_len = int(prefix_mask.long().sum(dim=-1).max().item())
        prefix_positions = torch.arange(prefix_len, device=prefix_mask.device)[None, :]
        prefix_visible = prefix_positions < prefix_mask.long().sum(dim=-1)[:, None]
        prefix = prefix_visible[:, None, None, :].expand(batch, 1, action_horizon, prefix_len)
        memory = torch.ones(
            batch, 1, action_horizon, memory_token_count, dtype=torch.bool, device=prefix_mask.device
        )
        action_visible = torch.ones(action_horizon, action_horizon, dtype=torch.bool, device=prefix_mask.device).tril()
        action_visible = action_visible[None, None, :, :].expand(batch, 1, action_horizon, action_horizon)
        visible = torch.cat([prefix, memory, action_visible], dim=-1)
        return torch.where(visible, torch.tensor(0.0, device=prefix_mask.device), torch.tensor(-2.3819763e38, device=prefix_mask.device))
