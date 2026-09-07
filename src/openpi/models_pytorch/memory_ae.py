from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.gemma import modeling_gemma

from openpi.models_pytorch.futuremamba import ActionHistoryState, FutureMambaPluginPytorch
from openpi.models_pytorch.gemma_pytorch import PrefixKVView, _cache_layers_as_key_values
from openpi.models_pytorch.mamba_memory import Mamba2MemoryBackend
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks


class MemoryAEPytorch(nn.Module):
    """Original pi0.5 Action Expert with executed-action memory and no PE.

    The frozen VLM supplies all prefix layers online. The existing AE owns every
    transformer, action projection and time-conditioning parameter; memory is an
    additional read-only K/V token, not another transformer or action query.
    """

    def __init__(
        self, config, *, base: nn.Module | None = None, futuremamba: FutureMambaPluginPytorch | None = None
    ) -> None:
        super().__init__()
        if not config.pi05:
            raise ValueError("MemoryAEPytorch requires the original pi0.5 Action Expert")
        if float(config.handoff_ratio) != 0.0:
            raise ValueError("MemoryAEPytorch has no Progress Expert or denoising handoff")
        self.config = config
        self.base = base if base is not None else PI0Pytorch(config)
        self.futuremamba = (
            futuremamba if futuremamba is not None else FutureMambaPluginPytorch(config, include_progress_expert=False)
        )
        if getattr(self.futuremamba, "progress_expert", None) is not None:
            raise ValueError("MemoryAEPytorch must not allocate a Progress Expert")
        if self.futuremamba.progress_memory_tokens != 1:
            raise ValueError("MemoryAEPytorch requires one executed-action memory token")
        self.freeze_base()

    def freeze_base(self) -> None:
        """Freeze only the VLM, leaving the original AE and memory trainable."""
        self.base.requires_grad_(True)
        self.base.paligemma_with_expert.paligemma.requires_grad_(False)
        self.base.paligemma_with_expert.paligemma.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.paligemma_with_expert.paligemma.eval()
        return self

    def initial_history_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> ActionHistoryState:
        return self.futuremamba.initial_history_state(batch_size, device, dtype)

    def _memory_token_from_history(self, history_actions: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        """Encode committed chunks plus one partial chunk with full backward.

        Rows are grouped by their number of valid chunk tokens. Padding never
        enters Mamba, and the partial chunk is encoded once after all committed
        chunks, exactly as an online peek from the committed inference state.
        No inference cache or step kernel participates in this training path.
        """
        plugin = self.futuremamba
        if history_actions.ndim != 3 or history_actions.shape[-1] != plugin.action_dim:
            raise ValueError("history_actions must have shape [batch, history, action_dim]")
        if history_mask.shape != history_actions.shape[:2] or history_mask.dtype != torch.bool:
            raise ValueError("history_mask must be boolean [batch, history]")
        if history_actions.shape[0] == 0:
            raise ValueError("history batch must not be empty")
        history_mask = history_mask.to(device=history_actions.device)
        lengths = history_mask.long().sum(dim=-1)
        expected = torch.arange(history_mask.shape[1], device=history_mask.device)[None] < lengths[:, None]
        if not torch.equal(history_mask, expected):
            raise ValueError("history_mask must be a right-padded prefix")
        backend = plugin.memory_backend
        if not isinstance(backend, Mamba2MemoryBackend):
            raise ValueError("MemoryAEPytorch sequence training requires the Mamba-2 backend")
        chunk_size = plugin.chunk_size
        chunk_counts = (lengths + chunk_size - 1) // chunk_size
        memory_output = plugin.empty_history[None].expand(history_actions.shape[0], -1)
        for count in torch.unique(chunk_counts).tolist():
            if count == 0:
                continue
            rows = torch.nonzero(chunk_counts == count, as_tuple=False).flatten()
            padded_length = count * chunk_size
            actions = history_actions.index_select(0, rows)[:, :padded_length]
            masks = history_mask.index_select(0, rows)[:, :padded_length]
            missing = padded_length - actions.shape[1]
            if missing:
                actions = F.pad(actions, (0, 0, 0, missing))
                masks = F.pad(masks, (0, missing), value=False)
            chunks = actions.reshape(-1, chunk_size, plugin.action_dim)
            chunk_masks = masks.reshape(-1, chunk_size)
            encoded = plugin.encode_action_chunk(chunks, chunk_masks)
            hidden = encoded.reshape(rows.numel(), count, plugin.memory_width)
            hidden = hidden.to(dtype=next(backend.parameters()).dtype)
            residual = None
            for block in backend.layers:
                hidden, residual = block(hidden, residual, inference_params=None)
            hidden = hidden + residual
            hidden = backend.norm(hidden.to(dtype=backend.norm.weight.dtype))
            memory_output = memory_output.index_copy(0, rows, hidden[:, -1].to(dtype=memory_output.dtype))
        return plugin.memory_token_projection(
            memory_output.to(dtype=plugin.memory_token_projection.weight.dtype)
        ).reshape(history_actions.shape[0], 1, plugin.action_expert_width)

    @staticmethod
    def _adaptive_norm(module, hidden: torch.Tensor, condition: torch.Tensor):
        # pi0.5 retains AdaRMS (including dense modulation) in FP32 while the
        # attention/MLP matrices are BF16. Do not cast the norm parameters.
        dense = getattr(module, "dense", None)
        if dense is not None:
            condition = condition.to(dtype=dense.weight.dtype)
        return module(hidden, cond=condition)

    def memory_velocity(
        self,
        prefix_cache,
        prefix_mask: torch.Tensor,
        memory_token: torch.Tensor | None,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Original AE suffix-only velocity, with optional differentiable memory.

        ``memory_token=None`` is the original no-memory AE parity path. No cache
        update, cache clone, or detached memory view is used. Only the frozen
        prefix is detached before concatenating the differentiable memory K/V.
        """
        expert = self.base.paligemma_with_expert.gemma_expert.model
        batch_size, horizon, _ = x_t.shape
        if horizon != int(self.config.action_horizon):
            raise ValueError("noisy action horizon differs from the original AE horizon")
        if prefix_mask.ndim != 2 or prefix_mask.shape[0] != batch_size or prefix_mask.dtype != torch.bool:
            raise ValueError("prefix_mask must be boolean [batch, prefix]")
        if prefix_mask.device != x_t.device:
            raise ValueError("prefix mask and noisy actions must share a device")
        if timestep.shape != (batch_size,):
            raise ValueError("timestep must have shape [batch]")
        if isinstance(prefix_cache, PrefixKVView):
            prefix_layers = prefix_cache.layers
        elif prefix_cache is None and prefix_mask.shape[1] == 0:
            prefix_layers = None
        else:
            prefix_layers = _cache_layers_as_key_values(prefix_cache)
        if prefix_layers is not None and len(prefix_layers) != len(expert.layers):
            raise ValueError("MemoryAE requires the full VLM prefix, one cache layer per original AE layer")
        memory_count = 0 if memory_token is None else 1
        if memory_token is not None and memory_token.shape != (batch_size, 1, self.futuremamba.action_expert_width):
            raise ValueError("memory_token must have shape [batch, 1, action_expert_width]")

        # pi0.5 does not use state in embed_suffix. Reuse its action/time modules
        # and block-mask convention instead of the causal PE action mask.
        hidden, suffix_mask, suffix_attention, condition = self.base.embed_suffix(
            None,
            x_t.to(dtype=self.base.action_in_proj.weight.dtype),
            timestep.to(dtype=self.base.time_mlp_in.weight.dtype),
        )
        hidden = hidden.to(dtype=expert.layers[0].self_attn.q_proj.weight.dtype)
        prefix_attention = prefix_mask[:, None].expand(batch_size, horizon, -1)
        action_attention = make_att_2d_masks(suffix_mask, suffix_attention)
        mask_parts = [prefix_attention]
        if memory_count:
            mask_parts.append(torch.ones(batch_size, horizon, 1, dtype=torch.bool, device=x_t.device))
        mask_parts.append(action_attention)
        attention_mask = self.base._prepare_attention_masks_4d(torch.cat(mask_parts, dim=-1))
        valid_prefix_lengths = prefix_mask.long().sum(dim=-1, keepdim=True)
        action_positions = valid_prefix_lengths + memory_count + suffix_mask.long().cumsum(dim=-1) - 1
        cos, sin = expert.rotary_emb(hidden, action_positions)
        if memory_count:
            memory_cos, memory_sin = expert.rotary_emb(hidden, valid_prefix_lengths)

        for layer_index, layer in enumerate(expert.layers):
            residual = hidden
            normalized, attention_gate = self._adaptive_norm(layer.input_layernorm, hidden, condition)
            attention = layer.self_attn
            shape = (batch_size, horizon, -1, attention.head_dim)
            query = attention.q_proj(normalized.to(dtype=attention.q_proj.weight.dtype)).view(shape).transpose(1, 2)
            key = attention.k_proj(normalized.to(dtype=attention.k_proj.weight.dtype)).view(shape).transpose(1, 2)
            value = attention.v_proj(normalized.to(dtype=attention.v_proj.weight.dtype)).view(shape).transpose(1, 2)
            query, key = modeling_gemma.apply_rotary_pos_emb(query, key, cos, sin, unsqueeze_dim=1)
            keys, values = [], []
            if prefix_layers is not None:
                prefix_key, prefix_value = prefix_layers[layer_index]
                expected_shape = (batch_size, key.shape[1], prefix_mask.shape[1], attention.head_dim)
                if tuple(prefix_key.shape) != expected_shape or tuple(prefix_value.shape) != expected_shape:
                    raise ValueError(f"VLM prefix layer {layer_index} must have shape {expected_shape}")
                keys.append(prefix_key.detach().to(dtype=key.dtype))
                values.append(prefix_value.detach().to(dtype=value.dtype))
            if memory_count:
                memory_shape = (batch_size, 1, -1, attention.head_dim)
                memory_key = attention.k_proj(memory_token.to(dtype=attention.k_proj.weight.dtype))
                memory_value = attention.v_proj(memory_token.to(dtype=attention.v_proj.weight.dtype))
                memory_key = memory_key.view(memory_shape).transpose(1, 2)
                memory_value = memory_value.view(memory_shape).transpose(1, 2)
                # RoPE acts only on keys, at the valid (not padded) prefix end.
                memory_key = (
                    memory_key * memory_cos[:, None] + modeling_gemma.rotate_half(memory_key) * memory_sin[:, None]
                )
                keys.append(memory_key)
                values.append(memory_value)
            keys.append(key)
            values.append(value)
            attended, _ = modeling_gemma.eager_attention_forward(
                attention,
                query,
                torch.cat(keys, dim=2),
                torch.cat(values, dim=2),
                attention_mask,
                attention.scaling,
                dropout=attention.attention_dropout if self.training else 0.0,
            )
            attended = attended.reshape(batch_size, horizon, -1).contiguous()
            attended = attention.o_proj(attended.to(dtype=attention.o_proj.weight.dtype))
            hidden = modeling_gemma._gated_residual(residual, attended, attention_gate)
            residual = hidden
            normalized, mlp_gate = self._adaptive_norm(layer.post_attention_layernorm, hidden, condition)
            transformed = layer.mlp(normalized.to(dtype=layer.mlp.up_proj.weight.dtype))
            hidden = modeling_gemma._gated_residual(residual, transformed, mlp_gate)
        hidden, _ = self._adaptive_norm(expert.norm, hidden, condition)
        return self.base.action_out_proj(hidden.to(dtype=self.base.action_out_proj.weight.dtype))

    def compute_query_loss(
        self, batch, *, noise: torch.Tensor | None = None, time: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        actions = batch.actions.to(dtype=torch.float32)
        expected = (actions.shape[0], int(self.config.action_horizon), int(self.config.action_dim))
        if tuple(actions.shape) != expected or actions.shape[0] == 0:
            raise ValueError(f"query actions must be a nonempty batch with shape {expected}")
        mask = batch.action_mask
        if mask.shape != actions.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("query action mask must be boolean [batch, horizon]")
        counts = mask.long().sum(dim=-1)
        if bool((counts == 0).any().item()):
            raise ValueError("every selected query must contain a valid target action")
        if noise is None:
            noise = self.base.sample_noise(actions.shape, actions.device)
        if tuple(noise.shape) != tuple(actions.shape):
            raise ValueError("noise shape must match query actions")
        if time is None:
            time = self.base.sample_time(actions.shape[0], actions.device)
        if time.shape != (actions.shape[0],):
            raise ValueError("sampled time must have shape [batch]")
        time = time.to(device=actions.device, dtype=torch.float32)
        noise = noise.to(device=actions.device, dtype=torch.float32)
        safe_actions = torch.where(mask[..., None], actions, 0.0)
        safe_noise = torch.where(mask[..., None], noise, 0.0)
        x_t = time[:, None, None] * safe_noise + (1.0 - time[:, None, None]) * safe_actions
        target = safe_noise - safe_actions
        with torch.no_grad():
            frozen = self.base.encode_frozen_prefix(batch.observation, train=False)
        memory_token = self._memory_token_from_history(batch.history_actions, batch.history_mask)
        predicted = self.memory_velocity(frozen.kv_cache, frozen.pad_mask, memory_token, x_t, time)
        error = (predicted.float() - target).square().mean(dim=-1)
        query_losses = torch.where(mask, error, 0.0).sum(dim=-1) / counts.to(dtype=error.dtype)
        flow_loss = query_losses.mean()
        return {
            "loss": flow_loss,
            "flow_loss": flow_loss,
            "sample_time_mean": time.mean(),
            "sample_time_min": time.min(),
            "sample_time_max": time.max(),
        }

    @torch.no_grad()
    def sample_actions_with_memory(
        self,
        observation,
        history: ActionHistoryState,
        executed_actions: torch.Tensor,
        executed_action_mask: torch.Tensor,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        handoff_ratio: float | None = None,
    ):
        num_steps = int(self.config.num_denoise_steps if num_steps is None else num_steps)
        ratio = float(self.config.handoff_ratio if handoff_ratio is None else handoff_ratio)
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")
        if ratio != 0.0:
            raise ValueError("MemoryAEPytorch has no Progress Expert; handoff_ratio must be zero")
        device = observation.state.device
        batch_size = int(observation.state.shape[0])
        if history.memory.batch_size != batch_size:
            raise ValueError("history batch must match observation")
        expected = (batch_size, int(self.config.action_horizon), int(self.config.action_dim))
        if noise is None:
            noise = self.base.sample_noise(expected, device)
        elif tuple(noise.shape) != expected:
            raise ValueError(f"noise must have shape {expected}")
        memory_token, next_history, history_diagnostics = self.futuremamba.advance_history(
            history, executed_actions, executed_action_mask
        )
        frozen = self.base.encode_frozen_prefix(observation, train=False)
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
        timestep = torch.tensor(1.0, dtype=torch.float32, device=device)
        x_t = noise
        # Match original pi0.5's FP32 Euler time accumulation, using the AE on
        # every step (ten by default), with one history advance per query.
        for _ in range(num_steps):
            velocity = self.memory_velocity(
                frozen.kv_cache, frozen.pad_mask, memory_token, x_t, timestep.expand(batch_size)
            )
            x_t = x_t + dt * velocity
            timestep = timestep + dt
        return (
            x_t,
            next_history,
            {
                "progress_calls": 0,
                "action_calls": num_steps,
                "handoff_steps": 0,
                **history_diagnostics,
            },
        )
