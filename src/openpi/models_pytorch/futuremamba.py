from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PrefixKVView, detach_cache, slice_cache_batch
from openpi.models_pytorch.mamba_memory import (
    FrameStackMemoryBackend,
    GRUMemoryBackend,
    LSTMMemoryBackend,
    Mamba2MemoryBackend,
    MemorySnapshot,
    NoMemoryBackend,
)
from openpi.models_pytorch.pi0_pytorch import FrozenPrefix, PI0Pytorch
from openpi.models_pytorch.progress_expert import ProgressExpertPytorch


class FutureMambaPluginPytorch(nn.Module):
    def __init__(
        self,
        config,
        *,
        memory_backend: nn.Module | None = None,
        progress_expert: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.memory_width = int(config.memory.d_model)
        self.action_expert_width = _action_expert_width(config, self.memory_width)
        self.progress_layer_indices = tuple(config.resolved_progress_layer_indices)

        self.vlm_width = _vlm_width(config, self.memory_width)
        self.vlm_memory_in_proj = nn.Linear(self.vlm_width, self.memory_width)
        self.progress_expert = progress_expert if progress_expert is not None else ProgressExpertPytorch(config)
        self.memory_backend = memory_backend if memory_backend is not None else _make_memory_backend(config)
        self.memory_token_proj = nn.Linear(self.memory_width, self.action_expert_width)
        self.to(dtype=_dtype_from_config(config))

    def initial_memory_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> MemorySnapshot:
        return self.memory_backend.initial_state(batch_size, device=device, dtype=dtype)

    def compute_memory_token(
        self,
        last_vlm_token: torch.Tensor,
        memory_state: MemorySnapshot,
    ) -> tuple[torch.Tensor, MemorySnapshot]:
        if last_vlm_token.ndim != 2 or last_vlm_token.shape[1] != self.vlm_width:
            raise ValueError(f"last_vlm_token must have shape [batch, {self.vlm_width}], got {tuple(last_vlm_token.shape)}")
        memory_input = self.vlm_memory_in_proj(last_vlm_token.to(dtype=self.vlm_memory_in_proj.weight.dtype))
        memory_output, next_state = self.memory_backend.step(memory_input, memory_state)
        memory_token = self.memory_token_proj(memory_output.to(dtype=self.memory_token_proj.weight.dtype)).unsqueeze(1)
        return memory_token, next_state

    def forward_progress(
        self,
        prefix_cache: PrefixKVView,
        prefix_mask: torch.BoolTensor,
        memory_token: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.progress_expert(prefix_cache, prefix_mask, memory_token, noisy_actions, timestep)

    def memory_dependency_diagnostics(
        self,
        prefix_cache: PrefixKVView,
        prefix_mask: torch.BoolTensor,
        memory_token: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> dict[str, float]:
        """Measure early-velocity sensitivity while changing only the Memory Token."""
        with torch.no_grad():
            memory_velocity = self.forward_progress(
                prefix_cache, prefix_mask, memory_token, noisy_actions, timestep
            )
            zero_velocity = self.forward_progress(
                prefix_cache, prefix_mask, torch.zeros_like(memory_token), noisy_actions, timestep
            )
            delta = memory_velocity - zero_velocity
            memory_norm = torch.sqrt(torch.mean(torch.square(memory_token.float())))
            delta_norm = torch.sqrt(torch.mean(torch.square(delta.float())))
            memory_flat = memory_velocity.float().reshape(memory_velocity.shape[0], -1)
            zero_flat = zero_velocity.float().reshape(zero_velocity.shape[0], -1)
            denominator = torch.linalg.vector_norm(memory_flat, dim=-1) * torch.linalg.vector_norm(
                zero_flat, dim=-1
            )
            cosine = torch.where(
                denominator > 0,
                torch.sum(memory_flat * zero_flat, dim=-1) / denominator,
                torch.zeros_like(denominator),
            ).mean()
        return {
            "memory_token_rms": float(memory_norm.cpu().item()),
            "memory_velocity_delta_rms": float(delta_norm.cpu().item()),
            "memory_velocity_cosine_to_zero": float(cosine.cpu().item()),
        }



class FutureMambaPytorch(nn.Module):
    def __init__(
        self,
        config,
        *,
        base: nn.Module | None = None,
        futuremamba: FutureMambaPluginPytorch | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.base = base if base is not None else PI0Pytorch(config)
        self.futuremamba = futuremamba if futuremamba is not None else FutureMambaPluginPytorch(config)
        self.freeze_base()

    def freeze_base(self) -> None:
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.base.eval()

    def initialize_progress_from_action_expert(self) -> None:
        """Initialize the trainable Progress Expert from the frozen Action Expert."""
        expert = self.base.paligemma_with_expert.gemma_expert.model
        self.futuremamba.progress_expert.initialize_from_action_expert(
            action_layers=expert.layers,
            action_norm=expert.norm,
            action_in_proj=self.base.action_in_proj,
            time_mlp_in=self.base.time_mlp_in,
            time_mlp_out=self.base.time_mlp_out,
            action_out_proj=self.base.action_out_proj,
        )
        self.freeze_base()

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.base.eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        return self

    def base_checksum(self) -> str:
        digest = hashlib.sha256()
        for name, tensor in self.base.state_dict().items():
            digest.update(name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(tensor.shape)).encode("utf-8"))
            raw_tensor = tensor.detach().cpu().contiguous().view(torch.uint8)
            digest.update(raw_tensor.numpy().tobytes())
        return digest.hexdigest()

    def initial_memory_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> MemorySnapshot:
        return self.futuremamba.initial_memory_state(batch_size, device, dtype)

    def sample_actions_with_memory(
        self,
        observation,
        memory_state: MemorySnapshot,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        handoff_ratio: float | None = None,
    ) -> tuple[torch.Tensor, MemorySnapshot, dict[str, int]]:
        num_steps = int(self.config.num_denoise_steps if num_steps is None else num_steps)
        ratio = float(self.config.handoff_ratio if handoff_ratio is None else handoff_ratio)
        self._validate_sampling_inputs(observation, memory_state, noise, num_steps, ratio)
        device = noise.device if noise is not None else observation.state.device
        batch_size = int(observation.state.shape[0])
        if noise is None:
            noise = self.base.sample_noise((batch_size, int(self.config.action_horizon), int(self.config.action_dim)), device)

        with torch.no_grad():
            *_, processed_state = self.base._preprocess_observation(observation, train=False)
            frozen = self._detached_frozen_prefix(self.base.extract_prefix_context(observation, train=False))
            last_vlm = self.base.last_valid_prefix(frozen).detach()
            memory_token, next_memory_state = self.futuremamba.compute_memory_token(last_vlm, memory_state)
            prefix_cache = PrefixKVView.from_cache(
                frozen.kv_cache, self.futuremamba.progress_layer_indices, frozen.pad_mask
            )

            dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
            handoff_steps = _handoff_steps(ratio, num_steps)
            x_t = noise
            progress_calls = 0
            action_calls = 0
            for step in range(num_steps):
                timestep = torch.full((batch_size,), 1.0 + step * float(dt.item()), dtype=torch.float32, device=device)
                if step < handoff_steps:
                    velocity = self.futuremamba.forward_progress(
                        prefix_cache, frozen.pad_mask, memory_token, x_t, timestep
                    )
                    progress_calls += 1
                else:
                    velocity = self.base.action_expert_velocity(processed_state, frozen.pad_mask, frozen.kv_cache, x_t, timestep)
                    action_calls += 1
                x_t = x_t + dt * velocity
        diagnostics = {
            "progress_calls": progress_calls,
            "action_calls": action_calls,
            "handoff_steps": handoff_steps,
        }
        return x_t, next_memory_state, diagnostics

    def compute_episode_loss(
        self,
        batch,
        *,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        train: bool = True,
    ) -> dict[str, torch.Tensor]:
        from openpi.training import episode_data_loader as _episode_loader

        device = batch.actions.device if torch.is_tensor(batch.actions) else torch.device("cpu")
        batch = _episode_loader.episode_batch_to_torch(batch, device)
        actions = batch.actions
        batch_size, num_queries = actions.shape[:2]
        device = actions.device
        dtype = actions.dtype
        if noise is None:
            noise = self.base.sample_noise(tuple(actions.shape), device)
        else:
            noise = noise.to(device=device, dtype=dtype)
        if tuple(noise.shape) != tuple(actions.shape):
            raise ValueError(f"noise must have shape {tuple(actions.shape)}, got {tuple(noise.shape)}")
        handoff_steps = _handoff_steps(float(getattr(self.config, "handoff_ratio", 0.0)), int(self.config.num_denoise_steps))
        if time is None:
            time = self._sample_high_noise_time((batch_size, num_queries), handoff_steps=handoff_steps, device=device)
        else:
            time = time.to(device=device, dtype=torch.float32)
        if tuple(time.shape) != (batch_size, num_queries):
            raise ValueError(f"time must have shape {(batch_size, num_queries)}, got {tuple(time.shape)}")

        query_mask = batch.query_mask
        train_query_mask = batch.train_query_mask
        if train_query_mask is None:
            raise ValueError("train_query_mask must be materialized before computing episode loss")
        valid_action_mask = batch.action_mask & train_query_mask[:, :, None]
        safe_actions = torch.where(valid_action_mask[:, :, :, None], actions, torch.zeros_like(actions))
        safe_noise = torch.where(valid_action_mask[:, :, :, None], noise, torch.zeros_like(noise))
        x_t = time[:, :, None, None] * safe_noise + (1.0 - time[:, :, None, None]) * safe_actions
        target_velocity = safe_noise - safe_actions

        flow_error = torch.zeros(batch_size, num_queries, int(self.config.action_horizon), dtype=dtype, device=device)
        terminal_error = torch.zeros((), dtype=dtype, device=device)
        handoff_error = torch.zeros((), dtype=dtype, device=device)
        boundary_error = torch.zeros((), dtype=dtype, device=device)
        terminal_weight = float(getattr(self.config, "terminal_loss_weight", 0.0))
        handoff_weight = float(getattr(self.config, "handoff_loss_weight", 0.0))
        boundary_weight = float(getattr(self.config, "boundary_loss_weight", 0.0))
        terminal_error_steps = torch.zeros_like(flow_error)
        handoff_error_steps = torch.zeros_like(flow_error)
        boundary_error_steps = torch.zeros_like(flow_error)
        terminal_episode_limit = max(
            1,
            int(math.ceil(batch_size * float(getattr(self.config, "terminal_loss_batch_fraction", 1.0)))),
        )
        terminal_query_mask = torch.zeros_like(train_query_mask)
        terminal_query_limit = getattr(self.config, "terminal_loss_queries_per_episode", None)
        if terminal_weight > 0.0:
            for episode_index in range(terminal_episode_limit):
                eligible_queries = torch.nonzero(train_query_mask[episode_index], as_tuple=False).flatten()
                if terminal_query_limit is not None and eligible_queries.numel() > terminal_query_limit:
                    permutation = torch.randperm(eligible_queries.numel(), device=eligible_queries.device)
                    eligible_queries = eligible_queries[permutation[:terminal_query_limit]]
                terminal_query_mask[episode_index, eligible_queries] = True


        prefix_microbatch_size = int(getattr(self.config, "frozen_prefix_microbatch_size", 1))
        for episode_index in range(batch_size):
            valid_queries = int(query_mask[episode_index].long().sum().item())
            memory_state = self.initial_memory_state(1, device, dtype)
            for query_start in range(0, valid_queries, prefix_microbatch_size):
                query_stop = min(query_start + prefix_microbatch_size, valid_queries)
                observation_batch = _slice_episode_observation_range(
                    batch.observation, episode_index, query_start, query_stop
                )
                with torch.no_grad():
                    *_, processed_states = self.base._preprocess_observation(observation_batch, train=train)
                    frozen_batch = self._detached_frozen_prefix(
                        self.base.encode_frozen_prefix(observation_batch, train=train)
                    )
                    last_vlm_batch = self.base.last_valid_prefix(frozen_batch).detach()
                    prefix_cache_batch = PrefixKVView.from_cache(
                        frozen_batch.kv_cache,
                        self.futuremamba.progress_layer_indices,
                        frozen_batch.pad_mask,
                    )
                for local_index, query_index in enumerate(range(query_start, query_stop)):
                    if bool(batch.reset_mask[episode_index, query_index].item()):
                        memory_state = self.initial_memory_state(1, device, dtype)
                    processed_state = processed_states[local_index : local_index + 1]
                    frozen = FrozenPrefix(
                        hidden=frozen_batch.hidden[local_index : local_index + 1],
                        pad_mask=frozen_batch.pad_mask[local_index : local_index + 1],
                        kv_cache=slice_cache_batch(frozen_batch.kv_cache, local_index),
                    )
                    last_vlm = last_vlm_batch[local_index : local_index + 1]
                    prefix_cache = prefix_cache_batch.batch_slice(local_index)
                    if bool(train_query_mask[episode_index, query_index].item()):
                        memory_token, memory_state = self.futuremamba.compute_memory_token(last_vlm, memory_state)
                    else:
                        with torch.no_grad():
                            _, memory_state = self.futuremamba.compute_memory_token(last_vlm, memory_state)
                        continue
                    current_x = x_t[episode_index, query_index : query_index + 1]
                    current_time = time[episode_index, query_index : query_index + 1]
                    pred_velocity = self.futuremamba.forward_progress(
                        prefix_cache, frozen.pad_mask, memory_token, current_x, current_time
                    )
                    flow_error[episode_index, query_index] = torch.mean(
                        torch.square(pred_velocity - target_velocity[episode_index, query_index : query_index + 1]),
                        dim=-1,
                    ).squeeze(0)

                    terminal_selected = bool(terminal_query_mask[episode_index, query_index].item())
                    if handoff_steps > 0 and (
                        (terminal_weight > 0.0 and terminal_selected)
                        or handoff_weight > 0.0
                        or boundary_weight > 0.0
                    ):
                        boundary_state = safe_noise[episode_index, query_index : query_index + 1]
                        dt = torch.tensor(
                            -1.0 / int(self.config.num_denoise_steps), dtype=torch.float32, device=device
                        )
                        for step in range(handoff_steps):
                            step_time = torch.full(
                                (1,), 1.0 + step * float(dt.item()), dtype=torch.float32, device=device
                            )
                            boundary_state = boundary_state + dt.to(
                                dtype=boundary_state.dtype
                            ) * self.futuremamba.forward_progress(
                                prefix_cache, frozen.pad_mask, memory_token, boundary_state, step_time
                            )
                        boundary_time_value = 1.0 - handoff_steps / int(self.config.num_denoise_steps)
                        boundary_time = torch.full(
                            (1,), boundary_time_value, dtype=torch.float32, device=device
                        )
                        if terminal_weight > 0.0 and terminal_selected:
                            terminal_state = boundary_state
                            checkpoint_action_expert = bool(
                                getattr(self.config, "action_expert_gradient_checkpointing", False)
                            )
                            for step in range(handoff_steps, int(self.config.num_denoise_steps)):
                                step_time = torch.full(
                                    (1,), 1.0 + step * float(dt.item()), dtype=torch.float32, device=device
                                )
                                if checkpoint_action_expert:
                                    terminal_velocity = torch.utils.checkpoint.checkpoint(
                                        lambda current: self.base.denoise_step(
                                            processed_state,
                                            frozen.pad_mask,
                                            frozen.kv_cache,
                                            current,
                                            step_time,
                                        ),
                                        terminal_state,
                                        use_reentrant=False,
                                        preserve_rng_state=False,
                                    )
                                else:
                                    terminal_velocity = self.base.denoise_step(
                                        processed_state,
                                        frozen.pad_mask,
                                        frozen.kv_cache,
                                        terminal_state,
                                        step_time,
                                    )
                                terminal_state = terminal_state + dt.to(
                                    dtype=terminal_state.dtype
                                ) * terminal_velocity
                            terminal_error_steps[episode_index, query_index] = torch.mean(
                                torch.square(
                                    terminal_state - safe_actions[episode_index, query_index : query_index + 1]
                                ),
                                dim=-1,
                            ).squeeze(0)
                        if handoff_weight > 0.0:
                            target_boundary = (
                                boundary_time[:, None, None].to(dtype=dtype)
                                * safe_noise[episode_index, query_index : query_index + 1]
                                + (1.0 - boundary_time[:, None, None].to(dtype=dtype))
                                * safe_actions[episode_index, query_index : query_index + 1]
                            )
                            handoff_error_steps[episode_index, query_index] = torch.mean(
                                torch.square(boundary_state - target_boundary), dim=-1
                            ).squeeze(0)
                        if boundary_weight > 0.0:
                            detached_boundary = boundary_state.detach()
                            progress_boundary = self.futuremamba.forward_progress(
                                prefix_cache,
                                frozen.pad_mask,
                                memory_token,
                                detached_boundary,
                                boundary_time,
                            )
                            action_boundary = self.base.denoise_step(
                                processed_state,
                                frozen.pad_mask,
                                frozen.kv_cache,
                                detached_boundary,
                                boundary_time,
                            ).detach()
                            boundary_error_steps[episode_index, query_index] = torch.mean(
                                torch.square(progress_boundary - action_boundary), dim=-1
                            ).squeeze(0)

        flow_loss = _mean_masked_action_error(flow_error, batch.action_mask, train_query_mask)
        if terminal_weight > 0.0:
            terminal_error = _mean_masked_action_error(
                terminal_error_steps[:terminal_episode_limit],
                batch.action_mask[:terminal_episode_limit],
                terminal_query_mask[:terminal_episode_limit],
            )
        if handoff_weight > 0.0:
            handoff_error = _mean_masked_action_error(handoff_error_steps, batch.action_mask, train_query_mask)
        if boundary_weight > 0.0:
            boundary_error = _mean_masked_action_error(boundary_error_steps, batch.action_mask, train_query_mask)
        terminal_loss = terminal_error * torch.as_tensor(terminal_weight, dtype=dtype, device=device)
        handoff_loss = handoff_error * torch.as_tensor(handoff_weight, dtype=dtype, device=device)
        boundary_loss = boundary_error * torch.as_tensor(boundary_weight, dtype=dtype, device=device)
        loss = flow_loss + terminal_loss + handoff_loss + boundary_loss
        sampled_time = time[train_query_mask]
        return {
            "loss": loss,
            "flow_loss": flow_loss,
            "terminal_loss": terminal_loss,
            "terminal_error": terminal_error,
            "handoff_loss": handoff_loss,
            "handoff_error": handoff_error,
            "boundary_loss": boundary_loss,
            "boundary_error": boundary_error,
            "sample_time_mean": sampled_time.mean(),
            "sample_time_min": sampled_time.min(),
            "sample_time_max": sampled_time.max(),
        }

    def _sample_high_noise_time(self, shape: tuple[int, ...], *, handoff_steps: int, device: torch.device) -> torch.Tensor:
        lower = 1.0 - (float(handoff_steps) / float(self.config.num_denoise_steps))
        lower = min(max(lower, 0.0), 1.0)
        alpha = torch.as_tensor(1.5, dtype=torch.float32, device=device)
        beta = torch.as_tensor(1.0, dtype=torch.float32, device=device)
        sample = torch.distributions.Beta(alpha, beta).sample(shape)
        return lower + (1.0 - lower) * sample

    def _detached_frozen_prefix(self, frozen: FrozenPrefix) -> FrozenPrefix:
        return FrozenPrefix(
            hidden=frozen.hidden.detach(),
            pad_mask=frozen.pad_mask.detach(),
            kv_cache=detach_cache(frozen.kv_cache),
        )

    def _validate_sampling_inputs(
        self,
        observation,
        memory_state: MemorySnapshot,
        noise: torch.Tensor | None,
        num_steps: int,
        handoff_ratio: float,
    ) -> None:
        if not 0.0 <= handoff_ratio <= 1.0:
            raise ValueError(f"handoff_ratio must be in [0, 1], got {handoff_ratio}")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if not isinstance(memory_state, MemorySnapshot):
            raise ValueError("memory_state must be a MemorySnapshot")
        batch_size = int(observation.state.shape[0])
        expected_action_shape = (batch_size, int(self.config.action_horizon), int(self.config.action_dim))
        if noise is not None and tuple(noise.shape) != expected_action_shape:
            raise ValueError(f"noise must have shape {expected_action_shape}, got {tuple(noise.shape)}")
        if memory_state.batch_size != batch_size:
            raise ValueError("memory_state batch must match observation")


def _action_expert_width(config, fallback: int) -> int:
    try:
        return int(_gemma.get_config(config.action_expert_variant).width)
    except (KeyError, AttributeError):
        return fallback

def _vlm_width(config, fallback: int) -> int:
    configured = getattr(config, "vlm_width", None)
    if configured is not None:
        return int(configured)
    try:
        return int(_gemma.get_config(config.paligemma_variant).width)
    except (KeyError, AttributeError):
        return fallback


def _dtype_from_config(config) -> torch.dtype:
    value = getattr(config, "dtype", "float32")
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    return torch.float32


def _make_memory_backend(config) -> nn.Module:
    backend = config.memory_backend
    if backend == "gru":
        return GRUMemoryBackend(config.memory)
    if backend == "lstm":
        return LSTMMemoryBackend(config.memory)
    if backend == "frame_stack":
        return FrameStackMemoryBackend(config.memory)
    if backend == "none":
        return NoMemoryBackend(config.memory)
    if backend == "mamba2":
        return Mamba2MemoryBackend(config.memory)
    if backend == "mamba3_siso":
        raise NotImplementedError("memory_backend 'mamba3_siso' has no PyTorch MemoryBackend implementation")
    raise ValueError(f"unknown memory_backend {backend!r}")




def _handoff_steps(ratio: float, num_steps: int) -> int:
    return max(0, min(num_steps, int(math.ceil(ratio * num_steps))))


def _mean_masked_action_error(error: torch.Tensor, action_mask: torch.Tensor, query_mask: torch.Tensor) -> torch.Tensor:
    action_weights = action_mask.to(dtype=error.dtype)
    query_error = torch.where(action_mask, error, torch.zeros_like(error)).sum(dim=-1) / action_weights.sum(dim=-1).clamp_min(1.0)
    query_weights = query_mask.to(dtype=error.dtype)
    episode_error = (query_error * query_weights).sum(dim=-1) / query_weights.sum(dim=-1).clamp_min(1.0)
    return episode_error.mean()




def _slice_episode_observation(observation: Any, batch_index: int, query_index: int) -> Any:
    if observation is None:
        return None
    if isinstance(observation, Mapping):
        return {key: _slice_episode_observation(value, batch_index, query_index) for key, value in observation.items()}
    if hasattr(observation, "__dataclass_fields__"):
        import dataclasses

        return dataclasses.replace(
            observation,
            **{
                field.name: _slice_episode_observation(getattr(observation, field.name), batch_index, query_index)
                for field in dataclasses.fields(observation)
            },
        )
    if torch.is_tensor(observation):
        return observation[batch_index : batch_index + 1, query_index]
    return observation



def _slice_episode_observation_range(
    observation: Any, batch_index: int, query_start: int, query_stop: int
) -> Any:
    if observation is None:
        return None
    if isinstance(observation, Mapping):
        return {
            key: _slice_episode_observation_range(value, batch_index, query_start, query_stop)
            for key, value in observation.items()
        }
    if hasattr(observation, "__dataclass_fields__"):
        import dataclasses

        return dataclasses.replace(
            observation,
            **{
                field.name: _slice_episode_observation_range(
                    getattr(observation, field.name), batch_index, query_start, query_stop
                )
                for field in dataclasses.fields(observation)
            },
        )
    if torch.is_tensor(observation):
        return observation[batch_index, query_start:query_stop]
    return observation