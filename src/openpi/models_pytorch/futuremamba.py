from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import math
from typing import Any

import torch
from torch import nn

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PrefixKVView, detach_cache
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


@dataclasses.dataclass(frozen=True)
class ActionHistoryState:
    """Committed 20-action Mamba state plus one uncommitted right-padded chunk."""

    memory: MemorySnapshot
    committed_output: torch.Tensor
    committed_chunks: torch.LongTensor
    pending_actions: torch.Tensor
    pending_mask: torch.BoolTensor


class FutureMambaPluginPytorch(nn.Module):
    """Executed-action Mamba conditioning the early-denoising Progress Expert."""

    def __init__(
        self,
        config,
        *,
        memory_backend: nn.Module | None = None,
        progress_expert: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.chunk_size = int(config.action_history_chunk_size)
        self.action_dim = int(config.action_dim)
        self.memory_width = int(config.memory.d_model)
        self.action_expert_width = _action_expert_width(config, self.memory_width)
        self.progress_layer_indices = tuple(config.resolved_progress_layer_indices)
        flattened_width = self.chunk_size * (self.action_dim + 1)
        self.action_chunk_projection = nn.Sequential(
            nn.Linear(flattened_width, self.memory_width),
            nn.SiLU(),
            nn.Linear(self.memory_width, self.memory_width),
        )
        self.empty_history = nn.Parameter(torch.zeros(self.memory_width))
        self.memory_backend = memory_backend if memory_backend is not None else _make_memory_backend(config)
        self.memory_token_projection = nn.Linear(self.memory_width, self.action_expert_width)
        self.progress_expert = progress_expert if progress_expert is not None else ProgressExpertPytorch(config)
        self.progress_num_key_value_heads = _action_expert_num_key_value_heads(config, 1)
        self.progress_head_dim = _action_expert_head_dim(
            config, max(1, self.action_expert_width // self.progress_num_key_value_heads)
        )
        self.to(dtype=_dtype_from_config(config))

    def initial_history_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> ActionHistoryState:
        memory = self.memory_backend.initial_state(batch_size, device=device, dtype=dtype)
        empty = self.empty_history.to(device=device, dtype=dtype)[None].expand(batch_size, -1).clone()
        return ActionHistoryState(
            memory=memory,
            committed_output=empty,
            committed_chunks=torch.zeros(batch_size, dtype=torch.long, device=device),
            pending_actions=torch.zeros(
                batch_size, self.chunk_size, self.action_dim, dtype=dtype, device=device
            ),
            pending_mask=torch.zeros(batch_size, self.chunk_size, dtype=torch.bool, device=device),
        )

    def encode_action_chunk(
        self, actions: torch.Tensor, action_mask: torch.BoolTensor
    ) -> torch.Tensor:
        self._validate_chunk(actions, action_mask)
        mask = action_mask.to(device=actions.device, dtype=torch.bool)
        values = actions.to(dtype=self.action_chunk_projection[0].weight.dtype)
        values = torch.where(mask[..., None], values, torch.zeros_like(values))
        features = torch.cat((values, mask.to(values.dtype)[..., None]), dim=-1)
        return self.action_chunk_projection(features.flatten(start_dim=1))

    def advance_history(
        self,
        history: ActionHistoryState,
        executed_actions: torch.Tensor,
        executed_action_mask: torch.BoolTensor,
    ) -> tuple[torch.Tensor, ActionHistoryState, dict[str, int]]:
        """Append new actions and expose all prior actions without double-counting partial chunks."""
        self._validate_history(history)
        self._validate_executed_actions(executed_actions, executed_action_mask, history)
        actions = executed_actions.to(
            device=history.pending_actions.device, dtype=history.pending_actions.dtype
        )
        mask = executed_action_mask.to(device=actions.device, dtype=torch.bool)
        memory = history.memory
        committed_output = history.committed_output
        committed_chunks = history.committed_chunks
        pending_actions = history.pending_actions
        pending_mask = history.pending_mask
        commit_count = 0

        for action_index in range(actions.shape[1]):
            valid_rows = mask[:, action_index]
            if not bool(valid_rows.any().item()):
                continue
            counts = pending_mask.long().sum(dim=-1)
            if bool(torch.any(valid_rows & (counts >= self.chunk_size)).item()):
                raise RuntimeError("pending action-history chunk overflow")
            next_actions = pending_actions.clone()
            next_mask = pending_mask.clone()
            rows = torch.nonzero(valid_rows, as_tuple=False).flatten()
            next_actions[rows, counts[rows]] = actions[rows, action_index]
            next_mask[rows, counts[rows]] = True
            pending_actions, pending_mask = next_actions, next_mask

            full_rows = pending_mask.all(dim=-1)
            if bool(full_rows.any().item()):
                encoded = self.encode_action_chunk(pending_actions, pending_mask)
                candidate_output, candidate_memory = self.memory_backend.step(encoded, memory)
                memory = _merge_memory_rows(memory, candidate_memory, full_rows)
                committed_output = torch.where(
                    full_rows[:, None], candidate_output.to(committed_output.dtype), committed_output
                )
                committed_chunks = committed_chunks + full_rows.long()
                pending_actions = torch.where(
                    full_rows[:, None, None], torch.zeros_like(pending_actions), pending_actions
                )
                pending_mask = torch.where(
                    full_rows[:, None], torch.zeros_like(pending_mask), pending_mask
                )
                commit_count += int(full_rows.long().sum().item())

        partial_rows = pending_mask.any(dim=-1)
        memory_output = committed_output
        if bool(partial_rows.any().item()):
            encoded = self.encode_action_chunk(pending_actions, pending_mask)
            partial_output, _ = self.memory_backend.step(encoded, memory)
            memory_output = torch.where(
                partial_rows[:, None], partial_output.to(memory_output.dtype), memory_output
            )
        has_history = partial_rows | (committed_chunks > 0)
        empty = self.empty_history.to(device=memory_output.device, dtype=memory_output.dtype)[None]
        memory_output = torch.where(has_history[:, None], memory_output, empty)
        memory_token = self.memory_token_projection(
            memory_output.to(dtype=self.memory_token_projection.weight.dtype)
        ).unsqueeze(1)
        next_history = ActionHistoryState(
            memory=memory,
            committed_output=committed_output,
            committed_chunks=committed_chunks,
            pending_actions=pending_actions,
            pending_mask=pending_mask,
        )
        history_actions = committed_chunks * self.chunk_size + pending_mask.long().sum(dim=-1)
        return memory_token, next_history, {
            "memory_commits": commit_count,
            "history_actions": int(history_actions.sum().item()),
            "pending_actions": int(pending_mask.long().sum().item()),
        }

    def progress_prefix_from_cache(
        self, prefix_cache: object, prefix_mask: torch.BoolTensor
    ) -> tuple[PrefixKVView, torch.BoolTensor]:
        if prefix_mask.ndim != 2 or prefix_mask.dtype is not torch.bool:
            raise ValueError("prefix_mask must have shape [batch, prefix] and bool dtype")
        if prefix_mask.shape[1] == 0:
            return self._empty_progress_prefix(prefix_mask.shape[0], prefix_mask.device), prefix_mask
        return PrefixKVView.from_cache(prefix_cache, self.progress_layer_indices, prefix_mask), prefix_mask

    def forward_progress(
        self,
        prefix_cache: PrefixKVView,
        prefix_mask: torch.BoolTensor,
        memory_token: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        return self.progress_expert(
            prefix_cache, prefix_mask, memory_token, noisy_actions, timestep
        )

    def _empty_progress_prefix(self, batch_size: int, device: torch.device) -> PrefixKVView:
        dtype = self.memory_token_projection.weight.dtype
        layers = tuple(
            (
                torch.empty(
                    batch_size,
                    self.progress_num_key_value_heads,
                    0,
                    self.progress_head_dim,
                    dtype=dtype,
                    device=device,
                ),
                torch.empty(
                    batch_size,
                    self.progress_num_key_value_heads,
                    0,
                    self.progress_head_dim,
                    dtype=dtype,
                    device=device,
                ),
            )
            for _ in self.progress_layer_indices
        )
        return PrefixKVView(
            layers=layers,
            valid_lengths=torch.zeros(batch_size, dtype=torch.long, device=device),
        )

    def _validate_chunk(self, actions: torch.Tensor, mask: torch.Tensor) -> None:
        expected = (actions.shape[0], self.chunk_size, self.action_dim)
        if actions.ndim != 3 or tuple(actions.shape) != expected:
            raise ValueError(f"action-history chunk must have shape {expected}, got {tuple(actions.shape)}")
        if mask.shape != actions.shape[:2] or mask.dtype is not torch.bool:
            raise ValueError("action-history mask shape or dtype is invalid")
        counts = mask.long().sum(dim=-1)
        expected_mask = torch.arange(self.chunk_size, device=mask.device)[None] < counts[:, None]
        if not torch.equal(mask, expected_mask):
            raise ValueError("action-history mask must be a right-padded prefix")

    def _validate_history(self, history: ActionHistoryState) -> None:
        if not isinstance(history, ActionHistoryState):
            raise TypeError("history must be ActionHistoryState")
        batch = history.memory.batch_size
        if history.committed_output.shape != (batch, self.memory_width):
            raise ValueError("committed_output shape mismatch")
        if history.committed_chunks.shape != (batch,) or history.committed_chunks.dtype is not torch.long:
            raise ValueError("committed_chunks shape or dtype mismatch")
        self._validate_chunk(history.pending_actions, history.pending_mask)
        if history.pending_actions.shape[0] != batch:
            raise ValueError("pending history batch mismatch")

    def _validate_executed_actions(
        self,
        actions: torch.Tensor,
        mask: torch.Tensor,
        history: ActionHistoryState,
    ) -> None:
        if actions.ndim != 3 or actions.shape[0] != history.memory.batch_size:
            raise ValueError("executed_actions must have shape [batch, time, action_dim]")
        if actions.shape[1] > self.chunk_size or actions.shape[2] != self.action_dim:
            raise ValueError(
                f"executed_actions must fit [batch, <= {self.chunk_size}, {self.action_dim}], got {tuple(actions.shape)}"
            )
        if mask.shape != actions.shape[:2] or mask.dtype is not torch.bool:
            raise ValueError("executed_action_mask shape or dtype is invalid")
        counts = mask.long().sum(dim=-1)
        expected = torch.arange(actions.shape[1], device=mask.device)[None] < counts[:, None]
        if not torch.equal(mask, expected):
            raise ValueError("executed_action_mask must be a right-padded prefix")


class FutureMambaPytorch(nn.Module):
    """Frozen visual VLA/Action Expert with PE-to-AE denoising handoff."""

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
            digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def initial_history_state(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> ActionHistoryState:
        return self.futuremamba.initial_history_state(batch_size, device, dtype)

    def sample_actions_with_memory(
        self,
        observation,
        history: ActionHistoryState,
        executed_actions: torch.Tensor,
        executed_action_mask: torch.BoolTensor,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
        handoff_ratio: float | None = None,
    ) -> tuple[torch.Tensor, ActionHistoryState, dict[str, int]]:
        num_steps = int(self.config.num_denoise_steps if num_steps is None else num_steps)
        ratio = float(self.config.handoff_ratio if handoff_ratio is None else handoff_ratio)
        self._validate_sampling_inputs(
            observation, history, executed_actions, executed_action_mask, noise, num_steps, ratio
        )
        device = observation.state.device
        batch_size = int(observation.state.shape[0])
        if noise is None:
            noise = self.base.sample_noise(
                (batch_size, int(self.config.action_horizon), int(self.config.action_dim)), device
            )
        memory_token, next_history, history_diagnostics = self.futuremamba.advance_history(
            history, executed_actions, executed_action_mask
        )

        with torch.no_grad():
            *_, processed_state = self.base._preprocess_observation(observation, train=False)
            frozen = self._detached_frozen_prefix(
                self.base.extract_prefix_context(observation, train=False)
            )
            prefix_cache, progress_prefix_mask = self.futuremamba.progress_prefix_from_cache(
                frozen.kv_cache, frozen.pad_mask
            )
            dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
            handoff_steps = _handoff_steps(ratio, num_steps)
            x_t = noise
            progress_calls = 0
            action_calls = 0
            for step in range(num_steps):
                timestep = torch.full(
                    (batch_size,),
                    1.0 + step * float(dt.item()),
                    dtype=torch.float32,
                    device=device,
                )
                if step < handoff_steps:
                    velocity = self.futuremamba.forward_progress(
                        prefix_cache, progress_prefix_mask, memory_token, x_t, timestep
                    )
                    progress_calls += 1
                else:
                    velocity = self.base.action_expert_velocity(
                        processed_state, frozen.pad_mask, frozen.kv_cache, x_t, timestep
                    )
                    action_calls += 1
                x_t = x_t + dt.to(dtype=x_t.dtype) * velocity
        return x_t, next_history, {
            "progress_calls": progress_calls,
            "action_calls": action_calls,
            "handoff_steps": handoff_steps,
            **history_diagnostics,
        }

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
        dtype = actions.dtype
        if noise is None:
            noise = self.base.sample_noise(tuple(actions.shape), device)
        else:
            noise = noise.to(device=device, dtype=dtype)
        if tuple(noise.shape) != tuple(actions.shape):
            raise ValueError(f"noise must have shape {tuple(actions.shape)}, got {tuple(noise.shape)}")
        handoff_steps = _handoff_steps(
            float(self.config.handoff_ratio), int(self.config.num_denoise_steps)
        )
        if time is None:
            time = self._sample_high_noise_time(
                (batch_size, num_queries), handoff_steps=handoff_steps, device=device
            )
        else:
            time = time.to(device=device, dtype=torch.float32)
        if tuple(time.shape) != (batch_size, num_queries):
            raise ValueError(f"time must have shape {(batch_size, num_queries)}, got {tuple(time.shape)}")

        train_query_mask = batch.train_query_mask
        if train_query_mask is None:
            raise ValueError("train_query_mask must be materialized before computing episode loss")
        valid_action_mask = batch.action_mask & train_query_mask[:, :, None]
        safe_actions = torch.where(valid_action_mask[..., None], actions, torch.zeros_like(actions))
        safe_noise = torch.where(valid_action_mask[..., None], noise, torch.zeros_like(noise))
        x_t = time[..., None, None] * safe_noise + (1.0 - time[..., None, None]) * safe_actions
        target_velocity = safe_noise - safe_actions
        flow_error = torch.zeros(
            batch_size, num_queries, int(self.config.action_horizon), dtype=dtype, device=device
        )

        history_tokens: list[torch.Tensor] = []
        for episode_index in range(batch_size):
            valid_queries = int(batch.query_mask[episode_index].long().sum().item())
            history = self.initial_history_state(1, device, next(self.futuremamba.parameters()).dtype)
            episode_tokens = []
            for query_index in range(valid_queries):
                if bool(batch.reset_mask[episode_index, query_index].item()):
                    history = self.initial_history_state(
                        1, device, next(self.futuremamba.parameters()).dtype
                    )
                if bool(train_query_mask[episode_index, query_index].item()):
                    token, history, _ = self.futuremamba.advance_history(
                        history,
                        batch.executed_actions[episode_index, query_index : query_index + 1],
                        batch.executed_action_mask[episode_index, query_index : query_index + 1],
                    )
                else:
                    with torch.no_grad():
                        token, history, _ = self.futuremamba.advance_history(
                            history,
                            batch.executed_actions[episode_index, query_index : query_index + 1],
                            batch.executed_action_mask[episode_index, query_index : query_index + 1],
                        )
                    history = _detach_history(history)
                episode_tokens.append(token)
            if valid_queries < num_queries:
                zero = torch.zeros(
                    1,
                    1,
                    self.futuremamba.action_expert_width,
                    dtype=episode_tokens[0].dtype,
                    device=device,
                )
                episode_tokens.extend([zero] * (num_queries - valid_queries))
            history_tokens.append(torch.cat(episode_tokens, dim=0))
        memory_tokens = torch.stack(history_tokens, dim=0)

        if batch.conditioning_cache is not None:
            cache = batch.conditioning_cache
            prefix_microbatch_size = int(self.config.frozen_prefix_microbatch_size)
            for episode_index in range(batch_size):
                valid_queries = int(batch.query_mask[episode_index].long().sum().item())
                for query_start in range(0, valid_queries, prefix_microbatch_size):
                    query_stop = min(query_start + prefix_microbatch_size, valid_queries)
                    selected_mask = train_query_mask[episode_index, query_start:query_stop]
                    if not bool(selected_mask.any().item()):
                        continue
                    local_indices = torch.nonzero(selected_mask, as_tuple=False).flatten()
                    query_indices = query_start + local_indices
                    prefix_mask = cache["prefix_mask"][episode_index, query_indices].to(torch.bool)
                    keys = cache["action_expert_keys"][episode_index, query_indices]
                    values = cache["action_expert_values"][episode_index, query_indices]
                    layers = tuple((keys[:, layer], values[:, layer]) for layer in range(keys.shape[1]))
                    prefix_cache = PrefixKVView.from_layers(layers, prefix_mask)
                    predicted = self.futuremamba.forward_progress(
                        prefix_cache,
                        prefix_mask,
                        memory_tokens[episode_index, query_indices],
                        x_t[episode_index, query_indices],
                        time[episode_index, query_indices],
                    )
                    flow_error[episode_index, query_indices] = torch.mean(
                        torch.square(predicted - target_velocity[episode_index, query_indices]),
                        dim=-1,
                    )
        else:
            prefix_microbatch_size = int(self.config.frozen_prefix_microbatch_size)
            for episode_index in range(batch_size):
                valid_queries = int(batch.query_mask[episode_index].long().sum().item())
                for query_start in range(0, valid_queries, prefix_microbatch_size):
                    query_stop = min(query_start + prefix_microbatch_size, valid_queries)
                    observation_batch = _slice_episode_observation_range(
                        batch.observation, episode_index, query_start, query_stop
                    )
                    with torch.no_grad():
                        frozen = self._detached_frozen_prefix(
                            self.base.encode_frozen_prefix(observation_batch, train=train)
                        )
                        prefix_cache, prefix_mask = self.futuremamba.progress_prefix_from_cache(
                            frozen.kv_cache, frozen.pad_mask
                        )
                    predicted = self.futuremamba.forward_progress(
                        prefix_cache,
                        prefix_mask,
                        memory_tokens[episode_index, query_start:query_stop],
                        x_t[episode_index, query_start:query_stop],
                        time[episode_index, query_start:query_stop],
                    )
                    flow_error[episode_index, query_start:query_stop] = torch.mean(
                        torch.square(
                            predicted - target_velocity[episode_index, query_start:query_stop]
                        ),
                        dim=-1,
                    )

        flow_loss = _mean_masked_action_error(
            flow_error, batch.action_mask, train_query_mask
        )
        zero = torch.zeros((), dtype=flow_loss.dtype, device=flow_loss.device)
        sampled_time = time[train_query_mask]
        return {
            "loss": flow_loss,
            "flow_loss": flow_loss,
            "terminal_loss": zero,
            "terminal_error": zero,
            "handoff_loss": zero,
            "handoff_error": zero,
            "boundary_loss": zero,
            "boundary_error": zero,
            "sample_time_mean": sampled_time.mean(),
            "sample_time_min": sampled_time.min(),
            "sample_time_max": sampled_time.max(),
        }

    def _sample_high_noise_time(
        self, shape: tuple[int, ...], *, handoff_steps: int, device: torch.device
    ) -> torch.Tensor:
        lower = 1.0 - float(handoff_steps) / float(self.config.num_denoise_steps)
        alpha = torch.as_tensor(1.5, dtype=torch.float32, device=device)
        beta = torch.as_tensor(1.0, dtype=torch.float32, device=device)
        return lower + (1.0 - lower) * torch.distributions.Beta(alpha, beta).sample(shape)

    @staticmethod
    def _detached_frozen_prefix(frozen: FrozenPrefix) -> FrozenPrefix:
        return FrozenPrefix(
            hidden=frozen.hidden.detach(),
            pad_mask=frozen.pad_mask.detach(),
            kv_cache=detach_cache(frozen.kv_cache),
        )

    def _validate_sampling_inputs(
        self,
        observation,
        history: ActionHistoryState,
        executed_actions: torch.Tensor,
        executed_action_mask: torch.Tensor,
        noise: torch.Tensor | None,
        num_steps: int,
        handoff_ratio: float,
    ) -> None:
        if not 0.0 <= handoff_ratio <= 1.0:
            raise ValueError(f"handoff_ratio must be in [0, 1], got {handoff_ratio}")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        batch_size = int(observation.state.shape[0])
        if history.memory.batch_size != batch_size:
            raise ValueError("history batch must match observation")
        expected = (batch_size, int(self.config.action_horizon), int(self.config.action_dim))
        if noise is not None and tuple(noise.shape) != expected:
            raise ValueError(f"noise must have shape {expected}, got {tuple(noise.shape)}")
        if executed_actions.shape[0] != batch_size or executed_action_mask.shape != executed_actions.shape[:2]:
            raise ValueError("executed action batch must match observation")


def _detach_history(history: ActionHistoryState) -> ActionHistoryState:
    return ActionHistoryState(
        memory=MemorySnapshot(
            history.memory.backend_id,
            history.memory.state_schema_version,
            history.memory.batch_size,
            tuple(tuple(t.detach() for t in layer) for layer in history.memory.layers),
        ),
        committed_output=history.committed_output.detach(),
        committed_chunks=history.committed_chunks.detach(),
        pending_actions=history.pending_actions.detach(),
        pending_mask=history.pending_mask.detach(),
    )


def _merge_memory_rows(
    previous: MemorySnapshot, candidate: MemorySnapshot, update_mask: torch.BoolTensor
) -> MemorySnapshot:
    layers = []
    for previous_layer, candidate_layer in zip(previous.layers, candidate.layers, strict=True):
        tensors = []
        for previous_tensor, candidate_tensor in zip(previous_layer, candidate_layer, strict=True):
            view = update_mask.reshape(update_mask.shape[0], *([1] * (previous_tensor.ndim - 1)))
            tensors.append(torch.where(view, candidate_tensor, previous_tensor))
        layers.append(tuple(tensors))
    return MemorySnapshot(previous.backend_id, previous.state_schema_version, previous.batch_size, tuple(layers))


def _handoff_steps(ratio: float, num_steps: int) -> int:
    return max(0, min(num_steps, int(math.ceil(ratio * num_steps))))


def _mean_masked_action_error(
    error: torch.Tensor, action_mask: torch.Tensor, query_mask: torch.Tensor
) -> torch.Tensor:
    action_weights = action_mask.to(dtype=error.dtype)
    query_error = torch.where(action_mask, error, torch.zeros_like(error)).sum(dim=-1)
    query_error = query_error / action_weights.sum(dim=-1).clamp_min(1.0)
    query_weights = query_mask.to(dtype=error.dtype)
    episode_error = (query_error * query_weights).sum(dim=-1) / query_weights.sum(dim=-1).clamp_min(1.0)
    return episode_error.mean()


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


def _action_expert_width(config, fallback: int) -> int:
    try:
        return int(_gemma.get_config(config.action_expert_variant).width)
    except (KeyError, AttributeError):
        return fallback


def _action_expert_num_key_value_heads(config, fallback: int) -> int:
    try:
        return int(_gemma.get_config(config.action_expert_variant).num_kv_heads)
    except (KeyError, AttributeError):
        return fallback


def _action_expert_head_dim(config, fallback: int) -> int:
    try:
        return int(_gemma.get_config(config.action_expert_variant).head_dim)
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
        raise NotImplementedError("memory_backend 'mamba3_siso' has no PyTorch implementation")
    raise ValueError(f"unknown memory_backend {backend!r}")
