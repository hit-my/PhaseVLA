from collections.abc import Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.gemma as _gemma
from openpi.models.pi0 import posemb_sincos
import openpi.shared.array_typing as at


def _validate_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def default_progress_depth(action_depth: int) -> int:
    """Returns the default lightweight Progress Expert depth for an action expert depth."""
    _validate_positive_int("action_depth", action_depth)
    if action_depth == 1:
        return 1
    return min(action_depth, max(2, int(round(action_depth / 4))))


def make_layer_mapping(action_depth: int, progress_depth: int) -> tuple[int, ...]:
    """Evenly maps progress layers to zero-based frozen action/VLM prefix-cache layers."""
    _validate_positive_int("action_depth", action_depth)
    _validate_positive_int("progress_depth", progress_depth)
    if progress_depth > action_depth:
        raise ValueError(f"progress_depth ({progress_depth}) must be <= action_depth ({action_depth})")
    if progress_depth == 1:
        return (0,)
    return tuple((index * (action_depth - 1)) // (progress_depth - 1) for index in range(progress_depth))


@at.typecheck
class _ProgressStack(nn.Module):
    config: _gemma.Config
    layer_mapping: tuple[int, ...]
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @nn.compact
    def __call__(
        self,
        action_tokens: at.Float[at.Array, "b a d"],
        memory_token: at.Float[at.Array, "b 1 d"],
        prefix_kv_cache: _gemma.KVCache,
        memory_positions: at.Int[at.Array, "b 1"],
        action_positions: at.Int[at.Array, "b a"],
        attn_mask: at.Bool[at.Array, "b 1 a s"],
        adarms_cond: at.Float[at.Array, "b d"],
        deterministic: bool = True,  # noqa: FBT002
    ) -> at.Float[at.Array, "b a d"]:
        prefix_k, prefix_v = prefix_kv_cache
        for progress_index, action_layer_index in enumerate(self.layer_mapping):
            action_tokens = _gemma.CachedPrefixActionBlock(
                config=self.config,
                dropout=self.dropout,
                dropout_bdims=self.dropout_bdims,
                name=f"layers_{progress_index}",
            )(
                action_tokens,
                memory_token,
                (prefix_k[action_layer_index], prefix_v[action_layer_index]),
                memory_positions,
                action_positions,
                attn_mask,
                adarms_cond,
                deterministic,
            )
        return action_tokens


@at.typecheck
class ProgressExpert(nn.Module):
    """Lightweight action-only Gemma expert conditioned on a frozen per-layer prefix KV cache."""

    action_expert_config: _gemma.Config
    action_dim: int
    action_horizon: int
    progress_depth: int | None = None
    layer_mapping_override: Sequence[int] | None = None
    embed_dtype: str = "bfloat16"
    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()

    @property
    def resolved_progress_depth(self) -> int:
        depth = default_progress_depth(self.action_expert_config.depth) if self.progress_depth is None else self.progress_depth
        return _validate_positive_int("progress_depth", depth)

    @property
    def layer_mapping(self) -> tuple[int, ...]:
        if self.layer_mapping_override is None:
            mapping = make_layer_mapping(self.action_expert_config.depth, self.resolved_progress_depth)
        else:
            mapping = tuple(int(index) for index in self.layer_mapping_override)
        if len(mapping) != self.resolved_progress_depth:
            raise ValueError(
                f"layer_mapping length ({len(mapping)}) must equal progress_depth ({self.resolved_progress_depth})"
            )
        if any(index < 0 or index >= self.action_expert_config.depth for index in mapping):
            raise ValueError(f"layer_mapping entries must be in [0, {self.action_expert_config.depth}), got {mapping}")
        if any(left >= right for left, right in zip(mapping, mapping[1:])):
            raise ValueError(f"layer_mapping must be strictly increasing, got {mapping}")
        if len(mapping) > 1 and (mapping[0] != 0 or mapping[-1] != self.action_expert_config.depth - 1):
            raise ValueError(
                "layer_mapping must cover the first and last action layers, "
                f"got {mapping} for depth {self.action_expert_config.depth}"
            )
        return mapping

    def checkpoint_metadata(self) -> dict[str, int | tuple[int, ...]]:
        return {"progress_depth": self.resolved_progress_depth, "layer_mapping": self.layer_mapping}

    @nn.compact
    @at.typecheck
    def __call__(
        self,
        prefix_kv_cache: _gemma.KVCache,
        prefix_mask: at.Bool[at.Array, "b c"],
        memory_token: at.Float[at.Array, "b 1 d"],
        noisy_actions: at.Float[at.Array, "b h a"],
        timestep: at.Float[at.Array, "b"],
        *,
        use_prefix_cache: bool = True,
        deterministic: bool = True,
    ) -> at.Float[at.Array, "b h a"]:
        if noisy_actions.shape[1] != self.action_horizon:
            raise ValueError(f"noisy_actions horizon must be {self.action_horizon}, got {noisy_actions.shape[1]}")
        if noisy_actions.shape[2] != self.action_dim:
            raise ValueError(f"noisy_actions action_dim must be {self.action_dim}, got {noisy_actions.shape[2]}")
        if memory_token.shape[1] != 1:
            raise ValueError(f"memory_token must contain exactly one token, got shape {memory_token.shape}")

        prefix_enabled = jnp.asarray(use_prefix_cache, dtype=jnp.bool_)
        prefix_lengths = jnp.where(prefix_enabled, jnp.sum(prefix_mask.astype(jnp.int32), axis=-1), 0)
        memory_positions = prefix_lengths[:, None]
        action_positions = prefix_lengths[:, None] + 1 + jnp.arange(self.action_horizon, dtype=jnp.int32)[None, :]

        prefix_attn_mask = jnp.logical_and(prefix_mask[:, None, :], prefix_enabled)
        prefix_attn_mask = jnp.broadcast_to(prefix_attn_mask, (prefix_mask.shape[0], self.action_horizon, prefix_mask.shape[1]))
        memory_attn_mask = jnp.ones((prefix_mask.shape[0], self.action_horizon, 1), dtype=jnp.bool_)
        action_attn_mask = jnp.ones((prefix_mask.shape[0], self.action_horizon, self.action_horizon), dtype=jnp.bool_)
        attn_mask = jnp.concatenate([prefix_attn_mask, memory_attn_mask, action_attn_mask], axis=-1)[:, None, :, :]

        action_tokens = nn.Dense(self.action_expert_config.width, name="action_in_proj")(noisy_actions)
        action_tokens = action_tokens.astype(self.embed_dtype)
        memory_token = memory_token.astype(self.embed_dtype)

        time_emb = posemb_sincos(timestep, self.action_expert_config.width, min_period=4e-3, max_period=4.0)
        time_emb = nn.Dense(self.action_expert_config.width, name="time_mlp_in")(time_emb)
        time_emb = nn.swish(time_emb)
        time_emb = nn.Dense(self.action_expert_config.width, name="time_mlp_out")(time_emb)
        adarms_cond = nn.swish(time_emb).astype(self.embed_dtype)

        action_tokens = _ProgressStack(
            config=self.action_expert_config,
            layer_mapping=self.layer_mapping,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
            name="progress",
        )(
            action_tokens,
            memory_token,
            jax.tree.map(jax.lax.stop_gradient, prefix_kv_cache),
            memory_positions,
            action_positions,
            attn_mask,
            adarms_cond,
            deterministic,
        )
        action_tokens = _gemma.RMSNorm(name="final_norm")(action_tokens, adarms_cond)[0]
        velocity = nn.Dense(self.action_dim, name="action_out_proj")(action_tokens)
        return velocity.astype(noisy_actions.dtype)
