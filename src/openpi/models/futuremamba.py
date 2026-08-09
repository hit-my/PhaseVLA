from flax import nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import futuremamba_config
from openpi.models import gemma as _gemma
from openpi.models.mamba import SelectiveMamba
from openpi.models.pi0 import Pi0
from openpi.models.progress_expert import ProgressExpert
import openpi.shared.array_typing as at


class _FutureMambaPlugin(nnx.Module):
    def __init__(self, config: futuremamba_config.FutureMambaConfig, *, rngs: nnx.Rngs):
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.config = config
        self.vlm_memory_in_proj = nnx.Linear(action_expert_config.width, config.memory.d_model, rngs=rngs)
        self.executed_action_encoder = nnx.Linear(config.action_dim, config.memory.d_model, rngs=rngs)
        self.memory = SelectiveMamba(config.memory, rngs=rngs)
        self.memory_token_proj = nnx.Linear(config.memory.d_model, action_expert_config.width, rngs=rngs)

        progress_expert = nnx_bridge.ToNNX(
            ProgressExpert(
                action_expert_config=action_expert_config,
                action_dim=config.action_dim,
                action_horizon=config.action_horizon,
                progress_depth=config.progress_depth,
                layer_mapping_override=config.resolved_progress_layer_indices,
                embed_dtype=config.dtype,
            )
        )
        batch_size = 1
        prefix_len = 1
        prefix_kv_cache = (
            jnp.zeros(
                (
                    action_expert_config.depth,
                    batch_size,
                    prefix_len,
                    action_expert_config.num_kv_heads,
                    action_expert_config.head_dim,
                ),
                dtype=jnp.dtype(config.dtype),
            ),
            jnp.zeros(
                (
                    action_expert_config.depth,
                    batch_size,
                    prefix_len,
                    action_expert_config.num_kv_heads,
                    action_expert_config.head_dim,
                ),
                dtype=jnp.dtype(config.dtype),
            ),
        )
        prefix_mask = jnp.ones((batch_size, prefix_len), dtype=jnp.bool_)
        memory_token = jnp.zeros((batch_size, 1, action_expert_config.width), dtype=jnp.dtype(config.dtype))
        noisy_actions = jnp.zeros((batch_size, config.action_horizon, config.action_dim), dtype=jnp.float32)
        timestep = jnp.zeros((batch_size,), dtype=jnp.float32)
        progress_expert.lazy_init(prefix_kv_cache, prefix_mask, memory_token, noisy_actions, timestep, rngs=rngs)
        self.progress_expert = progress_expert

    def vlm_to_memory(self, prefix_out: jax.Array) -> jax.Array:
        return self.vlm_memory_in_proj(prefix_out)

    def encode_executed_actions(self, actions: jax.Array) -> jax.Array:
        return self.executed_action_encoder(actions)

    def project_memory_token(self, memory_token: jax.Array) -> jax.Array:
        return self.memory_token_proj(memory_token)


class FutureMamba(Pi0):
    def __init__(self, config: futuremamba_config.FutureMambaConfig, rngs: nnx.Rngs):
        if config.memory_backend != "mamba":
            raise NotImplementedError(f"FutureMamba memory_backend={config.memory_backend!r} is not implemented")
        if config.decoder_mode != "handoff":
            raise NotImplementedError(f"FutureMamba decoder_mode={config.decoder_mode!r} is not implemented")
        super().__init__(config, rngs)
        self.futuremamba = _FutureMambaPlugin(config, rngs=rngs)

    @at.typecheck
    def compute_loss(self, rng: at.KeyArrayLike, observation, actions, *, train: bool = False):
        return super().compute_loss(rng, observation, actions, train=train)

    def sample_actions(self, rng: at.KeyArrayLike, observation, **kwargs):
        return super().sample_actions(rng, observation, **kwargs)
