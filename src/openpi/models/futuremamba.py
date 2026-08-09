from flax import nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp

from openpi.models import futuremamba_config
from openpi.models import gemma as _gemma
from openpi.models import model as _model
from openpi.models.mamba import SelectiveMamba
from openpi.models.pi0 import Pi0
from openpi.models.progress_expert import ProgressExpert
import openpi.shared.array_typing as at


def _batch_where(mask: jax.Array, if_true, if_false):
    def select(true_leaf, false_leaf):
        leaf_mask = mask
        while leaf_mask.ndim < true_leaf.ndim:
            leaf_mask = leaf_mask[..., None]
        return jnp.where(leaf_mask, true_leaf, false_leaf)

    return jax.tree.map(select, if_true, if_false)


def _masked_mean(tokens: jax.Array, mask: jax.Array) -> jax.Array:
    weights = mask.astype(tokens.dtype)[..., None]
    total = jnp.sum(jnp.where(weights.astype(jnp.bool_), tokens, 0), axis=-2)
    count = jnp.sum(weights, axis=-2)
    return total / jnp.maximum(count, 1)


def _last_valid(tokens: jax.Array, mask: jax.Array) -> jax.Array:
    if tokens.shape[-2] == 0:
        return jnp.zeros((*tokens.shape[:-2], tokens.shape[-1]), dtype=tokens.dtype)
    positions = jnp.arange(tokens.shape[-2], dtype=jnp.int32)
    indices = jnp.max(jnp.where(mask, positions, 0), axis=-1)
    count = jnp.sum(mask.astype(jnp.int32), axis=-1)
    gathered = jnp.take_along_axis(tokens, indices[..., None, None], axis=-2)[..., 0, :]
    return jnp.where((count > 0)[..., None], gathered, jnp.zeros_like(gathered))


def _mean_last_k_valid(tokens: jax.Array, mask: jax.Array, k: int) -> jax.Array:
    valid_index = jnp.cumsum(mask.astype(jnp.int32), axis=-1)
    valid_count = jnp.sum(mask.astype(jnp.int32), axis=-1)
    first_kept = jnp.maximum(valid_count - k, 0)
    keep = jnp.logical_and(mask, valid_index > first_kept[..., None])
    return _masked_mean(tokens, keep)


class _FutureMambaPlugin(nnx.Module):
    def __init__(self, config: futuremamba_config.FutureMambaConfig, *, rngs: nnx.Rngs):
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.config = config
        self.vlm_memory_in_proj = nnx.Linear(paligemma_config.width, config.memory.d_model, rngs=rngs)
        self.executed_action_encoder = nnx.Linear(config.action_dim, config.memory.d_model, rngs=rngs)
        self.executed_action_mlp = nnx.Linear(config.memory.d_model, config.memory.d_model, rngs=rngs)
        self.executed_action_fusion = nnx.Linear(2 * config.memory.d_model, config.memory.d_model, rngs=rngs)
        self.memory_input_fusion = nnx.Linear(2 * config.memory.d_model, config.memory.d_model, rngs=rngs)
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
        encoded = self.executed_action_encoder(actions)
        encoded = nnx.swish(encoded)
        encoded = self.executed_action_mlp(encoded)
        return nnx.swish(encoded)

    def summarize_executed_actions(self, actions: jax.Array, action_mask: jax.Array) -> jax.Array:
        if actions.shape[-2] == 0:
            return jnp.zeros((*actions.shape[:-2], self.config.memory.d_model), dtype=actions.dtype)
        encoded = self.encode_executed_actions(actions)
        masked_mean = _masked_mean(encoded, action_mask)
        last_valid = _last_valid(encoded, action_mask)
        return self.executed_action_fusion(jnp.concatenate([masked_mean, last_valid], axis=-1))

    def fuse_memory_input(self, prefix_input: jax.Array, action_summary: jax.Array) -> jax.Array:
        return self.memory_input_fusion(jnp.concatenate([prefix_input, action_summary], axis=-1))

    def project_memory_token(self, memory_token: jax.Array) -> jax.Array:
        return self.memory_token_proj(memory_token)


def integrate_handoff(
    noise: jax.Array,
    *,
    num_steps: int,
    handoff_steps: int,
    progress_velocity,
    action_velocity,
) -> tuple[jax.Array, dict[str, jax.Array | int]]:
    num_steps = int(num_steps)
    handoff_steps = min(max(int(handoff_steps), 0), num_steps)
    dt = -1.0 / num_steps
    x_t = noise
    handoff_state = noise

    for step in range(num_steps):
        time = jnp.broadcast_to(jnp.asarray(1.0 + step * dt, dtype=noise.dtype), (noise.shape[0],))
        if step < handoff_steps:
            velocity = progress_velocity(x_t, time)
        else:
            if step == handoff_steps:
                handoff_state = x_t
            velocity = action_velocity(x_t, time)
        x_t = x_t + dt * velocity

    if handoff_steps == num_steps:
        handoff_state = x_t

    diagnostics = {
        "progress_calls": handoff_steps,
        "action_calls": num_steps - handoff_steps,
        "handoff_steps": handoff_steps,
        "handoff_state": handoff_state,
    }
    return x_t, diagnostics


class FutureMamba(Pi0):
    def __init__(self, config: futuremamba_config.FutureMambaConfig, rngs: nnx.Rngs):
        if config.memory_backend != "mamba":
            raise NotImplementedError(f"FutureMamba memory_backend={config.memory_backend!r} is not implemented")
        if config.decoder_mode != "handoff":
            raise NotImplementedError(f"FutureMamba decoder_mode={config.decoder_mode!r} is not implemented")
        super().__init__(config, rngs)
        self.futuremamba = _FutureMambaPlugin(config, rngs=rngs)

    @property
    def _futuremamba_config(self) -> futuremamba_config.FutureMambaConfig:
        return self.futuremamba.config

    def _handoff_steps(self, handoff_ratio: float | None, num_steps: int) -> int:
        ratio = self._futuremamba_config.handoff_ratio if handoff_ratio is None else handoff_ratio
        return min(max(int(round(float(ratio) * int(num_steps))), 0), int(num_steps))

    def initial_memory_state(self, batch_size: int):
        return self.futuremamba.memory.initial_state(int(batch_size), dtype=jnp.float32)

    def _encode_memory_inputs(self, prefix_out: jax.Array, prefix_mask: jax.Array, mode: str | None = None) -> jax.Array:
        mode = self._futuremamba_config.conditioning_pool if mode is None else mode
        if mode == "last_valid":
            return _last_valid(prefix_out, prefix_mask)
        if mode == "attention":
            return _masked_mean(prefix_out, prefix_mask)
        if mode == "tokens4":
            return _mean_last_k_valid(prefix_out, prefix_mask, 4)
        if mode == "tokens8":
            return _mean_last_k_valid(prefix_out, prefix_mask, 8)
        raise ValueError(f"Unknown conditioning_pool: {mode!r}")

    def _memory_step_input(
        self,
        prefix_input: jax.Array,
        executed_actions: jax.Array,
        executed_action_mask: jax.Array,
    ) -> jax.Array:
        if self._futuremamba_config.memory_input == "token_only":
            return prefix_input
        action_summary = self.futuremamba.summarize_executed_actions(executed_actions, executed_action_mask)
        return self.futuremamba.fuse_memory_input(prefix_input, action_summary)

    def _scan_memory(
        self,
        prefix_inputs: jax.Array,
        executed_actions: jax.Array,
        executed_action_mask: jax.Array,
        query_mask: jax.Array,
        reset_mask: jax.Array,
        state,
    ):
        batch_size, num_queries = prefix_inputs.shape[:2]
        if num_queries == 0:
            return jnp.zeros_like(prefix_inputs), state

        zero_state = self.initial_memory_state(batch_size)
        carry = state
        previous_token = jnp.zeros((batch_size, self._futuremamba_config.memory.d_model), dtype=prefix_inputs.dtype)
        memory_tokens = []

        for query_index in range(num_queries):
            valid = query_mask[:, query_index]
            reset = reset_mask[:, query_index]
            step_state = _batch_where(reset, zero_state, carry)
            step_input = self._memory_step_input(
                prefix_inputs[:, query_index],
                executed_actions[:, query_index],
                executed_action_mask[:, query_index],
            )
            token, stepped_state = self.futuremamba.memory.step(step_input, step_state)
            carry = _batch_where(valid, stepped_state, carry)
            token = jnp.where(valid[:, None], token, previous_token)
            previous_token = token
            memory_tokens.append(token)

        return jnp.stack(memory_tokens, axis=1), carry

    def _prepare_memory_context(
        self,
        observation: _model.Observation,
        memory_state,
        executed_actions: jax.Array,
        executed_action_mask: jax.Array,
    ):
        prefix_out, prefix_mask, _, kv_cache = self.encode_prefix(observation)
        prefix_out = jax.lax.stop_gradient(prefix_out)
        prefix_mask = jax.lax.stop_gradient(prefix_mask)
        kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)

        pooled_prefix = self._encode_memory_inputs(prefix_out, prefix_mask, self._futuremamba_config.conditioning_pool)
        memory_input = self.futuremamba.vlm_to_memory(pooled_prefix)
        query_mask = jnp.ones((memory_input.shape[0], 1), dtype=jnp.bool_)
        reset_value = bool(self._futuremamba_config.reset_memory_every_query)
        reset_mask = jnp.full((memory_input.shape[0], 1), reset_value, dtype=jnp.bool_)
        memory_tokens, next_state = self._scan_memory(
            memory_input[:, None, :],
            executed_actions[:, None, :, :],
            executed_action_mask[:, None, :],
            query_mask,
            reset_mask,
            memory_state,
        )
        memory_token = self.futuremamba.project_memory_token(memory_tokens[:, -1])[:, None, :]
        return prefix_mask, kv_cache, memory_token, next_state

    @at.typecheck
    def compute_loss(self, rng: at.KeyArrayLike, observation, actions, *, train: bool = False):
        return super().compute_loss(rng, observation, actions, train=train)

    def _progress_velocity(
        self,
        prefix_mask: jax.Array,
        kv_cache,
        memory_token: jax.Array,
        x_t: jax.Array,
        timestep: jax.Array,
    ) -> jax.Array:
        return self.futuremamba.progress_expert(
            kv_cache,
            prefix_mask,
            memory_token,
            x_t,
            timestep,
            use_prefix_cache=self._futuremamba_config.use_prefix_cache,
            deterministic=self.deterministic,
        )

    def sample_actions_with_memory(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        memory_state=None,
        executed_actions: jax.Array | None = None,
        executed_action_mask: jax.Array | None = None,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        handoff_ratio: float | None = None,
        noise: jax.Array | None = None,
    ):
        observation = _model.preprocess_observation(None, observation, train=False)
        batch_size = observation.state.shape[0]
        if memory_state is None:
            memory_state = self.initial_memory_state(batch_size)
        if executed_actions is None:
            executed_actions = jnp.zeros(
                (batch_size, self._futuremamba_config.executed_horizon, self.action_dim), dtype=jnp.float32
            )
        if executed_action_mask is None:
            executed_action_mask = jnp.zeros(executed_actions.shape[:-1], dtype=jnp.bool_)
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        prefix_mask, kv_cache, memory_token, next_state = self._prepare_memory_context(
            observation, memory_state, executed_actions, executed_action_mask
        )
        handoff_steps = self._handoff_steps(handoff_ratio, int(num_steps))

        def progress_velocity(x_t, timestep):
            return self._progress_velocity(prefix_mask, kv_cache, memory_token, x_t, timestep)

        def action_velocity(x_t, timestep):
            return self.action_velocity(observation, x_t, timestep, prefix_mask, kv_cache)

        actions, diagnostics = integrate_handoff(
            noise,
            num_steps=int(num_steps),
            handoff_steps=handoff_steps,
            progress_velocity=progress_velocity,
            action_velocity=action_velocity,
        )
        return actions, next_state, diagnostics

    def sample_actions(self, rng: at.KeyArrayLike, observation, **kwargs):
        memory_state = kwargs.pop("memory_state", None)
        executed_actions = kwargs.pop("executed_actions", None)
        executed_action_mask = kwargs.pop("executed_action_mask", None)
        return_memory = bool(kwargs.pop("return_memory", False))
        handoff_ratio = kwargs.pop("handoff_ratio", None)
        if memory_state is None and executed_actions is None and executed_action_mask is None and handoff_ratio is None and not return_memory:
            return super().sample_actions(rng, observation, **kwargs)
        actions, next_state, diagnostics = self.sample_actions_with_memory(
            rng,
            observation,
            memory_state,
            executed_actions,
            executed_action_mask,
            handoff_ratio=handoff_ratio,
            **kwargs,
        )
        if return_memory:
            return actions, next_state, diagnostics
        return actions

    def _flatten_episode_observation(self, observation: _model.Observation, batch_size: int, num_queries: int):
        def flatten(value):
            if value is None:
                return None
            return jnp.reshape(value, (batch_size * num_queries, *value.shape[2:]))

        return _model.Observation(
            images={key: flatten(value) for key, value in observation.images.items()},
            image_masks={key: flatten(value) for key, value in observation.image_masks.items()},
            state=flatten(observation.state),
            tokenized_prompt=flatten(observation.tokenized_prompt),
            tokenized_prompt_mask=flatten(observation.tokenized_prompt_mask),
            token_ar_mask=flatten(observation.token_ar_mask),
            token_loss_mask=flatten(observation.token_loss_mask),
        )

    def _mean_masked_action_error(self, error: jax.Array, action_mask: jax.Array, query_mask: jax.Array) -> jax.Array:
        valid = jnp.logical_and(action_mask, query_mask[..., None])
        total = jnp.sum(jnp.where(valid, error, 0.0))
        count = jnp.sum(valid.astype(error.dtype))
        return total / jnp.maximum(count, 1.0)

    def _sample_high_noise_time(self, rng: at.KeyArrayLike, shape: tuple[int, ...], *, handoff_steps: int, num_steps: int):
        lower = 1.0 - (float(handoff_steps) / float(num_steps))
        lower = min(max(lower, 0.0), 1.0)
        return lower + (1.0 - lower) * jax.random.uniform(rng, shape, dtype=jnp.float32)

    def compute_episode_loss(self, rng: at.KeyArrayLike, batch, *, train: bool = False) -> dict[str, jax.Array]:
        config = self._futuremamba_config
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        actions = batch.actions
        batch_size, num_queries = actions.shape[:2]
        flat_observation = self._flatten_episode_observation(batch.observation, batch_size, num_queries)
        flat_observation = _model.preprocess_observation(preprocess_rng, flat_observation, train=train)

        query_mask = getattr(batch, "query_mask", jnp.ones((batch_size, num_queries), dtype=jnp.bool_))
        action_mask = getattr(
            batch,
            "action_mask",
            jnp.ones((batch_size, num_queries, self.action_horizon), dtype=jnp.bool_),
        )
        executed_actions = getattr(
            batch,
            "executed_actions",
            jnp.zeros((batch_size, num_queries, config.executed_horizon, self.action_dim), dtype=actions.dtype),
        )
        executed_action_mask = getattr(
            batch,
            "executed_action_mask",
            jnp.zeros(executed_actions.shape[:-1], dtype=jnp.bool_),
        )
        reset_mask = getattr(batch, "reset_mask", jnp.zeros((batch_size, num_queries), dtype=jnp.bool_))
        if config.reset_memory_every_query:
            reset_mask = jnp.ones_like(reset_mask)

        valid_action_mask = jnp.logical_and(action_mask, query_mask[..., None])
        safe_actions = jnp.where(valid_action_mask[..., None], actions, 0.0)
        noise = jax.random.normal(noise_rng, actions.shape)
        noise = jnp.where(valid_action_mask[..., None], noise, 0.0)
        handoff_steps = self._handoff_steps(config.handoff_ratio, config.num_denoise_steps)
        time = self._sample_high_noise_time(time_rng, (batch_size, num_queries), handoff_steps=handoff_steps, num_steps=config.num_denoise_steps)
        x_t = time[..., None, None] * noise + (1.0 - time[..., None, None]) * safe_actions
        target_velocity = noise - safe_actions

        prefix_out, prefix_mask, _, kv_cache = self.encode_prefix(flat_observation)
        prefix_out = jax.lax.stop_gradient(prefix_out)
        prefix_mask = jax.lax.stop_gradient(prefix_mask)
        kv_cache = jax.tree.map(jax.lax.stop_gradient, kv_cache)
        pooled_prefix = self._encode_memory_inputs(prefix_out, prefix_mask, config.conditioning_pool)
        memory_inputs = self.futuremamba.vlm_to_memory(pooled_prefix).reshape(batch_size, num_queries, -1)
        memory_tokens, _ = self._scan_memory(
            memory_inputs,
            executed_actions,
            executed_action_mask,
            query_mask,
            reset_mask,
            self.initial_memory_state(batch_size),
        )
        flat_memory_token = self.futuremamba.project_memory_token(memory_tokens.reshape(batch_size * num_queries, -1))[:, None, :]

        flat_x_t = x_t.reshape(batch_size * num_queries, self.action_horizon, self.action_dim)
        flat_time = time.reshape(batch_size * num_queries)
        pred_velocity = self._progress_velocity(prefix_mask, kv_cache, flat_memory_token, flat_x_t, flat_time)
        pred_velocity = pred_velocity.reshape(batch_size, num_queries, self.action_horizon, self.action_dim)
        flow_error_steps = jnp.mean(jnp.square(pred_velocity - target_velocity), axis=-1)
        flow_loss = self._mean_masked_action_error(flow_error_steps, action_mask, query_mask)

        handoff_error = jnp.asarray(0.0, dtype=jnp.float32)
        boundary_error = jnp.asarray(0.0, dtype=jnp.float32)
        boundary_state = noise
        boundary_time = jnp.full((batch_size, num_queries), 1.0, dtype=jnp.float32)
        if handoff_steps > 0 and (config.handoff_loss_weight > 0 or config.boundary_loss_weight > 0):
            dt = -1.0 / config.num_denoise_steps
            flat_boundary = noise.reshape(batch_size * num_queries, self.action_horizon, self.action_dim)
            for step in range(handoff_steps):
                step_time = jnp.full((batch_size * num_queries,), 1.0 + step * dt, dtype=jnp.float32)
                flat_boundary = flat_boundary + dt * self._progress_velocity(
                    prefix_mask, kv_cache, flat_memory_token, flat_boundary, step_time
                )
            boundary_state = flat_boundary.reshape(batch_size, num_queries, self.action_horizon, self.action_dim)
            boundary_time_value = 1.0 - handoff_steps / config.num_denoise_steps
            boundary_time = jnp.full((batch_size, num_queries), boundary_time_value, dtype=jnp.float32)

        if config.handoff_loss_weight > 0:
            target_boundary = boundary_time[..., None, None] * noise + (1.0 - boundary_time[..., None, None]) * safe_actions
            handoff_error_steps = jnp.mean(jnp.square(boundary_state - target_boundary), axis=-1)
            handoff_error = self._mean_masked_action_error(handoff_error_steps, action_mask, query_mask)

        if config.boundary_loss_weight > 0:
            flat_boundary = boundary_state.reshape(batch_size * num_queries, self.action_horizon, self.action_dim)
            flat_boundary_time = boundary_time.reshape(batch_size * num_queries)
            progress_boundary = self._progress_velocity(prefix_mask, kv_cache, flat_memory_token, flat_boundary, flat_boundary_time)
            action_boundary = self.action_velocity(
                flat_observation,
                flat_boundary,
                flat_boundary_time,
                prefix_mask,
                kv_cache,
            )
            action_boundary = jax.lax.stop_gradient(action_boundary)
            boundary_error_steps = jnp.mean(
                jnp.square(
                    progress_boundary.reshape(batch_size, num_queries, self.action_horizon, self.action_dim)
                    - action_boundary.reshape(batch_size, num_queries, self.action_horizon, self.action_dim)
                ),
                axis=-1,
            )
            boundary_error = self._mean_masked_action_error(boundary_error_steps, action_mask, query_mask)

        handoff_loss = jnp.asarray(config.handoff_loss_weight, dtype=jnp.float32) * handoff_error
        boundary_loss = jnp.asarray(config.boundary_loss_weight, dtype=jnp.float32) * boundary_error
        total_loss = flow_loss + handoff_loss + boundary_loss
        return {
            "loss": total_loss,
            "flow_loss": flow_loss,
            "handoff_loss": handoff_loss,
            "handoff_error": handoff_error,
            "boundary_loss": boundary_loss,
            "boundary_error": boundary_error,
        }
