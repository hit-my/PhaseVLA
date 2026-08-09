import dataclasses
import math

from flax import nnx
from flax import struct
import jax
import jax.numpy as jnp


def _validate_positive(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


@dataclasses.dataclass(frozen=True)
class MambaConfig:
    d_model: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    depth: int = 2
    dt_rank: int | None = None

    def __post_init__(self):
        _validate_positive("d_model", self.d_model)
        _validate_positive("d_state", self.d_state)
        _validate_positive("d_conv", self.d_conv)
        _validate_positive("expand", self.expand)
        _validate_positive("depth", self.depth)
        _validate_positive("dt_rank", self.dt_rank)
        if self.dt_rank is None:
            object.__setattr__(self, "dt_rank", math.ceil(self.d_model / 16))


@struct.dataclass
class MambaLayerState:
    ssm: jax.Array
    conv: jax.Array


@struct.dataclass
class MambaState:
    layers: tuple[MambaLayerState, ...]


class _RMSNorm(nnx.Module):
    def __init__(self, width: int, *, rngs: nnx.Rngs, eps: float = 1e-6):
        self.width = width
        self.eps = eps
        self.scale = nnx.Param(jnp.zeros((width,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        dtype = x.dtype
        x_f32 = x.astype(jnp.float32)
        normed = x_f32 * jax.lax.rsqrt(jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True) + self.eps)
        return (normed * (1.0 + self.scale.value)).astype(dtype)


class _MambaLayer(nnx.Module):
    def __init__(self, config: MambaConfig, *, rngs: nnx.Rngs):
        self.config = config
        self.d_inner = config.d_model * config.expand
        self.norm = _RMSNorm(config.d_model, rngs=rngs)
        self.in_proj = nnx.Linear(config.d_model, 2 * self.d_inner, rngs=rngs)
        self.x_proj = nnx.Linear(self.d_inner, config.dt_rank + 2 * config.d_state, use_bias=False, rngs=rngs)
        self.dt_proj = nnx.Linear(config.dt_rank, self.d_inner, rngs=rngs)
        self.out_proj = nnx.Linear(self.d_inner, config.d_model, rngs=rngs)

        conv_scale = 1.0 / math.sqrt(max(config.d_conv, 1))
        self.conv_kernel = nnx.Param(
            jax.random.normal(rngs.params(), (config.d_conv, self.d_inner), dtype=jnp.float32) * conv_scale
        )
        self.conv_bias = nnx.Param(jnp.zeros((self.d_inner,), dtype=jnp.float32))
        self.A_log = nnx.Param(
            jnp.log(jnp.broadcast_to(jnp.arange(1, config.d_state + 1, dtype=jnp.float32), (self.d_inner, config.d_state)))
        )
        self.D = nnx.Param(jnp.ones((self.d_inner,), dtype=jnp.float32))

        dt = jnp.exp(
            jax.random.uniform(
                rngs.params(),
                (self.d_inner,),
                minval=math.log(1e-3),
                maxval=math.log(0.1),
                dtype=jnp.float32,
            )
        )
        inverse_softplus_dt = dt + jnp.log(-jnp.expm1(-dt))
        self.dt_proj.bias.value = inverse_softplus_dt

    def _validate_layer_state(self, x: jax.Array, state: MambaLayerState, *, layer_index: int) -> None:
        expected_ssm = (x.shape[0], self.d_inner, self.config.d_state)
        expected_conv = (x.shape[0], self.config.d_conv - 1, self.d_inner)
        if state.ssm.shape != expected_ssm:
            raise ValueError(
                f"state layer {layer_index} ssm shape must be {expected_ssm}, got {state.ssm.shape}"
            )
        if state.conv.shape != expected_conv:
            raise ValueError(
                f"state layer {layer_index} conv shape must be {expected_conv}, got {state.conv.shape}"
            )

    def __call__(self, x: jax.Array, state: MambaLayerState, *, layer_index: int) -> tuple[jax.Array, MambaLayerState]:
        self._validate_layer_state(x, state, layer_index=layer_index)
        residual = x
        x = self.norm(x)
        projected, gate = jnp.split(self.in_proj(x), 2, axis=-1)

        conv_window = jnp.concatenate([state.conv, projected[:, None, :]], axis=1)
        conv = jnp.sum(conv_window * self.conv_kernel.value[None, :, :], axis=1) + self.conv_bias.value
        conv = jax.nn.silu(conv)
        next_conv = conv_window[:, 1:, :]

        dt_bc = self.x_proj(conv)
        dt, b, c = jnp.split(dt_bc, [self.config.dt_rank, self.config.dt_rank + self.config.d_state], axis=-1)
        dt = jax.nn.softplus(self.dt_proj(dt))

        A = -jnp.exp(self.A_log.value)
        dA = jnp.exp(dt[:, :, None] * A[None, :, :])
        input_update = dt[:, :, None] * b[:, None, :] * conv[:, :, None]
        next_ssm = dA * state.ssm + input_update
        ssm_y = jnp.sum(next_ssm * c[:, None, :], axis=-1) + self.D.value * conv
        y = self.out_proj(ssm_y * jax.nn.silu(gate))
        return residual + y, MambaLayerState(ssm=next_ssm, conv=next_conv)


class SelectiveMamba(nnx.Module):
    def __init__(self, config: MambaConfig, rngs: nnx.Rngs):
        self.config = config
        self.d_inner = config.d_model * config.expand
        self.layers = tuple(_MambaLayer(config, rngs=rngs) for _ in range(config.depth))

    def initial_state(self, batch_size: int, dtype=jnp.float32) -> MambaState:
        _validate_positive("batch_size", batch_size)
        layers = tuple(
            MambaLayerState(
                ssm=jnp.zeros((batch_size, self.d_inner, self.config.d_state), dtype=dtype),
                conv=jnp.zeros((batch_size, self.config.d_conv - 1, self.d_inner), dtype=dtype),
            )
            for _ in range(self.config.depth)
        )
        return MambaState(layers=layers)

    def _validate_state(self, batch_size: int, state: MambaState) -> None:
        if not isinstance(state, MambaState):
            raise ValueError(f"state must be a MambaState, got {type(state).__name__}")
        if len(state.layers) != self.config.depth:
            raise ValueError(f"state must have {self.config.depth} layers, got {len(state.layers)}")
        for index, layer_state in enumerate(state.layers):
            expected_ssm = (batch_size, self.d_inner, self.config.d_state)
            expected_conv = (batch_size, self.config.d_conv - 1, self.d_inner)
            if layer_state.ssm.shape != expected_ssm:
                raise ValueError(f"state layer {index} ssm shape must be {expected_ssm}, got {layer_state.ssm.shape}")
            if layer_state.conv.shape != expected_conv:
                raise ValueError(f"state layer {index} conv shape must be {expected_conv}, got {layer_state.conv.shape}")

    def step(self, x: jax.Array, state: MambaState) -> tuple[jax.Array, MambaState]:
        if x.ndim != 2:
            raise ValueError(f"x must have rank 2 [batch, d_model], got shape {x.shape}")
        if x.shape[-1] != self.config.d_model:
            raise ValueError(f"x d_model must be {self.config.d_model}, got shape {x.shape}")
        self._validate_state(x.shape[0], state)

        next_layers = []
        y = x
        for index, layer in enumerate(self.layers):
            y, layer_state = layer(y, state.layers[index], layer_index=index)
            next_layers.append(layer_state)
        return y, MambaState(layers=tuple(next_layers))

    def scan(self, x: jax.Array, state: MambaState) -> tuple[jax.Array, MambaState]:
        if x.ndim != 3:
            raise ValueError(f"x must have rank 3 [batch, time, d_model], got shape {x.shape}")
        if x.shape[-1] != self.config.d_model:
            raise ValueError(f"x d_model must be {self.config.d_model}, got shape {x.shape}")
        self._validate_state(x.shape[0], state)

        def scan_step(carry: MambaState, token: jax.Array) -> tuple[MambaState, jax.Array]:
            y, next_carry = self.step(token, carry)
            return next_carry, y

        next_state, y_time_major = jax.lax.scan(scan_step, state, jnp.swapaxes(x, 0, 1))
        return jnp.swapaxes(y_time_major, 0, 1), next_state
