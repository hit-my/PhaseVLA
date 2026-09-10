from __future__ import annotations

import dataclasses
import functools
import math
from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from openpi.models_pytorch.futuremamba_config import MambaMemoryConfig


_PARAMETER_MATCH_TOLERANCE = 0.05


def _clone_layers(
    layers: tuple[tuple[torch.Tensor, ...], ...], *, detach: bool = True
) -> tuple[tuple[torch.Tensor, ...], ...]:
    return tuple(tuple((tensor.detach() if detach else tensor).clone() for tensor in layer) for layer in layers)


def _convert_layers(
    layers: tuple[tuple[torch.Tensor, ...], ...],
    *,
    device: torch.device,
    dtype: torch.dtype,
    clone: bool,
) -> tuple[tuple[torch.Tensor, ...], ...]:
    converted_layers = []
    for layer in layers:
        converted_tensors = []
        for tensor in layer:
            converted = tensor.to(device=device, dtype=dtype)
            converted_tensors.append(converted.clone() if clone else converted)
        converted_layers.append(tuple(converted_tensors))
    return tuple(converted_layers)


@dataclasses.dataclass(frozen=True, eq=False)
class MemorySnapshot:
    backend_id: str
    state_schema_version: int
    batch_size: int
    layers: tuple[tuple[torch.Tensor, ...], ...]

    def clone(self) -> "MemorySnapshot":
        return MemorySnapshot(
            self.backend_id,
            self.state_schema_version,
            self.batch_size,
            _clone_layers(self.layers),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemorySnapshot):
            return NotImplemented
        if (
            self.backend_id != other.backend_id
            or self.state_schema_version != other.state_schema_version
            or self.batch_size != other.batch_size
            or len(self.layers) != len(other.layers)
        ):
            return False
        return all(
            len(left) == len(right)
            and all(torch.equal(left_tensor, right_tensor) for left_tensor, right_tensor in zip(left, right, strict=True))
            for left, right in zip(self.layers, other.layers, strict=True)
        )


def _mamba2_parameter_count(config: MambaMemoryConfig) -> int:
    d_model = int(config.d_model)
    d_inner = d_model * int(config.expand)
    nheads = d_inner // int(config.headdim)
    conv_dim = d_inner + 2 * int(config.ngroups) * int(config.d_state)
    in_projection = d_model * (2 * d_inner + 2 * int(config.ngroups) * int(config.d_state) + nheads)
    convolution = conv_dim * int(config.d_conv) + conv_dim
    state_parameters = 3 * nheads
    gated_norm = d_inner if config.rms_norm else 0
    output_projection = d_inner * d_model
    block_norm = d_model
    return int(config.depth) * (
        in_projection + convolution + state_parameters + gated_norm + output_projection + block_norm
    ) + d_model


def _nearest_hidden_width(target_count: int, counter) -> int:
    max_width = max(4096, int(math.ceil(math.sqrt(max(target_count, 1)))) * 4)
    return min(range(1, max_width + 1), key=lambda width: abs(counter(width) - target_count))


def _module_parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _module_device_dtype(module: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(module.parameters(), None)
    if parameter is None:
        return torch.device("cpu"), torch.float32
    return parameter.device, parameter.dtype


class _MemoryBackend(nn.Module):
    backend_id = "base"
    state_schema_version = 1

    def __init__(self, config: MambaMemoryConfig) -> None:
        super().__init__()
        self.config = config
        self.mamba_parameter_count = _mamba2_parameter_count(config)

    @property
    def parameter_count(self) -> int:
        return _module_parameter_count(self)

    @property
    def parameter_error(self) -> float:
        target = self.mamba_parameter_count
        return 0.0 if target == 0 else abs(self.parameter_count - target) / target

    @property
    def parameter_matched(self) -> bool:
        return self.parameter_count > 0 and self.parameter_error <= _PARAMETER_MATCH_TOLERANCE

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> MemorySnapshot:
        raise NotImplementedError

    def _step_impl(self, x: torch.Tensor, state: MemorySnapshot) -> tuple[torch.Tensor, MemorySnapshot]:
        raise NotImplementedError

    def _expected_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> MemorySnapshot:
        return self.initial_state(batch_size, device=device, dtype=dtype)

    def _validate_state(
        self,
        state: MemorySnapshot,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if state.backend_id != self.backend_id:
            raise ValueError(f"memory state backend mismatch: expected {self.backend_id!r}, got {state.backend_id!r}")
        if state.state_schema_version != self.state_schema_version:
            raise ValueError(
                "memory state schema mismatch: "
                f"expected {self.state_schema_version}, got {state.state_schema_version}"
            )
        if state.batch_size != batch_size:
            raise ValueError(f"memory state batch mismatch: expected {batch_size}, got {state.batch_size}")
        expected = self._expected_state(batch_size, device=device, dtype=dtype)
        if len(state.layers) != len(expected.layers):
            raise ValueError(f"memory state shape mismatch: expected {len(expected.layers)} layers, got {len(state.layers)}")
        for layer_index, (actual_layer, expected_layer) in enumerate(zip(state.layers, expected.layers, strict=True)):
            if len(actual_layer) != len(expected_layer):
                raise ValueError(
                    f"memory state shape mismatch at layer {layer_index}: "
                    f"expected {len(expected_layer)} tensors, got {len(actual_layer)}"
                )
            for tensor_index, (actual, expected_tensor) in enumerate(
                zip(actual_layer, expected_layer, strict=True)
            ):
                if actual.shape != expected_tensor.shape:
                    raise ValueError(
                        f"memory state shape mismatch at layer {layer_index} tensor {tensor_index}: "
                        f"expected {tuple(expected_tensor.shape)}, got {tuple(actual.shape)}"
                    )
                if actual.dtype != expected_tensor.dtype:
                    raise ValueError(
                        f"memory state dtype mismatch at layer {layer_index} tensor {tensor_index}: "
                        f"expected {expected_tensor.dtype}, got {actual.dtype}"
                    )
                if actual.device != expected_tensor.device:
                    raise ValueError(
                        f"memory state device mismatch at layer {layer_index} tensor {tensor_index}: "
                        f"expected {expected_tensor.device}, got {actual.device}"
                    )


    @staticmethod
    def _state_tensor_dtype(state: MemorySnapshot, fallback: torch.dtype) -> torch.dtype:
        return state.layers[0][0].dtype if state.layers else fallback

    @staticmethod
    def _state_for_compute(state: MemorySnapshot, *, device: torch.device, dtype: torch.dtype) -> MemorySnapshot:
        return MemorySnapshot(
            state.backend_id,
            state.state_schema_version,
            state.batch_size,
            _convert_layers(state.layers, device=device, dtype=dtype, clone=True),
        )

    @staticmethod
    def _state_for_external(state: MemorySnapshot, *, device: torch.device, dtype: torch.dtype) -> MemorySnapshot:
        return MemorySnapshot(
            state.backend_id,
            state.state_schema_version,
            state.batch_size,
            _convert_layers(state.layers, device=device, dtype=dtype, clone=False),
        )

    def _step_preserving_grad(self, x: torch.Tensor, state: MemorySnapshot) -> tuple[torch.Tensor, MemorySnapshot]:
        state_dtype = self._state_tensor_dtype(state, x.dtype)
        self._validate_state(state, batch_size=x.shape[0], device=x.device, dtype=state_dtype)
        _, module_dtype = _module_device_dtype(self)
        output, next_state = self._step_impl(
            x.to(dtype=module_dtype), self._state_for_compute(state, device=x.device, dtype=module_dtype)
        )
        if output.shape != x.shape:
            raise RuntimeError(f"memory backend output shape must be {tuple(x.shape)}, got {tuple(output.shape)}")
        return output.to(dtype=x.dtype), self._state_for_external(next_state, device=x.device, dtype=x.dtype)

    def step(self, x: torch.Tensor, state: MemorySnapshot) -> tuple[torch.Tensor, MemorySnapshot]:
        if x.ndim != 2 or x.shape[1] != self.config.d_model:
            raise ValueError(f"memory input shape must be [batch, {self.config.d_model}], got {tuple(x.shape)}")
        return self._step_preserving_grad(x, state)

    @staticmethod
    def _validate_query_mask(x: torch.Tensor, query_mask: torch.Tensor | None) -> torch.Tensor:
        if query_mask is None:
            return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if query_mask.shape != x.shape[:2]:
            raise ValueError(f"query_mask shape must be {tuple(x.shape[:2])}, got {tuple(query_mask.shape)}")
        query_mask = query_mask.to(device=x.device, dtype=torch.bool)
        if torch.any((~query_mask[:, :-1]) & query_mask[:, 1:]):
            raise ValueError("query_mask must use right padding")
        if torch.any(query_mask.sum(dim=1) == 0):
            raise ValueError("query_mask contains a row with zero valid queries")
        return query_mask

    @staticmethod
    def _merge_state_rows(
        previous: MemorySnapshot, candidate: MemorySnapshot, update_mask: torch.Tensor
    ) -> MemorySnapshot:
        mask = update_mask
        layers = []
        for previous_layer, candidate_layer in zip(previous.layers, candidate.layers, strict=True):
            tensors = []
            for previous_tensor, candidate_tensor in zip(previous_layer, candidate_layer, strict=True):
                view = mask.reshape(mask.shape[0], *([1] * (previous_tensor.ndim - 1)))
                tensors.append(torch.where(view, candidate_tensor, previous_tensor))
            layers.append(tuple(tensors))
        return MemorySnapshot(previous.backend_id, previous.state_schema_version, previous.batch_size, tuple(layers))

    def forward_sequence(
        self, x: torch.Tensor, *, query_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, MemorySnapshot]:
        if x.ndim != 3 or x.shape[2] != self.config.d_model:
            raise ValueError(
                f"memory sequence shape must be [batch, queries, {self.config.d_model}], got {tuple(x.shape)}"
            )
        mask = self._validate_query_mask(x, query_mask)
        state = self.initial_state(x.shape[0], device=x.device, dtype=x.dtype)
        outputs = []
        for query_index in range(x.shape[1]):
            output, candidate = self._step_preserving_grad(x[:, query_index], state)
            update_mask = mask[:, query_index]
            outputs.append(torch.where(update_mask[:, None], output, torch.zeros_like(output)))
            state = self._merge_state_rows(state, candidate, update_mask)
        return torch.stack(outputs, dim=1), state

    def reset(self, state: MemorySnapshot, reset_mask: torch.Tensor) -> MemorySnapshot:
        if reset_mask.ndim != 1:
            raise ValueError(f"reset_mask must be one-dimensional, got {tuple(reset_mask.shape)}")
        dtype = state.layers[0][0].dtype if state.layers else torch.float32
        device = state.layers[0][0].device if state.layers else reset_mask.device
        self._validate_state(state, batch_size=reset_mask.shape[0], device=device, dtype=dtype)
        reset_mask = reset_mask.to(device=device, dtype=torch.bool)
        zero = self.initial_state(state.batch_size, device=device, dtype=dtype)
        return self._merge_state_rows(state, zero, reset_mask)

    def snapshot(self, state: MemorySnapshot) -> MemorySnapshot:
        dtype = state.layers[0][0].dtype if state.layers else torch.float32
        device = state.layers[0][0].device if state.layers else torch.device("cpu")
        self._validate_state(state, batch_size=state.batch_size, device=device, dtype=dtype)
        return state.clone()

    def restore(
        self, snapshot: MemorySnapshot, *, batch_size: int, device: torch.device
    ) -> MemorySnapshot:
        _, module_dtype = _module_device_dtype(self)
        dtype = snapshot.layers[0][0].dtype if snapshot.layers else module_dtype
        self._validate_state(snapshot, batch_size=batch_size, device=device, dtype=dtype)
        return snapshot.clone()


class GRUMemoryBackend(_MemoryBackend):
    backend_id = "gru"

    def __init__(self, config: MambaMemoryConfig) -> None:
        super().__init__(config)
        target = self.mamba_parameter_count

        def count(width: int) -> int:
            return 3 * width * config.d_model + 3 * width * width + 6 * width + width * config.d_model + config.d_model

        self.hidden_width = _nearest_hidden_width(target, count)
        self.cell = nn.GRUCell(config.d_model, self.hidden_width)
        self.output = nn.Linear(self.hidden_width, config.d_model)

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> MemorySnapshot:
        hidden = torch.zeros(batch_size, self.hidden_width, device=device, dtype=dtype)
        return MemorySnapshot(self.backend_id, self.state_schema_version, batch_size, ((hidden,),))

    def _step_impl(self, x: torch.Tensor, state: MemorySnapshot) -> tuple[torch.Tensor, MemorySnapshot]:
        hidden = self.cell(x, state.layers[0][0])
        output = self.output(hidden)
        return output, MemorySnapshot(self.backend_id, self.state_schema_version, x.shape[0], ((hidden,),))


class LSTMMemoryBackend(_MemoryBackend):
    backend_id = "lstm"

    def __init__(self, config: MambaMemoryConfig) -> None:
        super().__init__(config)
        target = self.mamba_parameter_count

        def count(width: int) -> int:
            return 4 * width * config.d_model + 4 * width * width + 8 * width + width * config.d_model + config.d_model

        self.hidden_width = _nearest_hidden_width(target, count)
        self.cell = nn.LSTMCell(config.d_model, self.hidden_width)
        self.output = nn.Linear(self.hidden_width, config.d_model)

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> MemorySnapshot:
        hidden = torch.zeros(batch_size, self.hidden_width, device=device, dtype=dtype)
        cell = torch.zeros_like(hidden)
        return MemorySnapshot(self.backend_id, self.state_schema_version, batch_size, ((hidden, cell),))

    def _step_impl(self, x: torch.Tensor, state: MemorySnapshot) -> tuple[torch.Tensor, MemorySnapshot]:
        hidden, cell = self.cell(x, state.layers[0])
        output = self.output(hidden)
        return output, MemorySnapshot(self.backend_id, self.state_schema_version, x.shape[0], ((hidden, cell),))


class FrameStackMemoryBackend(_MemoryBackend):
    backend_id = "frame_stack"

    def __init__(self, config: MambaMemoryConfig, *, frame_stack_window: int = 4) -> None:
        if frame_stack_window <= 0:
            raise ValueError(f"frame_stack_window must be positive, got {frame_stack_window}")
        super().__init__(config)
        self.frame_stack_window = frame_stack_window
        target = self.mamba_parameter_count

        def count(width: int) -> int:
            return frame_stack_window * config.d_model * width + width + width * config.d_model + config.d_model

        self.hidden_width = _nearest_hidden_width(target, count)
        self.input = nn.Linear(frame_stack_window * config.d_model, self.hidden_width)
        self.output = nn.Linear(self.hidden_width, config.d_model)

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> MemorySnapshot:
        frames = torch.zeros(
            batch_size,
            self.frame_stack_window,
            self.config.d_model,
            device=device,
            dtype=dtype,
        )
        return MemorySnapshot(self.backend_id, self.state_schema_version, batch_size, ((frames,),))

    def _step_impl(self, x: torch.Tensor, state: MemorySnapshot) -> tuple[torch.Tensor, MemorySnapshot]:
        frames = torch.cat([state.layers[0][0][:, 1:], x[:, None]], dim=1)
        hidden = torch.nn.functional.silu(self.input(frames.flatten(start_dim=1)))
        output = self.output(hidden)
        return output, MemorySnapshot(self.backend_id, self.state_schema_version, x.shape[0], ((frames,),))


class NoMemoryBackend(_MemoryBackend):
    backend_id = "none"

    @property
    def parameter_matched(self) -> bool:
        return False

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> MemorySnapshot:
        del device, dtype
        return MemorySnapshot(self.backend_id, self.state_schema_version, batch_size, ())

    def _step_impl(self, x: torch.Tensor, state: MemorySnapshot) -> tuple[torch.Tensor, MemorySnapshot]:
        return torch.zeros_like(x), state.clone()


class Mamba2MemoryBackend(_MemoryBackend):
    backend_id = "mamba2"

    def __init__(self, config: MambaMemoryConfig) -> None:
        super().__init__(config)
        try:
            from mamba_ssm.models.mixer_seq_simple import create_block
            from mamba_ssm.ops.triton.layer_norm import RMSNorm
            from mamba_ssm.utils.generation import InferenceParams
        except ImportError as exc:
            raise ImportError(
                "mamba_ssm is required for the official Mamba-2 memory backend; "
                "install the pinned FutureMamba runtime"
            ) from exc

        self._inference_params_type = InferenceParams
        ssm_config = {
            "layer": "Mamba2",
            "d_state": config.d_state,
            "d_conv": config.d_conv,
            "expand": config.expand,
            "headdim": config.headdim,
            "ngroups": config.ngroups,
            "rmsnorm": config.rms_norm,
            "use_mem_eff_path": config.use_mem_eff_path,
        }
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model=config.d_model,
                    d_intermediate=0,
                    ssm_cfg=ssm_config,
                    rms_norm=config.rms_norm,
                    residual_in_fp32=config.residual_in_fp32,
                    fused_add_norm=config.fused_add_norm,
                    layer_idx=layer_index,
                )
                for layer_index in range(config.depth)
            ]
        )
        norm_type = RMSNorm if config.rms_norm else nn.LayerNorm
        self.norm = norm_type(config.d_model, eps=1e-5)

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> MemorySnapshot:
        layers = []
        for block in self.layers:
            cache = block.mixer.allocate_inference_cache(batch_size, max_seqlen=1, dtype=dtype)
            layers.append(tuple(tensor.detach().to(device=device, dtype=dtype).clone() for tensor in cache))
        return MemorySnapshot(self.backend_id, self.state_schema_version, batch_size, tuple(layers))

    def _step_impl(self, x: torch.Tensor, state: MemorySnapshot) -> tuple[torch.Tensor, MemorySnapshot]:
        hidden = x[:, None]
        residual = None
        next_layers = []
        for block, layer_state in zip(self.layers, state.layers, strict=True):
            residual = hidden if residual is None else hidden + residual
            hidden = self._norm(block.norm, residual.to(dtype=block.norm.weight.dtype), None)
            if block.residual_in_fp32:
                residual = residual.float()
            conv_state, ssm_state = (tensor.clone() for tensor in layer_state)
            if torch.is_grad_enabled():
                hidden, conv_state, ssm_state = self._step_mixer_differentiable(
                    block.mixer, hidden, conv_state, ssm_state
                )
            elif x.device.type == "cpu":
                hidden, conv_state, ssm_state = self._step_mixer_cpu(
                    block.mixer, hidden, conv_state, ssm_state
                )
            else:
                hidden, conv_state, ssm_state = block.mixer.step(hidden, conv_state, ssm_state)
            next_layers.append((conv_state, ssm_state))
        hidden = hidden + residual
        hidden = self._norm(self.norm, hidden.to(dtype=self.norm.weight.dtype), None)
        next_state = MemorySnapshot(self.backend_id, self.state_schema_version, x.shape[0], tuple(next_layers))
        return hidden[:, 0].to(dtype=x.dtype), next_state

    @staticmethod
    def _norm(module: nn.Module, x: torch.Tensor, z: torch.Tensor | None) -> torch.Tensor:
        if x.device.type != "cpu":
            return module(x, z)
        if z is not None and not getattr(module, "norm_before_gate", True):
            x = x * F.silu(z)
        output = x.float()
        output = output * torch.rsqrt(output.square().mean(dim=-1, keepdim=True) + float(module.eps))
        output = output * module.weight.float()
        bias = getattr(module, "bias", None)
        if bias is not None:
            output = output + bias.float()
        if z is not None and getattr(module, "norm_before_gate", True):
            output = output * F.silu(z.float())
        return output.to(dtype=x.dtype)

    @staticmethod
    def _step_mixer_differentiable(
        mixer: nn.Module,
        hidden: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Functional recurrent update for autograd; incoming caches are read-only.

        The pinned CUDA inference kernels update caches in place and do not
        propagate gradients through earlier states. Keep their FP32 SSM
        discretization, but build new states with differentiable tensor ops.
        This path supports the single-group, non-distributed reference model.
        """
        if mixer.ngroups != 1:
            raise ValueError("Differentiable Mamba-2 step requires ngroups=1")
        if getattr(mixer, "process_group", None) is not None:
            raise ValueError("Differentiable Mamba-2 step does not support tensor parallelism")
        projected = mixer.in_proj(hidden.squeeze(1))
        d_mlp = (projected.shape[-1] - 2 * mixer.d_ssm - 2 * mixer.d_state - mixer.nheads) // 2
        z0, x0, z, xbc, dt = torch.split(
            projected,
            [d_mlp, d_mlp, mixer.d_ssm, mixer.d_ssm + 2 * mixer.d_state, mixer.nheads],
            dim=-1,
        )
        next_conv = torch.cat((conv_state[:, :, 1:], xbc[:, :, None].to(conv_state.dtype)), dim=-1)
        xbc = (next_conv * mixer.conv1d.weight.squeeze(1)).sum(dim=-1)
        if mixer.conv1d.bias is not None:
            xbc = xbc + mixer.conv1d.bias
        xbc = mixer.act(xbc).to(hidden.dtype)
        x, b, c = torch.split(xbc, [mixer.d_ssm, mixer.d_state, mixer.d_state], dim=-1)
        dt = F.softplus(dt.float() + mixer.dt_bias.float())
        decay = torch.exp(dt * (-torch.exp(mixer.A_log.float())))
        x = x.reshape(x.shape[0], mixer.nheads, mixer.headdim)
        contribution = torch.einsum("bh,bn,bhp->bhpn", dt, b.float(), x.float())
        updated = ssm_state.float() * decay[:, :, None, None] + contribution
        next_ssm = updated.to(ssm_state.dtype)
        y = torch.einsum("bhpn,bn->bhp", updated, c.float())
        d = mixer.D.float().reshape(1, mixer.nheads, mixer.headdim if mixer.D_has_hdim else 1)
        y = (y + d * x.float()).reshape(x.shape[0], mixer.d_ssm).to(hidden.dtype)
        y = Mamba2MemoryBackend._norm(mixer.norm, y, z) if mixer.rmsnorm else y * mixer.act(z)
        if d_mlp:
            y = torch.cat((F.silu(z0) * x0, y), dim=-1)
        return mixer.out_proj(y).unsqueeze(1), next_conv, next_ssm


    @staticmethod
    def _step_mixer_cpu(
        mixer: nn.Module,
        hidden_states: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mixer.ngroups != 1:
            raise ValueError("CPU Mamba-2 fallback requires ngroups=1")
        dtype = hidden_states.dtype
        zxbcdt = mixer.in_proj(hidden_states.squeeze(1))
        d_mlp = (zxbcdt.shape[-1] - 2 * mixer.d_ssm - 2 * mixer.ngroups * mixer.d_state - mixer.nheads) // 2
        z0, x0, z, xBC, dt = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, mixer.d_ssm, mixer.d_ssm + 2 * mixer.ngroups * mixer.d_state, mixer.nheads],
            dim=-1,
        )
        conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
        conv_state[:, :, -1] = xBC
        xBC = torch.sum(conv_state * mixer.conv1d.weight.squeeze(1), dim=-1)
        if mixer.conv1d.bias is not None:
            xBC = xBC + mixer.conv1d.bias
        xBC = mixer.act(xBC).to(dtype=dtype)
        x, B, C = torch.split(
            xBC,
            [mixer.d_ssm, mixer.ngroups * mixer.d_state, mixer.ngroups * mixer.d_state],
            dim=-1,
        )
        A = -torch.exp(mixer.A_log.float())
        dt = F.softplus(dt + mixer.dt_bias.to(dtype=dt.dtype))
        dA = torch.exp(dt * A)
        x = x.reshape(x.shape[0], mixer.nheads, mixer.headdim)
        dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x)
        ssm_state.copy_(ssm_state * dA.reshape(x.shape[0], mixer.nheads, 1, 1) + dBx)
        y = torch.einsum("bhpn,bn->bhp", ssm_state.to(dtype), C)
        if mixer.D_has_hdim:
            y = y + mixer.D.to(dtype).reshape(1, mixer.nheads, mixer.headdim) * x
        else:
            y = y + mixer.D.to(dtype).reshape(1, mixer.nheads, 1) * x
        y = y.reshape(y.shape[0], mixer.d_ssm)
        y = Mamba2MemoryBackend._norm(mixer.norm, y, z) if mixer.rmsnorm else y * mixer.act(z)
        if d_mlp > 0:
            y = torch.cat([F.silu(z0) * x0, y], dim=-1)
        return mixer.out_proj(y).unsqueeze(1), conv_state, ssm_state

    def _forward_valid_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, MemorySnapshot]:
        _, module_dtype = _module_device_dtype(self)
        inference_params = self._inference_params_type(max_seqlen=x.shape[1], max_batch_size=x.shape[0])
        hidden = x.to(dtype=module_dtype)
        residual = None
        for block in self.layers:
            hidden, residual = block(hidden, residual, inference_params=inference_params)
        hidden = hidden + residual
        hidden = self.norm(hidden.to(dtype=self.norm.weight.dtype)).to(dtype=x.dtype)
        layers = tuple(
            tuple(tensor.to(dtype=x.dtype).clone() for tensor in inference_params.key_value_memory_dict[index])
            for index in range(len(self.layers))
        )
        return hidden, MemorySnapshot(self.backend_id, self.state_schema_version, x.shape[0], layers)

    def forward_sequence(
        self, x: torch.Tensor, *, query_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, MemorySnapshot]:
        if x.ndim != 3 or x.shape[2] != self.config.d_model:
            raise ValueError(
                f"memory sequence shape must be [batch, queries, {self.config.d_model}], got {tuple(x.shape)}"
            )
        mask = self._validate_query_mask(x, query_mask)
        outputs = torch.zeros_like(x)
        row_states: list[MemorySnapshot] = []
        for row in range(x.shape[0]):
            valid_length = int(mask[row].sum().item())
            row_output, row_state = self._forward_valid_sequence(x[row : row + 1, :valid_length])
            outputs[row, :valid_length] = row_output[0]
            row_states.append(row_state)
        layers = tuple(
            tuple(
                torch.cat([row_state.layers[layer_index][tensor_index] for row_state in row_states], dim=0)
                for tensor_index in range(len(row_states[0].layers[layer_index]))
            )
            for layer_index in range(len(row_states[0].layers))
        )
        return outputs, MemorySnapshot(self.backend_id, self.state_schema_version, x.shape[0], layers)
