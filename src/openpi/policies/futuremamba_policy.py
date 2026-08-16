from __future__ import annotations

from collections.abc import Sequence
import dataclasses
import hashlib
import time
from typing import Any
import uuid

import numpy as np
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models_pytorch.mamba_memory import MemorySnapshot
from openpi_client import base_policy as _base_policy


@dataclasses.dataclass(frozen=True)
class FutureMambaPolicyState:
    backend_id: str
    state_schema_version: int
    memory: MemorySnapshot
    episode_count: int
    query_count: int
    client_id: str


class FutureMambaPolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        model: Any,
        *,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str | torch.device | None = None,
        client_id: str | None = None,
        _sample_actions_with_memory=None,
    ) -> None:
        if not isinstance(model, torch.nn.Module):
            raise TypeError("FutureMambaPolicy requires a PyTorch FutureMamba model")
        if not hasattr(model, "initial_memory_state") or not hasattr(model, "sample_actions_with_memory"):
            raise AttributeError("FutureMambaPolicy model must define PyTorch memory initialization and sampling")
        self._device = torch.device(
            pytorch_device or (next(model.parameters(), torch.empty(0)).device if any(True for _ in model.parameters()) else "cpu")
        )
        self._model = model.to(self._device)
        self._model.eval()
        self._transforms = tuple(transforms)
        self._output_transforms = tuple(output_transforms)
        self._input_transform = _transforms.compose(self._transforms)
        self._output_transform = _transforms.compose(self._output_transforms)
        self._sample_kwargs = dict(sample_kwargs or {})
        self._metadata = metadata or {}
        self._client_id = client_id or uuid.uuid4().hex
        self._next_fork_id = 0
        self._sample_actions_with_memory = _sample_actions_with_memory or model.sample_actions_with_memory
        self._episode_count = 0
        self._query_count = 0
        self._memory_state = self._new_memory_state()
        self._buffer_diagnostics: list[dict[str, Any]] = []

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = self._input_transform(dict(obs))
        inputs.pop("executed_actions", None)
        inputs.pop("executed_action_mask", None)
        batched_inputs = _tree_to_torch_batch(inputs, self._device)
        observation = _model.Observation.from_dict(batched_inputs)
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise_tensor = _as_torch(noise, self._device, torch.float32)
            sample_kwargs["noise"] = noise_tensor[None, ...] if noise_tensor.ndim == 2 else noise_tensor

        start_time = time.monotonic()
        with torch.no_grad():
            actions, next_memory, diagnostics = self._sample_actions_with_memory(
                observation,
                self._memory_state,
                **sample_kwargs,
            )
        model_time = time.monotonic() - start_time
        self._memory_state = self._validate_memory(next_memory, expected_batch=1, device=self._device)
        self._query_count += 1

        outputs = {
            "state": np.asarray(batched_inputs["state"][0].detach().cpu()),
            "actions": np.asarray(actions[0].detach().cpu()),
        }
        outputs = self._output_transform(outputs)
        outputs["handoff_step"] = _handoff_step_diagnostic(diagnostics)
        outputs["memory_state_bytes"] = _memory_nbytes(self._memory_state)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    def add_buffer(self, payload: dict[str, Any]) -> dict[str, Any]:
        diagnostic = _validate_buffer_payload(payload)
        self._buffer_diagnostics.append(diagnostic)
        return dict(diagnostic)

    @property
    def buffer_diagnostics(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(item) for item in self._buffer_diagnostics)

    @override
    def reset(self) -> None:
        self._episode_count += 1
        self._query_count = 0
        self._memory_state = self._new_memory_state()
        self._buffer_diagnostics.clear()
    @override
    def snapshot_state(self) -> FutureMambaPolicyState:
        memory = _snapshot_memory_cpu(self._memory_state)
        return FutureMambaPolicyState(
            backend_id=memory.backend_id,
            state_schema_version=memory.state_schema_version,
            memory=memory,
            episode_count=self._episode_count,
            query_count=self._query_count,
            client_id=self._client_id,
        )

    @override
    def restore_state(self, state: FutureMambaPolicyState) -> None:
        if not isinstance(state, FutureMambaPolicyState):
            raise TypeError("restore_state requires FutureMambaPolicyState")
        expected = self._new_memory_state()
        if state.backend_id != expected.backend_id or state.memory.backend_id != expected.backend_id:
            raise ValueError("memory state backend mismatch")
        if state.state_schema_version != expected.state_schema_version or state.memory.state_schema_version != expected.state_schema_version:
            raise ValueError("memory state schema mismatch")
        restored = _clone_memory_to(state.memory, self._device)
        self._memory_state = self._validate_memory(restored, expected_batch=1, device=self._device)
        if state.episode_count < 0 or state.query_count < 0:
            raise ValueError("episode_count and query_count must be non-negative")
        self._episode_count = state.episode_count
        self._query_count = state.query_count

    @override
    def fork(self) -> "FutureMambaPolicy":
        self._next_fork_id += 1
        return FutureMambaPolicy(
            self._model,
            transforms=self._transforms,
            output_transforms=self._output_transforms,
            sample_kwargs=self._sample_kwargs,
            metadata=self._metadata,
            pytorch_device=self._device,
            client_id=f"{self._client_id}:{self._next_fork_id}:{uuid.uuid4().hex}",
            _sample_actions_with_memory=self._sample_actions_with_memory,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _new_memory_state(self) -> MemorySnapshot:
        dtype = next(self._model.futuremamba.parameters(), torch.empty(0, dtype=torch.float32)).dtype
        memory = self._model.initial_memory_state(1, self._device, dtype)
        return self._validate_memory(memory, expected_batch=1, device=self._device)


    def _validate_memory(self, memory: MemorySnapshot, *, expected_batch: int, device: torch.device) -> MemorySnapshot:
        if not isinstance(memory, MemorySnapshot):
            raise TypeError("memory must be MemorySnapshot")
        backend = self._model.futuremamba.memory_backend
        if memory.backend_id != backend.backend_id:
            raise ValueError("memory state backend mismatch")
        if memory.state_schema_version != backend.state_schema_version:
            raise ValueError("memory state schema mismatch")
        if memory.batch_size != expected_batch:
            raise ValueError("memory state batch mismatch")
        expected_dtype = next(self._model.futuremamba.parameters(), torch.empty(0, dtype=torch.float32)).dtype
        expected = backend.initial_state(expected_batch, device=device, dtype=expected_dtype)
        if len(memory.layers) != len(expected.layers):
            raise ValueError("memory state layer shape mismatch")
        for index, (actual_layer, expected_layer) in enumerate(zip(memory.layers, expected.layers, strict=True)):
            if len(actual_layer) != len(expected_layer):
                raise ValueError(f"memory state layer {index} shape mismatch")
            for actual, reference in zip(actual_layer, expected_layer, strict=True):
                if actual.shape != reference.shape:
                    raise ValueError("memory state shape mismatch")
                if actual.dtype != reference.dtype:
                    raise ValueError("memory state dtype mismatch")
                if not _devices_match(actual.device, device):
                    raise ValueError("memory state device mismatch")
        return memory

def _devices_match(actual: torch.device, expected: torch.device) -> bool:
    """Compare devices while honoring an unspecified accelerator index."""
    if actual.type != expected.type:
        return False
    return expected.index is None or actual.index == expected.index



def _as_torch(value: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)


def _tree_to_torch_batch(value: Any, device: torch.device) -> Any:
    if isinstance(value, dict):
        return {key: _tree_to_torch_batch(item, device) for key, item in value.items()}
    if value is None:
        return None
    array = np.asarray(value)
    tensor = torch.as_tensor(array, device=device)
    if tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    return tensor[None, ...]


def _snapshot_memory_cpu(memory: MemorySnapshot) -> MemorySnapshot:
    return MemorySnapshot(
        memory.backend_id,
        memory.state_schema_version,
        memory.batch_size,
        tuple(tuple(tensor.detach().cpu().clone() for tensor in layer) for layer in memory.layers),
    )


def _clone_memory_to(memory: MemorySnapshot, device: torch.device) -> MemorySnapshot:
    return MemorySnapshot(
        memory.backend_id,
        memory.state_schema_version,
        memory.batch_size,
        tuple(tuple(tensor.detach().to(device=device).clone() for tensor in layer) for layer in memory.layers),
    )


def _memory_dtype(memory: MemorySnapshot) -> torch.dtype:
    return memory.layers[0][0].dtype if memory.layers else torch.float32


def _validate_buffer_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict) or payload.get("add_buffer") is not True:
        raise ValueError("RoboMME buffer payload must set add_buffer=true")
    if "images" not in payload or "state" not in payload or "exec_start_idx" not in payload:
        raise ValueError("RoboMME buffer payload requires images, state, and exec_start_idx")
    images = np.asarray(payload["images"])
    state = np.asarray(payload["state"])
    if images.ndim < 1 or state.ndim < 1 or images.shape[0] != state.shape[0] or images.shape[0] == 0:
        raise ValueError("RoboMME buffer images and state must have the same nonzero time dimension")
    exec_start_idx = int(payload["exec_start_idx"])
    if exec_start_idx < 0 or exec_start_idx >= images.shape[0]:
        raise ValueError("RoboMME buffer exec_start_idx must select a buffered frame")
    digest = hashlib.sha256()
    for name, array in (("images", images), ("state", state)):
        contiguous = np.ascontiguousarray(array)
        digest.update(name.encode("utf-8"))
        digest.update(str(contiguous.dtype).encode("utf-8"))
        digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
        digest.update(contiguous.tobytes())
    digest.update(str(exec_start_idx).encode("utf-8"))
    return {
        "buffer_checksum": digest.hexdigest(),
        "exec_start_idx": exec_start_idx,
        "num_frames": int(images.shape[0]),
    }


def _handoff_step_diagnostic(diagnostics: Mapping[str, Any] | None):
    if not diagnostics:
        raise ValueError("FutureMamba diagnostics must include handoff_step")
    value = diagnostics.get("handoff_steps", diagnostics.get("handoff_step"))
    if value is None:
        raise ValueError("FutureMamba diagnostics must include handoff_step")
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1)[0].item()
    array = np.asarray(value)
    return array.item() if array.shape == () else array.reshape(-1)[0].item()


def _memory_nbytes(memory: MemorySnapshot) -> int:
    return sum(tensor.numel() * tensor.element_size() for layer in memory.layers for tensor in layer)
