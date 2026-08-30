from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from openpi.models_pytorch.futuremamba import ActionHistoryState
from openpi.models_pytorch.mamba_memory import MemorySnapshot
from openpi_client import base_policy as _base_policy


@dataclasses.dataclass(frozen=True)
class FutureMambaPolicyState:
    backend_id: str
    state_schema_version: int
    history: ActionHistoryState
    episode_count: int
    query_count: int
    client_id: str


class FutureMambaPolicy(_base_policy.BasePolicy):
    """Visual PE-to-AE handoff policy with recurrent executed-action history."""

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
        if not hasattr(model, "initial_history_state") or not hasattr(model, "sample_actions_with_memory"):
            raise AttributeError("FutureMambaPolicy model must define history initialization and sampling")
        first_parameter = next(model.parameters(), None)
        self._device = torch.device(pytorch_device or ("cpu" if first_parameter is None else first_parameter.device))
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
        self._history = self._new_history()
        self._buffer_diagnostics: list[dict[str, Any]] = []

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = self._input_transform(dict(obs))
        executed_actions, executed_mask = self._executed_chunk(inputs)
        inputs.pop("executed_actions", None)
        inputs.pop("executed_action_mask", None)
        batched_inputs = _tree_to_torch_batch(inputs, self._device)
        observation = _model.Observation.from_dict(batched_inputs)
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise_tensor = _as_torch(noise, self._device, torch.float32)
            sample_kwargs["noise"] = noise_tensor[None] if noise_tensor.ndim == 2 else noise_tensor

        started = time.monotonic()
        with torch.no_grad():
            actions, next_history, diagnostics = self._sample_actions_with_memory(
                observation,
                self._history,
                executed_actions,
                executed_mask,
                **sample_kwargs,
            )
        elapsed = time.monotonic() - started
        self._history = self._validate_history(next_history, expected_batch=1, device=self._device)
        self._query_count += 1

        outputs = {
            "state": np.asarray(batched_inputs["state"][0].detach().cpu()),
            "actions": np.asarray(actions[0].detach().cpu()),
        }
        outputs = self._output_transform(outputs)
        for key in (
            "handoff_steps",
            "progress_calls",
            "action_calls",
            "memory_commits",
            "history_actions",
            "pending_actions",
        ):
            outputs[key if key != "handoff_steps" else "handoff_step"] = int(diagnostics[key])
        outputs["memory_state_bytes"] = _history_nbytes(self._history)
        outputs["policy_timing"] = {"infer_ms": elapsed * 1000}
        return outputs

    def _executed_chunk(self, transformed: dict[str, Any]) -> tuple[torch.Tensor, torch.BoolTensor]:
        chunk_size = int(self._model.config.action_history_chunk_size)
        action_dim = int(self._model.config.action_dim)
        raw = np.asarray(
            transformed.get("executed_actions", np.zeros((0, action_dim), dtype=np.float32)),
            dtype=np.float32,
        )
        if raw.ndim != 2 or raw.shape[0] > chunk_size or raw.shape[1] > action_dim:
            raise ValueError(
                f"executed_actions must fit [<= {chunk_size}, <= {action_dim}], got {raw.shape}"
            )
        supplied_mask = transformed.get("executed_action_mask")
        if supplied_mask is None:
            mask = np.ones(raw.shape[0], dtype=np.bool_)
        else:
            mask = np.asarray(supplied_mask, dtype=np.bool_)
            if mask.shape != (raw.shape[0],):
                raise ValueError(
                    f"executed_action_mask must have shape {(raw.shape[0],)}, got {mask.shape}"
                )
        valid_count = int(mask.sum())
        if not np.array_equal(mask, np.arange(raw.shape[0]) < valid_count):
            raise ValueError("executed_action_mask must be a right-padded prefix")
        padded = np.zeros((raw.shape[0], action_dim), dtype=np.float32)
        padded[:, : raw.shape[1]] = raw
        return (
            torch.as_tensor(padded, device=self._device)[None],
            torch.as_tensor(mask, device=self._device, dtype=torch.bool)[None],
        )

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
        self._history = self._new_history()
        self._buffer_diagnostics.clear()

    @override
    def snapshot_state(self) -> FutureMambaPolicyState:
        history = _clone_history_to(self._history, torch.device("cpu"))
        return FutureMambaPolicyState(
            backend_id=history.memory.backend_id,
            state_schema_version=history.memory.state_schema_version,
            history=history,
            episode_count=self._episode_count,
            query_count=self._query_count,
            client_id=self._client_id,
        )

    @override
    def restore_state(self, state: FutureMambaPolicyState) -> None:
        if not isinstance(state, FutureMambaPolicyState):
            raise TypeError("restore_state requires FutureMambaPolicyState")
        if state.episode_count < 0 or state.query_count < 0:
            raise ValueError("episode_count and query_count must be non-negative")
        restored = _clone_history_to(state.history, self._device)
        restored = self._validate_history(restored, expected_batch=1, device=self._device)
        if state.backend_id != restored.memory.backend_id:
            raise ValueError("history backend identity mismatch")
        if state.state_schema_version != restored.memory.state_schema_version:
            raise ValueError("history schema identity mismatch")
        self._history = restored
        self._episode_count = int(state.episode_count)
        self._query_count = int(state.query_count)

    @override
    def fork(self) -> "FutureMambaPolicy":
        self._next_fork_id += 1
        fork = FutureMambaPolicy(
            self._model,
            transforms=self._transforms,
            output_transforms=self._output_transforms,
            sample_kwargs=self._sample_kwargs,
            metadata=self._metadata,
            pytorch_device=self._device,
            client_id=f"{self._client_id}:{self._next_fork_id}:{uuid.uuid4().hex}",
            _sample_actions_with_memory=self._sample_actions_with_memory,
        )
        fork._history = _clone_history_to(self._history, self._device)
        fork._episode_count = self._episode_count
        fork._query_count = self._query_count
        return fork

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _new_history(self) -> ActionHistoryState:
        dtype = next(self._model.futuremamba.parameters()).dtype
        history = self._model.initial_history_state(1, self._device, dtype)
        return self._validate_history(history, expected_batch=1, device=self._device)

    def _validate_history(
        self, history: ActionHistoryState, *, expected_batch: int, device: torch.device
    ) -> ActionHistoryState:
        if not isinstance(history, ActionHistoryState):
            raise TypeError("history must be ActionHistoryState")
        backend = self._model.futuremamba.memory_backend
        memory = history.memory
        if memory.backend_id != backend.backend_id or memory.state_schema_version != backend.state_schema_version:
            raise ValueError("history memory identity mismatch")
        if memory.batch_size != expected_batch:
            raise ValueError("history memory batch mismatch")
        if history.committed_output.shape != (expected_batch, self._model.futuremamba.memory_width):
            raise ValueError("history committed output shape mismatch")
        if history.committed_chunks.shape != (expected_batch,):
            raise ValueError("history committed chunk shape mismatch")
        if history.pending_actions.shape != (
            expected_batch,
            int(self._model.config.action_history_chunk_size),
            int(self._model.config.action_dim),
        ):
            raise ValueError("history pending action shape mismatch")
        if history.pending_mask.shape != history.pending_actions.shape[:2]:
            raise ValueError("history pending mask shape mismatch")
        tensors = [
            history.committed_output,
            history.committed_chunks,
            history.pending_actions,
            history.pending_mask,
            *(tensor for layer in memory.layers for tensor in layer),
        ]
        if any(tensor.device.type != device.type for tensor in tensors):
            raise ValueError("history device mismatch")
        return history


def _as_torch(value: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(np.asarray(value), device=device, dtype=dtype)


def _tree_to_torch_batch(value: Any, device: torch.device) -> Any:
    if isinstance(value, Mapping):
        return {key: _tree_to_torch_batch(item, device) for key, item in value.items()}
    if value is None:
        return None
    tensor = torch.as_tensor(np.asarray(value), device=device)
    if tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    return tensor[None]


def _clone_history_to(history: ActionHistoryState, device: torch.device) -> ActionHistoryState:
    return ActionHistoryState(
        memory=MemorySnapshot(
            history.memory.backend_id,
            history.memory.state_schema_version,
            history.memory.batch_size,
            tuple(
                tuple(tensor.detach().to(device=device).clone() for tensor in layer)
                for layer in history.memory.layers
            ),
        ),
        committed_output=history.committed_output.detach().to(device=device).clone(),
        committed_chunks=history.committed_chunks.detach().to(device=device).clone(),
        pending_actions=history.pending_actions.detach().to(device=device).clone(),
        pending_mask=history.pending_mask.detach().to(device=device).clone(),
    )


def _history_nbytes(history: ActionHistoryState) -> int:
    tensors = [
        history.committed_output,
        history.committed_chunks,
        history.pending_actions,
        history.pending_mask,
        *(tensor for layer in history.memory.layers for tensor in layer),
    ]
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _validate_buffer_payload(payload: dict[str, Any]) -> dict[str, Any]:
    digest = hashlib.sha256()
    for key in sorted(payload):
        value = np.ascontiguousarray(np.asarray(payload[key]))
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(value.tobytes())
    return {
        "buffer_checksum": digest.hexdigest(),
        "exec_start_idx": int(payload.get("exec_start_idx", 0)),
        "num_frames": int(np.asarray(payload.get("state", ())).shape[0]),
    }
