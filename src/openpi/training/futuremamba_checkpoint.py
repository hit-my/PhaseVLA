from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import json
import os
from pathlib import Path
import shutil
from typing import Any
import uuid

import safetensors.torch
import torch
from torch import nn


_METADATA_FIELDS = (
    "schema_version",
    "architecture",
    "base_checkpoint_uri",
    "base_checkpoint_checksum",
    "base_assets_checksum",
    "assets_uri",
    "dataset_uri",
    "dataset_checksum",
    "task_name",
    "train_seed",
    "mamba_repo_commit",
    "memory_backend",
    "memory_state_schema_version",
    "history_state_schema_version",
    "memory_config",
    "progress_depth",
    "progress_layer_mapping",
    "handoff_ratio",
    "denoising_order",
    "num_denoise_steps",
    "progress_denoise_steps",
    "prediction_horizon",
    "execution_horizon",
    "memory_input_source",
    "action_history_encoding",
    "action_history_chunk_size",
    "training_query_stride",
    "memory_update_timing",
    "partial_chunk_behavior",
    "empty_history_behavior",
    "uses_vlm_hidden_for_memory",
    "uses_prefix_kv_for_progress",
    "loss_weights",
    "frozen_prefix_microbatch_size",
    "training_dtype",
    "state_dtypes",
    "kernel_mode",
    "torch_version",
    "triton_version",
    "cuda_version",
    "gpu_name",
    "compute_capability",
)
_FILES = (
    "plugin.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pt",
    "metadata.json",
)


@dataclasses.dataclass(frozen=True)
class RestoredFutureMambaCheckpoint:
    step: int
    data_iterator_step: int
    metadata: dict[str, Any]


def save_futuremamba_checkpoint(
    directory: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    step: int,
    metadata: Mapping[str, Any],
    data_iterator_step: int | None = None,
) -> None:
    if step < 0:
        raise ValueError("step must be non-negative")
    if data_iterator_step is None:
        data_iterator_step = step
    if data_iterator_step < 0:
        raise ValueError("data_iterator_step must be non-negative")
    normalized = _validate_metadata(metadata)
    if hasattr(model, "base_checksum"):
        actual_base_checksum = model.base_checksum()
        if normalized["base_checkpoint_checksum"] != actual_base_checksum:
            raise ValueError(
                "base_checkpoint_checksum mismatch before save: "
                f"metadata={normalized['base_checkpoint_checksum']!r}, model={actual_base_checksum!r}"
            )
    plugin_state = _plugin_state_for_save(model)
    if not plugin_state:
        raise ValueError("FutureMamba plugin has no parameters or buffers")

    destination = Path(directory)
    if destination.exists():
        raise ValueError(f"checkpoint directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir(exist_ok=False)
    try:
        safetensors.torch.save_file(plugin_state, staging / "plugin.safetensors")
        torch.save(optimizer.state_dict(), staging / "optimizer.pt")
        torch.save(scheduler.state_dict(), staging / "scheduler.pt")
        torch.save(_capture_rng_state(), staging / "rng_state.pt")
        saved_metadata = {**normalized, "step": int(step), "data_iterator_step": int(data_iterator_step)}
        _atomic_json(staging / "metadata.json", saved_metadata)
        missing = [name for name in _FILES if not (staging / name).is_file()]
        if missing:
            raise RuntimeError(f"incomplete FutureMamba checkpoint: missing {missing}")
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_futuremamba_checkpoint(
    directory: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    *,
    expected_metadata: Mapping[str, Any],
    map_location: torch.device | str | None = None,
) -> RestoredFutureMambaCheckpoint:
    root = Path(directory)
    missing = [name for name in _FILES if not (root / name).is_file()]
    if missing:
        raise ValueError(f"incomplete FutureMamba checkpoint: missing {missing[0]}")
    try:
        saved_metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid FutureMamba metadata.json: {error}") from error
    saved_identity = _validate_metadata(saved_metadata)
    expected = _validate_metadata(expected_metadata)
    for field in _METADATA_FIELDS:
        if saved_identity[field] != expected[field]:
            raise ValueError(
                f"FutureMamba checkpoint identity mismatch for {field}: "
                f"expected {expected[field]!r}, got {saved_identity[field]!r}"
            )
    if hasattr(model, "base_checksum"):
        actual_base_checksum = model.base_checksum()
        if expected["base_checkpoint_checksum"] != actual_base_checksum:
            raise ValueError(
                "base_checkpoint_checksum mismatch for current model: "
                f"expected {expected['base_checkpoint_checksum']!r}, got {actual_base_checksum!r}"
            )

    device = _model_device(model) if map_location is None else torch.device(map_location)
    try:
        state = safetensors.torch.load_file(root / "plugin.safetensors", device=str(device))
    except Exception as error:
        raise ValueError(f"invalid plugin.safetensors: {error}") from error
    _strict_load_plugin(model, state)
    try:
        optimizer.load_state_dict(torch.load(root / "optimizer.pt", map_location=device, weights_only=True))
        scheduler.load_state_dict(torch.load(root / "scheduler.pt", map_location="cpu", weights_only=True))
        rng_state = torch.load(root / "rng_state.pt", map_location="cpu", weights_only=True)
        _restore_rng_state(rng_state)
    except Exception as error:
        raise ValueError(f"invalid FutureMamba runtime state: {error}") from error
    step = saved_metadata.get("step")
    data_iterator_step = saved_metadata.get("data_iterator_step")
    if not isinstance(step, int) or step < 0:
        raise ValueError("metadata field step must be a non-negative integer")
    if not isinstance(data_iterator_step, int) or data_iterator_step < 0:
        raise ValueError("metadata field data_iterator_step must be a non-negative integer")
    return RestoredFutureMambaCheckpoint(step=step, data_iterator_step=data_iterator_step, metadata=saved_metadata)


def _validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise ValueError("FutureMamba checkpoint metadata must be a mapping")
    missing = [field for field in _METADATA_FIELDS if field not in metadata]
    if missing:
        raise ValueError(f"FutureMamba checkpoint metadata missing {missing[0]}")
    return {field: _json_value(metadata[field], field=field) for field in _METADATA_FIELDS}


def _json_value(value: Any, *, field: str) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"FutureMamba checkpoint metadata field {field} is not JSON serializable") from error
    return value


def _base_checksum(model: nn.Module) -> str:
    checksum = getattr(model, "base_checksum", None)
    if not callable(checksum):
        raise ValueError("FutureMamba model must provide base_checksum()")
    value = checksum()
    if not isinstance(value, str) or not value:
        raise ValueError("base_checksum() must return a non-empty string")
    return value


def _plugin_module(model: nn.Module) -> nn.Module:
    plugin = getattr(model, "futuremamba", None)
    if not isinstance(plugin, nn.Module):
        raise ValueError("model.futuremamba must be a torch module")
    return plugin


def _plugin_state_for_save(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        f"futuremamba.{name}": tensor.detach().cpu().contiguous().clone()
        for name, tensor in _plugin_module(model).state_dict().items()
    }


def _strict_load_plugin(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    if not state or any(not key.startswith("futuremamba.") for key in state):
        raise ValueError("plugin.safetensors keys must all start with futuremamba.")
    plugin_state = {key.removeprefix("futuremamba."): tensor for key, tensor in state.items()}
    expected = _plugin_module(model).state_dict()
    missing = sorted(set(expected) - set(plugin_state))
    unexpected = sorted(set(plugin_state) - set(expected))
    if missing or unexpected:
        raise ValueError(f"strict plugin load failed: missing={missing}, unexpected={unexpected}")
    for name, tensor in plugin_state.items():
        reference = expected[name]
        if tensor.shape != reference.shape:
            raise ValueError(f"strict plugin load shape mismatch for {name}: {tuple(tensor.shape)} != {tuple(reference.shape)}")
        if tensor.dtype != reference.dtype:
            raise ValueError(f"strict plugin load dtype mismatch for {name}: {tensor.dtype} != {reference.dtype}")
    _plugin_module(model).load_state_dict(plugin_state, strict=True)


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {"cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping) or "cpu" not in state or not torch.is_tensor(state["cpu"]):
        raise ValueError("rng_state.pt is missing Torch CPU RNG")
    torch.set_rng_state(state["cpu"].cpu())
    cuda_state = state.get("cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA RNG but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda_state)


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return torch.device("cpu") if parameter is None else parameter.device


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    content = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
