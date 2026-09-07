from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

import safetensors.torch
import torch

from openpi.training.futuremamba_checkpoint import _capture_rng_state, _restore_rng_state

WEIGHTS = "memory_ae.safetensors"
TRAINABLE_PREFIXES = (
    "futuremamba.",
    "base.paligemma_with_expert.gemma_expert.",
    "base.action_in_proj.",
    "base.action_out_proj.",
    "base.time_mlp_in.",
    "base.time_mlp_out.",
)


def adaptation_state(model):
    return {name: tensor for name, tensor in model.state_dict().items() if name.startswith(TRAINABLE_PREFIXES)}


def trainable_parameters(model):
    names, parameters = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            if not name.startswith(TRAINABLE_PREFIXES) or "progress_expert" in name:
                raise ValueError(f"Unexpected no-PE trainable parameter: {name}")
            names.append(name)
            parameters.append(parameter)
    if not parameters:
        raise ValueError("MemoryAE has no trainable parameters")
    return names, parameters


def frozen_vlm_checksum(model):
    digest = hashlib.sha256()
    for name, tensor in model.base.paligemma_with_expert.paligemma.state_dict().items():
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_base(model_config, device, *, weight_path=None):
    path = Path(weight_path or model_config.base_checkpoint_uri).expanduser()
    weights = path if path.name == "model.safetensors" else path / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    model = model_config.create_pytorch().to(device)
    safetensors.torch.load_model(model.base, weights, strict=True, device=str(device))
    model.freeze_base()
    return model, weights


def source_checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_policy_bundle(model_config, directory, device):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    validate_metadata(metadata, model_config.checkpoint_metadata())
    model, weights = load_base(model_config, device)
    if source_checksum(weights) != metadata["source_checkpoint_sha256"]:
        raise ValueError("MemoryAE source checkpoint identity mismatch")
    if frozen_vlm_checksum(model) != metadata["frozen_vlm_checksum"]:
        raise ValueError("MemoryAE frozen VLM identity mismatch")
    load_adaptation(model, directory, device)
    model.eval()
    return model, metadata, weights.parent


def load_adaptation(model, directory, device):
    directory = Path(directory)
    saved = safetensors.torch.load_file(directory / WEIGHTS, device=str(device))
    expected = adaptation_state(model)
    if saved.keys() != expected.keys():
        raise ValueError(
            f"MemoryAE weights mismatch: missing={expected.keys() - saved.keys()}, unexpected={saved.keys() - expected.keys()}"
        )
    for name, tensor in saved.items():
        if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
            raise ValueError(f"MemoryAE tensor shape/dtype mismatch: {name}")
    result = model.load_state_dict(saved, strict=False)
    if result.unexpected_keys or any(name in expected for name in result.missing_keys):
        raise ValueError("Incomplete MemoryAE adaptation load")


def validate_metadata(saved, expected):
    for key, value in expected.items():
        if value is not None and saved.get(key) != value:
            raise ValueError(f"MemoryAE checkpoint mismatch for {key}: expected {value!r}, got {saved.get(key)!r}")


def save_checkpoint(
    directory, model, optimizer, scheduler, *, step, metadata, data_iterator_step, processed_queries, elapsed_seconds
):
    save_started = time.perf_counter()
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    actual = frozen_vlm_checksum(model)
    if actual != metadata["frozen_vlm_checksum"]:
        raise ValueError("Frozen VLM changed during MemoryAE training")
    state = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in adaptation_state(model).items()}
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = directory.parent / f".{directory.name}.tmp-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        safetensors.torch.save_file(state, staging / WEIGHTS)
        torch.save(optimizer.state_dict(), staging / "optimizer.pt")
        torch.save(scheduler.state_dict(), staging / "scheduler.pt")
        torch.save(_capture_rng_state(), staging / "rng_state.pt")
        saved = dict(
            metadata,
            step=int(step),
            data_iterator_step=int(data_iterator_step),
            processed_queries=int(processed_queries),
            elapsed_seconds=float(elapsed_seconds) + time.perf_counter() - save_started,
        )
        saved["checkpoint_seconds"] = time.perf_counter() - save_started
        saved["cost_cutoff"] = "checkpoint_ready_before_atomic_rename"
        (staging / "metadata.json").write_text(json.dumps(saved, sort_keys=True) + "\n")
        os.replace(staging, directory)
    except BaseException:
        shutil.rmtree(staging)
        raise


def restore_checkpoint(directory, model, optimizer, scheduler, *, expected_metadata, device):
    directory = Path(directory)
    saved = json.loads((directory / "metadata.json").read_text())
    validate_metadata(saved, expected_metadata)
    if frozen_vlm_checksum(model) != saved["frozen_vlm_checksum"]:
        raise ValueError("Frozen VLM checksum mismatch")
    load_adaptation(model, directory, device)
    optimizer.load_state_dict(torch.load(directory / "optimizer.pt", map_location=device, weights_only=True))
    scheduler.load_state_dict(torch.load(directory / "scheduler.pt", map_location="cpu", weights_only=True))
    _restore_rng_state(torch.load(directory / "rng_state.pt", map_location="cpu", weights_only=True))
    return saved
