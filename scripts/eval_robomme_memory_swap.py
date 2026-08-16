#!/usr/bin/env python3
from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from openpi.models import model as _model
from openpi.models_pytorch.futuremamba_config import FutureMambaPytorchConfig, MambaMemoryConfig
from openpi.models_pytorch.gemma_pytorch import PrefixKVView, detach_cache
from openpi.models_pytorch.mamba_memory import MemorySnapshot
from openpi.policies.policy_config import _load_futuremamba_bundle

OUTPUT_SCHEMA_VERSION = "futuremamba.robomme.memory_swap.v1"
ARTIFACT_SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class InterventionLabels:
    progress_probe: Sequence[Any] | None = None
    branch: Any | None = None
    stop: Any | None = None
    episode_success: bool | None = None
    provenance: Mapping[str, Any] | None = None


@dataclasses.dataclass(frozen=True)
class InterventionArtifact:
    path: Path
    artifact_schema_version: int
    episode_id: str
    query_a_id: str
    query_b_id: str
    current_query_id: str
    observation: Mapping[str, Any]
    physical_state: Any
    prompt: str
    noise: torch.Tensor
    num_steps: int
    handoff_step: int
    memory_a: MemorySnapshot
    memory_b: MemorySnapshot
    provenance: Mapping[str, Any]
    labels: InterventionLabels | None = None


@dataclasses.dataclass(frozen=True)
class Args:
    bundle: str
    artifact: str
    output: str
    device: str = "cpu"


def tensor_checksum(tensor: torch.Tensor | np.ndarray) -> str:
    tensor = _as_cpu_tensor(tensor)
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tree_checksum(value: Any) -> str:
    digest = hashlib.sha256()
    _update_tree_digest(digest, value)
    return digest.hexdigest()


def memory_snapshot_checksum(memory: MemorySnapshot) -> str:
    validate_memory_snapshot_structure(memory)
    digest = hashlib.sha256()
    digest.update(memory.backend_id.encode("utf-8"))
    digest.update(str(memory.state_schema_version).encode("utf-8"))
    digest.update(str(memory.batch_size).encode("utf-8"))
    for layer in memory.layers:
        digest.update(str(len(layer)).encode("utf-8"))
        for tensor in layer:
            digest.update(tensor_checksum(tensor).encode("utf-8"))
    return digest.hexdigest()


def serialize_memory_snapshot(memory: MemorySnapshot) -> dict[str, Any]:
    validate_memory_snapshot_structure(memory)
    return {
        "backend_id": memory.backend_id,
        "state_schema_version": memory.state_schema_version,
        "batch_size": memory.batch_size,
        "layers": [
            [
                {
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "shape": list(tensor.shape),
                    "data_b64": base64.b64encode(
                        tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    ).decode("ascii"),
                }
                for tensor in layer
            ]
            for layer in memory.layers
        ],
    }


def deserialize_memory_snapshot(payload: Any) -> MemorySnapshot:
    if not isinstance(payload, Mapping):
        raise ValueError("memory snapshot must be an object")
    for field in ("backend_id", "state_schema_version", "batch_size", "layers"):
        if field not in payload:
            raise ValueError(f"memory snapshot missing {field}")
    backend_id = payload["backend_id"]
    if not isinstance(backend_id, str) or not backend_id:
        raise ValueError("memory backend_id must be a non-empty string")
    schema = _strict_int(payload["state_schema_version"], "memory state_schema_version")
    batch = _strict_int(payload["batch_size"], "memory batch_size")
    raw_layers = payload["layers"]
    if not isinstance(raw_layers, Sequence) or isinstance(raw_layers, (str, bytes, bytearray)):
        raise ValueError("memory layers must be a sequence")
    layers = []
    for layer_index, raw_layer in enumerate(raw_layers):
        if not isinstance(raw_layer, Sequence) or isinstance(raw_layer, (str, bytes, bytearray)):
            raise ValueError(f"memory layer {layer_index} must be a sequence")
        tensors = []
        for tensor_index, raw_tensor in enumerate(raw_layer):
            tensors.append(_deserialize_tensor(raw_tensor, f"memory layer {layer_index} tensor {tensor_index}"))
        layers.append(tuple(tensors))
    memory = MemorySnapshot(backend_id, schema, batch, tuple(layers))
    validate_memory_snapshot_structure(memory)
    return memory


def validate_memory_snapshot_structure(memory: MemorySnapshot) -> None:
    if not isinstance(memory, MemorySnapshot):
        raise TypeError("memory must be MemorySnapshot")
    if not isinstance(memory.backend_id, str) or not memory.backend_id:
        raise ValueError("memory backend_id must be a non-empty string")
    if not isinstance(memory.state_schema_version, int):
        raise ValueError("memory state_schema_version must be an integer")
    if not isinstance(memory.batch_size, int) or memory.batch_size <= 0:
        raise ValueError("memory batch_size must be positive")
    if not isinstance(memory.layers, tuple):
        raise ValueError("memory layers must be a tuple")
    for layer_index, layer in enumerate(memory.layers):
        if not isinstance(layer, tuple):
            raise ValueError(f"memory layer {layer_index} must be a tuple")
        for tensor_index, tensor in enumerate(layer):
            if not torch.is_tensor(tensor):
                raise ValueError(f"memory layer {layer_index} tensor {tensor_index} must be a tensor")
            if tensor.shape[:1] != (memory.batch_size,):
                raise ValueError(
                    f"memory layer {layer_index} tensor {tensor_index} batch mismatch: "
                    f"expected {memory.batch_size}, got shape {tuple(tensor.shape)}"
                )
            if not tensor.is_floating_point():
                raise ValueError(f"memory layer {layer_index} tensor {tensor_index} must be floating point")
            if torch.any(~torch.isfinite(tensor.detach())):
                raise ValueError(f"memory layer {layer_index} tensor {tensor_index} contains non-finite values")


def validate_memory_snapshot(memory: MemorySnapshot, expected: MemorySnapshot) -> None:
    validate_memory_snapshot_structure(memory)
    validate_memory_snapshot_structure(expected)
    if memory.backend_id != expected.backend_id:
        raise ValueError(f"memory backend mismatch: expected {expected.backend_id!r}, got {memory.backend_id!r}")
    if memory.state_schema_version != expected.state_schema_version:
        raise ValueError(
            "memory schema mismatch: "
            f"expected {expected.state_schema_version}, got {memory.state_schema_version}"
        )
    if memory.batch_size != expected.batch_size:
        raise ValueError(f"memory batch mismatch: expected {expected.batch_size}, got {memory.batch_size}")
    if len(memory.layers) != len(expected.layers):
        raise ValueError(f"memory layer shape mismatch: expected {len(expected.layers)} layers, got {len(memory.layers)}")
    for layer_index, (actual_layer, expected_layer) in enumerate(zip(memory.layers, expected.layers, strict=True)):
        if len(actual_layer) != len(expected_layer):
            raise ValueError(
                f"memory layer {layer_index} shape mismatch: expected {len(expected_layer)} tensors, got {len(actual_layer)}"
            )
        for tensor_index, (actual, reference) in enumerate(zip(actual_layer, expected_layer, strict=True)):
            if actual.shape != reference.shape:
                raise ValueError(
                    f"memory shape mismatch at layer {layer_index} tensor {tensor_index}: "
                    f"expected {tuple(reference.shape)}, got {tuple(actual.shape)}"
                )
            if actual.dtype != reference.dtype:
                raise ValueError(
                    f"memory dtype mismatch at layer {layer_index} tensor {tensor_index}: "
                    f"expected {reference.dtype}, got {actual.dtype}"
                )


def load_intervention_artifact(path: str | Path) -> InterventionArtifact:
    path = Path(path)
    payload = _load_json_object(path, "intervention artifact")
    if "error" in payload:
        raise ValueError("intervention artifact contains error")
    for field in (
        "artifact_schema_version",
        "episode_id",
        "query_a_id",
        "query_b_id",
        "current_query_id",
        "observation",
        "physical_state",
        "prompt",
        "noise",
        "num_steps",
        "handoff_step",
        "memory_a",
        "memory_b",
        "provenance",
    ):
        if field not in payload:
            raise ValueError(f"intervention artifact missing {field}")
    schema = _strict_int(payload["artifact_schema_version"], "artifact_schema_version")
    if schema != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported artifact_schema_version {schema}")
    episode_id = _non_empty_str(payload["episode_id"], "episode_id")
    query_a_id = _non_empty_str(payload["query_a_id"], "query_a_id")
    query_b_id = _non_empty_str(payload["query_b_id"], "query_b_id")
    current_query_id = _non_empty_str(payload["current_query_id"], "current_query_id")
    if query_a_id == query_b_id:
        raise ValueError("memory snapshots must come from different queries")
    prompt = payload["prompt"]
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("prompt must be a non-empty string")
    observation = payload["observation"]
    if not isinstance(observation, Mapping):
        raise ValueError("observation must be an object")
    physical_state = payload["physical_state"]
    provenance = payload["provenance"]
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("provenance must be a non-empty object")
    noise = _tensor_from_json(payload["noise"], "noise")
    if noise.ndim != 3:
        raise ValueError(f"noise must have shape [batch, horizon, action_dim], got {tuple(noise.shape)}")
    if torch.any(~torch.isfinite(noise)):
        raise ValueError("noise contains non-finite values")
    num_steps = _strict_int(payload["num_steps"], "num_steps")
    handoff_step = _strict_int(payload["handoff_step"], "handoff_step")
    if num_steps <= 0:
        raise ValueError("num_steps must be positive")
    if not 0 <= handoff_step <= num_steps:
        raise ValueError("handoff_step must be in [0, num_steps]")
    memory_a = deserialize_memory_snapshot(payload["memory_a"])
    memory_b = deserialize_memory_snapshot(payload["memory_b"])
    validate_memory_snapshot(memory_b, memory_a)
    if memory_snapshot_checksum(memory_a) == memory_snapshot_checksum(memory_b):
        raise ValueError("memory checksums must differ")
    labels = _parse_labels(payload.get("labels"))
    return InterventionArtifact(
        path=path,
        artifact_schema_version=schema,
        episode_id=episode_id,
        query_a_id=query_a_id,
        query_b_id=query_b_id,
        current_query_id=current_query_id,
        observation=observation,
        physical_state=physical_state,
        prompt=prompt,
        noise=noise,
        num_steps=num_steps,
        handoff_step=handoff_step,
        memory_a=memory_a,
        memory_b=memory_b,
        provenance=provenance,
        labels=labels,
    )


def intervention_identity(artifact: InterventionArtifact) -> dict[str, Any]:
    memory_a = _memory_identity(artifact.memory_a)
    memory_b = _memory_identity(artifact.memory_b)
    inputs = {
        "observation_checksum": tree_checksum(artifact.observation),
        "physical_state_checksum": tree_checksum(artifact.physical_state),
        "prompt_checksum": tree_checksum(artifact.prompt),
        "noise_checksum": tensor_checksum(artifact.noise),
        "solver_schedule_checksum": tree_checksum({"num_steps": artifact.num_steps, "handoff_step": artifact.handoff_step}),
    }
    return {
        "episode_id": artifact.episode_id,
        "current_query_id": artifact.current_query_id,
        "query_a_id": artifact.query_a_id,
        "query_b_id": artifact.query_b_id,
        "inputs": inputs,
        "memory_a": memory_a,
        "memory_b": memory_b,
        "only_memory_differs": memory_a["checksum"] != memory_b["checksum"],
    }


def evaluate_memory_swap(model: torch.nn.Module, artifact: InterventionArtifact, *, device: torch.device) -> dict[str, Any]:
    _validate_model_contract(model)
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    dtype = next(model.futuremamba.parameters(), torch.empty(0, dtype=torch.float32)).dtype
    expected = model.initial_memory_state(artifact.memory_a.batch_size, device, dtype)
    memory_a = _memory_to_device(artifact.memory_a, device=device)
    memory_b = _memory_to_device(artifact.memory_b, device=device)
    validate_memory_snapshot(memory_a, expected)
    validate_memory_snapshot(memory_b, expected)
    observation = _observation_to_device(artifact.observation, device=device, batch_size=artifact.memory_a.batch_size)
    _validate_observation_contract(observation, batch_size=artifact.memory_a.batch_size, device=device)
    noise = artifact.noise.to(device=device, dtype=torch.float32)
    expected_noise_shape = (
        artifact.memory_a.batch_size,
        int(model.config.action_horizon),
        int(model.config.action_dim),
    )
    if tuple(noise.shape) != expected_noise_shape:
        raise ValueError(f"noise must have shape {expected_noise_shape}, got {tuple(noise.shape)}")
    configured_steps = int(getattr(model.config, "num_denoise_steps", artifact.num_steps))
    if configured_steps != artifact.num_steps:
        raise ValueError(f"artifact num_steps {artifact.num_steps} does not match model config {configured_steps}")
    with torch.no_grad():
        rollout_a = _instrumented_rollout(model, observation, memory_a, noise, artifact.num_steps, artifact.handoff_step)
        rollout_b = _instrumented_rollout(model, observation, memory_b, noise, artifact.num_steps, artifact.handoff_step)
    identity = intervention_identity(artifact)
    if not identity["only_memory_differs"]:
        raise ValueError("intervention identity is invalid: memory checksums must differ")
    step_metrics = _step_metrics(rollout_a, rollout_b)
    handoff = _handoff_metrics(step_metrics, artifact.handoff_step)
    reports, unsupported = _optional_reports(artifact.labels, step_metrics)
    output = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "artifact_schema_version": artifact.artifact_schema_version,
        "input_identity": identity,
        "backend": artifact.memory_a.backend_id,
        "state_schema_version": artifact.memory_a.state_schema_version,
        "handoff_step": artifact.handoff_step,
        "num_steps": artifact.num_steps,
        "steps": step_metrics,
        "trajectory_a": _tensor_sequence_to_json(rollout_a["states"]),
        "trajectory_b": _tensor_sequence_to_json(rollout_b["states"]),
        "handoff": handoff,
        "final_action_chunk_distance_l2": step_metrics[-1]["state_distance_l2"] if step_metrics else 0.0,
        "final_action_chunk_distance_mean_abs": step_metrics[-1]["state_distance_mean_abs"] if step_metrics else 0.0,
        "reports": reports,
        "unsupported_reports": unsupported,
        "provenance": {"artifact": dict(artifact.provenance)},
    }
    if artifact.labels and artifact.labels.provenance:
        output["provenance"]["labels"] = dict(artifact.labels.provenance)
    return _canonicalize(output)


def main(args: Args) -> None:
    device = torch.device(args.device)
    artifact = load_intervention_artifact(args.artifact)
    bundle_dir = Path(args.bundle)
    metadata_hint = _read_bundle_metadata(bundle_dir)
    config = _config_from_metadata(metadata_hint, artifact=artifact)

    model, bundle_metadata, base_root = _load_futuremamba_bundle(config, bundle_dir, device)
    result = evaluate_memory_swap(model, artifact, device=device)
    result["bundle_metadata"] = _canonicalize(bundle_metadata)
    result["base_checkpoint_root"] = str(base_root)
    result = _canonicalize(result)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def _instrumented_rollout(
    model: torch.nn.Module,
    observation: _model.Observation,
    memory_state: MemorySnapshot,
    noise: torch.Tensor,
    num_steps: int,
    handoff_step: int,
) -> dict[str, Any]:
    batch_size = int(observation.state.shape[0])
    *_, processed_state = model.base._preprocess_observation(observation, train=False)
    frozen = model._detached_frozen_prefix(model.base.encode_frozen_prefix(observation, train=False)) if hasattr(model, "_detached_frozen_prefix") else _detach_frozen_prefix(model.base.encode_frozen_prefix(observation, train=False))
    last_vlm = model.base.last_valid_prefix(frozen).detach()
    memory_token, next_memory = model.futuremamba.compute_memory_token(last_vlm, memory_state)
    prefix_cache = PrefixKVView.from_cache(frozen.kv_cache, model.futuremamba.progress_layer_indices, frozen.pad_mask) if hasattr(model.futuremamba, "progress_layer_indices") else frozen.kv_cache
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=noise.device)
    x_t = noise.detach().clone()
    states: list[torch.Tensor] = []
    velocities: list[torch.Tensor] = []
    branches: list[str] = []
    for step in range(num_steps):
        timestep = torch.full((batch_size,), 1.0 + step * float(dt.item()), dtype=torch.float32, device=noise.device)
        if step < handoff_step:
            branch = "progress"
            velocity = model.futuremamba.forward_progress(prefix_cache, frozen.pad_mask, memory_token, x_t, timestep)
        else:
            branch = "action"
            velocity = model.base.denoise_step(processed_state, frozen.pad_mask, frozen.kv_cache, x_t, timestep)
        if tuple(velocity.shape) != tuple(x_t.shape):
            raise ValueError(f"{branch} velocity shape must be {tuple(x_t.shape)}, got {tuple(velocity.shape)}")
        if velocity.dtype != x_t.dtype:
            velocity = velocity.to(dtype=x_t.dtype)
        if torch.any(~torch.isfinite(velocity)):
            raise ValueError(f"{branch} velocity contains non-finite values")
        x_t = x_t + dt.to(dtype=x_t.dtype) * velocity
        velocities.append(velocity.detach().cpu())
        states.append(x_t.detach().cpu())
        branches.append(branch)
    return {"states": states, "velocities": velocities, "branches": branches, "next_memory": next_memory}


def _step_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, Any]]:
    metrics = []
    for index, (left_state, right_state, left_velocity, right_velocity, left_branch, right_branch) in enumerate(
        zip(left["states"], right["states"], left["velocities"], right["velocities"], left["branches"], right["branches"], strict=True)
    ):
        if left_branch != right_branch:
            raise ValueError(f"rollout branch mismatch at step {index}: {left_branch} != {right_branch}")
        state_delta = left_state.float() - right_state.float()
        velocity_delta = left_velocity.float() - right_velocity.float()
        metrics.append(
            {
                "step": index,
                "branch": left_branch,
                "state_distance_l2": _l2(state_delta),
                "state_distance_mean_abs": _mean_abs(state_delta),
                "velocity_distance_l2": _l2(velocity_delta),
                "velocity_distance_mean_abs": _mean_abs(velocity_delta),
                "action_state_checksum_a": tensor_checksum(left_state),
                "action_state_checksum_b": tensor_checksum(right_state),
            }
        )
    return metrics


def _handoff_metrics(step_metrics: Sequence[Mapping[str, Any]], handoff_step: int) -> dict[str, float | int]:
    if not step_metrics:
        return {"pre_step": -1, "post_step": -1, "pre_error_l2": 0.0, "post_error_l2": 0.0}
    if handoff_step <= 0:
        return {
            "pre_step": -1,
            "post_step": 0,
            "pre_error_l2": 0.0,
            "post_error_l2": float(step_metrics[0]["state_distance_l2"]),
        }
    pre_index = min(handoff_step - 1, len(step_metrics) - 1)
    post_index = min(handoff_step, len(step_metrics) - 1)
    return {
        "pre_step": pre_index,
        "post_step": post_index,
        "pre_error_l2": float(step_metrics[pre_index]["state_distance_l2"]),
        "post_error_l2": float(step_metrics[post_index]["state_distance_l2"]),
    }


def _optional_reports(labels: InterventionLabels | None, step_metrics: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    if labels is None:
        return {}, [
            "progress_probe_accuracy requires artifact.labels.progress_probe with provenance",
            "branch_stop_behavior requires artifact.labels.branch/stop with provenance",
            "episode_success requires artifact.labels.episode_success with provenance",
        ]
    reports: dict[str, Any] = {}
    unsupported = []
    if labels.progress_probe is None:
        unsupported.append("progress_probe_accuracy requires artifact.labels.progress_probe with provenance")
    else:
        pairs = _progress_probe_pairs(labels.progress_probe)
        correct = sum(predicted == truth for predicted, truth in pairs)
        reports["progress_probe_accuracy"] = correct / len(pairs)
    if labels.branch is None or labels.stop is None:
        unsupported.append("branch_stop_behavior requires artifact.labels.branch/stop with provenance")
    else:
        reports["branch_stop_behavior"] = {"branch": labels.branch, "stop": labels.stop}
    if labels.episode_success is None:
        unsupported.append("episode_success requires artifact.labels.episode_success with provenance")
    else:
        reports["episode_success"] = bool(labels.episode_success)
    return reports, unsupported


def _progress_probe_pairs(payload: Sequence[Any]) -> list[tuple[Any, Any]]:
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise ValueError("progress_probe labels must be a sequence")
    pairs: list[tuple[Any, Any]] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping) or "predicted" not in item or "label" not in item:
            raise ValueError(f"progress_probe item {index} must contain predicted and label")
        pairs.append((item["predicted"], item["label"]))
    if not pairs:
        raise ValueError("progress_probe labels must contain at least one item")
    return pairs


def _parse_labels(payload: Any) -> InterventionLabels | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping):
        raise ValueError("labels must be an object")
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError("label provenance is required when labels are provided")
    progress_probe = payload.get("progress_probe")
    if progress_probe is not None:
        _progress_probe_pairs(progress_probe)
    episode_success = payload.get("episode_success")
    if episode_success is not None and not isinstance(episode_success, bool):
        raise ValueError("episode_success label must be boolean")
    return InterventionLabels(
        progress_probe=progress_probe,
        branch=payload.get("branch"),
        stop=payload.get("stop"),
        episode_success=episode_success,
        provenance=provenance,
    )


def _validate_model_contract(model: torch.nn.Module) -> None:
    for attr in ("base", "futuremamba", "config", "initial_memory_state"):
        if not hasattr(model, attr):
            raise TypeError(f"FutureMamba model missing {attr}")
    for attr in ("_preprocess_observation", "encode_frozen_prefix", "last_valid_prefix", "denoise_step"):
        if not hasattr(model.base, attr):
            raise TypeError(f"FutureMamba base missing {attr}")
    for attr in ("compute_memory_token", "forward_progress", "memory_backend"):
        if not hasattr(model.futuremamba, attr):
            raise TypeError(f"FutureMamba plugin missing {attr}")


def _validate_observation_contract(observation: _model.Observation, *, batch_size: int, device: torch.device) -> None:
    if not torch.is_tensor(observation.state):
        raise ValueError("observation.state must be a tensor")
    if observation.state.ndim != 2:
        raise ValueError(f"observation.state must have shape [batch, state_dim], got {tuple(observation.state.shape)}")
    if observation.state.shape[0] != batch_size:
        raise ValueError(f"observation batch mismatch: expected {batch_size}, got {observation.state.shape[0]}")
    if observation.state.device != device:
        raise ValueError(f"observation device mismatch: expected {device}, got {observation.state.device}")
    if torch.any(~torch.isfinite(observation.state)):
        raise ValueError("observation.state contains non-finite values")


def _observation_to_device(value: Mapping[str, Any], *, device: torch.device, batch_size: int) -> _model.Observation:
    converted = _tree_to_torch(value, device=device)
    if not isinstance(converted, dict):
        raise ValueError("observation must be an object")
    if "image" not in converted:
        converted["image"] = {}
    _normalize_observation_tensors(converted, batch_size=batch_size)
    if "image_mask" not in converted:
        converted["image_mask"] = {
            key: torch.ones(batch_size, dtype=torch.bool, device=device) for key in converted["image"]
        }
    return _model.Observation.from_dict(converted)


def _normalize_observation_tensors(value: dict[str, Any], *, batch_size: int) -> None:
    if "state" not in value or not torch.is_tensor(value["state"]):
        raise ValueError("observation.state must be present as a tensor")
    value["state"] = _normalize_state_batch(value["state"], batch_size=batch_size)
    images = value.get("image", {})
    if not isinstance(images, Mapping):
        raise ValueError("observation.image must be an object")
    for key, image in list(images.items()):
        if not torch.is_tensor(image):
            raise ValueError(f"observation.image[{key!r}] must be a tensor")
        image = _normalize_image_batch(image, batch_size=batch_size, field=f"observation.image[{key!r}]")
        if not image.is_floating_point() and image.dtype != torch.uint8:
            if torch.any((image < 0) | (image > 255)):
                raise ValueError(f"observation.image[{key!r}] integer values must be in [0, 255]")
            image = image.to(torch.uint8)
        images[key] = image
    masks = value.get("image_mask")
    if masks is not None:
        if not isinstance(masks, Mapping):
            raise ValueError("observation.image_mask must be an object")
        for key, mask in list(masks.items()):
            if not torch.is_tensor(mask):
                raise ValueError(f"observation.image_mask[{key!r}] must be a tensor")
            masks[key] = _normalize_vector_batch(
                mask.to(torch.bool), batch_size=batch_size, field=f"observation.image_mask[{key!r}]"
            )
    for key in ("tokenized_prompt", "tokenized_prompt_mask"):
        if key in value and value[key] is not None and torch.is_tensor(value[key]):
            dtype = torch.bool if key.endswith("_mask") else torch.long
            value[key] = _normalize_sequence_batch(value[key].to(dtype), batch_size=batch_size, field=f"observation.{key}")


def _normalize_state_batch(tensor: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    tensor = tensor.to(torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"observation.state must have shape [batch, state_dim], got {tuple(tensor.shape)}")
    if tensor.shape[0] != batch_size:
        raise ValueError(f"observation.state batch mismatch: expected {batch_size}, got {tensor.shape[0]}")
    return tensor


def _normalize_image_batch(tensor: torch.Tensor, *, batch_size: int, field: str) -> torch.Tensor:
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"{field} must have shape [batch, height, width, channels], got {tuple(tensor.shape)}")
    if tensor.shape[0] != batch_size:
        raise ValueError(f"{field} batch mismatch: expected {batch_size}, got {tensor.shape[0]}")
    return tensor


def _normalize_sequence_batch(tensor: torch.Tensor, *, batch_size: int, field: str) -> torch.Tensor:
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"{field} must have shape [batch, length], got {tuple(tensor.shape)}")
    if tensor.shape[0] != batch_size:
        raise ValueError(f"{field} batch mismatch: expected {batch_size}, got {tensor.shape[0]}")
    return tensor


def _normalize_vector_batch(tensor: torch.Tensor, *, batch_size: int, field: str) -> torch.Tensor:
    if tensor.ndim == 0:
        if batch_size != 1:
            raise ValueError(f"{field} scalar mask is only valid for batch size 1")
        tensor = tensor.reshape(1)
    if tensor.ndim != 1:
        raise ValueError(f"{field} must have shape [batch], got {tuple(tensor.shape)}")
    if tensor.shape[0] != batch_size:
        raise ValueError(f"{field} batch mismatch: expected {batch_size}, got {tensor.shape[0]}")
    return tensor


def _tree_to_torch(value: Any, *, device: torch.device) -> Any:
    if _is_tensor_payload(value):
        return _deserialize_tensor(value, "tensor").to(device=device)
    if isinstance(value, Mapping):
        return {str(key): _tree_to_torch(item, device=device) for key, item in value.items()}
    if isinstance(value, str):
        return value
    tensor = torch.as_tensor(np.asarray(value), device=device)
    if tensor.is_floating_point():
        tensor = tensor.to(torch.float32)
    return tensor


def _is_tensor_payload(value: Any) -> bool:
    return isinstance(value, Mapping) and {"dtype", "shape", "data_b64"} <= set(value.keys())



def _memory_to_device(memory: MemorySnapshot, *, device: torch.device) -> MemorySnapshot:
    return MemorySnapshot(
        memory.backend_id,
        memory.state_schema_version,
        memory.batch_size,
        tuple(tuple(tensor.detach().to(device=device).clone() for tensor in layer) for layer in memory.layers),
    )


def _detach_frozen_prefix(frozen: Any) -> Any:
    return dataclasses.replace(frozen, hidden=frozen.hidden.detach(), pad_mask=frozen.pad_mask.detach(), kv_cache=detach_cache(frozen.kv_cache)) if dataclasses.is_dataclass(frozen) else type(frozen)(hidden=frozen.hidden.detach(), pad_mask=frozen.pad_mask.detach(), kv_cache=detach_cache(frozen.kv_cache))


def _tensor_from_json(value: Any, field: str) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(value), dtype=torch.float32)
    if tensor.numel() == 0:
        raise ValueError(f"{field} must not be empty")
    return tensor


def _deserialize_tensor(payload: Any, field: str) -> torch.Tensor:
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field} must be an object")
    for key in ("dtype", "shape", "data_b64"):
        if key not in payload:
            raise ValueError(f"{field} missing {key}")
    dtype = _dtype_from_name(payload["dtype"])
    shape = tuple(_strict_int(item, f"{field} shape") for item in payload["shape"])
    raw = base64.b64decode(payload["data_b64"], validate=True)
    element_size = torch.empty((), dtype=dtype).element_size()
    expected = (int(np.prod(shape, dtype=np.int64)) if shape else 1) * element_size
    if len(raw) != expected:
        raise ValueError(f"{field} data size mismatch: expected {expected} bytes, got {len(raw)}")
    byte_array = np.frombuffer(raw, dtype=np.uint8).copy()
    return torch.from_numpy(byte_array).view(dtype).reshape(shape).clone()



def _dtype_from_name(name: Any) -> torch.dtype:
    if not isinstance(name, str):
        raise ValueError("tensor dtype must be a string")
    normalized = name.removeprefix("torch.")
    mapping = {"float16": torch.float16, "float32": torch.float32, "float64": torch.float64, "bfloat16": torch.bfloat16}
    if normalized not in mapping:
        raise ValueError(f"unsupported tensor dtype {name!r}")
    return mapping[normalized]


def _numpy_dtype(dtype: torch.dtype) -> np.dtype:
    if dtype == torch.float16:
        return np.dtype("float16")
    if dtype == torch.float32:
        return np.dtype("float32")
    if dtype == torch.float64:
        return np.dtype("float64")
    if dtype == torch.bfloat16:
        return np.dtype("uint16")
    raise ValueError(f"unsupported tensor dtype {dtype}")


def _as_cpu_tensor(value: torch.Tensor | np.ndarray) -> torch.Tensor:
    return torch.as_tensor(value).detach().cpu()


def _update_tree_digest(digest: "hashlib._Hash", value: Any) -> None:
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        digest.update(b"tensor")
        digest.update(tensor_checksum(value).encode("utf-8"))
    elif _is_tensor_payload(value):
        digest.update(b"encoded_tensor")
        digest.update(tensor_checksum(_deserialize_tensor(value, "checksum tensor")).encode("utf-8"))
    elif isinstance(value, Mapping):
        digest.update(b"dict")
        for key in sorted(value):
            digest.update(str(key).encode("utf-8"))
            _update_tree_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(b"list")
        digest.update(str(len(value)).encode("utf-8"))
        for item in value:
            _update_tree_digest(digest, item)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    else:
        array = np.asarray(value)
        if array.dtype != object:
            digest.update(tensor_checksum(array).encode("utf-8"))
        else:
            raise ValueError(f"unsupported value for checksum: {type(value).__name__}")


def _memory_identity(memory: MemorySnapshot) -> dict[str, Any]:
    return {
        "backend_id": memory.backend_id,
        "state_schema_version": memory.state_schema_version,
        "batch_size": memory.batch_size,
        "layers": [[{"shape": list(tensor.shape), "dtype": str(tensor.dtype)} for tensor in layer] for layer in memory.layers],
        "checksum": memory_snapshot_checksum(memory),
    }


def _tensor_sequence_to_json(values: Sequence[torch.Tensor]) -> list[Any]:
    return [_tensor_to_json(value) for value in values]


def _tensor_to_json(tensor: torch.Tensor) -> Any:
    return tensor.detach().cpu().to(torch.float32).numpy().tolist()


def _l2(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.reshape(-1)).item())


def _mean_abs(tensor: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(tensor)).item())


def _strict_int(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    raise ValueError(f"{field} must be an integer")


def _non_empty_str(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _load_json_object(path: Path, description: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_object_pairs)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid {description} JSON: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{description} must be a JSON object")
    return payload


def _reject_duplicate_object_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _read_bundle_metadata(bundle_dir: Path) -> Mapping[str, Any] | None:
    path = bundle_dir / "metadata.json"
    if not path.is_file():
        return None
    return _load_json_object(path, "bundle metadata")


def _config_from_metadata(
    metadata: Mapping[str, Any] | None, *, artifact: InterventionArtifact | None = None
) -> FutureMambaPytorchConfig:
    if metadata is None:
        kwargs: dict[str, Any] = {"pytorch_compile_mode": None}
        if artifact is not None:
            kwargs["action_horizon"] = int(artifact.noise.shape[1])
            kwargs["action_dim"] = int(artifact.noise.shape[2])
            kwargs["execution_horizon"] = min(16, int(artifact.noise.shape[1]))
        return FutureMambaPytorchConfig(**kwargs)
    memory_payload = metadata.get("memory_config", {})
    if not isinstance(memory_payload, Mapping):
        raise ValueError("bundle memory_config must be an object")
    memory_fields = {field.name for field in dataclasses.fields(MambaMemoryConfig)}
    memory = MambaMemoryConfig(**{key: value for key, value in memory_payload.items() if key in memory_fields})
    config_fields = {field.name for field in dataclasses.fields(FutureMambaPytorchConfig)}
    kwargs = {
        "memory": memory,
        "pytorch_compile_mode": None,
    }
    mapping = {
        "memory_backend": "memory_backend",
        "handoff_ratio": "handoff_ratio",
        "num_denoise_steps": "num_denoise_steps",
        "prediction_horizon": "action_horizon",
        "action_dim": "action_dim",
        "execution_horizon": "execution_horizon",
        "progress_depth": "progress_depth",
        "schema_version": "schema_version",
        "base_checkpoint_uri": "base_checkpoint_uri",
        "base_checkpoint_checksum": "base_checkpoint_checksum",
        "base_assets_checksum": "base_assets_checksum",
        "robomme_policy_commit": "robomme_policy_commit",
        "robomme_benchmark_commit": "robomme_benchmark_commit",
        "robomme_dataset_checksum": "robomme_dataset_checksum",
        "robomme_task_suite": "robomme_task_suite",
        "mamba_repo_commit": "mamba_repo_commit",
        "memory_state_schema_version": "memory_state_schema_version",
        "training_dtype": "dtype",
        "state_dtypes": "state_dtypes",
        "kernel_mode": "kernel_mode",
        "torch_version": "torch_version",
        "triton_version": "triton_version",
        "cuda_version": "cuda_version",
        "gpu_name": "gpu_name",
        "compute_capability": "compute_capability",
        "action_expert_gradient_checkpointing": "action_expert_gradient_checkpointing",
        "terminal_loss_batch_fraction": "terminal_loss_batch_fraction",
        "terminal_loss_queries_per_episode": "terminal_loss_queries_per_episode",
        "frozen_prefix_microbatch_size": "frozen_prefix_microbatch_size",
    }
    for source, target in mapping.items():
        if source in metadata and target in config_fields:
            kwargs[target] = metadata[source]
    if artifact is not None:
        kwargs.setdefault("action_horizon", int(artifact.noise.shape[1]))
        kwargs.setdefault("action_dim", int(artifact.noise.shape[2]))
        kwargs.setdefault("execution_horizon", min(16, int(artifact.noise.shape[1])))
    losses = metadata.get("loss_weights")
    if isinstance(losses, Mapping):
        kwargs["terminal_loss_weight"] = losses.get("terminal", 1.0)
        kwargs["handoff_loss_weight"] = losses.get("handoff", 0.0)
        kwargs["boundary_loss_weight"] = losses.get("boundary", 0.0)
    return FutureMambaPytorchConfig(**kwargs)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_canonicalize(item) for item in value]
    if isinstance(value, list):
        return [_canonicalize(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return _tensor_to_json(value)
    return value


if __name__ == "__main__":
    import tyro

    main(tyro.cli(Args))
