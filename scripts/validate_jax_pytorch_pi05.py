#!/usr/bin/env python3
"""Numerically compare a task-adapted pi0.5 JAX checkpoint with its PyTorch conversion."""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import sys
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import safetensors.torch
import torch

from openpi.models import model as model_api
from openpi.models_pytorch import pi0_pytorch
from openpi.training import config as training_config


DEFAULT_MEAN_TOLERANCE = 1e-4
DEFAULT_MAX_TOLERANCE = 5e-4


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float32).numpy()
    return np.asarray(value, dtype=np.float32)


def error_metrics(reference: Any, candidate: Any) -> dict[str, Any]:
    reference_np = _to_numpy(reference)
    candidate_np = _to_numpy(candidate)
    if reference_np.shape != candidate_np.shape:
        return {
            "shape_match": False,
            "reference_shape": list(reference_np.shape),
            "candidate_shape": list(candidate_np.shape),
            "mean_absolute_error": None,
            "max_absolute_error": None,
        }
    absolute_error = np.abs(reference_np.astype(np.float64) - candidate_np.astype(np.float64))
    return {
        "shape_match": True,
        "reference_shape": list(reference_np.shape),
        "candidate_shape": list(candidate_np.shape),
        "mean_absolute_error": float(absolute_error.mean()) if absolute_error.size else 0.0,
        "max_absolute_error": float(absolute_error.max()) if absolute_error.size else 0.0,
    }


def _passes(metric: dict[str, Any], mean_tolerance: float, max_tolerance: float) -> bool:
    return bool(
        metric["shape_match"]
        and metric["mean_absolute_error"] <= mean_tolerance
        and metric["max_absolute_error"] <= max_tolerance
    )


def _fixed_observations(model_config, seed: int, device: torch.device):
    rng = np.random.default_rng(seed)
    images_hwc = {
        key: rng.uniform(-1.0, 1.0, size=(1, 224, 224, 3)).astype(np.float32)
        for key in model_api.IMAGE_KEYS
    }
    image_masks = {key: np.ones((1,), dtype=bool) for key in model_api.IMAGE_KEYS}
    state = np.linspace(-1.0, 1.0, model_config.action_dim, dtype=np.float32)[None, :]
    tokenized_prompt = np.zeros((1, model_config.max_token_len), dtype=np.int32)
    tokenized_prompt[:, :8] = np.arange(1, 9, dtype=np.int32)
    tokenized_prompt_mask = np.zeros((1, model_config.max_token_len), dtype=bool)
    tokenized_prompt_mask[:, :8] = True

    jax_observation = model_api.Observation(
        images={key: jnp.asarray(value) for key, value in images_hwc.items()},
        image_masks={key: jnp.asarray(value) for key, value in image_masks.items()},
        state=jnp.asarray(state),
        tokenized_prompt=jnp.asarray(tokenized_prompt),
        tokenized_prompt_mask=jnp.asarray(tokenized_prompt_mask),
    )
    torch_observation = model_api.Observation(
        images={
            key: torch.from_numpy(value).permute(0, 3, 1, 2).to(device)
            for key, value in images_hwc.items()
        },
        image_masks={key: torch.from_numpy(value).to(device) for key, value in image_masks.items()},
        state=torch.from_numpy(state).to(device),
        tokenized_prompt=torch.from_numpy(tokenized_prompt).to(device),
        tokenized_prompt_mask=torch.from_numpy(tokenized_prompt_mask).to(device),
    )
    return jax_observation, torch_observation


def _last_valid_token(hidden: Any, mask: Any):
    hidden_np = _to_numpy(hidden)
    mask_np = np.asarray(mask, dtype=bool)
    indices = mask_np.sum(axis=1) - 1
    if np.any(indices < 0):
        raise ValueError("prefix mask contains an example without a valid token")
    return hidden_np[np.arange(hidden_np.shape[0]), indices]


def _jax_cache_layers(cache: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    keys, values = cache
    keys_np = _to_numpy(keys)
    values_np = _to_numpy(values)
    if keys_np.ndim != 5 or values_np.ndim != 5:
        raise ValueError(f"unexpected JAX KV cache shapes: {keys_np.shape}, {values_np.shape}")
    return [
        (keys_np[layer].transpose(0, 2, 1, 3), values_np[layer].transpose(0, 2, 1, 3))
        for layer in range(keys_np.shape[0])
    ]


def _torch_cache_layers(cache: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        cache = tuple(zip(cache.key_cache, cache.value_cache, strict=True))
    if not isinstance(cache, (tuple, list)):
        raise TypeError(f"unsupported PyTorch KV cache type: {type(cache).__name__}")
    layers = []
    for entry in cache:
        if not isinstance(entry, (tuple, list)) or len(entry) < 2:
            raise TypeError("each PyTorch KV cache layer must contain key and value tensors")
        layers.append((_to_numpy(entry[0]), _to_numpy(entry[1])))
    return layers


def _torch_prefix(model, observation):
    images, image_masks, language_tokens, language_masks, state = model._preprocess_observation(
        observation, train=False
    )
    prefix_embeddings, prefix_mask, prefix_ar_mask = model.embed_prefix(
        images, image_masks, language_tokens, language_masks
    )
    attention = model._prepare_attention_masks_4d(
        pi0_pytorch.make_att_2d_masks(prefix_mask, prefix_ar_mask)
    )
    positions = torch.cumsum(prefix_mask, dim=1) - 1
    (prefix_hidden, suffix_hidden), cache = model.paligemma_with_expert.forward(
        attention_mask=attention,
        position_ids=positions,
        past_key_values=None,
        inputs_embeds=[prefix_embeddings, None],
        use_cache=True,
    )
    if suffix_hidden is not None:
        raise AssertionError("prefix-only PyTorch forward unexpectedly returned suffix hidden states")
    return prefix_hidden, prefix_mask, cache, state


def run_parity(
    *,
    jax_checkpoint: pathlib.Path,
    pytorch_checkpoint: pathlib.Path,
    config_name: str,
    dtype: str,
    seed: int,
    device: torch.device,
    mean_tolerance: float,
    max_tolerance: float,
) -> dict[str, Any]:
    if dtype != "float32":
        raise ValueError("strict pi0.5 parity currently requires --dtype float32")
    train_config = training_config.get_config(config_name)
    config = dataclasses.replace(train_config.model, dtype="float32", pytorch_compile_mode=None)

    jax_params = model_api.restore_params(
        jax_checkpoint / "params", restore_type=jax.Array, dtype=jnp.float32
    )
    jax_model = config.load(jax_params, remove_extra_params=False)
    jax_model.eval()

    torch_model = pi0_pytorch.PI0Pytorch(config).to(device=device, dtype=torch.float32).eval()
    weight_path = pytorch_checkpoint / "model.safetensors"
    safetensors.torch.load_model(torch_model, str(weight_path), strict=True, device=str(device))

    jax_observation, torch_observation = _fixed_observations(config, seed, device)
    jax_processed = model_api.preprocess_observation(None, jax_observation, train=False)
    jax_prefix_hidden, jax_prefix_mask, _, jax_cache = jax_model.encode_prefix(jax_processed)
    with torch.inference_mode():
        torch_prefix_hidden, torch_prefix_mask, torch_cache, torch_state = _torch_prefix(
            torch_model, torch_observation
        )

    comparisons: dict[str, dict[str, Any]] = {}
    comparisons["last_valid_prefix_token"] = error_metrics(
        _last_valid_token(jax_prefix_hidden, jax_prefix_mask),
        _last_valid_token(torch_prefix_hidden, _to_numpy(torch_prefix_mask)),
    )

    jax_layers = _jax_cache_layers(jax_cache)
    torch_layers = _torch_cache_layers(torch_cache)
    if len(jax_layers) != len(torch_layers):
        comparisons["prefix_kv_layer_count"] = {
            "shape_match": False,
            "reference_shape": [len(jax_layers)],
            "candidate_shape": [len(torch_layers)],
            "mean_absolute_error": None,
            "max_absolute_error": None,
        }
    else:
        for layer, ((jax_key, jax_value), (torch_key, torch_value)) in enumerate(
            zip(jax_layers, torch_layers, strict=True)
        ):
            comparisons[f"prefix_kv.layer_{layer:02d}.key"] = error_metrics(jax_key, torch_key)
            comparisons[f"prefix_kv.layer_{layer:02d}.value"] = error_metrics(jax_value, torch_value)

    rng = np.random.default_rng(seed + 1)
    actions_np = np.linspace(
        -0.5,
        0.5,
        config.action_horizon * config.action_dim,
        dtype=np.float32,
    ).reshape(1, config.action_horizon, config.action_dim)
    noise_np = rng.standard_normal(actions_np.shape, dtype=np.float32)
    time_np = np.array([0.75], dtype=np.float32)
    x_t_np = time_np[:, None, None] * noise_np + (1.0 - time_np[:, None, None]) * actions_np

    jax_velocity = jax_model.action_velocity(
        jax_processed,
        jnp.asarray(x_t_np),
        jnp.asarray(time_np),
        jax_prefix_mask,
        jax_cache,
    )
    with torch.inference_mode():
        torch_velocity = torch_model.denoise_step(
            torch_state,
            torch_prefix_mask,
            torch_cache,
            torch.from_numpy(x_t_np).to(device),
            torch.from_numpy(time_np).to(device),
        )
    comparisons["action_expert_velocity"] = error_metrics(jax_velocity, torch_velocity)

    jax_actions = jax_model.sample_actions(
        jax.random.key(seed),
        jax_observation,
        noise=jnp.asarray(noise_np),
        num_steps=10,
    )
    with torch.inference_mode():
        torch_actions = torch_model.sample_actions(
            device,
            torch_observation,
            noise=torch.from_numpy(noise_np).to(device),
            num_steps=10,
        )
    comparisons["action_chunk_10_step"] = error_metrics(jax_actions, torch_actions)

    first_failure = next(
        (
            name
            for name, metric in comparisons.items()
            if not _passes(metric, mean_tolerance, max_tolerance)
        ),
        None,
    )
    return {
        "config": config_name,
        "dtype": dtype,
        "seed": seed,
        "device": str(device),
        "mean_tolerance": mean_tolerance,
        "max_tolerance": max_tolerance,
        "passed": first_failure is None,
        "first_divergence": first_failure,
        "comparisons": comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jax-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--pytorch-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--mean-tolerance", type=float, default=DEFAULT_MEAN_TOLERANCE)
    parser.add_argument("--max-tolerance", type=float, default=DEFAULT_MAX_TOLERANCE)
    args = parser.parse_args()

    try:
        result = run_parity(
            jax_checkpoint=args.jax_checkpoint,
            pytorch_checkpoint=args.pytorch_checkpoint,
            config_name=args.config,
            dtype=args.dtype,
            seed=args.seed,
            device=torch.device(args.device),
            mean_tolerance=args.mean_tolerance,
            max_tolerance=args.max_tolerance,
        )
    except Exception as exc:
        result = {
            "passed": False,
            "first_divergence": "runtime_error",
            "error": f"{exc.__class__.__name__}: {exc}",
        }
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
