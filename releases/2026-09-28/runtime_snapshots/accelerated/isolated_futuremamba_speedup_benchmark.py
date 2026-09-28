from __future__ import annotations

import dataclasses
import gc
import json
import importlib.util
import statistics
import math
import sys
from pathlib import Path
from types import MethodType

import safetensors.torch
import torch

import openpi.models.model as openpi_model
from openpi.models_pytorch.gemma_pytorch import PrefixKVView
from openpi.models_pytorch.mamba_memory import MemorySnapshot
from openpi.training import config as training_config
from openpi.training import episode_data_loader
TRAIN_ENTRYPOINT = Path("/home/nvidia/zyx/PhaseVLA/scripts/train_futuremamba_pytorch.py")


def load_training_module():
    spec = importlib.util.spec_from_file_location("isolated_futuremamba_train", TRAIN_ENTRYPOINT)
    if spec is None or spec.loader is None:
        raise ImportError(TRAIN_ENTRYPOINT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


futuremamba_train = load_training_module()

CONFIG_NAME = "futuremamba_action_history_handoff04_libero_mem_bowl_t1"
CHECKPOINT = Path(
    "/data/libero_mem_baseline/checkpoints/action_history_handoff04_futuremamba/"
    "futuremamba_action_history_handoff04_libero_mem_bowl_t1/1000"
)
EPISODE_CACHE = Path(
    "/data/libero_mem_baseline/conditioning_cache_handoff04/"
    "futuremamba_action_history_handoff04_libero_mem_bowl_t1/episode_000045.pt"
)
OUTPUT = Path("/data/libero_mem_baseline/isolated_speedup/history_fastpath_t1_step1000.json")


def repeat_snapshot(snapshot: MemorySnapshot, count: int) -> MemorySnapshot:
    return MemorySnapshot(
        snapshot.backend_id,
        snapshot.state_schema_version,
        count,
        tuple(
            tuple(tensor.expand(count, *tensor.shape[1:]) for tensor in layer)
            for layer in snapshot.layers
        ),
    )


def slice_snapshot(snapshot: MemorySnapshot, index: int) -> MemorySnapshot:
    return MemorySnapshot(
        snapshot.backend_id,
        snapshot.state_schema_version,
        1,
        tuple(tuple(tensor[index : index + 1] for tensor in layer) for layer in snapshot.layers),
    )


def fast_history_tokens(plugin, batch, train_query_mask: torch.BoolTensor) -> torch.Tensor:
    batch_size, num_queries = batch.actions.shape[:2]
    if batch_size != 1:
        raise ValueError("isolated fast path currently requires batch_size=1")
    valid_queries = int(batch.query_mask[0].long().sum().item())
    if valid_queries <= 0:
        raise ValueError("empty episode")
    expected_query_mask = torch.arange(num_queries, device=batch.query_mask.device) < valid_queries
    if not torch.equal(batch.query_mask[0], expected_query_mask):
        raise ValueError("query mask must be one right-padded episode")
    if not torch.equal(train_query_mask[0], batch.query_mask[0]):
        raise ValueError("isolated fast path requires every valid query to train")
    if not bool(batch.reset_mask[0, 0].item()) or bool(batch.reset_mask[0, 1:valid_queries].any().item()):
        raise ValueError("isolated fast path requires one reset at the first query")
    if batch.executed_actions.shape[2] != 1:
        raise ValueError("isolated fast path requires executed_horizon=1")
    executed_mask = batch.executed_action_mask[0, :valid_queries, 0]
    expected_executed_mask = torch.ones_like(executed_mask)
    expected_executed_mask[0] = False
    if not torch.equal(executed_mask, expected_executed_mask):
        raise ValueError("isolated fast path requires exactly one previous action after query zero")

    parameter = next(plugin.parameters())
    dtype = parameter.dtype
    device = batch.actions.device
    history = plugin.initial_history_state(1, device, dtype)
    empty_token = plugin.memory_token_projection(
        history.committed_output.to(dtype=plugin.memory_token_projection.weight.dtype)
    ).reshape(1, plugin.progress_memory_tokens, plugin.action_expert_width)
    tokens = [empty_token]
    actions = batch.executed_actions[0, 1:valid_queries, 0].to(dtype=dtype)
    chunk_size = int(plugin.chunk_size)
    committed_memory = history.memory

    for start in range(0, actions.shape[0], chunk_size):
        block = actions[start : start + chunk_size]
        count = int(block.shape[0])
        row = torch.arange(count, device=device)[:, None]
        column = torch.arange(count, device=device)[None, :]
        prefix_mask = column <= row
        candidates = torch.zeros(
            count,
            chunk_size,
            int(plugin.action_dim),
            dtype=dtype,
            device=device,
        )
        candidates[:, :count] = torch.where(
            prefix_mask[..., None],
            block[None].expand(count, count, int(plugin.action_dim)),
            torch.zeros((), dtype=dtype, device=device),
        )
        candidate_mask = torch.zeros(count, chunk_size, dtype=torch.bool, device=device)
        candidate_mask[:, :count] = prefix_mask
        encoded = plugin.encode_action_chunk(candidates, candidate_mask)
        outputs, candidate_memory = plugin.memory_backend.step(
            encoded, repeat_snapshot(committed_memory, count)
        )
        tokens.append(
            plugin.memory_token_projection(
                outputs.to(dtype=plugin.memory_token_projection.weight.dtype)
            ).reshape(count, plugin.progress_memory_tokens, plugin.action_expert_width)
        )
        if count == chunk_size:
            committed_memory = slice_snapshot(candidate_memory, count - 1)

    result = torch.cat(tokens, dim=0)
    if result.shape[0] != valid_queries:
        raise RuntimeError(f"history token count mismatch: {result.shape[0]} != {valid_queries}")
    if valid_queries < num_queries:
        result = torch.cat(
            (
                result,
                torch.zeros(
                    num_queries - valid_queries,
                    int(plugin.progress_memory_tokens),
                    int(plugin.action_expert_width),
                    dtype=result.dtype,
                    device=result.device,
                ),
            ),
            dim=0,
        )
    return result.unsqueeze(0)

def mean_masked_action_error(
    error: torch.Tensor, action_mask: torch.Tensor, query_mask: torch.Tensor
) -> torch.Tensor:
    action_weights = action_mask.to(dtype=error.dtype)
    query_error = torch.where(action_mask, error, torch.zeros_like(error)).sum(dim=-1)
    query_error = query_error / action_weights.sum(dim=-1).clamp_min(1.0)
    query_weights = query_mask.to(dtype=error.dtype)
    episode_error = (query_error * query_weights).sum(dim=-1)
    episode_error = episode_error / query_weights.sum(dim=-1).clamp_min(1.0)
    return episode_error.mean()



def fast_compute_episode_loss(self, batch, *, noise=None, time=None, train=True):
    device = batch.actions.device if torch.is_tensor(batch.actions) else torch.device("cpu")
    batch = episode_data_loader.episode_batch_to_torch(batch, device)
    actions = batch.actions
    batch_size, num_queries = actions.shape[:2]
    dtype = actions.dtype
    if noise is None:
        noise = self.base.sample_noise(tuple(actions.shape), device)
    else:
        noise = noise.to(device=device, dtype=dtype)
    if tuple(noise.shape) != tuple(actions.shape):
        raise ValueError(f"noise must have shape {tuple(actions.shape)}, got {tuple(noise.shape)}")
    handoff_steps = max(
        0,
        min(
            int(self.config.num_denoise_steps),
            int(
                math.ceil(
                    float(self.config.handoff_ratio)
                    * int(self.config.num_denoise_steps)
                )
            ),
        ),
    )
    if time is None:
        time = self._sample_high_noise_time(
            (batch_size, num_queries), handoff_steps=handoff_steps, device=device
        )
    else:
        time = time.to(device=device, dtype=torch.float32)
    if tuple(time.shape) != (batch_size, num_queries):
        raise ValueError(f"time must have shape {(batch_size, num_queries)}, got {tuple(time.shape)}")
    train_query_mask = batch.train_query_mask
    if train_query_mask is None:
        raise ValueError("train query mask is required")

    valid_action_mask = batch.action_mask & train_query_mask[:, :, None]
    safe_actions = torch.where(valid_action_mask[..., None], actions, torch.zeros_like(actions))
    safe_noise = torch.where(valid_action_mask[..., None], noise, torch.zeros_like(noise))
    x_t = time[..., None, None] * safe_noise + (1.0 - time[..., None, None]) * safe_actions
    target_velocity = safe_noise - safe_actions
    flow_error = torch.zeros(
        batch_size,
        num_queries,
        int(self.config.action_horizon),
        dtype=dtype,
        device=device,
    )
    memory_tokens = fast_history_tokens(self.futuremamba, batch, train_query_mask)

    if batch.conditioning_cache is None:
        raise ValueError("isolated benchmark requires the production conditioning cache")
    cache = batch.conditioning_cache
    prefix_microbatch_size = int(self.config.frozen_prefix_microbatch_size)
    for episode_index in range(batch_size):
        valid_queries = int(batch.query_mask[episode_index].long().sum().item())
        for query_start in range(0, valid_queries, prefix_microbatch_size):
            query_stop = min(query_start + prefix_microbatch_size, valid_queries)
            selected_mask = train_query_mask[episode_index, query_start:query_stop]
            if not bool(selected_mask.any().item()):
                continue
            local_indices = torch.nonzero(selected_mask, as_tuple=False).flatten()
            query_indices = query_start + local_indices
            prefix_mask = cache["prefix_mask"][episode_index, query_indices].to(torch.bool)
            keys = cache["action_expert_keys"][episode_index, query_indices]
            values = cache["action_expert_values"][episode_index, query_indices]
            layers = tuple((keys[:, layer], values[:, layer]) for layer in range(keys.shape[1]))
            prefix_cache = PrefixKVView.from_layers(layers, prefix_mask)
            predicted = self.futuremamba.forward_progress(
                prefix_cache,
                prefix_mask,
                memory_tokens[episode_index, query_indices],
                x_t[episode_index, query_indices],
                time[episode_index, query_indices],
            )
            flow_error[episode_index, query_indices] = torch.mean(
                torch.square(predicted - target_velocity[episode_index, query_indices]), dim=-1
            )

    flow_loss = mean_masked_action_error(flow_error, batch.action_mask, train_query_mask)
    zero = torch.zeros((), dtype=flow_loss.dtype, device=flow_loss.device)
    sampled_time = time[train_query_mask]
    return {
        "loss": flow_loss,
        "flow_loss": flow_loss,
        "terminal_loss": zero,
        "terminal_error": zero,
        "handoff_loss": zero,
        "handoff_error": zero,
        "boundary_loss": zero,
        "boundary_error": zero,
        "sample_time_mean": sampled_time.mean(),
        "sample_time_min": sampled_time.min(),
        "sample_time_max": sampled_time.max(),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def clear_step(model) -> None:
    model.zero_grad(set_to_none=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def timed_step(model, method, batch, noise, sample_time, device):
    clear_step(model)
    synchronize(device)
    started = time.perf_counter()
    outputs = method(batch, noise=noise, time=sample_time)
    outputs["loss"].backward()
    synchronize(device)
    elapsed = time.perf_counter() - started
    metrics = {key: float(value.detach().float().cpu()) for key, value in outputs.items()}
    return elapsed, metrics


def collect_gradients(model):
    result = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            result[name] = (
                None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            )
    return result


def compare_gradients(reference, candidate):
    if reference.keys() != candidate.keys():
        raise RuntimeError("gradient key mismatch")
    per_parameter = {}
    allclose = True
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    max_abs = 0.0
    for name in reference:
        left = reference[name]
        right = candidate[name]
        if left is None or right is None:
            close = left is None and right is None
            allclose = allclose and close
            per_parameter[name] = {
                "allclose": close,
                "old_has_gradient": left is not None,
                "fast_has_gradient": right is not None,
                "max_abs": None,
            }
            continue
        left = left.float()
        right = right.float()
        delta = (left - right).abs()
        local_max = float(delta.max()) if delta.numel() else 0.0
        close = bool(torch.allclose(left, right, rtol=5e-3, atol=5e-4))
        allclose = allclose and close
        max_abs = max(max_abs, local_max)
        dot += float(torch.sum(left * right))
        left_norm += float(torch.sum(left.square()))
        right_norm += float(torch.sum(right.square()))
        per_parameter[name] = {
            "allclose": close,
            "old_has_gradient": True,
            "fast_has_gradient": True,
            "max_abs": local_max,
        }
    cosine = dot / max((left_norm * right_norm) ** 0.5, 1e-30)
    return {
        "allclose": allclose,
        "max_abs": max_abs,
        "cosine": cosine,
        "parameters": per_parameter,
    }


def load_production_cache_batch() -> episode_data_loader.TorchEpisodeBatch:
    payload = torch.load(EPISODE_CACHE, map_location="cpu", weights_only=True)
    required = {
        "actions",
        "action_mask",
        "executed_actions",
        "executed_action_mask",
        "last_valid_hidden",
        "prefix_mask",
        "action_expert_keys",
        "action_expert_values",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"production cache keys mismatch: {EPISODE_CACHE}")
    converted = {
        key: value.float() if value.dtype is torch.bfloat16 else value
        for key, value in payload.items()
    }
    queries = int(converted["actions"].shape[0])
    query_mask = torch.ones(1, queries, dtype=torch.bool)
    reset_mask = torch.zeros(1, queries, dtype=torch.bool)
    reset_mask[:, 0] = True
    return episode_data_loader.TorchEpisodeBatch(
        observation=openpi_model.Observation(
            images={},
            image_masks={},
            state=torch.zeros(1, queries, 1, dtype=torch.float32),
        ),
        actions=converted["actions"].unsqueeze(0),
        action_mask=converted["action_mask"].unsqueeze(0).to(torch.bool),
        executed_actions=converted["executed_actions"].unsqueeze(0),
        executed_action_mask=converted["executed_action_mask"].unsqueeze(0).to(torch.bool),
        query_mask=query_mask,
        reset_mask=reset_mask,
        episode_index=torch.tensor([45], dtype=torch.int64),
        train_query_mask=query_mask.clone(),
        conditioning_cache={
            key: converted[key].unsqueeze(0)
            for key in (
                "last_valid_hidden",
                "prefix_mask",
                "action_expert_keys",
                "action_expert_values",
            )
        },
    )


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    config = training_config.get_config(CONFIG_NAME)

    load_started = time.perf_counter()
    model = futuremamba_train.load_training_model(config, device)
    plugin_state = safetensors.torch.load_file(CHECKPOINT / "plugin.safetensors", device=str(device))
    model.futuremamba.load_state_dict(
        {key.removeprefix("futuremamba."): value for key, value in plugin_state.items()}, strict=True
    )
    model.train(True)
    model_load_sec = time.perf_counter() - load_started

    data_started = time.perf_counter()
    data = futuremamba_train.create_training_data(config, shuffle=False)
    cpu_batch = next(iter(data))
    data_load_sec = time.perf_counter() - data_started
    transfer_started = time.perf_counter()
    batch = futuremamba_train._to_device(cpu_batch, device)
    synchronize(device)
    transfer_sec = time.perf_counter() - transfer_started
    generator = torch.Generator(device=device).manual_seed(20260827)
    noise = torch.randn(batch.actions.shape, generator=generator, device=device, dtype=batch.actions.dtype)
    lower = 1.0 - 4.0 / 10.0
    sample_time = torch.linspace(
        lower + 0.01,
        0.99,
        batch.actions.shape[1],
        device=device,
        dtype=torch.float32,
    ).unsqueeze(0)
    production_method = model.compute_episode_loss
    fast_method = MethodType(fast_compute_episode_loss, model)

    warmup_old_sec, _ = timed_step(
        model, production_method, batch, noise, sample_time, device
    )
    warmup_fast_sec, _ = timed_step(model, fast_method, batch, noise, sample_time, device)

    old_sec, old_metrics = timed_step(
        model, production_method, batch, noise, sample_time, device
    )
    old_gradients = collect_gradients(model)
    fast_sec, fast_metrics = timed_step(model, fast_method, batch, noise, sample_time, device)
    fast_gradients = collect_gradients(model)
    gradient_comparison = compare_gradients(old_gradients, fast_gradients)

    metric_delta = {
        key: {
            "old": old_metrics[key],
            "fast": fast_metrics[key],
            "abs_delta": abs(old_metrics[key] - fast_metrics[key]),
        }
        for key in old_metrics
    }
    report = {
        "status": "completed",
        "scope": "isolated_no_optimizer_step_no_training_process_changes",
        "config": CONFIG_NAME,
        "checkpoint": str(CHECKPOINT),
        "device": torch.cuda.get_device_name(device),
        "episode_index": batch.episode_index.detach().cpu().tolist(),
        "queries": int(batch.query_mask.long().sum()),
        "timing_sec": {
            "model_load": model_load_sec,
            "data_load_cpu": data_load_sec,
            "host_to_device": transfer_sec,
            "warmup_old": warmup_old_sec,
            "warmup_fast": warmup_fast_sec,
            "measured_old_forward_backward": old_sec,
            "measured_fast_forward_backward": fast_sec,
            "speedup": old_sec / fast_sec,
        },
        "metrics": metric_delta,
        "gradients": gradient_comparison,
        "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "output": str(OUTPUT)}, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
