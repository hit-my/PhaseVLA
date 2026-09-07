from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch

from openpi.models_pytorch.memory_ae_config import MemoryAEConfig
from openpi.training import config as training_config
from openpi.training import memory_ae_checkpoint as checkpoints
from openpi.training.memory_ae_data_loader import create_memory_ae_data

LOGGER = logging.getLogger(__name__)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def parameter_counts(model):
    groups = {"vlm_frozen": 0, "action_expert": 0, "memory_and_projection": 0}
    for name, parameter in model.named_parameters():
        if name.startswith("base.paligemma_with_expert.paligemma."):
            if parameter.requires_grad:
                raise ValueError(f"VLM parameter is trainable: {name}")
            groups["vlm_frozen"] += parameter.numel()
        elif name.startswith("futuremamba."):
            groups["memory_and_projection"] += parameter.numel()
        elif name.startswith(checkpoints.TRAINABLE_PREFIXES):
            groups["action_expert"] += parameter.numel()
        elif parameter.requires_grad:
            raise ValueError(f"Unexpected trainable module: {name}")
    _, parameters = checkpoints.trainable_parameters(model)
    groups["trainable_total"] = sum(parameter.numel() for parameter in parameters)
    groups["total"] = sum(parameter.numel() for parameter in model.parameters())
    return groups


def backward_update(model, batch, device, microbatch_size):
    query_count = len(batch.actions)
    if query_count <= 0 or microbatch_size <= 0:
        raise ValueError("Batch and microbatch size must be positive")
    totals = {}
    forward_seconds = 0.0
    backward_seconds = 0.0
    transfer_seconds = 0.0
    history_actions = 0
    for start in range(0, query_count, microbatch_size):
        stop = min(start + microbatch_size, query_count)
        started = time.perf_counter()
        part = batch.slice(start, stop).to(device)
        synchronize(device)
        transfer_seconds += time.perf_counter() - started
        history_actions += int(part.history_mask.sum().item())
        started = time.perf_counter()
        outputs = model.compute_query_loss(part)
        loss = outputs["loss"]
        if not torch.is_tensor(loss) or loss.ndim != 0 or not torch.isfinite(loss).item():
            raise FloatingPointError("MemoryAE loss must be a finite scalar")
        synchronize(device)
        forward_seconds += time.perf_counter() - started
        weight = (stop - start) / query_count
        started = time.perf_counter()
        (loss * weight).backward()
        synchronize(device)
        backward_seconds += time.perf_counter() - started
        for name in ("loss", "flow_loss", "sample_time_mean", "sample_time_min", "sample_time_max"):
            if name not in outputs:
                continue
            value = float(outputs[name].detach().item())
            if not np.isfinite(value):
                raise FloatingPointError(f"Nonfinite MemoryAE metric: {name}")
            if name == "sample_time_min":
                totals[name] = min(totals.get(name, value), value)
            elif name == "sample_time_max":
                totals[name] = max(totals.get(name, value), value)
            else:
                totals[name] = totals.get(name, 0.0) + value * weight
        del part, outputs, loss
    totals.update(
        forward_seconds=forward_seconds,
        backward_seconds=backward_seconds,
        transfer_seconds=transfer_seconds,
        history_actions=history_actions,
    )
    return totals


def run_training(
    model,
    data,
    *,
    checkpoint_root,
    metadata,
    steps,
    save_interval,
    microbatch_size,
    learning_rate,
    weight_decay,
    clip_gradient_norm,
    device,
    resume=False,
    metric_stream=None,
    wandb_run=None,
    startup_seconds=0.0,
):
    if min(steps, save_interval, microbatch_size) <= 0:
        raise ValueError("Training steps and intervals must be positive")
    root = Path(checkpoint_root)
    if not resume and root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Use a fresh experiment directory or --resume: {root}")
    root.mkdir(parents=True, exist_ok=True)
    model.train()
    _, parameters = checkpoints.trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    start = 0
    processed_queries = 0
    prior_elapsed = 0.0
    if resume:
        available = sorted(int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdigit())
        if not available:
            raise FileNotFoundError(f"No checkpoint to resume in {root}")
        saved = checkpoints.restore_checkpoint(
            root / str(available[-1]),
            model,
            optimizer,
            scheduler,
            expected_metadata=metadata,
            device=device,
        )
        start = int(saved["step"])
        processed_queries = int(saved["processed_queries"])
        prior_elapsed = float(saved["elapsed_seconds"])
        if int(saved["data_iterator_step"]) != start:
            raise ValueError("MemoryAE checkpoint update/data iterator mismatch")
        if start > steps:
            raise ValueError("Requested final step precedes the restored checkpoint")
    iterator = data.iter_from_update(start)
    session_started = time.perf_counter()
    print(
        f"TRAINING_READY start_step={start} final_step={steps} queries_per_update={data.queries_per_update}", flush=True
    )
    for update in range(start, steps):
        started = time.perf_counter()
        batch = next(iterator)
        data_seconds = time.perf_counter() - started
        if len(batch.actions) != data.queries_per_update:
            raise ValueError("Query count changed within training")
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metrics = backward_update(model, batch, device, microbatch_size)
        optimizer_started = time.perf_counter()
        norm = torch.nn.utils.clip_grad_norm_(parameters, clip_gradient_norm)
        if not torch.isfinite(norm).item():
            raise FloatingPointError("Nonfinite MemoryAE gradient norm")
        optimizer.step()
        scheduler.step()
        synchronize(device)
        optimizer_seconds = time.perf_counter() - optimizer_started
        completed = update + 1
        processed_queries += len(batch.actions)
        step_seconds = time.perf_counter() - started
        elapsed = prior_elapsed + startup_seconds + time.perf_counter() - session_started
        metrics.update(
            step=completed,
            processed_queries=processed_queries,
            queries_per_update=len(batch.actions),
            data_seconds=data_seconds,
            optimizer_seconds=optimizer_seconds,
            step_seconds=step_seconds,
            queries_per_second=len(batch.actions) / step_seconds,
            grad_norm=float(norm.item()),
            learning_rate=optimizer.param_groups[0]["lr"],
            elapsed_seconds=elapsed,
            allocated_gpu_hours=elapsed / 3600.0 if device.type == "cuda" else 0.0,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0,
        )
        if completed % save_interval == 0 or completed == steps:
            save_started = time.perf_counter()
            checkpoints.save_checkpoint(
                root / str(completed),
                model,
                optimizer,
                scheduler,
                step=completed,
                metadata=metadata,
                data_iterator_step=completed,
                processed_queries=processed_queries,
                elapsed_seconds=elapsed,
            )
            metrics["checkpoint_seconds"] = time.perf_counter() - save_started
            metrics["elapsed_seconds"] = prior_elapsed + startup_seconds + time.perf_counter() - session_started
            metrics["allocated_gpu_hours"] = metrics["elapsed_seconds"] / 3600.0 if device.type == "cuda" else 0.0
        if metric_stream is not None:
            metric_stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            metric_stream.flush()
        if wandb_run is not None:
            wandb_run.log({f"train/{key}": value for key, value in metrics.items()}, step=completed)
        LOGGER.info(
            "step=%d/%d loss=%.8f grad_norm=%.6f processed_queries=%d step_time=%.4f peak_gib=%.3f gpu_hours=%.5f",
            completed,
            steps,
            metrics["loss"],
            metrics["grad_norm"],
            processed_queries,
            step_seconds,
            metrics["peak_allocated_bytes"] / 2**30,
            metrics["allocated_gpu_hours"],
        )
        del batch
    return {"step": steps, "processed_queries": processed_queries}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train original pi05 AE with causal Mamba memory KV; frozen VLM, no PE"
    )
    parser.add_argument("config")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--queries-per-update", type=int)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args(argv)
    if args.cpu_threads <= 0 or args.microbatch_size <= 0:
        parser.error("CPU threads and microbatch size must be positive")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be positive")
    if args.save_interval is not None and args.save_interval <= 0:
        parser.error("--save-interval must be positive")
    started = time.perf_counter()
    cfg = training_config.get_config(args.config)
    if not isinstance(cfg.model, MemoryAEConfig):
        raise TypeError("Expected a registered MemoryAEConfig")
    torch.set_num_threads(args.cpu_threads)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.seed)
    root = args.checkpoint_root or Path(cfg.checkpoint_base_dir) / cfg.name
    log_dir = args.log_dir or Path(cfg.log_file).parent / cfg.name
    metrics_path = log_dir / "metrics.jsonl"
    if not args.resume and (metrics_path.exists() or (root.exists() and any(root.iterdir()))):
        raise FileExistsError("Existing experiment artifacts; use --resume or separate experiment paths")
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, handlers=[logging.StreamHandler(), logging.FileHandler(log_dir / "train.log")], force=True
    )
    model, weights = checkpoints.load_base(cfg.model, device, weight_path=cfg.pytorch_weight_path)
    data = create_memory_ae_data(cfg, queries_per_update=args.queries_per_update)
    metadata = cfg.model.checkpoint_metadata()
    metadata.update(
        source_checkpoint_sha256=checkpoints.source_checksum(weights),
        frozen_vlm_checksum=checkpoints.frozen_vlm_checksum(model),
        query_sampling_protocol="uniform_episode_then_query_with_full_causal_history",
        queries_per_update=data.queries_per_update,
        episode_count=data.episode_count,
        mean_episode_queries=data.mean_episode_queries,
        microbatch_size=args.microbatch_size,
        cpu_threads=args.cpu_threads,
        prefix_mode="online_frozen_vlm_all_layers",
        optimizer="AdamW",
        learning_rate=float(getattr(cfg.lr_schedule, "peak_lr", 2.5e-5)),
        weight_decay=float(getattr(cfg.optimizer, "weight_decay", 1e-10)),
        parameter_counts=parameter_counts(model),
        cost_definition="single_gpu_allocated_wall_time_including_startup; not utilization_integral",
    )
    if device.type == "cuda":
        metadata["gpu_name"] = torch.cuda.get_device_name(device)
    (log_dir / "identity.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    LOGGER.info("identity=%s", json.dumps(metadata))
    run = None
    if args.wandb:
        import wandb

        run_id_path = log_dir / "wandb_run_id"
        if args.resume and not run_id_path.exists():
            raise FileNotFoundError("Missing W&B run ID for resume")
        run_id = run_id_path.read_text().strip() if run_id_path.exists() else uuid.uuid4().hex[:8]
        run_id_path.write_text(run_id + "\n")
        run = wandb.init(
            entity="2023112993-harbin-institute-of-technology",
            project="futuremamba",
            id=run_id,
            name=cfg.wandb_run_name,
            config=metadata,
            resume="must" if args.resume else "allow",
            dir=str(log_dir),
        )
    try:
        with metrics_path.open("a", buffering=1) as stream:
            result = run_training(
                model,
                data,
                checkpoint_root=root,
                metadata=metadata,
                steps=cfg.num_train_steps if args.steps is None else args.steps,
                save_interval=cfg.save_interval if args.save_interval is None else args.save_interval,
                microbatch_size=args.microbatch_size,
                learning_rate=metadata["learning_rate"],
                weight_decay=metadata["weight_decay"],
                clip_gradient_norm=float(getattr(cfg.optimizer, "clip_gradient_norm", 1.0)),
                device=device,
                resume=args.resume,
                metric_stream=stream,
                wandb_run=run,
                startup_seconds=time.perf_counter() - started,
            )
        print("TRAINING_COMPLETE " + json.dumps(result), flush=True)
    except BaseException:
        if run is not None:
            run.finish(exit_code=1)
        raise
    else:
        if run is not None:
            run.finish()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
