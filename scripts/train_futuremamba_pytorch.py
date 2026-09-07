from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Mapping
import dataclasses
import importlib.metadata
import json
import random

import numpy as np
import logging
from pathlib import Path
import shutil
from typing import Any
import time

import safetensors.torch
import torch
from torch import nn

from openpi.models_pytorch.futuremamba_config import FutureMambaPytorchConfig
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import episode_data_loader as _episode_data_loader
from openpi.training.futuremamba_checkpoint import load_futuremamba_checkpoint
from openpi.training.futuremamba_checkpoint import save_futuremamba_checkpoint


LOGGER = logging.getLogger(__name__)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO)
    LOGGER.setLevel(logging.INFO)


@dataclasses.dataclass(frozen=True)
class TrainingResult:
    start_step: int
    end_step: int
    data_iterator_step: int
    losses: list[float]
    checkpoint: Path | None


_LOSS_METRIC_NAMES = (
    "loss",
    "flow_loss",
    "terminal_loss",
    "terminal_error",
    "handoff_loss",
    "handoff_error",
    "boundary_loss",
    "boundary_error",
    "sample_time_mean",
    "sample_time_min",
    "sample_time_max",
)


def _scalar_training_metrics(outputs: Mapping[str, Any]) -> dict[str, float]:
    metrics = {}
    for name in _LOSS_METRIC_NAMES:
        value = outputs.get(name)
        if value is None:
            continue
        if not torch.is_tensor(value) or value.ndim != 0:
            raise ValueError(f"training loss output {name!r} must be a scalar tensor")
        if not torch.isfinite(value):
            raise FloatingPointError(f"training metric {name!r} must be finite")
        metrics[f"train/{name}"] = float(value.detach().cpu().item())
    return metrics


def _training_protocol_metadata(model: nn.Module, batches: Iterable[Any]) -> dict[str, Any]:
    if getattr(getattr(model, "config", None), "memory_backend", None) != "none":
        return {"query_sampling_protocol": "episode"}
    queries_per_update = getattr(batches, "queries_per_update", None)
    if not isinstance(queries_per_update, int) or queries_per_update <= 0:
        raise ValueError("no-memory training requires cached independent-query data")
    return {
        "query_sampling_protocol": "independent_query",
        "queries_per_update": queries_per_update,
        "mean_episode_queries": float(batches.mean_episode_queries),
        "episode_count": int(batches.episode_count),
    }


def _validate_training_protocol(checkpoint: Path, expected: Mapping[str, Any]) -> None:
    saved = json.loads((checkpoint / "metadata.json").read_text(encoding="utf-8"))
    saved_protocol = saved.get("query_sampling_protocol", "episode")
    expected_protocol = expected["query_sampling_protocol"]
    if saved_protocol != expected_protocol:
        raise ValueError(
            "checkpoint training protocol mismatch: "
            f"expected {expected_protocol!r}, got {saved_protocol!r}; use a new checkpoint root"
        )
    if expected_protocol == "independent_query":
        for field in ("queries_per_update", "mean_episode_queries", "episode_count"):
            if saved.get(field) != expected[field]:
                raise ValueError(
                    f"checkpoint training protocol mismatch for {field}: "
                    f"expected {expected[field]!r}, got {saved.get(field)!r}"
                )


def _backward_cached_queries(
    model: nn.Module, batch: Any, device: torch.device, microbatch_size: int
) -> dict[str, float]:
    query_count = int(batch.actions.shape[0])
    if query_count <= 0 or microbatch_size <= 0:
        raise ValueError("cached query batch and microbatch size must be positive")
    aggregated: dict[str, torch.Tensor] = {}
    for start in range(0, query_count, microbatch_size):
        stop = min(start + microbatch_size, query_count)
        # Slice CPU KV rows before transfer; never stage the complete update on the GPU.
        microbatch = batch.slice(start, stop).to(device)
        outputs = model.compute_cached_query_loss(microbatch)
        weight = (stop - start) / query_count
        loss = outputs["loss"]
        if not torch.is_tensor(loss) or loss.ndim != 0:
            raise ValueError("cached query loss must be a scalar tensor")
        (loss * weight).backward()
        for name in _LOSS_METRIC_NAMES:
            value = outputs.get(name)
            if value is None:
                continue
            if not torch.is_tensor(value) or value.ndim != 0:
                raise ValueError(f"training loss output {name!r} must be a scalar tensor")
            value = value.detach()
            if name == "sample_time_min":
                aggregated[name] = value if name not in aggregated else torch.minimum(aggregated[name], value)
            elif name == "sample_time_max":
                aggregated[name] = value if name not in aggregated else torch.maximum(aggregated[name], value)
            else:
                contribution = value * weight
                aggregated[name] = contribution if name not in aggregated else aggregated[name] + contribution
        del microbatch, outputs, loss
    return _scalar_training_metrics(aggregated)


def plugin_trainable_parameters(model: nn.Module) -> tuple[list[str], list[nn.Parameter]]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    names = [name for name, _ in trainable]
    if not trainable or any(not name.startswith("futuremamba.") for name in names):
        raise RuntimeError(f"invalid trainable parameters: {names}")
    return names, [parameter for _, parameter in trainable]


class WandbMetricLogger:
    def __init__(self, run) -> None:
        self._run = run

    def __call__(self, metrics: Mapping[str, Any], step: int) -> None:
        self._run.log(dict(metrics), step=step)

    def finish(self) -> None:
        self._run.finish()


def create_wandb_metric_logger(*, enabled: bool, wandb_module=None, project: str, name: str, config: Mapping[str, Any]):
    if not enabled:
        return None, None
    if wandb_module is None:
        import wandb as wandb_module
    run = wandb_module.init(
        project=project,
        name=name,
        config=dict(config),
        resume="allow",
    )
    return run, WandbMetricLogger(run)


def _architecture_metadata(model_config: FutureMambaPytorchConfig) -> dict[str, Any]:
    return {
        "architecture": model_config.architecture,
        "memory_input_source": model_config.memory_input_source,
        "action_history_encoding": model_config.action_history_encoding,
    }


def _attach_file_log_handler(log_file: str | Path | None) -> logging.Handler | None:
    if log_file is None:
        return None
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    LOGGER.addHandler(handler)
    return handler


def _wandb_run_name(train_config: _config.TrainConfig) -> str:
    return train_config.wandb_run_name or f"{train_config.name}-seed{train_config.seed}"


def _log_training_identity(
    train_config: _config.TrainConfig, metadata: Mapping[str, Any], checkpoint_root: Path
) -> None:
    LOGGER.info(
        "config=%s task=%s seed=%d checkpoint_root=%s log_file=%s wandb_run_name=%s "
        "architecture=%s memory_input_source=%s query_sampling_protocol=%s "
        "queries_per_update=%s mean_episode_queries=%s",
        train_config.name,
        metadata.get("task_name") or "unknown",
        train_config.seed,
        checkpoint_root,
        train_config.log_file,
        _wandb_run_name(train_config),
        metadata["architecture"],
        metadata["memory_input_source"],
        metadata["query_sampling_protocol"],
        metadata.get("queries_per_update"),
        metadata.get("mean_episode_queries"),
    )


def run_training(
    model: nn.Module,
    batches: Iterable[Any],
    *,
    checkpoint_root: str | Path,
    metadata: Mapping[str, Any],
    num_train_steps: int,
    save_interval: int,
    log_interval: int = 100,
    learning_rate: float = 2.5e-5,
    weight_decay: float = 1e-10,
    clip_gradient_norm: float = 1.0,
    resume: bool = False,
    device: torch.device | str | None = None,
    metric_logger=None,
    terminal_monitor_enabled: bool = False,
) -> TrainingResult:
    if num_train_steps <= 0:
        raise ValueError("num_train_steps must be positive")
    if save_interval <= 0:
        raise ValueError("save_interval must be positive")
    if log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if clip_gradient_norm <= 0:
        raise ValueError("clip_gradient_norm must be positive")
    metadata = {**metadata, **_training_protocol_metadata(model, batches)}
    cached_queries = metadata["query_sampling_protocol"] == "independent_query"
    if cached_queries and terminal_monitor_enabled:
        raise ValueError("terminal monitoring is not supported for independent-query training")
    device = _model_device(model) if device is None else torch.device(device)
    model.to(device)
    model.train(True)
    _, parameters = plugin_trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    checkpoint_root = Path(checkpoint_root)
    start_step = 0
    data_iterator_step = 0
    processed_queries = 0
    if resume:
        latest = _latest_checkpoint(checkpoint_root)
        _validate_training_protocol(latest, metadata)
        restored = load_futuremamba_checkpoint(
            latest, model, optimizer, scheduler, expected_metadata=metadata, map_location=device
        )
        start_step = restored.step
        data_iterator_step = restored.data_iterator_step
        processed_queries = int(
            restored.metadata.get(
                "processed_queries",
                data_iterator_step * metadata["queries_per_update"] if cached_queries else 0,
            )
        )
        if start_step > num_train_steps:
            raise ValueError(f"checkpoint step {start_step} exceeds requested num_train_steps {num_train_steps}")
    else:
        if checkpoint_root.exists():
            if any(checkpoint_root.iterdir()):
                raise ValueError(f"checkpoint root already contains files: {checkpoint_root}")
        else:
            checkpoint_root.mkdir(parents=True)

    if cached_queries and hasattr(batches, "iter_from_update"):
        batch_iterator = iter(batches.iter_from_update(data_iterator_step))
    else:
        batch_iterator = _iterator_at_step(batches, data_iterator_step)
    losses: list[float] = []
    last_checkpoint: Path | None = None
    for step in range(start_step, num_train_steps):
        step_started = time.perf_counter()
        batch = next(batch_iterator)
        data_time = time.perf_counter() - step_started
        optimizer.zero_grad(set_to_none=True)
        if cached_queries:
            query_count = int(batch.actions.shape[0])
            if query_count != metadata["queries_per_update"]:
                raise ValueError("cached batch size must equal queries_per_update")
            metrics = _backward_cached_queries(model, batch, device, int(model.config.frozen_prefix_microbatch_size))
        else:
            batch = _to_device(batch, device)
            outputs = model.compute_episode_loss(batch)
            loss = outputs["loss"]
            metrics = _scalar_training_metrics(outputs)
            loss.backward()
            query_mask = getattr(batch, "train_query_mask", None)
            if query_mask is None:
                query_mask = getattr(batch, "query_mask", None)
            query_count = 0 if query_mask is None else int(torch.as_tensor(query_mask).sum().item())
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_gradient_norm)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(f"plugin gradient norm must be finite at step {step}")
        optimizer.step()
        scheduler.step()
        data_iterator_step += 1
        completed_step = step + 1
        processed_queries += query_count
        losses.append(metrics["train/loss"])
        metrics.update(
            {
                "train/grad_norm": float(torch.as_tensor(grad_norm).detach().cpu().item()),
                "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train/data_iterator_step": data_iterator_step,
                "train/processed_queries": processed_queries,
                "train/data_time": data_time,
            }
        )
        if terminal_monitor_enabled and (completed_step % log_interval == 0 or completed_step == num_train_steps):
            cuda_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
            was_training = model.training
            with torch.random.fork_rng(devices=cuda_devices):
                torch.manual_seed(10_000_000)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(10_000_000)
                model.eval()
                try:
                    terminal_monitor = model.compute_cached_terminal_monitor_loss(batch)
                finally:
                    model.train(was_training)
            if not torch.isfinite(terminal_monitor):
                raise FloatingPointError(f"terminal monitor loss must be finite at step {completed_step}")
            metrics["train/terminal_monitor_loss"] = float(terminal_monitor.cpu().item())
            metrics["train/terminal_monitor_step"] = completed_step
            if metrics["train/loss"] != metrics["train/flow_loss"]:
                raise RuntimeError("terminal monitor experiment must optimize flow loss only")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        metrics["train/step_time"] = time.perf_counter() - step_started
        if cached_queries:
            metrics["train/queries_per_update"] = metadata["queries_per_update"]
        if metric_logger is not None:
            metric_logger(metrics, completed_step)
        if completed_step % log_interval == 0 or completed_step == num_train_steps:
            LOGGER.info(
                "step=%d/%d loss=%.8f flow_loss=%.8f terminal_loss=%.8f "
                "handoff_loss=%.8f boundary_loss=%.8f grad_norm=%.8f "
                "processed_queries=%d data_time=%.6f step_time=%.6f",
                completed_step,
                num_train_steps,
                metrics["train/loss"],
                metrics.get("train/flow_loss", 0.0),
                metrics.get("train/terminal_loss", 0.0),
                metrics.get("train/handoff_loss", 0.0),
                metrics.get("train/boundary_loss", 0.0),
                metrics["train/grad_norm"],
                processed_queries,
                data_time,
                metrics["train/step_time"],
            )
        if completed_step % save_interval == 0 or completed_step == num_train_steps:
            last_checkpoint = checkpoint_root / str(completed_step)
            save_futuremamba_checkpoint(
                last_checkpoint,
                model,
                optimizer,
                scheduler,
                step=completed_step,
                metadata={**metadata, "processed_queries": processed_queries},
                data_iterator_step=data_iterator_step,
            )
    return TrainingResult(
        start_step=start_step,
        end_step=num_train_steps,
        data_iterator_step=data_iterator_step,
        losses=losses,
        checkpoint=last_checkpoint,
    )


def build_checkpoint_metadata(model: nn.Module, model_config: FutureMambaPytorchConfig) -> dict[str, Any]:
    metadata = model_config.checkpoint_metadata()
    metadata.update(_architecture_metadata(model_config))
    metadata["base_checkpoint_checksum"] = model.base_checksum()
    metadata["torch_version"] = torch.__version__
    metadata["cuda_version"] = torch.version.cuda
    try:
        metadata["triton_version"] = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        metadata["triton_version"] = None
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        metadata["gpu_name"] = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
        metadata["compute_capability"] = f"{major}.{minor}"
    return metadata


def create_training_data(train_config: _config.TrainConfig, *, shuffle: bool, queries_per_update: int | None = None):
    if train_config.model.memory_backend == "none":
        from openpi.training.cached_query_data_loader import create_cached_query_data

        return create_cached_query_data(train_config, queries_per_update=queries_per_update)
    if queries_per_update is not None:
        raise ValueError("queries_per_update only applies to no-memory training")
    if not isinstance(train_config.data, _config.LeRobotLiberoDataConfig):
        raise TypeError("action-history handoff FutureMamba requires LeRobotLiberoDataConfig")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = _episode_data_loader.create_lerobot_episode_dataset(
        data_config=data_config,
        episode_config=train_config.episode_data,
        action_horizon=int(train_config.model.action_horizon),
    )
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=train_config.batch_size,
        shuffle=shuffle,
        num_workers=train_config.num_workers,
        seed=train_config.seed,
        framework="pytorch",
        collate_fn=_episode_data_loader.EpisodeCollator(),
    )
    return _data_loader.EpisodeDataLoaderImpl(data_config, data_loader)


def load_training_model(train_config: _config.TrainConfig, device: torch.device) -> nn.Module:
    if not isinstance(train_config.model, FutureMambaPytorchConfig):
        raise TypeError("train config model must be FutureMambaPytorchConfig")
    base_checkpoint = train_config.pytorch_weight_path or train_config.model.base_checkpoint_uri
    if not base_checkpoint:
        raise ValueError("action-history handoff training requires a frozen base checkpoint")
    checkpoint_path = Path(base_checkpoint.removeprefix("file://")).expanduser()
    weight_path = (
        checkpoint_path if checkpoint_path.name == "model.safetensors" else checkpoint_path / "model.safetensors"
    )
    if not weight_path.is_file():
        raise FileNotFoundError(f"converted base model.safetensors not found: {weight_path}")
    model = train_config.model.create_pytorch().to(device)
    try:
        safetensors.torch.load_model(model.base, weight_path, strict=True)
    except Exception as error:
        raise ValueError(f"strict base checkpoint load failed for {weight_path}: {error}") from error
    model.initialize_progress_from_action_expert()
    model.freeze_base()
    return model


def _iterator_at_step(batches: Iterable[Any], step: int) -> Iterator[Any]:
    if step < 0:
        raise ValueError("data iterator step must be non-negative")
    iterator = iter(batches)
    for _ in range(step):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(batches)
            try:
                next(iterator)
            except StopIteration as error:
                raise ValueError("training batches must not be empty") from error

    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(batches)
            try:
                yield next(iterator)
            except StopIteration as error:
                raise ValueError("training batches must not be empty") from error


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value, **{field.name: _to_device(getattr(value, field.name), device) for field in dataclasses.fields(value)}
        )
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value


def _latest_checkpoint(root: Path) -> Path:
    if not root.is_dir():
        raise FileNotFoundError(f"checkpoint root does not exist: {root}")
    steps = sorted(int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdigit())
    if not steps:
        raise FileNotFoundError(f"no numeric FutureMamba checkpoints found in {root}")
    return root / str(steps[-1])


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return torch.device("cpu") if parameter is None else parameter.device


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train action-history Mamba + Progress Expert")
    parser.add_argument("config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-train-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--queries-per-update", type=int)
    parser.add_argument("--cpu-threads", type=int, help="No-memory CPU threads (default: 4)")
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--pytorch-training-precision", choices=("float32", "bfloat16"))
    parser.add_argument("--wandb-enabled", choices=("true", "false"))
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--device")
    return parser


def apply_cli_overrides(train_config: _config.TrainConfig, args: argparse.Namespace) -> _config.TrainConfig:
    replacements: dict[str, Any] = {}
    for name in (
        "seed",
        "num_train_steps",
        "batch_size",
        "save_interval",
        "log_interval",
        "pytorch_training_precision",
        "wandb_run_name",
    ):
        value = getattr(args, name, None)
        if value is not None:
            replacements[name] = value
    if getattr(args, "wandb_enabled", None) is not None:
        replacements["wandb_enabled"] = args.wandb_enabled == "true"
    if getattr(args, "log_file", None) is not None:
        replacements["log_file"] = str(args.log_file)
    for name in ("queries_per_update", "cpu_threads"):
        value = getattr(args, name, None)
        if value is not None:
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            if train_config.model.memory_backend != "none":
                raise ValueError(f"{name} only applies to no-memory training")
    effective_seed = replacements.get("seed", train_config.seed)
    replacements["model"] = dataclasses.replace(train_config.model, train_seed=effective_seed)
    return dataclasses.replace(train_config, **replacements) if replacements else train_config


def seed_training_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    train_config = apply_cli_overrides(_config.get_config(args.config), args)
    if train_config.model.memory_backend == "none":
        torch.set_num_threads(4 if args.cpu_threads is None else args.cpu_threads)
    seed_training_runtime(train_config.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_root = args.checkpoint_root or Path(train_config.checkpoint_base_dir) / train_config.name
    file_log_handler = _attach_file_log_handler(train_config.log_file)
    wandb_run = None
    try:
        if args.overwrite and checkpoint_root.exists():
            shutil.rmtree(checkpoint_root)
        model = load_training_model(train_config, device)
        metadata = build_checkpoint_metadata(model, train_config.model)
        metadata["memory_update_stride"] = train_config.episode_data.query_stride
        metadata["train_query_stride"] = train_config.episode_data.query_stride
        if train_config.model.memory_backend == "none":
            data = create_training_data(train_config, shuffle=True, queries_per_update=args.queries_per_update)
            metadata["cpu_threads"] = torch.get_num_threads()
        else:
            data = create_training_data(train_config, shuffle=True)
        metadata.update(_training_protocol_metadata(model, data))
        _log_training_identity(train_config, metadata, checkpoint_root)
        optimizer_config = train_config.optimizer
        schedule = train_config.lr_schedule
        learning_rate = float(getattr(schedule, "peak_lr", 2.5e-5))
        wandb_name = _wandb_run_name(train_config)
        wandb_run, metric_logger = create_wandb_metric_logger(
            enabled=train_config.wandb_enabled,
            project="futuremamba",
            name=wandb_name,
            config=metadata,
        )
        run_training(
            model,
            data,
            checkpoint_root=checkpoint_root,
            metadata=metadata,
            num_train_steps=train_config.num_train_steps,
            save_interval=train_config.save_interval,
            learning_rate=learning_rate,
            weight_decay=float(getattr(optimizer_config, "weight_decay", 1e-10)),
            log_interval=train_config.log_interval,
            clip_gradient_norm=float(getattr(optimizer_config, "clip_gradient_norm", 1.0)),
            resume=args.resume,
            device=device,
            metric_logger=metric_logger,
            terminal_monitor_enabled=False,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if file_log_handler is not None:
            LOGGER.removeHandler(file_log_handler)
            file_log_handler.close()
    return 0


if __name__ == "__main__":
    _configure_logging()
    raise SystemExit(main())
