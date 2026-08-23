from __future__ import annotations

import argparse
from collections.abc import Iterable, Iterator, Mapping
import dataclasses
import importlib.metadata
import random

import numpy as np
import logging
from pathlib import Path
import shutil
from typing import Any

import safetensors.torch
import torch
from torch import nn

from openpi.models_pytorch.futuremamba_config import FutureMambaPytorchConfig
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader
from openpi.training import episode_data_loader as _episode_data_loader
from openpi.training import robomme_episode_dataset as _robomme_episode_dataset
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
            raise ValueError(f"compute_episode_loss output {name!r} must be a scalar tensor")
        if not torch.isfinite(value):
            raise FloatingPointError(f"training metric {name!r} must be finite")
        metrics[f"train/{name}"] = float(value.detach().cpu().item())
    return metrics


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


def create_wandb_metric_logger(
    *, enabled: bool, wandb_module=None, project: str, name: str, config: Mapping[str, Any]
):
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
    device = _model_device(model) if device is None else torch.device(device)
    model.to(device)
    model.train(True)
    _, parameters = plugin_trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    checkpoint_root = Path(checkpoint_root)
    start_step = 0
    data_iterator_step = 0
    if resume:
        latest = _latest_checkpoint(checkpoint_root)
        restored = load_futuremamba_checkpoint(
            latest, model, optimizer, scheduler, expected_metadata=metadata, map_location=device
        )
        start_step = restored.step
        data_iterator_step = restored.data_iterator_step
        if start_step > num_train_steps:
            raise ValueError(f"checkpoint step {start_step} exceeds requested num_train_steps {num_train_steps}")
    else:
        if checkpoint_root.exists():
            if any(checkpoint_root.iterdir()):
                raise ValueError(f"checkpoint root already contains files: {checkpoint_root}")
        else:
            checkpoint_root.mkdir(parents=True)

    batch_iterator = _iterator_at_step(batches, data_iterator_step)
    losses: list[float] = []
    last_checkpoint: Path | None = None
    for step in range(start_step, num_train_steps):
        batch = _to_device(next(batch_iterator), device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model.compute_episode_loss(batch)
        loss = outputs["loss"]
        metrics = _scalar_training_metrics(outputs)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, clip_gradient_norm)
        if not torch.isfinite(torch.as_tensor(grad_norm)):
            raise FloatingPointError(f"plugin gradient norm must be finite at step {step}")
        optimizer.step()
        scheduler.step()
        data_iterator_step += 1
        completed_step = step + 1
        losses.append(float(loss.detach().cpu().item()))
        metrics.update(
            {
                "train/grad_norm": float(torch.as_tensor(grad_norm).detach().cpu().item()),
                "train/learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train/data_iterator_step": data_iterator_step,
            }
        )
        if terminal_monitor_enabled and (
            completed_step % log_interval == 0 or completed_step == num_train_steps
        ):
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
                raise FloatingPointError(
                    f"terminal monitor loss must be finite at step {completed_step}"
                )
            metrics["train/terminal_monitor_loss"] = float(terminal_monitor.cpu().item())
            metrics["train/terminal_monitor_step"] = completed_step
            if metrics["train/loss"] != metrics["train/flow_loss"]:
                raise RuntimeError("terminal monitor experiment must optimize flow loss only")
        if metric_logger is not None:
            metric_logger(metrics, completed_step)
        if completed_step % log_interval == 0 or completed_step == num_train_steps:
            LOGGER.info(
                "step=%d/%d loss=%.8f flow_loss=%.8f terminal_loss=%.8f "
                "handoff_loss=%.8f boundary_loss=%.8f grad_norm=%.8f",
                completed_step,
                num_train_steps,
                metrics["train/loss"],
                metrics.get("train/flow_loss", 0.0),
                metrics.get("train/terminal_loss", 0.0),
                metrics.get("train/handoff_loss", 0.0),
                metrics.get("train/boundary_loss", 0.0),
                metrics["train/grad_norm"],
            )
        if completed_step % save_interval == 0 or completed_step == num_train_steps:
            last_checkpoint = checkpoint_root / str(completed_step)
            save_futuremamba_checkpoint(
                last_checkpoint,
                model,
                optimizer,
                scheduler,
                step=completed_step,
                metadata=metadata,
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
    metadata["base_checkpoint_checksum"] = _base_checksum(model)
    metadata["base_checkpoint_uri"] = model_config.base_checkpoint_uri
    metadata["base_assets_checksum"] = model_config.base_assets_checksum
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

def create_training_data(train_config: _config.TrainConfig, *, shuffle: bool):
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if isinstance(train_config.data, _config.RoboMMEDataConfig):
        dataset = _robomme_episode_dataset.create_robomme_episode_dataset(
            data_config, train_config.model, train_config.episode_data
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
    return _data_loader.create_episode_data_loader(train_config, shuffle=shuffle)


def load_training_model(train_config: _config.TrainConfig, device: torch.device) -> nn.Module:
    if not isinstance(train_config.model, FutureMambaPytorchConfig):
        raise TypeError("train config model must be FutureMambaPytorchConfig")
    base_checkpoint = train_config.pytorch_weight_path or train_config.model.base_checkpoint_uri
    if not base_checkpoint:
        raise ValueError("FutureMamba PyTorch training requires pytorch_weight_path or base_checkpoint_uri")
    checkpoint_path = Path(base_checkpoint.removeprefix("file://")).expanduser()
    weight_path = checkpoint_path if checkpoint_path.name == "model.safetensors" else checkpoint_path / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"converted pi0.5 model.safetensors not found: {weight_path}")
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


def _base_checksum(model: nn.Module) -> str:
    checksum = getattr(model, "base_checksum", None)
    if not callable(checksum):
        raise ValueError("FutureMamba model must provide base_checksum()")
    return checksum()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the pure PyTorch FutureMamba plugin")
    parser.add_argument("config")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-train-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--log-interval", type=int)
    parser.add_argument("--pytorch-training-precision", choices=("float32", "bfloat16"))
    parser.add_argument("--wandb-enabled", choices=("true", "false"))
    parser.add_argument(
        "--terminal-monitor-enabled", choices=("true", "false"), default="false"
    )
    parser.add_argument("--episode-data-dir", type=Path)
    parser.add_argument("--conditioning-cache-dir", type=Path)
    parser.add_argument("--robomme-dataset-checksum")
    parser.add_argument("--robomme-task-suite")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--device")
    return parser

def apply_cli_overrides(train_config: _config.TrainConfig, args: argparse.Namespace) -> _config.TrainConfig:
    replacements: dict[str, Any] = {}
    if args.seed is not None:
        replacements["seed"] = args.seed
    if args.num_train_steps is not None:
        replacements["num_train_steps"] = args.num_train_steps
    if args.batch_size is not None:
        replacements["batch_size"] = args.batch_size
    if args.save_interval is not None:
        replacements["save_interval"] = args.save_interval
    if args.log_interval is not None:
        replacements["log_interval"] = args.log_interval
    if args.pytorch_training_precision is not None:
        replacements["pytorch_training_precision"] = args.pytorch_training_precision
    if args.wandb_enabled is not None:
        replacements["wandb_enabled"] = args.wandb_enabled == "true"
    if args.episode_data_dir is not None:
        if not isinstance(train_config.data, _config.RoboMMEDataConfig):
            raise ValueError("--episode-data-dir is only valid for RoboMME configs")
        replacements["data"] = dataclasses.replace(
            train_config.data, episode_data_dir=str(args.episode_data_dir)
        )
    if args.conditioning_cache_dir is not None:
        if not isinstance(train_config.data, _config.RoboMMEDataConfig):
            raise ValueError("--conditioning-cache-dir is only valid for RoboMME configs")
        replacements["data"] = dataclasses.replace(
            replacements.get("data", train_config.data),
            conditioning_cache_dir=str(args.conditioning_cache_dir),
        )
    if args.robomme_dataset_checksum is not None or args.robomme_task_suite is not None:
        if not isinstance(train_config.data, _config.RoboMMEDataConfig):
            raise ValueError("RoboMME provenance overrides are only valid for RoboMME configs")
        replacements["model"] = dataclasses.replace(
            replacements.get("model", train_config.model),
            robomme_dataset_checksum=args.robomme_dataset_checksum,
            robomme_task_suite=args.robomme_task_suite,
        )
    effective_seed = train_config.seed if args.seed is None else args.seed
    replacements["model"] = dataclasses.replace(
        replacements.get("model", train_config.model), train_seed=effective_seed
    )
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
    seed_training_runtime(train_config.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_root = args.checkpoint_root or Path(train_config.checkpoint_base_dir) / train_config.name
    if args.overwrite and checkpoint_root.exists():
        shutil.rmtree(checkpoint_root)
    model = load_training_model(train_config, device)
    metadata = build_checkpoint_metadata(model, train_config.model)
    terminal_monitor_enabled = args.terminal_monitor_enabled == "true"
    metadata["memory_update_stride"] = train_config.episode_data.query_stride
    metadata["train_query_stride"] = (
        train_config.episode_data.train_query_stride or train_config.episode_data.query_stride
    )
    metadata["terminal_monitor_only"] = terminal_monitor_enabled
    metadata["terminal_monitor_interval"] = (
        train_config.log_interval if terminal_monitor_enabled else None
    )
    data = create_training_data(train_config, shuffle=True)
    optimizer_config = train_config.optimizer
    schedule = train_config.lr_schedule
    learning_rate = float(getattr(schedule, "peak_lr", 2.5e-5))
    wandb_run, metric_logger = create_wandb_metric_logger(
        enabled=train_config.wandb_enabled,
        project="futuremamba",
        name=f"{train_config.name}-{train_config.model.robomme_task_suite or 'default'}-seed{train_config.seed}",
        config=metadata,
    )
    try:
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
            terminal_monitor_enabled=terminal_monitor_enabled,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    return 0


if __name__ == "__main__":
    _configure_logging()
    raise SystemExit(main())
