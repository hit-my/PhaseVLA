from __future__ import annotations

import argparse
import dataclasses
import json
import os
import logging
from pathlib import Path
from types import MethodType

import numpy as np
import torch

import openpi.models.model as openpi_model
from openpi.training import config as training_config
from openpi.training import episode_data_loader

from isolated_futuremamba_speedup_benchmark import fast_compute_episode_loss
from isolated_futuremamba_speedup_benchmark import load_training_module

LOGGER = logging.getLogger("accelerated_futuremamba_train")


@dataclasses.dataclass(frozen=True)
class CachedEpisodeSpec:
    episode_id: int
    action_file: Path
    conditioning_file: Path


class CompleteCachedEpisodeDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        *,
        config_name: str,
        action_root: Path,
        conditioning_root: Path,
        conditioning_cache_name: str | None = None,
        conditioning_alt_root: Path | None = None,
    ):
        manifest_path = action_root / config_name / "manifest.json"
        manifest = json.loads(manifest_path.read_text(errors="strict"))
        entries = manifest.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"invalid action-cache manifest: {manifest_path}")
        conditioning_name = conditioning_cache_name or config_name
        conditioning_roots = (conditioning_root,) if conditioning_alt_root is None else (conditioning_root, conditioning_alt_root)
        self._specs = tuple(
            CachedEpisodeSpec(
                episode_id=int(entry["episode_id"]),
                action_file=Path(entry["file"]),
                conditioning_file=next(
                    (
                        root / conditioning_name / f"episode_{int(entry['episode_id']):06d}.pt"
                        for root in conditioning_roots
                        if (root / conditioning_name / f"episode_{int(entry['episode_id']):06d}.pt").is_file()
                    ),
                    conditioning_roots[0] / conditioning_name / f"episode_{int(entry['episode_id']):06d}.pt",
                ),
            )
            for entry in entries
        )
        for spec in self._specs:
            if not spec.action_file.is_file():
                raise FileNotFoundError(spec.action_file)
            if not spec.conditioning_file.is_file():
                raise FileNotFoundError(spec.conditioning_file)

    def __len__(self) -> int:
        return len(self._specs)

    def __getitem__(self, index: int) -> episode_data_loader.EpisodeExample:
        spec = self._specs[int(index)]
        actions = torch.load(spec.action_file, map_location="cpu", weights_only=True)
        conditioning = torch.load(
            spec.conditioning_file, map_location="cpu", weights_only=True
        )
        required_actions = {
            "actions",
            "action_mask",
            "executed_actions",
            "executed_action_mask",
        }
        required_conditioning = {
            "last_valid_hidden",
            "prefix_mask",
            "action_expert_keys",
            "action_expert_values",
        }
        if set(actions) != required_actions:
            raise ValueError(f"action-cache keys mismatch: {spec.action_file}")
        if set(conditioning) != required_conditioning:
            raise ValueError(
                f"conditioning-cache keys mismatch: {spec.conditioning_file}"
            )
        action_arrays = {
            key: value.numpy() if torch.is_tensor(value) else np.asarray(value)
            for key, value in actions.items()
        }
        conditioning_arrays = {
            key: (
                value.float().numpy()
                if torch.is_tensor(value) and value.dtype is torch.bfloat16
                else value.numpy()
                if torch.is_tensor(value)
                else np.asarray(value)
            )
            for key, value in conditioning.items()
        }
        queries = int(action_arrays["actions"].shape[0])
        if any(value.shape[0] != queries for value in conditioning_arrays.values()):
            raise ValueError(
                f"action/conditioning query mismatch for episode {spec.episode_id}"
            )
        return episode_data_loader.EpisodeExample(
            observation=openpi_model.Observation(
                images={},
                image_masks={},
                state=np.zeros((queries, 1), dtype=np.float32),
            ),
            actions=action_arrays["actions"],
            action_mask=action_arrays["action_mask"].astype(np.bool_, copy=False),
            executed_actions=action_arrays["executed_actions"],
            executed_action_mask=action_arrays["executed_action_mask"].astype(
                np.bool_, copy=False
            ),
            episode_index=spec.episode_id,
            conditioning_cache=conditioning_arrays,
        )
class ReplayPositionedBatches:
    """Replay the original shuffled sampler cheaply before loading real batches."""

    def __init__(self, dataset, *, batch_size: int, seed: int, replay_prefix: int):
        self._dataset = dataset
        self._batch_size = int(batch_size)
        self._seed = int(seed)
        self._replay_prefix = int(replay_prefix)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self._seed)
        index_loader = torch.utils.data.DataLoader(
            range(len(self._dataset)),
            batch_size=self._batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            generator=generator,
            collate_fn=list,
        )
        position = 0
        collator = episode_data_loader.EpisodeCollator()
        while True:
            for indices in index_loader:
                position += 1
                if position <= self._replay_prefix:
                    yield None
                    continue
                episodes = [self._dataset[int(index)] for index in indices]
                collated = collator(episodes)
                yield episode_data_loader.episode_batch_to_torch(
                    collated, torch.device("cpu")
                )




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-checkpoint-root", type=Path)
    parser.add_argument("--source-step", type=int, default=0)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--wandb-run-name", required=True)
    parser.add_argument("--action-cache-root", type=Path, required=True)
    parser.add_argument("--conditioning-cache-root", type=Path, required=True)
    parser.add_argument("--action-cache-name")
    parser.add_argument("--conditioning-cache-name")
    parser.add_argument("--wandb-entity", required=True)
    parser.add_argument("--wandb-api-key-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-train-steps", type=int, default=3000)
    return parser.parse_args()


def prepare_independent_resume_root(args: argparse.Namespace) -> Path | None:
    if args.source_step == 0:
        if args.source_checkpoint_root is not None:
            raise ValueError("source-checkpoint-root is invalid when source-step is zero")
        args.checkpoint_root.mkdir(parents=True, exist_ok=True)
        if any(args.checkpoint_root.iterdir()):
            raise FileExistsError(
                f"new accelerated checkpoint root is not empty: {args.checkpoint_root}"
            )
        return None
    if args.source_checkpoint_root is None:
        raise ValueError("source-checkpoint-root is required when resuming")
    source = args.source_checkpoint_root / str(args.source_step)
    required = (
        "plugin.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pt",
        "metadata.json",
    )
    missing = [name for name in required if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"source checkpoint missing {missing[0]}: {source}")
    args.checkpoint_root.mkdir(parents=True, exist_ok=True)
    resume = args.checkpoint_root / str(args.source_step)
    if resume.exists() or resume.is_symlink():
        if resume.resolve() != source.resolve():
            raise FileExistsError(
                f"accelerated resume entry already exists with another identity: {resume}"
            )
    else:
        resume.symlink_to(source, target_is_directory=True)
    return source


def create_cached_training_data(training, config, args):
    del training
    dataset = CompleteCachedEpisodeDataset(
        config_name=args.action_cache_name or config.name,
        action_root=args.action_cache_root,
        conditioning_root=args.conditioning_cache_root,
        conditioning_cache_name=args.conditioning_cache_name or config.name,
    )
    return ReplayPositionedBatches(
        dataset,
        batch_size=config.batch_size,
        seed=config.seed,
        replay_prefix=args.source_step,
    )


def main() -> int:
    args = parse_args()
    if args.source_step < 0:
        raise ValueError("source-step must be non-negative")
    if args.num_train_steps <= args.source_step:
        raise ValueError("num-train-steps must exceed source-step")
    logging.basicConfig(level=logging.INFO)
    training = load_training_module()
    config = training_config.get_config(args.config)
    if config.batch_size != 1:
        raise ValueError("accelerated complete-cache training currently requires batch_size=1")
    if config.episode_data.query_stride != 1:
        raise ValueError("accelerated history path currently requires query_stride=1")
    if config.episode_data.executed_horizon != 1:
        raise ValueError("accelerated history path currently requires executed_horizon=1")
    config = dataclasses.replace(
        config,
        num_train_steps=args.num_train_steps,
        checkpoint_base_dir=str(args.checkpoint_root.parent),
        log_file=str(args.log_file),
        wandb_run_name=args.wandb_run_name,
        wandb_enabled=True,
    )
    training.seed_training_runtime(config.seed)
    source = prepare_independent_resume_root(args)
    api_key = args.wandb_api_key_file.read_text(encoding="utf-8").strip()
    if not api_key.startswith("wandb_v1_"):
        raise ValueError("invalid W&B API key file")
    os.environ["WANDB_API_KEY"] = api_key
    os.environ["WANDB_ENTITY"] = args.wandb_entity
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = training._attach_file_log_handler(args.log_file)
    wandb_run = None
    device = torch.device(args.device)
    try:
        model = training.load_training_model(config, device)
        metadata = training.build_checkpoint_metadata(model, config.model)
        metadata["memory_update_stride"] = config.episode_data.query_stride
        metadata["train_query_stride"] = config.episode_data.query_stride
        model.compute_episode_loss = MethodType(fast_compute_episode_loss, model)
        data = create_cached_training_data(training, config, args)
        optimizer_config = config.optimizer
        schedule = config.lr_schedule
        learning_rate = float(getattr(schedule, "peak_lr", 2.5e-5))
        import wandb

        wandb_run = wandb.init(
            entity=args.wandb_entity,
            project="futuremamba",
            name=args.wandb_run_name,
            config={
                **metadata,
                "accelerated_training": True,
                "accelerated_source_checkpoint": None if source is None else str(source),
                "accelerated_action_cache_root": str(args.action_cache_root),
                "accelerated_conditioning_cache_root": str(args.conditioning_cache_root),
                "accelerated_history_schedule": "block_batched_20_action_prefixes",
            },
            resume="allow",
            settings=wandb.Settings(api_key=api_key),
        )
        metric_logger = training.WandbMetricLogger(wandb_run)
        print(
            json.dumps(
                {
                    "event": "accelerated_training_ready",
                    "config": config.name,
                    "source_checkpoint": None if source is None else str(source),
                    "checkpoint_root": str(args.checkpoint_root),
                    "device": str(device),
                    "episodes": len(data._dataset),
                    "replay_prefix": args.source_step,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        result = training.run_training(
            model,
            data,
            checkpoint_root=args.checkpoint_root,
            metadata=metadata,
            num_train_steps=config.num_train_steps,
            save_interval=config.save_interval,
            learning_rate=learning_rate,
            weight_decay=float(
                getattr(optimizer_config, "weight_decay", 1e-10)
            ),
            log_interval=config.log_interval,
            clip_gradient_norm=float(
                getattr(optimizer_config, "clip_gradient_norm", 1.0)
            ),
            resume=args.source_step > 0,
            device=device,
            metric_logger=metric_logger,
            terminal_monitor_enabled=False,
        )
        print(
            json.dumps(
                {
                    "event": "accelerated_training_completed",
                    "config": config.name,
                    "start_step": result.start_step,
                    "end_step": result.end_step,
                    "data_iterator_step": result.data_iterator_step,
                    "checkpoint": str(result.checkpoint),
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        return 0
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if handler is not None:
            training.LOGGER.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    raise SystemExit(main())
