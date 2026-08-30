from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import time

import torch

from openpi.models_pytorch.gemma_pytorch import PrefixKVView
from openpi.training import config as _config
from openpi.training import episode_data_loader as _episode_loader
from scripts import train_futuremamba_pytorch as _train


def _slice_observation(observation, start: int, stop: int):
    return dataclasses.replace(
        observation,
        images={key: value[start:stop] for key, value in observation.images.items()},
        image_masks={key: value[start:stop] for key, value in observation.image_masks.items()},
        state=observation.state[start:stop],
        tokenized_prompt=None
        if observation.tokenized_prompt is None
        else observation.tokenized_prompt[start:stop],
        tokenized_prompt_mask=None
        if observation.tokenized_prompt_mask is None
        else observation.tokenized_prompt_mask[start:stop],
        token_ar_mask=None,
        token_loss_mask=None,
    )


def _atomic_torch_save(payload: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def precompute(args: argparse.Namespace) -> None:
    train_config = _config.get_config(args.config)
    data_factory = dataclasses.replace(train_config.data, conditioning_cache_dir=None)
    data_config = data_factory.create(train_config.assets_dirs, train_config.model)
    dataset = _episode_loader.create_lerobot_episode_dataset(
        data_config=data_config,
        episode_config=train_config.episode_data,
        action_horizon=int(train_config.model.action_horizon),
    )
    device = torch.device(args.device)
    model = _train.load_training_model(train_config, device).eval()
    mapping = tuple(train_config.model.resolved_progress_layer_indices)
    start = int(args.start_ordinal)
    stop = None if args.max_episodes is None else start + int(args.max_episodes)
    selected = dataset.episodes[start:stop]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    total_queries = 0

    for ordinal, record in enumerate(selected, start=1):
        output_path = args.output_dir / f"episode_{record.episode_index:06d}.pt"
        if output_path.is_file() and not args.overwrite:
            continue
        episode = dataset.episode_at(dataset.episodes.index(record))
        observation = _episode_loader._observation_to_torch(episode.observation, device)
        hidden_rows = []
        mask_rows = []
        key_rows = []
        value_rows = []
        for start in range(0, len(record.queries), args.microbatch_size):
            stop = min(start + args.microbatch_size, len(record.queries))
            current = _slice_observation(observation, start, stop)
            with torch.no_grad():
                frozen = model.base.extract_prefix_context(current, train=False)
                hidden = model.base.last_valid_prefix(frozen)
                view = PrefixKVView.from_cache(frozen.kv_cache, mapping, frozen.pad_mask)
            for row in range(stop - start):
                valid = int(view.valid_lengths[row].item())
                hidden_rows.append(hidden[row].detach().cpu().to(torch.bfloat16))
                mask_rows.append(torch.ones(valid, dtype=torch.bool))
                key_rows.append(
                    torch.stack(
                        [
                            layer[0][row, :, :valid].detach().cpu().to(torch.bfloat16)
                            for layer in view.layers
                        ],
                        dim=0,
                    )
                )
                value_rows.append(
                    torch.stack(
                        [
                            layer[1][row, :, :valid].detach().cpu().to(torch.bfloat16)
                            for layer in view.layers
                        ],
                        dim=0,
                    )
                )
        max_prefix = max(mask.numel() for mask in mask_rows)
        queries = len(mask_rows)
        layers, heads, _, head_dim = key_rows[0].shape
        prefix_mask = torch.zeros(queries, max_prefix, dtype=torch.bool)
        keys = torch.zeros(queries, layers, heads, max_prefix, head_dim, dtype=torch.bfloat16)
        values = torch.zeros_like(keys)
        for row, (mask, key, value) in enumerate(zip(mask_rows, key_rows, value_rows, strict=True)):
            valid = mask.numel()
            prefix_mask[row, :valid] = mask
            keys[row, :, :, :valid] = key
            values[row, :, :, :valid] = value
        _atomic_torch_save(
            {
                "last_valid_hidden": torch.stack(hidden_rows),
                "prefix_mask": prefix_mask,
                "action_expert_keys": keys,
                "action_expert_values": values,
            },
            output_path,
        )
        total_queries += queries
        print(
            json.dumps(
                {
                    "episode": record.episode_index,
                    "ordinal": ordinal,
                    "episodes": len(selected),
                    "queries": queries,
                    "total_queries": total_queries,
                    "bytes": output_path.stat().st_size,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    print(
        json.dumps(
            {
                "status": "completed",
                "config": args.config,
                "output_dir": str(args.output_dir),
                "episodes": len(selected),
                "queries": total_queries,
                "elapsed_seconds": time.perf_counter() - started,
            },
            separators=(",", ":"),
        ),
        flush=True,
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("config")
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--device", default="cuda")
    result.add_argument("--microbatch-size", type=int, default=16)
    result.add_argument("--start-ordinal", type=int, default=0)
    result.add_argument("--max-episodes", type=int)
    result.add_argument("--overwrite", action="store_true")
    return result


if __name__ == "__main__":
    precompute(parser().parse_args())
