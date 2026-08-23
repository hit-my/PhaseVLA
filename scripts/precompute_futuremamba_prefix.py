from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

from openpi.training.futuremamba_conditioning_cache import ConditioningCacheEntry
from openpi.training.futuremamba_conditioning_cache import ConditioningCacheWriter
from openpi.training.futuremamba_conditioning_cache import validate_cache_mode


def fake_tiny_expected_identity(*, dtype: str, base_checkpoint: str) -> dict[str, Any]:
    if dtype not in {"float32", "bfloat16"}:
        raise ValueError(f"unsupported dtype {dtype!r}")
    base_checksum = hashlib.sha256(base_checkpoint.encode("utf-8")).hexdigest()
    return {
        "base_checkpoint_checksum": base_checksum,
        "base_weights_checksum": hashlib.sha256(f"{base_checkpoint}:weights".encode()).hexdigest(),
        "assets_checksum": hashlib.sha256(b"fake-tiny-assets-v1").hexdigest(),
        "dataset_checksum": hashlib.sha256(b"fake-tiny-dataset-v1").hexdigest(),
        "task_suite": "fake_tiny",
        "query_stride": 1,
        "tokenizer_config_checksum": hashlib.sha256(b"fake-tiny-tokenizer-v1").hexdigest(),
        "preprocessing_checksum": hashlib.sha256(b"fake-tiny-deterministic-preprocessing-v1").hexdigest(),
        "layer_mapping": [0, 1, 2],
        "dtype": dtype,
    }


def _fake_tiny_entries(max_episodes: int, *, dtype: torch.dtype):
    for query_id in range(max_episodes):
        offset = float(query_id * 100)
        keys = torch.arange(3 * 2 * 4 * 3, dtype=dtype).reshape(3, 2, 4, 3) + offset
        yield ConditioningCacheEntry(
            episode_id=f"fake-episode-{query_id}",
            query_id=query_id,
            last_valid_hidden=torch.tensor([query_id, query_id + 0.5, query_id + 1.0], dtype=dtype),
            prefix_mask=torch.tensor([True, True, False, True]),
            action_expert_kv=(keys, keys + 1000),
        )


def _precompute_fake_tiny(args: argparse.Namespace) -> None:
    validate_cache_mode(train_image_augmentation=False)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    identity = fake_tiny_expected_identity(dtype=args.dtype, base_checkpoint=args.base_checkpoint)
    writer = ConditioningCacheWriter(args.output_dir, identity=identity, action_expert_layer_count=3)
    try:
        for entry in _fake_tiny_entries(args.max_episodes, dtype=dtype):
            writer.write_episode(entry)
        manifest = writer.commit()
    except BaseException:
        writer.abort()
        raise
    print(json.dumps({"output_dir": str(args.output_dir), "entries": len(manifest["entries"])}))


def _precompute_registered_config(args: argparse.Namespace) -> None:
    import dataclasses
    import hashlib
    import pathlib
    import numpy as np
    import safetensors.torch

    from openpi.models import model as _model
    from openpi.models_pytorch.gemma_pytorch import PrefixKVView
    from openpi.models_pytorch.futuremamba_config import FutureMambaPytorchConfig
    from openpi.training import config as _config
    from openpi.training import episode_data_loader as _episode_loader
    from openpi.training import robomme_episode_dataset as _dataset

    train_config = _config.get_config(args.config)
    if not isinstance(train_config.model, FutureMambaPytorchConfig):
        raise RuntimeError(f"config {args.config!r} is not a PyTorch FutureMamba config")
    model_config = train_config.model
    data_config = train_config.data.create(train_config.assets_dirs, model_config)
    data_config = dataclasses.replace(data_config, conditioning_cache_dir=None)
    data_dir = args.episode_data_dir or data_config.episode_data_dir
    if not data_dir:
        raise RuntimeError("--episode-data-dir is required when the registered config does not define one")
    data_config = dataclasses.replace(data_config, episode_data_dir=str(data_dir))
    query_stride = int(train_config.episode_data.query_stride)
    windows = _dataset.RoboMMEEpisodeDataset(
        data_dir,
        window_queries=1,
        action_horizon=int(model_config.action_horizon),
        query_stride=query_stride,
    )
    transformed = _dataset.RoboMMETransformedEpisodeDataset(windows, data_config, model_config)
    device = torch.device(args.device)
    base_path = pathlib.Path(train_config.pytorch_weight_path)
    weight_path = base_path if base_path.name == "model.safetensors" else base_path / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    requested_base = pathlib.Path(args.base_checkpoint)
    if requested_base.resolve() != base_path.resolve():
        raise ValueError(
            f"--base-checkpoint must match registered pytorch_weight_path: "
            f"{requested_base.resolve()} != {base_path.resolve()}"
        )
    model = model_config.create_pytorch().to(device)
    safetensors.torch.load_model(model.base, weight_path, strict=True, device=str(device))
    model.freeze_base()
    mapping = tuple(model_config.resolved_progress_layer_indices)
    base_checksum = model.base_checksum()
    dataset_checksum = args.dataset_checksum or getattr(model_config, "robomme_dataset_checksum", None)
    task_suite = args.task_suite or getattr(model_config, "robomme_task_suite", None)
    if not dataset_checksum or not task_suite:
        raise ValueError("registered config must define RoboMME dataset checksum and task suite")
    identity = {
        "base_checkpoint_checksum": base_checksum,
        "base_weights_checksum": base_checksum,
        "assets_checksum": str(model_config.base_assets_checksum),
        "dataset_checksum": str(dataset_checksum),
        "task_suite": str(task_suite),
        "query_stride": query_stride,
        "tokenizer_config_checksum": hashlib.sha256(b"transformers-4.53.2-pi05-replacement").hexdigest(),
        "preprocessing_checksum": hashlib.sha256(
            json.dumps(
                {"config": args.config, "query_stride": query_stride, "task_suite": task_suite},
                sort_keys=True,
            ).encode()
        ).hexdigest(),
        "layer_mapping": list(mapping),
        "dtype": args.dtype,
    }
    writer = ConditioningCacheWriter(args.output_dir, identity=identity, action_expert_layer_count=len(mapping))
    seen = 0
    selected_episodes = sorted({int(ref.epis_idx) for ref in windows.sample_refs})[: args.max_episodes]
    selected_episode_set = set(selected_episodes)
    try:
        for ref in windows.sample_refs:
            if int(ref.epis_idx) % args.num_shards != args.shard_id:
                continue
            if int(ref.epis_idx) not in selected_episode_set:
                continue
            sample = windows.sample_payload(ref)
            transformed_sample = transformed._transform(_dataset._mutable_copy(sample))
            observation_data = _episode_loader._copy_observation_fields(transformed_sample)
            observation_data["image_mask"] = {
                key: np.asarray(value, dtype=np.bool_) for key, value in observation_data["image_mask"].items()
            }
            observation = _model.Observation.from_dict(observation_data)
            observation = _episode_loader._stack_observations([observation])
            observation = _episode_loader._observation_to_torch(observation, device)
            with torch.no_grad():
                frozen = model.base.extract_prefix_context(observation, train=False)
                last_hidden = model.base.last_valid_prefix(frozen)
                prefix_view = PrefixKVView.from_cache(frozen.kv_cache, mapping, frozen.pad_mask)
            valid_length = int(prefix_view.valid_lengths[0].item())
            keys = torch.stack([layer[0][0, :, :valid_length] for layer in prefix_view.layers], dim=0)
            values = torch.stack([layer[1][0, :, :valid_length] for layer in prefix_view.layers], dim=0)
            prefix_mask = torch.ones(valid_length, dtype=torch.bool)
            writer.write_episode(
                ConditioningCacheEntry(
                    episode_id=f"e{ref.epis_idx}",
                    query_id=int(ref.step_idx),
                    last_valid_hidden=last_hidden[0].to(
                        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
                    ),
                    prefix_mask=prefix_mask,
                    action_expert_kv=(keys, values),
                )
            )
            seen += 1
            if seen % 100 == 0:
                print(json.dumps({"shard_id": args.shard_id, "cached": seen}), flush=True)
        manifest = writer.commit()
    except BaseException:
        writer.abort()
        raise
    print(json.dumps({"shard_id": args.shard_id, "entries": len(manifest["entries"]), "output_dir": str(args.output_dir)}))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute deterministic frozen FutureMamba prefix conditioning")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-episodes", required=True, type=int)
    parser.add_argument("--dtype", required=True, choices=("float32", "bfloat16"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--episode-data-dir", type=Path)
    parser.add_argument("--dataset-checksum")
    parser.add_argument("--task-suite")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")
    if args.num_shards <= 0 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("invalid shard selection")
    if args.config == "fake_tiny":
        if args.base_checkpoint != "fake://tiny-base":
            raise ValueError("fake_tiny requires --base-checkpoint=fake://tiny-base")
        _precompute_fake_tiny(args)
    else:
        _precompute_registered_config(args)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
