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
    # Production precomputation deliberately goes through the registered config and
    # converted PyTorch checkpoint.  The repository's Stage-B training registration
    # is created by the next migration task; refusing here is safer than publishing
    # a cache with guessed transforms or layer identities.
    try:
        from openpi.training import config as _config
    except ImportError as error:
        raise RuntimeError(f"cannot import OpenPI training config: {error}") from error
    try:
        train_config = _config.get_config(args.config)
    except Exception as error:
        raise RuntimeError(f"unknown or unavailable config {args.config!r}: {error}") from error
    raise RuntimeError(
        "real prefix precomputation requires the registered FutureMamba PyTorch episode-loader factory "
        f"for {getattr(train_config, 'name', args.config)!r}; use the completed Stage-B training registration"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute deterministic frozen FutureMamba prefix conditioning")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-episodes", required=True, type=int)
    parser.add_argument("--dtype", required=True, choices=("float32", "bfloat16"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")
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
