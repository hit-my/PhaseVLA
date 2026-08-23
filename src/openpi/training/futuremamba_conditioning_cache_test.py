from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

import pytest
import torch

from openpi.training import futuremamba_conditioning_cache as cache


def _identity(**overrides):
    identity = {
        "base_checkpoint_checksum": "base-sha256",
        "base_weights_checksum": "live-base-sha256",
        "assets_checksum": "assets-sha256",
        "dataset_checksum": "dataset-sha256",
        "task_suite": "Counting",
        "query_stride": 16,
        "tokenizer_config_checksum": "tokenizer-sha256",
        "preprocessing_checksum": "preprocess-sha256",
        "layer_mapping": [0, 1, 2],
        "dtype": "float32",
    }
    identity.update(overrides)
    return identity


def _kv(depth: int = 3, *, dtype: torch.dtype = torch.float32):
    keys = torch.arange(depth * 2 * 4 * 3, dtype=dtype).reshape(depth, 2, 4, 3)
    values = keys + 1000
    return keys, values


def _entry(*, episode_id: str = "episode-a", query_id: int = 7, dtype: torch.dtype = torch.float32):
    return cache.ConditioningCacheEntry(
        episode_id=episode_id,
        query_id=query_id,
        last_valid_hidden=torch.tensor([1.5, 2.5, 3.5], dtype=dtype),
        prefix_mask=torch.tensor([True, True, False, True]),
        action_expert_kv=_kv(dtype=dtype),
    )


def _writer(tmp_path: Path, **identity_overrides):
    return cache.ConditioningCacheWriter(
        tmp_path / "cache",
        identity=_identity(**identity_overrides),
        action_expert_layer_count=3,
    )


def test_validate_cache_mode_rejects_random_image_augmentation():
    with pytest.raises(ValueError, match="train_image_augmentation"):
        cache.validate_cache_mode(train_image_augmentation=True)


def test_boundary_loss_writer_requires_complete_prefix_kv(tmp_path: Path):
    writer = _writer(tmp_path)
    keys, values = _kv(depth=2)
    incomplete = dataclasses.replace(_entry(), action_expert_kv=(keys, values))

    with pytest.raises(ValueError, match="complete prefix KV"):
        writer.write_episode(incomplete)


def test_write_then_read_preserves_shape_dtype_value_and_checksum(tmp_path: Path):
    writer = _writer(tmp_path)
    entry = _entry()
    writer.write_episode(entry)
    manifest = writer.commit()

    loaded = cache.ConditioningCacheReader(tmp_path / "cache", expected_identity=_identity()).read_episode("episode-a")

    assert manifest["identity"]["base_weights_checksum"] == "live-base-sha256"
    assert loaded.episode_id == entry.episode_id
    assert loaded.query_id == entry.query_id
    assert loaded.last_valid_hidden.shape == (3,)
    assert loaded.last_valid_hidden.dtype == torch.float32
    assert loaded.prefix_mask.shape == (4,)
    assert loaded.prefix_mask.dtype is torch.bool
    assert loaded.action_expert_kv[0].shape == (3, 2, 4, 3)
    assert loaded.action_expert_kv[0].dtype == torch.float32
    torch.testing.assert_close(loaded.last_valid_hidden, entry.last_valid_hidden)
    torch.testing.assert_close(loaded.prefix_mask, entry.prefix_mask)
    torch.testing.assert_close(loaded.action_expert_kv[0], entry.action_expert_kv[0])
    torch.testing.assert_close(loaded.action_expert_kv[1], entry.action_expert_kv[1])
    assert loaded.content_checksum == cache.entry_checksum(entry)


def test_base_checkpoint_checksum_mismatch_is_rejected(tmp_path: Path):
    writer = _writer(tmp_path)
    writer.write_episode(_entry())
    writer.commit()

    with pytest.raises(ValueError, match="base_checkpoint_checksum"):
        cache.ConditioningCacheReader(
            tmp_path / "cache",
            expected_identity=_identity(base_checkpoint_checksum="different"),
        )


def test_dataset_and_query_identity_mismatches_are_rejected(tmp_path: Path):
    writer = _writer(tmp_path)
    writer.write_episode(_entry())
    writer.commit()

    for field, value in (("dataset_checksum", "different"), ("task_suite", "other"), ("query_stride", 8)):
        with pytest.raises(ValueError, match=field):
            cache.ConditioningCacheReader(
                tmp_path / "cache",
                expected_identity=_identity(**{field: value}),
            )

def test_manifest_identity_mismatch_is_rejected_by_field_name(tmp_path: Path):
    writer = _writer(tmp_path)
    writer.write_episode(_entry())
    writer.commit()

    with pytest.raises(ValueError, match="tokenizer_config_checksum"):
        cache.ConditioningCacheReader(
            tmp_path / "cache",
            expected_identity=_identity(tokenizer_config_checksum="different"),
        )


def test_missing_manifest_is_rejected(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    with pytest.raises(ValueError, match="manifest"):
        cache.ConditioningCacheReader(cache_dir, expected_identity=_identity())


def test_damaged_manifest_is_rejected(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text("not-json", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        cache.ConditioningCacheReader(cache_dir, expected_identity=_identity())


def test_missing_shard_is_rejected(tmp_path: Path):
    writer = _writer(tmp_path)
    writer.write_episode(_entry())
    writer.commit()
    (tmp_path / "cache" / "shards" / "episode-a__q7.pt").unlink()

    reader = cache.ConditioningCacheReader(tmp_path / "cache", expected_identity=_identity())
    with pytest.raises(ValueError, match="shard"):
        reader.read_episode("episode-a")


def test_damaged_shard_checksum_is_rejected(tmp_path: Path):
    writer = _writer(tmp_path)
    writer.write_episode(_entry())
    writer.commit()
    shard_path = tmp_path / "cache" / "shards" / "episode-a__q7.pt"
    shard_path.write_bytes(shard_path.read_bytes()[:-8] + b"corrupt!")

    reader = cache.ConditioningCacheReader(tmp_path / "cache", expected_identity=_identity())
    with pytest.raises(ValueError, match="checksum|shard"):
        reader.read_episode("episode-a")


def test_atomic_writer_does_not_publish_manifest_before_commit(tmp_path: Path):
    writer = _writer(tmp_path)
    writer.write_episode(_entry())

    assert not (tmp_path / "cache" / "manifest.json").exists()
    with pytest.raises(ValueError, match="manifest"):
        cache.ConditioningCacheReader(tmp_path / "cache", expected_identity=_identity())

    writer.commit()
    assert (tmp_path / "cache" / "manifest.json").exists()


def test_cli_fake_tiny_smoke_writes_readable_cache(tmp_path: Path):
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "precompute_futuremamba_prefix.py"
    spec = importlib.util.spec_from_file_location("precompute_futuremamba_prefix", script_path)
    assert spec is not None and spec.loader is not None
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    output_dir = tmp_path / "cli-cache"
    exit_code = cli.main(
        [
            "--config",
            "fake_tiny",
            "--base-checkpoint",
            "fake://tiny-base",
            "--output-dir",
            str(output_dir),
            "--max-episodes",
            "2",
            "--dtype",
            "float32",
        ]
    )

    assert exit_code == 0
    expected = cli.fake_tiny_expected_identity(dtype="float32", base_checkpoint="fake://tiny-base")
    reader = cache.ConditioningCacheReader(output_dir, expected_identity=expected)
    first = reader.read_episode("fake-episode-0")
    second = reader.read_episode("fake-episode-1")
    assert first.query_id == 0
    assert second.query_id == 1
    assert first.action_expert_kv[0].shape[0] == 3
    assert second.last_valid_hidden.dtype == torch.float32
