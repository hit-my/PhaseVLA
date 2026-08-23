from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
from typing import Any
import uuid

import torch


_SCHEMA_VERSION = 1
_REQUIRED_IDENTITY_FIELDS = (
    "base_checkpoint_checksum",
    "base_weights_checksum",
    "assets_checksum",
    "dataset_checksum",
    "task_suite",
    "query_stride",
    "tokenizer_config_checksum",
    "preprocessing_checksum",
    "layer_mapping",
    "dtype",
)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclasses.dataclass(frozen=True)
class ConditioningCacheEntry:
    episode_id: str
    query_id: int
    last_valid_hidden: torch.Tensor
    prefix_mask: torch.BoolTensor
    action_expert_kv: tuple[torch.Tensor, torch.Tensor]
    content_checksum: str = ""


def validate_cache_mode(*, train_image_augmentation: bool) -> None:
    if train_image_augmentation:
        raise ValueError("train_image_augmentation must be false for deterministic preprocessing cache")


def entry_checksum(entry: ConditioningCacheEntry) -> str:
    digest = hashlib.sha256()
    digest.update(b"futuremamba-conditioning-entry-v1\0")
    digest.update(entry.episode_id.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(entry.query_id).encode("ascii"))
    for name, tensor in (
        ("last_valid_hidden", entry.last_valid_hidden),
        ("prefix_mask", entry.prefix_mask),
        ("action_expert_keys", entry.action_expert_kv[0]),
        ("action_expert_values", entry.action_expert_kv[1]),
    ):
        value = tensor.detach().cpu().contiguous()
        digest.update(b"\0" + name.encode("ascii") + b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class ConditioningCacheWriter:
    def __init__(
        self,
        root: str | Path,
        *,
        identity: Mapping[str, Any],
        action_expert_layer_count: int,
    ) -> None:
        self.root = Path(root)
        self.identity = _validate_identity(identity)
        if action_expert_layer_count <= 0:
            raise ValueError("action_expert_layer_count must be positive")
        self.action_expert_layer_count = int(action_expert_layer_count)
        if self.root.exists():
            raise ValueError(f"cache output already exists: {self.root}")
        self._staging = self.root.parent / f".{self.root.name}.tmp-{uuid.uuid4().hex}"
        self._shards = self._staging / "shards"
        self._shards.mkdir(parents=True, exist_ok=False)
        self._entries: list[dict[str, Any]] = []
        self._entry_keys: set[tuple[str, int]] = set()
        self._committed = False

    def write_episode(self, entry: ConditioningCacheEntry) -> None:
        if self._committed:
            raise RuntimeError("conditioning cache is already committed")
        _validate_entry(entry, action_expert_layer_count=self.action_expert_layer_count)
        key = (entry.episode_id, int(entry.query_id))
        if key in self._entry_keys:
            raise ValueError(f"duplicate cache entry {entry.episode_id!r} query {entry.query_id}")
        if not _SAFE_ID.fullmatch(entry.episode_id):
            raise ValueError("episode_id may contain only letters, digits, dot, underscore, and hyphen")

        materialized = _entry_to_cpu(entry)
        content_checksum = entry_checksum(materialized)
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "episode_id": materialized.episode_id,
            "query_id": materialized.query_id,
            "last_valid_hidden": materialized.last_valid_hidden,
            "prefix_mask": materialized.prefix_mask,
            "action_expert_keys": materialized.action_expert_kv[0],
            "action_expert_values": materialized.action_expert_kv[1],
            "content_checksum": content_checksum,
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        shard_bytes = buffer.getvalue()
        relative_path = Path("shards") / f"{entry.episode_id}__q{entry.query_id}.pt"
        shard_path = self._staging / relative_path
        _atomic_write_bytes(shard_path, shard_bytes)
        self._entries.append(
            {
                "episode_id": entry.episode_id,
                "query_id": int(entry.query_id),
                "path": relative_path.as_posix(),
                "shard_checksum": hashlib.sha256(shard_bytes).hexdigest(),
                "content_checksum": content_checksum,
            }
        )
        self._entry_keys.add(key)

    def commit(self) -> dict[str, Any]:
        if self._committed:
            raise RuntimeError("conditioning cache is already committed")
        if not self._entries:
            raise ValueError("cannot commit an empty conditioning cache")
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "identity": self.identity,
            "action_expert_layer_count": self.action_expert_layer_count,
            "entries": sorted(self._entries, key=lambda row: (row["episode_id"], row["query_id"])),
        }
        manifest_bytes = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        _atomic_write_bytes(self._staging / "manifest.json", manifest_bytes)
        try:
            os.replace(self._staging, self.root)
        except OSError:
            if self.root.exists():
                raise ValueError(f"cache output already exists: {self.root}") from None
            raise
        self._committed = True
        return manifest

    def abort(self) -> None:
        if not self._committed:
            shutil.rmtree(self._staging, ignore_errors=True)


class ConditioningCacheReader:
    def __init__(self, root: str | Path, *, expected_identity: Mapping[str, Any]) -> None:
        self.root = Path(root)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"conditioning cache manifest is missing: {manifest_path}")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid conditioning cache manifest: {error}") from error
        if not isinstance(manifest, dict) or manifest.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError("invalid conditioning cache manifest schema_version")
        actual_identity = _validate_identity(manifest.get("identity"))
        expected = _validate_identity(expected_identity)
        for field in _REQUIRED_IDENTITY_FIELDS:
            if actual_identity[field] != expected[field]:
                raise ValueError(
                    f"conditioning cache identity mismatch for {field}: "
                    f"expected {expected[field]!r}, got {actual_identity[field]!r}"
                )
        layer_count = manifest.get("action_expert_layer_count")
        if not isinstance(layer_count, int) or layer_count <= 0:
            raise ValueError("invalid conditioning cache manifest action_expert_layer_count")
        rows = manifest.get("entries")
        if not isinstance(rows, list) or not rows:
            raise ValueError("invalid conditioning cache manifest entries")
        self.manifest = manifest
        self.action_expert_layer_count = layer_count
        self._entries: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            if not isinstance(row, dict) or not all(
                field in row for field in ("episode_id", "query_id", "path", "shard_checksum", "content_checksum")
            ):
                raise ValueError("invalid conditioning cache manifest shard entry")
            self._entries.setdefault(str(row["episode_id"]), []).append(row)

    def read_episode(self, episode_id: str, query_id: int | None = None) -> ConditioningCacheEntry:
        rows = self._entries.get(episode_id, [])
        if query_id is not None:
            rows = [row for row in rows if row["query_id"] == query_id]
        if len(rows) != 1:
            detail = "missing" if not rows else "ambiguous"
            raise ValueError(f"conditioning cache shard is {detail} for episode {episode_id!r}")
        row = rows[0]
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid conditioning cache shard path")
        shard_path = self.root / relative
        if not shard_path.is_file():
            raise ValueError(f"conditioning cache shard is missing: {shard_path}")
        try:
            shard_bytes = shard_path.read_bytes()
        except OSError as error:
            raise ValueError(f"cannot read conditioning cache shard: {error}") from error
        actual_shard_checksum = hashlib.sha256(shard_bytes).hexdigest()
        if actual_shard_checksum != row["shard_checksum"]:
            raise ValueError(f"conditioning cache shard checksum mismatch: {shard_path}")
        try:
            payload = torch.load(io.BytesIO(shard_bytes), map_location="cpu", weights_only=True)
            entry = ConditioningCacheEntry(
                episode_id=str(payload["episode_id"]),
                query_id=int(payload["query_id"]),
                last_valid_hidden=payload["last_valid_hidden"],
                prefix_mask=payload["prefix_mask"],
                action_expert_kv=(payload["action_expert_keys"], payload["action_expert_values"]),
                content_checksum=str(payload["content_checksum"]),
            )
            _validate_entry(entry, action_expert_layer_count=self.action_expert_layer_count)
        except (KeyError, TypeError, ValueError, RuntimeError, EOFError) as error:
            raise ValueError(f"invalid conditioning cache shard {shard_path}: {error}") from error
        if entry.episode_id != row["episode_id"] or entry.query_id != row["query_id"]:
            raise ValueError(f"conditioning cache shard identity mismatch: {shard_path}")
        actual_content_checksum = entry_checksum(entry)
        if actual_content_checksum != row["content_checksum"] or actual_content_checksum != entry.content_checksum:
            raise ValueError(f"conditioning cache shard content checksum mismatch: {shard_path}")
        return entry


def _validate_identity(identity: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError("conditioning cache identity must be a mapping")
    missing = [field for field in _REQUIRED_IDENTITY_FIELDS if field not in identity]
    if missing:
        raise ValueError(f"conditioning cache identity missing field {missing[0]}")
    normalized = dict(identity)
    layer_mapping = normalized["layer_mapping"]
    if not isinstance(layer_mapping, (list, tuple)) or not layer_mapping or not all(
        isinstance(layer, int) for layer in layer_mapping
    ):
        raise ValueError("conditioning cache identity layer_mapping must be non-empty integers")
    normalized["layer_mapping"] = list(layer_mapping)
    for field in _REQUIRED_IDENTITY_FIELDS:
        if field == "layer_mapping":
            continue
        if field == "query_stride":
            if not isinstance(normalized[field], int) or normalized[field] <= 0:
                raise ValueError("conditioning cache identity query_stride must be a positive integer")
            continue
        if not isinstance(normalized[field], str) or not normalized[field]:
            raise ValueError(f"conditioning cache identity field {field} must be a non-empty string")
    return normalized


def _validate_entry(entry: ConditioningCacheEntry, *, action_expert_layer_count: int) -> None:
    if not isinstance(entry.episode_id, str) or not entry.episode_id:
        raise ValueError("episode_id must be a non-empty string")
    if not isinstance(entry.query_id, int) or entry.query_id < 0:
        raise ValueError("query_id must be a non-negative integer")
    if not torch.is_tensor(entry.last_valid_hidden) or entry.last_valid_hidden.ndim < 1:
        raise ValueError("last_valid_hidden must be a tensor with at least one dimension")
    if not torch.is_tensor(entry.prefix_mask) or entry.prefix_mask.dtype is not torch.bool or entry.prefix_mask.ndim != 1:
        raise ValueError("prefix_mask must be a one-dimensional bool tensor")
    if not isinstance(entry.action_expert_kv, tuple) or len(entry.action_expert_kv) != 2:
        raise ValueError("action_expert_kv must contain key and value tensors")
    keys, values = entry.action_expert_kv
    if not torch.is_tensor(keys) or not torch.is_tensor(values) or keys.shape != values.shape:
        raise ValueError("action_expert_kv key/value tensors must have identical shapes")
    if keys.ndim < 2 or keys.shape[0] != action_expert_layer_count:
        raise ValueError(
            f"complete prefix KV requires {action_expert_layer_count} Action Expert layers, got "
            f"{keys.shape[0] if keys.ndim else 0}"
        )
    if keys.dtype != values.dtype:
        raise ValueError("action_expert_kv key/value tensors must have identical dtype")


def _entry_to_cpu(entry: ConditioningCacheEntry) -> ConditioningCacheEntry:
    return ConditioningCacheEntry(
        episode_id=entry.episode_id,
        query_id=int(entry.query_id),
        last_valid_hidden=entry.last_valid_hidden.detach().cpu().contiguous().clone(),
        prefix_mask=entry.prefix_mask.detach().cpu().to(dtype=torch.bool).contiguous().clone(),
        action_expert_kv=(
            entry.action_expert_kv[0].detach().cpu().contiguous().clone(),
            entry.action_expert_kv[1].detach().cpu().contiguous().clone(),
        ),
    )


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
