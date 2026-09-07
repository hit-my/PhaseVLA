from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import torch

import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class CachedQueryBatch:
    """Independent cached queries; only the sampled KV rows own CPU memory."""

    actions: torch.Tensor
    action_mask: torch.Tensor
    has_history: torch.Tensor
    prefix_mask: torch.Tensor
    action_expert_keys: torch.Tensor
    action_expert_values: torch.Tensor

    def to(self, device: torch.device | str) -> CachedQueryBatch:
        # In particular, do not promote the frozen BF16 cache to float32.
        return dataclasses.replace(
            self,
            **{field.name: getattr(self, field.name).to(device=device) for field in dataclasses.fields(self)},
        )

    def slice(self, start: int, stop: int) -> CachedQueryBatch:
        return dataclasses.replace(
            self,
            **{field.name: getattr(self, field.name)[start:stop] for field in dataclasses.fields(self)},
        )


@dataclasses.dataclass(frozen=True)
class _Episode:
    episode_index: int
    length: int
    queries: int
    data_path: Path
    cache_path: Path


@dataclasses.dataclass(frozen=True)
class _CacheSpec:
    layers: int
    heads: int
    prefix_tokens: int
    head_dim: int
    dtype: torch.dtype


class _ActionTransform:
    """Project the existing pipeline onto actions/state, without dummy images.

    LiberoInputs passes actions through and renames observation/state to state.
    All numerical transforms below are the original transform instances, in the
    original order. Unknown transforms (including action subsampling and FAST
    action tokenization) are rejected rather than guessing their dependencies.
    """

    def __init__(self, data_config: Any):
        from openpi.policies.libero_policy import LiberoInputs

        if data_config.norm_stats is None:
            raise ValueError("Cached query training requires the existing action normalization statistics")
        data_inputs = list(data_config.data_transforms.inputs)
        state_key = "state"
        if data_inputs and type(data_inputs[0]) is LiberoInputs:
            state_key = "observation/state"
            data_inputs.pop(0)

        sources = {"actions": "actions", "state": state_key}
        for transform in reversed(data_config.repack_transforms.inputs):
            if type(transform) is not _transforms.RepackTransform:
                raise ValueError(f"Unsupported cached-query repack transform: {type(transform).__qualname__}")
            structure = _transforms.flatten_dict(transform.structure)
            for key, source in sources.items():
                if source not in structure or not isinstance(structure[source], str):
                    raise ValueError(f"Cached-query repack must map {source!r} to a dataset column")
                sources[key] = structure[source]
        self.action_column = sources["actions"]
        self.state_column = sources["state"]
        if tuple(data_config.action_sequence_keys) != (self.action_column,):
            raise ValueError(
                "Cached-query targets require exactly the action column in action_sequence_keys; "
                f"got {data_config.action_sequence_keys!r}, expected {(self.action_column,)!r}"
            )

        numerical = (
            _transforms.Normalize,
            _transforms.DeltaActions,
            _transforms.AbsoluteActions,
            _transforms.PadStatesAndActions,
        )
        observation_only = (
            _transforms.InjectDefaultPrompt,
            _transforms.ResizeImages,
            _transforms.TokenizePrompt,
        )
        transforms = []
        pipeline = (
            *data_inputs,
            _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        )
        for transform in pipeline:
            if type(transform) in observation_only:
                continue
            if type(transform) not in numerical:
                raise ValueError(f"Unsupported cached-query action transform: {type(transform).__qualname__}")
            if type(transform) is _transforms.Normalize:
                # Observation statistics have no bearing on targets. Retain state
                # statistics too: a subsequent DeltaActions may depend on them.
                stats = transform.norm_stats
                if stats is not None:
                    stats = {key: value for key, value in stats.items() if key in {"actions", "state"}}
                transform = dataclasses.replace(transform, norm_stats=stats)
            transforms.append(transform)
        self._transform = _transforms.compose(transforms)

    def __call__(self, actions: np.ndarray, state: np.ndarray) -> np.ndarray:
        transformed = self._transform({"actions": actions, "state": state})
        return np.asarray(transformed["actions"], dtype=np.float32)


class _MappedEpisodes:
    """Small LRU of file-backed tensors, never eager episode-sized KV copies."""

    def __init__(self, capacity: int = 8):
        self._capacity = capacity
        self._entries: OrderedDict[int, dict[str, torch.Tensor]] = OrderedDict()

    def get(self, episode: _Episode) -> dict[str, torch.Tensor]:
        key = episode.episode_index
        if key in self._entries:
            self._entries.move_to_end(key)
            return self._entries[key]
        # Evict first so opening a new episode never exceeds the mapping bound.
        if len(self._entries) >= self._capacity:
            self._entries.popitem(last=False)
        payload = torch.load(episode.cache_path, map_location="cpu", weights_only=True, mmap=True)
        required = {"last_valid_hidden", "prefix_mask", "action_expert_keys", "action_expert_values"}
        if not isinstance(payload, dict) or set(payload) != required:
            raise ValueError(f"Invalid conditioning cache keys: {episode.cache_path}")
        for name, value in payload.items():
            if not torch.is_tensor(value) or value.device.type != "cpu" or value.layout != torch.strided:
                raise ValueError(f"Invalid cached tensor {name!r}: {episode.cache_path}")
            if value.ndim == 0 or value.shape[0] != episode.queries:
                raise ValueError(f"Conditioning cache query count mismatch: {episode.cache_path}")
        # The no-memory PE never consumes VLM hidden states.
        del payload["last_valid_hidden"]
        self._entries[key] = payload
        return payload


def _cache_spec(payload: dict[str, torch.Tensor], episode: _Episode) -> _CacheSpec:
    keys = payload["action_expert_keys"]
    values = payload["action_expert_values"]
    mask = payload["prefix_mask"]
    if keys.ndim != 5 or keys.shape != values.shape or keys.dtype != values.dtype:
        raise ValueError(f"Invalid cached KV layout: {episode.cache_path}")
    _, layers, heads, prefix_tokens, head_dim = keys.shape
    if min(layers, heads, prefix_tokens, head_dim) <= 0 or not keys.is_floating_point():
        raise ValueError(f"Invalid cached KV dimensions or dtype: {episode.cache_path}")
    if mask.dtype != torch.bool or mask.shape != (episode.queries, prefix_tokens):
        raise ValueError(f"Invalid cached prefix mask: {episode.cache_path}")
    return _CacheSpec(layers, heads, prefix_tokens, head_dim, keys.dtype)


def _episodes(metadata: Any, data_config: Any, query_stride: int) -> tuple[_Episode, ...]:
    import pyarrow.parquet as parquet

    records = []
    root = Path(metadata.root)
    cache_root = Path(data_config.conditioning_cache_dir)
    for position, (episode_id, episode) in enumerate(sorted(metadata.episodes.items())):
        if int(episode_id) != position:
            # The existing loader names cache files by the episode-data-index
            # position, so noncontiguous ids would make cache identity ambiguous.
            raise ValueError("Cached queries require contiguous LeRobot episode indices")
        data_path = root / metadata.get_data_file_path(episode_id)
        task_name = getattr(data_config, "task_name", None)
        if task_name is not None:
            tasks = episode.get("tasks", ())
            if len(tasks) == 1:
                episode_task = tasks[0]
            else:
                # Match the old loader's first-frame task for mixed-task episodes.
                with parquet.ParquetFile(data_path, memory_map=True) as source:
                    first_row = next(source.iter_batches(batch_size=1, columns=["task_index"], use_threads=False))
                    task_index = int(first_row.column(0)[0].as_py())
                episode_task = metadata.tasks.get(task_index)
            if episode_task != task_name:
                continue
        length = int(episode["length"])
        if length <= 0:
            raise ValueError(f"Episode {episode_id} has no valid queries")
        cache_path = cache_root / f"episode_{episode_id:06d}.pt"
        if not data_path.is_file():
            raise FileNotFoundError(f"LeRobot action data missing: {data_path}")
        if not cache_path.is_file():
            raise FileNotFoundError(f"Conditioning cache missing: {cache_path}")
        records.append(
            _Episode(int(episode_id), length, (length + query_stride - 1) // query_stride, data_path, cache_path)
        )
    if not records:
        raise ValueError(f"No cached episodes match task_name={getattr(data_config, 'task_name', None)!r}")
    return tuple(records)


class _CachedQueryData:
    query_sampling_protocol = "independent_query"

    def __init__(self, train_config: Any, data_config: Any, *, queries_per_update: int | None):
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

        episode_config = train_config.episode_data
        self._query_stride = int(episode_config.query_stride)
        self._action_horizon = int(train_config.model.action_horizon)
        self._action_dim = int(train_config.model.action_dim)
        executed_horizon = episode_config.executed_horizon
        self._executed_horizon = self._query_stride if executed_horizon is None else int(executed_horizon)
        if min(self._query_stride, self._action_horizon, self._action_dim, self._executed_horizon) <= 0:
            raise ValueError("Cached-query action dimensions and horizons must be positive")
        if self._executed_horizon > self._action_horizon:
            raise ValueError("executed_horizon must be <= action_horizon")
        self._seed = int(train_config.seed)
        if self._seed < 0:
            raise ValueError("Cached-query seed must be nonnegative")
        self._action_transform = _ActionTransform(data_config)
        metadata = lerobot_dataset.LeRobotDatasetMetadata(
            data_config.repo_id, root=getattr(data_config, "dataset_root", None)
        )
        self._column_widths = {}
        for column in (self._action_transform.action_column, self._action_transform.state_column):
            feature = metadata.features.get(column)
            if feature is None or feature["dtype"] != "float32" or len(feature["shape"]) != 1:
                raise ValueError(f"Cached-query action/state column {column!r} must be a float32 vector")
            self._column_widths[column] = int(feature["shape"][0])
        self._episodes = _episodes(metadata, data_config, self._query_stride)
        self.episode_count = len(self._episodes)
        self._query_counts = np.asarray([episode.queries for episode in self._episodes], dtype=np.int64)
        self.mean_episode_queries = float(self._query_counts.mean())
        if queries_per_update is None:
            queries_per_update = round(self.mean_episode_queries)
        if isinstance(queries_per_update, bool) or not isinstance(queries_per_update, (int, np.integer)):
            raise ValueError("queries_per_update must be a positive integer")
        self.queries_per_update = int(queries_per_update)
        if self.queries_per_update <= 0:
            raise ValueError("queries_per_update must be a positive integer")
        self._mapped = _MappedEpisodes()
        # Loading ZIP metadata with mmap does not fault the KV tensor pages into
        # RAM. Only these small descriptors are retained for all episodes.
        self._specs = tuple(_cache_spec(self._mapped.get(episode), episode) for episode in self._episodes)
        reference = self._specs[0]
        for spec in self._specs:
            if (spec.layers, spec.heads, spec.head_dim, spec.dtype) != (
                reference.layers,
                reference.heads,
                reference.head_dim,
                reference.dtype,
            ):
                raise ValueError("Conditioning caches disagree on KV layers, heads, width, or dtype")
        # These are tiny projected numerical columns, not images or cached KV.
        # Keep them after their first use rather than repeatedly decompressing
        # Parquet files when uniform sampling revisits an episode.
        self._action_columns: dict[int, dict[str, np.ndarray]] = {}

    def _columns(self, episode: _Episode) -> dict[str, np.ndarray]:
        import pyarrow as pa
        import pyarrow.parquet as parquet

        if episode.episode_index not in self._action_columns:
            with parquet.ParquetFile(episode.data_path, memory_map=True) as source:
                table = source.read(columns=list(self._column_widths), use_threads=False)
            if table.num_rows != episode.length:
                raise ValueError(f"LeRobot episode length mismatch: {episode.data_path}")
            arrays = {}
            for name, width in self._column_widths.items():
                column = table[name].combine_chunks()
                if not pa.types.is_fixed_size_list(column.type) or column.type.list_size != width:
                    raise ValueError(f"Unsupported action/state Arrow column {name!r}: {episode.data_path}")
                values = column.flatten()
                if column.null_count or values.null_count or values.type != pa.float32():
                    raise ValueError(f"Invalid action/state values in {name!r}: {episode.data_path}")
                arrays[name] = values.to_numpy(zero_copy_only=False).reshape(episode.length, width)
            self._action_columns[episode.episode_index] = arrays
        return self._action_columns[episode.episode_index]

    def __iter__(self) -> Iterator[CachedQueryBatch]:
        return self.iter_from_update(0)

    def iter_from_update(self, start_update: int) -> Iterator[CachedQueryBatch]:
        """Replay a resumed update directly, without reading discarded batches.

        Each update has its own PCG64 stream. Sampling never reads or mutates
        NumPy's global generator or the torch RNG used for flow noise/time.
        """
        if isinstance(start_update, bool) or not isinstance(start_update, (int, np.integer)) or start_update < 0:
            raise ValueError("start_update must be a nonnegative integer")
        update = int(start_update)
        while True:
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self._seed, update])))
            episode_indices = rng.integers(self.episode_count, size=self.queries_per_update)
            query_indices = rng.integers(self._query_counts[episode_indices])
            yield self._batch(episode_indices, query_indices)
            update += 1

    def _batch(self, episode_indices: np.ndarray, query_indices: np.ndarray) -> CachedQueryBatch:
        batch_size = len(episode_indices)
        unique_episodes = np.unique(episode_indices)
        prefix_tokens = max(self._specs[int(index)].prefix_tokens for index in unique_episodes)
        spec = self._specs[0]
        actions = torch.empty((batch_size, self._action_horizon, self._action_dim), dtype=torch.float32)
        action_mask = torch.empty((batch_size, self._action_horizon), dtype=torch.bool)
        has_history = torch.empty(batch_size, dtype=torch.bool)
        prefix_mask = torch.empty((batch_size, prefix_tokens), dtype=torch.bool)
        shape = (batch_size, spec.layers, spec.heads, prefix_tokens, spec.head_dim)
        keys = torch.empty(shape, dtype=spec.dtype)
        values = torch.empty(shape, dtype=spec.dtype)
        offsets = np.arange(self._action_horizon)
        executed_prefix = min(self._executed_horizon, self._query_stride, self._action_horizon)
        for episode_index in unique_episodes:
            episode = self._episodes[int(episode_index)]
            payload = self._mapped.get(episode)
            columns = self._columns(episode)
            raw_actions = columns[self._action_transform.action_column]
            raw_states = columns[self._action_transform.state_column]
            length = self._specs[int(episode_index)].prefix_tokens
            for batch_index in np.flatnonzero(episode_indices == episode_index):
                row = int(batch_index)
                query_index = int(query_indices[row])
                frame = query_index * self._query_stride
                # LeRobot clamps delta indices to the last frame before applying
                # transforms; EpisodeDataset subsequently zeros masked timesteps.
                indices = np.minimum(frame + offsets, episode.length - 1)
                transformed = self._action_transform(raw_actions[indices], raw_states[frame].copy())
                expected = (self._action_horizon, self._action_dim)
                if transformed.shape != expected:
                    raise ValueError(f"Action transforms produced {transformed.shape}, expected {expected}")
                valid = offsets < episode.length - frame
                transformed[~valid] = 0
                actions[row].copy_(torch.from_numpy(transformed))
                action_mask[row].copy_(torch.from_numpy(valid))
                previous_frame = (query_index - 1) * self._query_stride
                has_history[row] = query_index > 0 and min(executed_prefix, episode.length - previous_frame) > 0
                # Index the mmap first, then copy only this query. No NumPy
                # conversion of BF16 tensors and no episode-sized intermediate.
                prefix_mask[row, :length].copy_(payload["prefix_mask"][query_index])
                keys[row, :, :, :length, :].copy_(payload["action_expert_keys"][query_index])
                values[row, :, :, :length, :].copy_(payload["action_expert_values"][query_index])
                if length < prefix_tokens:
                    prefix_mask[row, length:] = False
                    keys[row, :, :, length:, :].zero_()
                    values[row, :, :, length:, :].zero_()
        return CachedQueryBatch(actions, action_mask, has_history, prefix_mask, keys, values)


def create_cached_query_data(train_config: Any, *, queries_per_update: int | None) -> _CachedQueryData:
    """Build the no-memory, episode-uniform then query-uniform cached stream."""
    if getattr(train_config.model, "memory_backend", None) != "none":
        raise ValueError("Cached independent-query training requires memory_backend='none'")
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if not data_config.repo_id or data_config.repo_id == "fake":
        raise ValueError("Cached-query training requires a real LeRobot dataset")
    if not getattr(data_config, "conditioning_cache_dir", None):
        raise ValueError("Cached-query training requires conditioning_cache_dir")
    if getattr(data_config, "rlds_data_dir", None) or getattr(data_config, "episode_data_dir", None):
        raise ValueError("Cached-query training supports LeRobot per-episode Parquet data only")
    return _CachedQueryData(train_config, data_config, queries_per_update=queries_per_update)
