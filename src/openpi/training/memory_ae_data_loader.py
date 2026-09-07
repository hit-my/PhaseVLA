from __future__ import annotations

from collections.abc import Iterator
import dataclasses
from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch

import openpi.models.model as _model
from openpi.training.cached_query_data_loader import _ActionTransform
import openpi.training.episode_data_loader as _episodes


@dataclasses.dataclass(frozen=True)
class MemoryAEQueryBatch:
    """Online observations and strictly earlier, right-padded executed actions.

    Slice the CPU batch before moving a microbatch to its device. Diagnostic
    indices stay on CPU, including after ``to``.
    """

    observation: _model.Observation
    actions: torch.Tensor | np.ndarray
    action_mask: torch.Tensor | np.ndarray
    history_actions: torch.Tensor | np.ndarray
    history_mask: torch.Tensor | np.ndarray
    episode_index: np.ndarray | None = None
    query_index: np.ndarray | None = None
    local_frame_index: np.ndarray | None = None
    episode_length: np.ndarray | None = None
    episode_position: np.ndarray | None = None
    frame_index: np.ndarray | None = None
    update_index: int | None = None

    def slice(self, start: int, stop: int) -> MemoryAEQueryBatch:
        values = {}
        for field in dataclasses.fields(self):
            if field.name in {"observation", "update_index"}:
                continue
            value = getattr(self, field.name)
            values[field.name] = None if value is None else value[start:stop]
        return dataclasses.replace(
            self,
            observation=jax.tree.map(lambda value: value[start:stop], self.observation),
            **values,
        )

    def to(self, device: torch.device | str) -> MemoryAEQueryBatch:
        device = torch.device(device)
        return dataclasses.replace(
            self,
            observation=_episodes._observation_to_torch(self.observation, device),
            actions=_episodes._tensor_to_torch(self.actions, device, dtype=torch.float32),
            action_mask=_episodes._tensor_to_torch(self.action_mask, device, dtype=torch.bool),
            history_actions=_episodes._tensor_to_torch(self.history_actions, device, dtype=torch.float32),
            history_mask=_episodes._tensor_to_torch(self.history_mask, device, dtype=torch.bool),
        )


class _TaskIndexedFrames:
    """Supply numeric task metadata without decoding episode-start images."""

    def __init__(self, dataset: Any, task_indices: np.ndarray):
        self._dataset = dataset
        self.episode_data_index = dataset.episode_data_index
        self.episode_task_index = task_indices
        self.meta = dataset.meta

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self._dataset[int(index)]


def _numeric_episode_metadata(metadata: Any, dataset: Any) -> tuple[dict[int, Path], np.ndarray]:
    """Check actual lengths and read only the first numeric task id."""
    import pyarrow.parquet as parquet

    starts = _episodes._to_int_array(dataset.episode_data_index["from"])
    ends = _episodes._to_int_array(dataset.episode_data_index["to"])
    records = sorted(metadata.episodes.items())
    if starts.ndim != 1 or starts.shape != ends.shape or len(records) != len(starts):
        raise ValueError("Memory-AE requires matching LeRobot episode metadata and frame indices")
    paths = {}
    tasks = np.empty(len(records), dtype=np.int64)
    for position, (episode_id, record) in enumerate(records):
        if int(episode_id) != position:
            raise ValueError("Memory-AE requires contiguous LeRobot episode indices")
        length = int(ends[position] - starts[position])
        if length <= 0 or length != int(record["length"]):
            raise ValueError(f"LeRobot episode length mismatch for episode {episode_id}")
        path = Path(metadata.root) / metadata.get_data_file_path(episode_id)
        with parquet.ParquetFile(path, memory_map=True) as source:
            if source.metadata.num_rows != length:
                raise ValueError(f"LeRobot per-episode Parquet length mismatch: {path}")
            first = next(source.iter_batches(batch_size=1, columns=["task_index"], use_threads=False))
            task_index = first.column(0)[0].as_py()
        if isinstance(task_index, bool) or not isinstance(task_index, int):
            raise ValueError(f"Invalid first-frame task_index in {path}")
        tasks[position] = task_index
        paths[position] = path
    return paths, tasks


class _MemoryAEData:
    query_sampling_protocol = "independent_query"

    def __init__(self, train_config: Any, data_config: Any, *, queries_per_update: int | None):
        import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

        episode_config = train_config.episode_data
        query_stride = int(episode_config.query_stride)
        executed_horizon = episode_config.executed_horizon
        executed_horizon = query_stride if executed_horizon is None else int(executed_horizon)
        if query_stride != 1 or executed_horizon != 1:
            raise ValueError("Memory-AE histories require query_stride=1 and executed_horizon=1")
        self._action_horizon = int(train_config.model.action_horizon)
        self._action_dim = int(train_config.model.action_dim)
        if min(self._action_horizon, self._action_dim) <= 0:
            raise ValueError("Memory-AE action dimensions and horizons must be positive")
        self._seed = int(train_config.seed)
        if self._seed < 0:
            raise ValueError("Memory-AE seed must be nonnegative")
        self._action_transform = _ActionTransform(data_config)
        root_kwargs = {}
        if getattr(data_config, "dataset_root", None) is not None:
            root_kwargs["root"] = data_config.dataset_root
        metadata = lerobot_dataset.LeRobotDatasetMetadata(data_config.repo_id, **root_kwargs)
        self._column_widths = {}
        for column in (self._action_transform.action_column, self._action_transform.state_column):
            feature = metadata.features.get(column)
            if feature is None or feature["dtype"] != "float32" or len(feature["shape"]) != 1:
                raise ValueError(f"Memory-AE action/state column {column!r} must be a float32 vector")
            self._column_widths[column] = int(feature["shape"][0])
        frames = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            delta_timestamps={
                key: [step / metadata.fps for step in range(self._action_horizon)]
                for key in data_config.action_sequence_keys
            },
            **root_kwargs,
        )
        self._data_paths, task_indices = _numeric_episode_metadata(metadata, frames)
        self._episode_dataset = _episodes.LeRobotEpisodeDataset(
            dataset=_TaskIndexedFrames(frames, task_indices),
            action_horizon=self._action_horizon,
            query_stride=query_stride,
            executed_horizon=executed_horizon,
            transforms=_episodes.make_transform_pipeline(data_config),
            action_sequence_keys=data_config.action_sequence_keys,
            conditioning_cache_dir=None,
            prompt_from_task=data_config.prompt_from_task,
            task_name=getattr(data_config, "task_name", None),
        )
        self._episodes = self._episode_dataset.episodes
        self.episode_count = len(self._episodes)
        if not self.episode_count:
            raise ValueError("Memory-AE requires at least one nonempty episode")
        self._episode_lengths = np.asarray(
            [record.end_frame - record.start_frame for record in self._episodes], dtype=np.int64
        )
        self._query_counts = np.asarray([len(record.queries) for record in self._episodes], dtype=np.int64)
        self.mean_episode_queries = float(self._query_counts.mean())
        if queries_per_update is None:
            queries_per_update = round(self.mean_episode_queries)
        if isinstance(queries_per_update, bool) or not isinstance(queries_per_update, (int, np.integer)):
            raise ValueError("queries_per_update must be a positive integer")
        self.queries_per_update = int(queries_per_update)
        if self.queries_per_update <= 0:
            raise ValueError("queries_per_update must be a positive integer")
        # Retain only small numeric columns, never images, observations, or KV.
        self._action_columns: dict[int, dict[str, np.ndarray]] = {}
        self._selected_frame_reads = 0

    def _columns(self, record: _episodes.EpisodeRecord) -> dict[str, np.ndarray]:
        import pyarrow as pa
        import pyarrow.parquet as parquet

        if record.episode_index not in self._action_columns:
            path = self._data_paths[record.episode_index]
            length = record.end_frame - record.start_frame
            with parquet.ParquetFile(path, memory_map=True) as source:
                table = source.read(columns=list(self._column_widths), use_threads=False)
            if table.num_rows != length:
                raise ValueError(f"LeRobot episode length mismatch: {path}")
            arrays = {}
            for name, width in self._column_widths.items():
                column = table[name].combine_chunks()
                if not pa.types.is_fixed_size_list(column.type) or column.type.list_size != width:
                    raise ValueError(f"Unsupported action/state Arrow column {name!r}: {path}")
                values = column.flatten()
                if column.null_count or values.null_count or values.type != pa.float32():
                    raise ValueError(f"Invalid action/state values in {name!r}: {path}")
                array = values.to_numpy(zero_copy_only=False).reshape(length, width)
                array.setflags(write=False)
                arrays[name] = array
            self._action_columns[record.episode_index] = arrays
        return self._action_columns[record.episode_index]

    def _history(self, episode_position: int, local_frame_index: int) -> np.ndarray:
        """Return [0, f), each executed action transformed with its own state.

        At stride=execution=1, these match the valid flattened prefixes from
        ``episode_at(position).executed_actions[:f + 1]``. A singleton horizon
        preserves DeltaActions/AbsoluteActions broadcasting. The supported
        numeric transforms are timestep-local; no future chunk is needed.
        """
        record = self._episodes[int(episode_position)]
        frame = int(local_frame_index)
        if not 0 <= frame < record.end_frame - record.start_frame:
            raise ValueError(f"History frame {frame} is outside episode {record.episode_index}")
        if frame == 0:
            return np.zeros((0, self._action_dim), dtype=np.float32)
        columns = self._columns(record)
        actions = columns[self._action_transform.action_column][:frame, None, :].copy()
        states = columns[self._action_transform.state_column][:frame].copy()
        transformed = self._action_transform(actions, states)
        expected = (frame, 1, self._action_dim)
        if transformed.shape != expected:
            raise ValueError(f"History transforms produced {transformed.shape}, expected {expected}")
        return transformed[:, 0, :]

    def __iter__(self) -> Iterator[MemoryAEQueryBatch]:
        return self.iter_from_update(0)

    def iter_from_update(self, start_update: int) -> Iterator[MemoryAEQueryBatch]:
        """Resume directly with an update-local PCG64 stream, without replay.

        Sampling matches the independent cached-query stream and never changes
        NumPy's global generator or the torch generator used for flow matching.
        """
        if isinstance(start_update, bool) or not isinstance(start_update, (int, np.integer)) or start_update < 0:
            raise ValueError("start_update must be a nonnegative integer")
        update = int(start_update)
        while True:
            rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([self._seed, update])))
            episode_indices = rng.integers(self.episode_count, size=self.queries_per_update)
            query_indices = rng.integers(self._query_counts[episode_indices])
            yield self._batch(episode_indices, query_indices, update_index=update)
            update += 1

    def _batch(
        self,
        episode_positions: np.ndarray,
        query_indices: np.ndarray,
        *,
        update_index: int | None = None,
    ) -> MemoryAEQueryBatch:
        """Explicit query selection for first/middle/final parity checks.

        Positions address the filtered ``_episode_dataset.episodes`` tuple;
        diagnostic episode_index retains the original unfiltered episode id.
        Exactly one LeRobot frame read occurs per selected query, never per
        history frame. No ``episode_at`` or conditioning-cache read occurs.
        """
        positions = np.asarray(episode_positions)
        queries = np.asarray(query_indices)
        if (
            positions.ndim != 1
            or positions.shape != queries.shape
            or not len(positions)
            or positions.dtype.kind not in "iu"
            or queries.dtype.kind not in "iu"
        ):
            raise ValueError("Episode positions and query indices must be nonempty matching integer vectors")
        if np.any(positions < 0) or np.any(positions >= self.episode_count):
            raise ValueError("Sampled episode position is out of range")
        if np.any(queries < 0) or np.any(queries >= self._query_counts[positions]):
            raise ValueError("Sampled query index is out of range")
        records = [self._episodes[int(position)] for position in positions]
        selected = [record.queries[int(query)] for record, query in zip(records, queries, strict=True)]
        frames = np.asarray([query.local_frame_index for query in selected], dtype=np.int64)
        batch_size = len(positions)
        actions = torch.empty((batch_size, self._action_horizon, self._action_dim), dtype=torch.float32)
        action_mask = torch.empty((batch_size, self._action_horizon), dtype=torch.bool)
        history_actions = torch.zeros((batch_size, int(frames.max()), self._action_dim), dtype=torch.float32)
        history_mask = torch.zeros(history_actions.shape[:2], dtype=torch.bool)
        # Transform each episode's largest selected prefix once; earlier rows
        # still receive only [0, f), regardless of any later selected queries.
        for position in np.unique(positions):
            rows = np.flatnonzero(positions == position)
            history = self._history(int(position), int(frames[rows].max()))
            for row in rows:
                frame = int(frames[row])
                history_actions[int(row), :frame].copy_(torch.from_numpy(history[:frame]))
                history_mask[int(row), :frame] = True
        observations = []
        dataset = self._episode_dataset
        for row, (record, query) in enumerate(zip(records, selected, strict=True)):
            raw = dataset._dataset[query.frame_index]
            self._selected_frame_reads += 1
            transformed = dataset._transform(dataset._with_prompt(raw, record))
            if np.asarray(transformed["actions"]).ndim != 2:
                # Reject _action_chunk's scalar-action fallback: it would decode
                # additional future observations to construct the target.
                raise ValueError("Memory-AE requires LeRobot delta-timestamp action chunks")
            chunk, mask = dataset._action_chunk(record, query, transformed)
            expected = (self._action_horizon, self._action_dim)
            if chunk.shape != expected:
                raise ValueError(f"Action transforms produced {chunk.shape}, expected {expected}")
            actions[row].copy_(torch.from_numpy(chunk))
            action_mask[row].copy_(torch.from_numpy(mask))
            observations.append(_model.Observation.from_dict(_episodes._copy_observation_fields(transformed)))
        return MemoryAEQueryBatch(
            observation=_episodes._stack_observations(observations),
            actions=actions,
            action_mask=action_mask,
            history_actions=history_actions,
            history_mask=history_mask,
            episode_index=np.asarray([record.episode_index for record in records], dtype=np.int64),
            query_index=queries.astype(np.int64, copy=True),
            local_frame_index=frames,
            episode_length=self._episode_lengths[positions].copy(),
            episode_position=positions.astype(np.int64, copy=True),
            frame_index=np.asarray([query.frame_index for query in selected], dtype=np.int64),
            update_index=update_index,
        )


def create_memory_ae_data(train_config: Any, *, queries_per_update: int | None = None) -> _MemoryAEData:
    """Build episode-uniform then query-uniform online Memory-AE batches."""
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if not data_config.repo_id or data_config.repo_id == "fake":
        raise ValueError("Memory-AE query training requires a real LeRobot dataset")
    if getattr(data_config, "rlds_data_dir", None) or getattr(data_config, "episode_data_dir", None):
        raise ValueError("Memory-AE query training supports LeRobot per-episode Parquet data only")
    return _MemoryAEData(train_config, data_config, queries_per_update=queries_per_update)
