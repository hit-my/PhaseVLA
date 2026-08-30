from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import openpi.models.model as _model
from openpi.training import episode_data_loader as _episode_loader


@dataclasses.dataclass(frozen=True)
class ActionNormStats:
    q01: np.ndarray
    q99: np.ndarray

    @classmethod
    def load(cls, path: str | Path) -> "ActionNormStats":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))["norm_stats"]["actions"]
        return cls(np.asarray(payload["q01"], dtype=np.float32), np.asarray(payload["q99"], dtype=np.float32))

    def normalize(self, actions: np.ndarray) -> np.ndarray:
        width = actions.shape[-1]
        return (actions - self.q01[:width]) / (self.q99[:width] - self.q01[:width] + 1e-6) * 2.0 - 1.0


class LiberoActionHistoryEpisodeDataset:
    """Action-only episode dataset built directly from an official LIBERO HDF5 file."""

    def __init__(
        self,
        hdf5_path: str | Path,
        norm_stats_path: str | Path,
        *,
        action_horizon: int,
        action_dim: int,
        query_stride: int,
        executed_horizon: int,
    ) -> None:
        self.path = Path(hdf5_path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        if action_horizon <= 0 or action_dim <= 0 or query_stride <= 0 or executed_horizon <= 0:
            raise ValueError("action_horizon, action_dim, query_stride, and executed_horizon must be positive")
        if executed_horizon > action_horizon:
            raise ValueError("executed_horizon must be <= action_horizon")
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.query_stride = int(query_stride)
        self.executed_horizon = int(executed_horizon)
        self.stats = ActionNormStats.load(norm_stats_path)
        with h5py.File(self.path, "r") as file:
            self._episode_keys = tuple(
                sorted(file["data"].keys(), key=lambda name: int(name.rsplit("_", 1)[-1]))
            )
        if not self._episode_keys:
            raise ValueError(f"no demonstrations in {self.path}")

    def __len__(self) -> int:
        return len(self._episode_keys)

    def __getitem__(self, index: int) -> _episode_loader.EpisodeExample:
        episode_index = int(index)
        key = self._episode_keys[episode_index]
        with h5py.File(self.path, "r") as file:
            raw_actions = np.asarray(file[f"data/{key}/actions"], dtype=np.float32)
        if raw_actions.ndim != 2 or raw_actions.shape[1] > self.action_dim:
            raise ValueError(f"invalid actions shape {raw_actions.shape} in {self.path}:{key}")
        normalized = self.stats.normalize(raw_actions)
        query_frames = tuple(range(0, len(normalized), self.query_stride))
        actions = []
        action_masks = []
        executed_actions = []
        executed_masks = []
        for query_index, frame in enumerate(query_frames):
            chunk, chunk_mask = self._padded_chunk(normalized[frame : frame + self.action_horizon])
            actions.append(chunk)
            action_masks.append(chunk_mask)
            if query_index == 0:
                previous = normalized[:0]
            else:
                previous_frame = query_frames[query_index - 1]
                previous = normalized[
                    previous_frame : min(previous_frame + self.executed_horizon, frame)
                ]
            executed, executed_mask = self._padded_executed(previous)
            executed_actions.append(executed)
            executed_masks.append(executed_mask)
        num_queries = len(query_frames)
        observation = _model.Observation(
            images={},
            image_masks={},
            state=np.zeros((num_queries, 1), dtype=np.float32),
        )
        return _episode_loader.EpisodeExample(
            observation=observation,
            actions=np.stack(actions),
            action_mask=np.stack(action_masks),
            executed_actions=np.stack(executed_actions),
            executed_action_mask=np.stack(executed_masks),
            episode_index=episode_index,
            train_query_mask=np.ones((num_queries,), dtype=np.bool_),
        )

    def _padded_chunk(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = np.zeros((self.action_horizon, self.action_dim), dtype=np.float32)
        count = min(len(values), self.action_horizon)
        if count:
            result[:count, : values.shape[1]] = values[:count]
        return result, np.arange(self.action_horizon) < count

    def _padded_executed(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        result = np.zeros((self.executed_horizon, self.action_dim), dtype=np.float32)
        count = min(len(values), self.executed_horizon)
        if count:
            result[:count, : values.shape[1]] = values[:count]
        return result, np.arange(self.executed_horizon) < count


def create_action_history_dataset(data_config: Any, model_config: Any, episode_config: Any):
    return LiberoActionHistoryEpisodeDataset(
        data_config.hdf5_path,
        data_config.norm_stats_path,
        action_horizon=int(model_config.action_horizon),
        action_dim=int(model_config.action_dim),
        query_stride=int(episode_config.query_stride),
        executed_horizon=int(episode_config.executed_horizon),
    )
