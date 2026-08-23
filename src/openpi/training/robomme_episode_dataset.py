from __future__ import annotations

import copy
import dataclasses
import json
import pathlib
import pickle
from typing import Any
from typing import SupportsIndex

import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.training import episode_data_loader as _episode_loader
from openpi.training.futuremamba_conditioning_cache import ConditioningCacheReader


_DEFAULT_ACTION_HORIZON = 20
_REQUIRED_FIELDS = (
    "epis_idx",
    "step_idx",
    "exec_start_idx",
    "is_demo",
    "actions",
    "image",
    "wrist_image",
    "state",
    "prompt",
    "simple_subgoal",
    "grounded_subgoal",
    "simple_subgoal_online",
    "grounded_subgoal_online",
)


@dataclasses.dataclass(frozen=True)
class RoboMMESampleRef:
    path: pathlib.Path
    sorted_index: int
    epis_idx: int
    step_idx: int
    exec_start_idx: int
    episode_start_step_idx: int


@dataclasses.dataclass(frozen=True)
class RoboMMEEpisodeWindow:
    epis_idx: int
    start_step_idx: int
    sample_refs: tuple[RoboMMESampleRef, ...]
    samples: tuple[dict[str, Any], ...]
    burn_in_refs: tuple[RoboMMESampleRef, ...]
    burn_in_step_indices: tuple[int, ...]
    burn_in_reset_mask: tuple[bool, ...]
    train_step_indices: tuple[int | None, ...]
    train_query_slice: slice
    padding_query_indices: tuple[int, ...]
    reset_before_burn_in: bool
    detach_after: bool
    actions: np.ndarray
    action_mask: np.ndarray
    query_mask: np.ndarray
    reset_mask: np.ndarray


@dataclasses.dataclass(frozen=True)
class _IndexedSample:
    ref: RoboMMESampleRef
    payload: dict[str, Any]
    action_width: int
    action_length: int


@dataclasses.dataclass(frozen=True)
class _EpisodeIndex:
    epis_idx: int
    samples: tuple[_IndexedSample, ...]


@dataclasses.dataclass(frozen=True)
class _WindowIndex:
    episode_index: int
    start_offset: int


class RoboMMEEpisodeDataset:
    """Pickle-backed RoboMME execution-step windows.

    The official RoboMME builder writes one pickle per execution query under
    ``data/*.pkl``. This dataset keeps those payloads available for downstream
    transforms and materializes fixed-size query windows without crossing an
    episode boundary.
    """

    def __init__(
        self,
        data_dir: str | pathlib.Path,
        *,
        window_queries: int = 1,
        action_horizon: int = _DEFAULT_ACTION_HORIZON,
        query_stride: int = 1,
        full_episodes: bool = False,
        train_query_stride: int | None = None,
    ) -> None:
        if window_queries <= 0:
            raise ValueError(f"window_queries must be positive, got {window_queries}")
        if action_horizon != _DEFAULT_ACTION_HORIZON:
            raise ValueError(f"action_horizon must be {_DEFAULT_ACTION_HORIZON}, got {action_horizon}")
        if query_stride <= 0:
            raise ValueError(f"query_stride must be positive, got {query_stride}")

        self._data_dir = pathlib.Path(data_dir)
        self._window_queries = int(window_queries)
        self._action_horizon = int(action_horizon)
        self._query_stride = int(query_stride)
        self._full_episodes = bool(full_episodes)
        self._train_query_stride = int(train_query_stride or query_stride)
        if self._train_query_stride % self._query_stride != 0:
            raise ValueError(
                "train_query_stride must be a multiple of query_stride, "
                f"got {self._train_query_stride} and {self._query_stride}"
            )
        indexed_episodes = self._build_index(self._data_dir)
        self._episodes = tuple(
            _EpisodeIndex(epis_idx=episode.epis_idx, samples=episode.samples[:: self._query_stride])
            for episode in indexed_episodes
        )
        self._windows = tuple(
            _WindowIndex(episode_index=episode_index, start_offset=start_offset)
            for episode_index, episode in enumerate(self._episodes)
            for start_offset in ([0] if self._full_episodes else range(len(episode.samples)))
        )
        self._sample_by_ref = {
            sample.ref: _readonly_copy(sample.payload) for episode in self._episodes for sample in episode.samples
        }
        self.sample_refs = tuple(sample.ref for episode in self._episodes for sample in episode.samples)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: SupportsIndex) -> RoboMMEEpisodeWindow:
        return self.window_at(index.__index__())

    @property
    def data_dir(self) -> pathlib.Path:
        return self._data_dir

    @property
    def window_queries(self) -> int:
        return self._window_queries

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def query_stride(self) -> int:
        return self._query_stride

    @property
    def episodes(self) -> tuple[tuple[RoboMMESampleRef, ...], ...]:
        return tuple(tuple(sample.ref for sample in episode.samples) for episode in self._episodes)

    def sample_payload(self, ref: RoboMMESampleRef) -> dict[str, Any]:
        return _readonly_copy(self._sample_by_ref[ref])

    def window_at(self, index: int) -> RoboMMEEpisodeWindow:
        window_index = self._windows[int(index)]
        episode = self._episodes[window_index.episode_index]
        start_offset = window_index.start_offset
        if self._full_episodes:
            window_samples = episode.samples[start_offset:]
            sequence_queries = len(window_samples)
        else:
            window_samples = episode.samples[start_offset : start_offset + self._window_queries]
            sequence_queries = self._window_queries
        if not window_samples:
            raise ValueError(f"RoboMME episode {episode.epis_idx} produced an empty query sequence")
        first_sample = episode.samples[0]
        action_width = first_sample.action_width

        actions = np.zeros((sequence_queries, self._action_horizon, action_width), dtype=np.float32)
        action_mask = np.zeros((sequence_queries, self._action_horizon), dtype=np.bool_)
        query_mask = np.zeros((sequence_queries,), dtype=np.bool_)
        reset_mask = np.zeros((sequence_queries,), dtype=np.bool_)

        for query_index, sample in enumerate(window_samples):
            action_array = _actions_as_2d(sample.payload["actions"], sample.ref.path)
            actions[query_index] = action_array
            action_mask[query_index] = True
            query_mask[query_index] = True
            reset_mask[query_index] = start_offset + query_index == 0

        burn_in_samples = episode.samples[:start_offset]
        valid_queries = len(window_samples)
        train_step_indices = tuple(sample.ref.step_idx for sample in window_samples) + (None,) * (
            sequence_queries - valid_queries
        )
        padding_query_indices = tuple(range(valid_queries, sequence_queries))
        return RoboMMEEpisodeWindow(
            epis_idx=episode.epis_idx,
            start_step_idx=episode.samples[start_offset].ref.step_idx,
            sample_refs=tuple(sample.ref for sample in window_samples),
            samples=tuple(_readonly_copy(sample.payload) for sample in window_samples),
            burn_in_refs=tuple(sample.ref for sample in burn_in_samples),
            burn_in_step_indices=tuple(sample.ref.step_idx for sample in burn_in_samples),
            burn_in_reset_mask=tuple(index == 0 for index in range(len(burn_in_samples))),
            train_step_indices=train_step_indices,
            train_query_slice=slice(0, valid_queries),
            padding_query_indices=padding_query_indices,
            reset_before_burn_in=True,
            detach_after=True,
            actions=actions,
            action_mask=action_mask,
            query_mask=query_mask,
            reset_mask=reset_mask,
        )

    @classmethod
    def _build_index(cls, data_dir: pathlib.Path) -> tuple[_EpisodeIndex, ...]:
        paths = sorted(data_dir.glob("*.pkl"))
        if not paths:
            raise ValueError(f"No RoboMME pickle samples found in {data_dir}")

        loaded: list[tuple[pathlib.Path, dict[str, Any], int, int, int, int, int]] = []
        for path in paths:
            payload = _load_sample(path)
            for field in _REQUIRED_FIELDS:
                if field not in payload:
                    raise ValueError(f"RoboMME sample {path} is missing required field {field!r}")
            epis_idx = _scalar_int(payload["epis_idx"], "epis_idx", path)
            step_idx = _scalar_int(payload["step_idx"], "step_idx", path)
            exec_start_idx = _scalar_int(payload["exec_start_idx"], "exec_start_idx", path)
            if _scalar_bool(payload["is_demo"], "is_demo", path):
                raise ValueError(f"RoboMME sample {path} is_demo must be False for execution samples")
            actions = _actions_as_2d(payload["actions"], path)
            action_length, action_width = actions.shape
            if action_length != _DEFAULT_ACTION_HORIZON:
                raise ValueError(
                    f"RoboMME sample {path} actions length must be exactly {_DEFAULT_ACTION_HORIZON}, got {action_length}"
                )
            loaded.append((path, payload, epis_idx, step_idx, exec_start_idx, action_width, action_length))
        loaded.sort(key=lambda item: (item[2], item[3], str(item[0])))
        episodes: list[_EpisodeIndex] = []
        cursor = 0
        sorted_index = 0
        while cursor < len(loaded):
            epis_idx = loaded[cursor][2]
            episode_items = []
            while cursor < len(loaded) and loaded[cursor][2] == epis_idx:
                episode_items.append(loaded[cursor])
                cursor += 1
            episodes.append(cls._build_episode(epis_idx, episode_items, sorted_index))
            sorted_index += len(episode_items)
        widths = {episode.samples[0].action_width for episode in episodes}
        if len(widths) != 1:
            raise ValueError(f"RoboMME dataset has inconsistent action width across episodes: {sorted(widths)}")
        return tuple(episodes)

    @staticmethod
    def _build_episode(
        epis_idx: int,
        items: list[tuple[pathlib.Path, dict[str, Any], int, int, int, int, int]],
        first_sorted_index: int,
    ) -> _EpisodeIndex:
        first_step = items[0][3]
        first_exec_start = items[0][4]
        if first_step != first_exec_start:
            raise ValueError(
                f"Episode {epis_idx} first step_idx must equal exec_start_idx, got step_idx={first_step}, "
                f"exec_start_idx={first_exec_start}"
            )

        action_width = items[0][5]
        samples: list[_IndexedSample] = []
        seen_steps: set[int] = set()
        for offset, (path, payload, _, step_idx, exec_start_idx, width, action_length) in enumerate(items):
            expected_step = first_step + offset
            if step_idx in seen_steps:
                raise ValueError(f"Episode {epis_idx} has duplicate step_idx {step_idx}")
            seen_steps.add(step_idx)
            if step_idx != expected_step:
                raise ValueError(
                    f"Episode {epis_idx} steps must be continuous from exec_start_idx {first_step}; "
                    f"expected {expected_step}, got {step_idx}"
                )
            if exec_start_idx != first_exec_start:
                raise ValueError(
                    f"Episode {epis_idx} has inconsistent exec_start_idx: expected {first_exec_start}, got {exec_start_idx}"
                )
            if width != action_width:
                raise ValueError(f"Episode {epis_idx} has inconsistent action width: expected {action_width}, got {width}")
            ref = RoboMMESampleRef(
                path=path,
                sorted_index=first_sorted_index + offset,
                epis_idx=epis_idx,
                step_idx=step_idx,
                exec_start_idx=exec_start_idx,
                episode_start_step_idx=first_step,
            )
            samples.append(_IndexedSample(ref=ref, payload=payload, action_width=width, action_length=action_length))
        return _EpisodeIndex(epis_idx=epis_idx, samples=tuple(samples))


def _load_sample(path: pathlib.Path) -> dict[str, Any]:
    with path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"RoboMME sample {path} must be a dict, got {type(payload).__name__}")
    return payload


def _scalar_int(value: Any, field: str, path: pathlib.Path) -> int:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"RoboMME sample {path} field {field!r} must be an integer scalar")
    return int(array.reshape(()).item())

def _scalar_bool(value: Any, field: str, path: pathlib.Path) -> bool:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind != "b":
        raise ValueError(f"RoboMME sample {path} field {field!r} must be a boolean scalar")
    return bool(array.reshape(()).item())


def _readonly_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _readonly_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_readonly_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_readonly_copy(item) for item in value)
    if isinstance(value, np.ndarray):
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result
    return copy.deepcopy(value)

def _mutable_copy(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _mutable_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_mutable_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_mutable_copy(item) for item in value)
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    return copy.deepcopy(value)


def _actions_as_2d(value: Any, path: pathlib.Path) -> np.ndarray:
    actions = np.asarray(value, dtype=np.float32)
    if actions.ndim != 2:
        raise ValueError(f"RoboMME sample {path} actions must have rank 2, got shape {actions.shape}")
    if actions.shape[0] != _DEFAULT_ACTION_HORIZON:
        raise ValueError(
            f"RoboMME sample {path} actions length must be exactly {_DEFAULT_ACTION_HORIZON}, got {actions.shape[0]}"
        )
    if actions.shape[1] <= 0:
        raise ValueError(f"RoboMME sample {path} actions must have positive action width, got {actions.shape}")
    return actions

class RoboMMETransformedEpisodeDataset:
    """Convert indexed RoboMME windows into the episode training contract."""

    def __init__(self, windows: RoboMMEEpisodeDataset, data_config: Any, model_config: Any) -> None:
        self._windows = windows
        self._model_config = model_config
        self._transform = _transforms.compose(
            _episode_loader.make_transform_pipeline(data_config, skip_norm_stats=False)
        )
        cache_dir = getattr(data_config, "conditioning_cache_dir", None)
        self._conditioning_cache = None
        if cache_dir:
            manifest = pathlib.Path(cache_dir) / "manifest.json"
            if not manifest.is_file():
                raise ValueError(f"conditioning cache manifest is missing: {manifest}")
            identity = json.loads(manifest.read_text(encoding="utf-8"))["identity"]
            expected_mapping = list(getattr(model_config, "resolved_progress_layer_indices"))
            if int(identity.get("query_stride", -1)) != self._windows.query_stride:
                raise ValueError("conditioning cache query_stride does not match dataset")
            if identity.get("layer_mapping") != expected_mapping:
                raise ValueError("conditioning cache layer_mapping does not match model config")
            if identity.get("dtype") != str(getattr(model_config, "dtype", "float32")):
                raise ValueError("conditioning cache dtype does not match model config")
            expected_assets = getattr(model_config, "base_assets_checksum", None)
            if expected_assets is not None and identity.get("assets_checksum") != expected_assets:
                raise ValueError("conditioning cache assets_checksum does not match model config")
            expected_dataset = getattr(model_config, "robomme_dataset_checksum", None)
            if expected_dataset is not None and identity.get("dataset_checksum") != expected_dataset:
                raise ValueError("conditioning cache dataset_checksum does not match model config")
            expected_task_suite = getattr(model_config, "robomme_task_suite", None)
            if expected_task_suite is not None and identity.get("task_suite") != expected_task_suite:
                raise ValueError("conditioning cache task_suite does not match model config")
            self._conditioning_cache = ConditioningCacheReader(cache_dir, expected_identity=identity)
        self._executed_horizon = int(
            getattr(model_config, "execution_horizon", getattr(model_config, "executed_horizon", 0))
        )
        if self._executed_horizon <= 0:
            raise ValueError("RoboMME model config must define a positive execution horizon")

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: SupportsIndex) -> _episode_loader.EpisodeExample:
        window = self._windows[index.__index__()]
        all_samples = tuple(self._windows.sample_payload(ref) for ref in window.burn_in_refs) + window.samples
        burn_in_queries = len(window.burn_in_refs)
        observations = []
        actions = []
        cached_hidden = []
        cached_masks = []
        cached_keys = []
        cached_values = []
        cache_enabled = self._conditioning_cache is not None
        refs = tuple(window.burn_in_refs) + tuple(window.sample_refs)
        for sample, ref in zip(all_samples, refs, strict=True):
            transformed = self._transform(_mutable_copy(sample))
            observation_data = _episode_loader._copy_observation_fields(transformed)
            observation_data["image_mask"] = {
                key: np.asarray(value, dtype=np.bool_) for key, value in observation_data["image_mask"].items()
            }
            observations.append(_model.Observation.from_dict(observation_data))
            action = np.asarray(transformed["actions"], dtype=np.float32)
            expected = (self._windows.action_horizon, int(self._model_config.action_dim))
            if action.shape != expected:
                raise ValueError(f"RoboMME transformed action target must have shape {expected}, got {action.shape}")
            actions.append(action)
            if cache_enabled:
                assert self._conditioning_cache is not None
                entry = self._conditioning_cache.read_episode(f"e{ref.epis_idx}", query_id=int(ref.step_idx))
                cached_hidden.append(entry.last_valid_hidden.float().numpy())
                cached_masks.append(entry.prefix_mask.numpy())
                cached_keys.append(entry.action_expert_kv[0].float().numpy())
                cached_values.append(entry.action_expert_kv[1].float().numpy())
        if not observations or burn_in_queries == len(observations):
            raise ValueError(f"RoboMME episode window {index.__index__()} has no training queries")
        stacked_actions = np.stack(actions, axis=0)
        num_queries = len(observations)
        training_target_mask = np.asarray(
            [
                index >= burn_in_queries
                and (ref.step_idx - ref.episode_start_step_idx) % self._windows._train_query_stride == 0
                for index, ref in enumerate(refs)
            ],
            dtype=np.bool_,
        )
        if not np.any(training_target_mask):
            raise ValueError(f"RoboMME episode window {index.__index__()} has no stride-aligned training query")
        train_query_mask = training_target_mask
        action_mask = np.broadcast_to(
            train_query_mask[:, None], (num_queries, self._windows.action_horizon)
        ).copy()
        conditioning_cache = None
        if cache_enabled:
            max_prefix = max(array.shape[2] for array in cached_keys)
            key_shape = (len(cached_keys), cached_keys[0].shape[0], cached_keys[0].shape[1], max_prefix, cached_keys[0].shape[3])
            padded_keys = np.zeros(key_shape, dtype=cached_keys[0].dtype)
            padded_values = np.zeros(key_shape, dtype=cached_values[0].dtype)
            padded_masks = np.zeros((len(cached_masks), max_prefix), dtype=np.bool_)
            for cache_index, (keys, values, mask) in enumerate(zip(cached_keys, cached_values, cached_masks, strict=True)):
                prefix_length = keys.shape[2]
                padded_keys[cache_index, :, :, :prefix_length] = keys
                padded_values[cache_index, :, :, :prefix_length] = values
                padded_masks[cache_index, :prefix_length] = True
            conditioning_cache = {
                "last_valid_hidden": np.stack(cached_hidden, axis=0),
                "prefix_mask": padded_masks,
                "action_expert_keys": padded_keys,
                "action_expert_values": padded_values,
            }
        return _episode_loader.EpisodeExample(
            observation=_episode_loader._stack_observations(observations),
            actions=stacked_actions,
            action_mask=action_mask,
            executed_actions=np.zeros(
                (num_queries, self._executed_horizon, int(self._model_config.action_dim)), dtype=np.float32
            ),
            executed_action_mask=np.zeros((num_queries, self._executed_horizon), dtype=np.bool_),
            episode_index=window.epis_idx,
            train_query_mask=train_query_mask,
            conditioning_cache=conditioning_cache,
        )


def create_robomme_episode_dataset(data_config: Any, model_config: Any, episode_config: Any):
    data_dir = getattr(data_config, "episode_data_dir", None)
    if not data_dir:
        raise ValueError(
            "RoboMME FutureMamba training requires data.episode_data_dir pointing to official preprocessed pickles"
        )
    windows = RoboMMEEpisodeDataset(
        data_dir,
        window_queries=int(getattr(episode_config, "window_queries", 1)),
        action_horizon=int(model_config.action_horizon),
        query_stride=int(getattr(episode_config, "query_stride", 1)),
        full_episodes=bool(getattr(episode_config, "full_episodes", False)),
        train_query_stride=getattr(episode_config, "train_query_stride", None),
    )
    return RoboMMETransformedEpisodeDataset(windows, data_config, model_config)
