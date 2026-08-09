from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
import dataclasses
import math
from typing import Any, Protocol, SupportsIndex

import jax
import numpy as np

import openpi.models.model as _model
import openpi.transforms as _transforms


@dataclasses.dataclass(frozen=True)
class EpisodeBatch:
    observation: _model.Observation
    actions: np.ndarray
    action_mask: np.ndarray
    executed_actions: np.ndarray
    executed_action_mask: np.ndarray
    query_mask: np.ndarray
    reset_mask: np.ndarray
    episode_index: np.ndarray


@dataclasses.dataclass(frozen=True)
class QueryRecord:
    episode_index: int
    query_index: int
    frame_index: int
    local_frame_index: int


@dataclasses.dataclass(frozen=True)
class EpisodeRecord:
    episode_index: int
    start_frame: int
    end_frame: int
    queries: tuple[QueryRecord, ...]
    task_index: int | None = None
    task_name: str | None = None


@dataclasses.dataclass(frozen=True)
class EpisodeExample:
    observation: _model.Observation
    actions: np.ndarray
    action_mask: np.ndarray
    executed_actions: np.ndarray
    executed_action_mask: np.ndarray
    episode_index: int


class _RandomAccessDataset(Protocol):
    episode_data_index: dict[str, Any]

    def __getitem__(self, index: SupportsIndex) -> dict[str, Any]: ...

    def __len__(self) -> int: ...


class LeRobotEpisodeDataset:
    """Episode-first wrapper for LeRobot-style datasets.

    The wrapped dataset must expose ``episode_data_index`` with ``"from"`` and
    ``"to"`` arrays. Each item returned by the wrapper is one complete episode
    represented as query chunks spaced by ``query_stride``.
    """

    def __init__(
        self,
        *,
        dataset: _RandomAccessDataset | None = None,
        repo_id: str | None = None,
        action_horizon: int,
        query_stride: int,
        executed_horizon: int | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        action_sequence_keys: Sequence[str] = ("actions",),
        prompt_from_task: bool = False,
        dataset_factory: Callable[..., _RandomAccessDataset] | None = None,
        dataset_metadata_factory: Callable[[str], Any] | None = None,
    ):
        if action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {action_horizon}")
        if query_stride <= 0:
            raise ValueError(f"query_stride must be positive, got {query_stride}")
        if executed_horizon is None:
            executed_horizon = query_stride
        if executed_horizon <= 0:
            raise ValueError(f"executed_horizon must be positive, got {executed_horizon}")
        if executed_horizon > action_horizon:
            raise ValueError("executed_horizon must be <= action_horizon")

        if dataset is None:
            if repo_id is None:
                raise ValueError("repo_id is required when dataset is not injected")
            if dataset_factory is None or dataset_metadata_factory is None:
                import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

                dataset_factory = lerobot_dataset.LeRobotDataset
                dataset_metadata_factory = lerobot_dataset.LeRobotDatasetMetadata
            metadata = dataset_metadata_factory(repo_id)
            dataset = dataset_factory(
                repo_id,
                delta_timestamps={
                    key: [t / metadata.fps for t in range(action_horizon)] for key in action_sequence_keys
                },
            )
        elif repo_id is not None:
            raise ValueError("Pass either dataset or repo_id, not both")

        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._action_horizon = int(action_horizon)
        self._query_stride = int(query_stride)
        self._executed_horizon = int(executed_horizon)
        self._prompt_from_task = bool(prompt_from_task)
        self._tasks = _tasks_from_dataset(dataset)
        self.episodes = self._build_episode_records(dataset)

    def __len__(self) -> int:
        return len(self.episodes)

    def __getitem__(self, index: SupportsIndex) -> EpisodeExample:
        return self.episode_at(index.__index__())

    @property
    def query_stride(self) -> int:
        return self._query_stride

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    @property
    def executed_horizon(self) -> int:
        return self._executed_horizon

    def query_record(self, episode_index: int, query_index: int) -> QueryRecord:
        return self.episodes[int(episode_index)].queries[int(query_index)]

    def episode_at(self, episode_index: int) -> EpisodeExample:
        record = self.episodes[int(episode_index)]
        observations = []
        actions = []
        action_masks = []
        for query in record.queries:
            raw_query_sample = self._dataset[query.frame_index]
            transformed = self._transform(self._with_prompt(raw_query_sample, record))
            observations.append(_model.Observation.from_dict(_copy_observation_fields(transformed)))
            action_chunk, mask = self._action_chunk(record, query, transformed)
            actions.append(action_chunk)
            action_masks.append(mask)

        if not observations:
            raise ValueError(f"Episode {episode_index} has no queries")

        stacked_actions = np.stack(actions, axis=0)
        stacked_action_mask = np.stack(action_masks, axis=0)
        executed_actions, executed_action_mask = self._executed_prefixes(stacked_actions, stacked_action_mask)
        return EpisodeExample(
            observation=_stack_observations(observations),
            actions=stacked_actions,
            action_mask=stacked_action_mask,
            executed_actions=executed_actions,
            executed_action_mask=executed_action_mask,
            episode_index=record.episode_index,
        )

    def _build_episode_records(self, dataset: _RandomAccessDataset) -> tuple[EpisodeRecord, ...]:
        if not hasattr(dataset, "episode_data_index"):
            raise ValueError("LeRobotEpisodeDataset requires episode_data_index")
        index = dataset.episode_data_index
        starts = _to_int_array(index["from"])
        ends = _to_int_array(index["to"])
        if starts.shape != ends.shape:
            raise ValueError("episode_data_index['from'] and ['to'] must have matching shapes")

        task_indices = getattr(dataset, "episode_task_index", None)
        records = []
        for episode_index, (start, end) in enumerate(zip(starts.tolist(), ends.tolist(), strict=True)):
            if end <= start:
                raise ValueError(f"Episode {episode_index} has non-positive length: from={start}, to={end}")
            queries = tuple(
                QueryRecord(
                    episode_index=episode_index,
                    query_index=query_index,
                    frame_index=frame_index,
                    local_frame_index=frame_index - start,
                )
                for query_index, frame_index in enumerate(range(start, end, self._query_stride))
            )
            task_index = None if task_indices is None else int(np.asarray(task_indices)[episode_index])
            records.append(
                EpisodeRecord(
                    episode_index=episode_index,
                    start_frame=int(start),
                    end_frame=int(end),
                    queries=queries,
                    task_index=task_index,
                    task_name=None if task_index is None else self._tasks.get(task_index),
                )
            )
        return tuple(records)

    def _with_prompt(self, sample: dict[str, Any], record: EpisodeRecord) -> dict[str, Any]:
        if not self._prompt_from_task:
            return sample
        if record.task_index is None or record.task_name is None:
            return sample
        return {**sample, "task_index": record.task_index, "prompt": record.task_name}

    def _action_chunk(
        self, record: EpisodeRecord, query: QueryRecord, transformed_query: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        transformed_actions = np.asarray(transformed_query["actions"], dtype=np.float32)
        if transformed_actions.ndim > 1:
            chunk = np.zeros((self._action_horizon, transformed_actions.shape[-1]), dtype=transformed_actions.dtype)
            available = min(self._action_horizon, transformed_actions.reshape(-1, transformed_actions.shape[-1]).shape[0])
            chunk[:available] = transformed_actions.reshape(-1, transformed_actions.shape[-1])[:available]
            valid = min(available, record.end_frame - query.frame_index)
            mask = np.arange(self._action_horizon) < valid
            chunk[~mask] = 0
            return chunk, mask

        first_action = _as_action_array(transformed_actions)
        chunk = np.zeros((self._action_horizon, first_action.shape[-1]), dtype=first_action.dtype)
        mask = np.zeros((self._action_horizon,), dtype=np.bool_)
        chunk[0] = first_action
        mask[0] = True
        for horizon_index in range(1, self._action_horizon):
            frame_index = query.frame_index + horizon_index
            if frame_index >= record.end_frame:
                break
            chunk[horizon_index] = _as_action_array(self._dataset[frame_index]["actions"])
            mask[horizon_index] = True
        return chunk, mask

    def _executed_prefixes(self, actions: np.ndarray, action_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        executed = np.zeros((actions.shape[0], self._executed_horizon, actions.shape[-1]), dtype=actions.dtype)
        executed_mask = np.zeros((actions.shape[0], self._executed_horizon), dtype=np.bool_)
        prefix_len = min(self._executed_horizon, self._query_stride, actions.shape[1])
        for query_index in range(1, actions.shape[0]):
            executed[query_index, :prefix_len] = actions[query_index - 1, :prefix_len]
            executed_mask[query_index, :prefix_len] = action_mask[query_index - 1, :prefix_len]
        return executed, executed_mask


class EpisodeCollator:
    """Collates episodes, padding only the query dimension."""

    def __call__(self, episodes: Sequence[EpisodeExample]) -> EpisodeBatch:
        if not episodes:
            raise ValueError("Cannot collate an empty episode batch")
        max_queries = max(episode.actions.shape[0] for episode in episodes)
        action_shape = (len(episodes), max_queries, *episodes[0].actions.shape[1:])
        executed_shape = (len(episodes), max_queries, *episodes[0].executed_actions.shape[1:])
        actions = np.zeros(action_shape, dtype=episodes[0].actions.dtype)
        action_mask = np.zeros((len(episodes), max_queries, episodes[0].action_mask.shape[1]), dtype=np.bool_)
        executed_actions = np.zeros(executed_shape, dtype=episodes[0].executed_actions.dtype)
        executed_action_mask = np.zeros(
            (len(episodes), max_queries, episodes[0].executed_action_mask.shape[1]), dtype=np.bool_
        )
        query_mask = np.zeros((len(episodes), max_queries), dtype=np.bool_)
        reset_mask = np.zeros((len(episodes), max_queries), dtype=np.bool_)
        episode_index = np.asarray([episode.episode_index for episode in episodes], dtype=np.int32)

        padded_observations = []
        for batch_index, episode in enumerate(episodes):
            num_queries = episode.actions.shape[0]
            actions[batch_index, :num_queries] = episode.actions
            action_mask[batch_index, :num_queries] = episode.action_mask
            executed_actions[batch_index, :num_queries] = episode.executed_actions
            executed_action_mask[batch_index, :num_queries] = episode.executed_action_mask
            query_mask[batch_index, :num_queries] = True
            reset_mask[batch_index, 0] = True
            padded_observations.append(_pad_observation_queries(episode.observation, max_queries))

        return EpisodeBatch(
            observation=_stack_observations(padded_observations),
            actions=actions,
            action_mask=action_mask,
            executed_actions=executed_actions,
            executed_action_mask=executed_action_mask,
            query_mask=query_mask,
            reset_mask=reset_mask,
            episode_index=episode_index,
        )


@dataclasses.dataclass(frozen=True)
class QueryEpisode:
    episode_id: str
    num_queries: int
    payload: Any = None


@dataclasses.dataclass(frozen=True)
class QueryTask:
    name: str
    episodes: Sequence[QueryEpisode]


@dataclasses.dataclass(frozen=True)
class QuerySuite:
    name: str
    weight: float
    tasks: Sequence[QueryTask]


@dataclasses.dataclass(frozen=True)
class BalancedQueryRecord:
    suite_name: str
    task_name: str
    episode_id: str
    query_index: int
    suite_index: int
    task_index: int
    episode_index: int
    payload: Any = None


class BalancedQueryDataset:
    """Deterministically samples suite, task, episode, then query uniformly at each level."""

    def __init__(self, suites: Sequence[QuerySuite], *, seed: int = 0):
        if not suites:
            raise ValueError("At least one suite is required")
        self._suites = tuple(suites)
        self._seed = int(seed)
        weights = []
        for suite in self._suites:
            if suite.weight <= 0 or not math.isfinite(suite.weight):
                raise ValueError(f"Suite {suite.name!r} weight must be positive")
            if not suite.tasks:
                raise ValueError(f"Suite {suite.name!r} must contain at least one task")
            for task in suite.tasks:
                if not task.episodes:
                    raise ValueError(f"Task {task.name!r} must contain at least one episode")
                for episode in task.episodes:
                    if episode.num_queries <= 0:
                        raise ValueError(f"Episode {episode.episode_id!r} num_queries must be positive")
            weights.append(float(suite.weight))
        self._suite_probabilities = np.asarray(weights, dtype=np.float64) / np.sum(weights)

    def __len__(self) -> int:
        return 2**63 - 1

    def __getitem__(self, index: SupportsIndex) -> BalancedQueryRecord:
        return self.record_at(index.__index__())

    def record_at(self, index: int) -> BalancedQueryRecord:
        rng = np.random.default_rng(np.random.SeedSequence([self._seed, int(index)]))
        suite_index = int(rng.choice(len(self._suites), p=self._suite_probabilities))
        suite = self._suites[suite_index]
        task_index = int(rng.integers(len(suite.tasks)))
        task = suite.tasks[task_index]
        episode_index = int(rng.integers(len(task.episodes)))
        episode = task.episodes[episode_index]
        query_index = int(rng.integers(episode.num_queries))
        return BalancedQueryRecord(
            suite_name=suite.name,
            task_name=task.name,
            episode_id=episode.episode_id,
            query_index=query_index,
            suite_index=suite_index,
            task_index=task_index,
            episode_index=episode_index,
            payload=episode.payload,
        )


def make_transform_pipeline(data_config: Any, *, skip_norm_stats: bool = False) -> tuple[_transforms.DataTransformFn, ...]:
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats
    return (
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    )


def _to_int_array(value: Any) -> np.ndarray:
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=np.int64)


def _tasks_from_dataset(dataset: Any) -> dict[int, str]:
    if hasattr(dataset, "meta") and hasattr(dataset.meta, "tasks"):
        return {int(key): value for key, value in dataset.meta.tasks.items()}
    if hasattr(dataset, "tasks"):
        return {int(key): value for key, value in dataset.tasks.items()}
    return {}


def _as_action_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        raise ValueError("actions must have at least one dimension")
    if array.ndim == 1:
        return array
    return array.reshape(-1, array.shape[-1])[0]


def _copy_observation_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "image": data["image"],
        "image_mask": data["image_mask"],
        "state": data["state"],
        **({"tokenized_prompt": data["tokenized_prompt"]} if "tokenized_prompt" in data else {}),
        **({"tokenized_prompt_mask": data["tokenized_prompt_mask"]} if "tokenized_prompt_mask" in data else {}),
        **({"token_ar_mask": data["token_ar_mask"]} if "token_ar_mask" in data else {}),
        **({"token_loss_mask": data["token_loss_mask"]} if "token_loss_mask" in data else {}),
    }


def _stack_observations(observations: Sequence[_model.Observation]) -> _model.Observation:
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *observations)


def _pad_tree_query_axis(tree: Any, target_queries: int) -> Any:
    def pad_leaf(value: Any) -> np.ndarray:
        array = np.asarray(value)
        if array.shape[0] == target_queries:
            return array
        pad_width = [(0, target_queries - array.shape[0]), *[(0, 0)] * (array.ndim - 1)]
        return np.pad(array, pad_width, mode="constant")

    return jax.tree.map(pad_leaf, tree)


def _pad_observation_queries(observation: _model.Observation, target_queries: int) -> _model.Observation:
    return _model.Observation(
        images=_pad_tree_query_axis(observation.images, target_queries),
        image_masks=_pad_tree_query_axis(observation.image_masks, target_queries),
        state=_pad_tree_query_axis(observation.state, target_queries),
        tokenized_prompt=None
        if observation.tokenized_prompt is None
        else _pad_tree_query_axis(observation.tokenized_prompt, target_queries),
        tokenized_prompt_mask=None
        if observation.tokenized_prompt_mask is None
        else _pad_tree_query_axis(observation.tokenized_prompt_mask, target_queries),
        token_ar_mask=None if observation.token_ar_mask is None else _pad_tree_query_axis(observation.token_ar_mask, target_queries),
        token_loss_mask=None
        if observation.token_loss_mask is None
        else _pad_tree_query_axis(observation.token_loss_mask, target_queries),
    )
