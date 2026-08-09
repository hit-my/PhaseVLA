from __future__ import annotations

import collections
import dataclasses

import numpy as np

from openpi.training import episode_data_loader as _episode_loader
from openpi.training import config as _config


ACTION_DIM = 3
ACTION_HORIZON = 10
QUERY_STRIDE = 5


def _frame_action(frame_index: int, episode_index: int) -> np.ndarray:
    return np.asarray(
        [1000.0 * episode_index + 10.0 * frame_index + dim for dim in range(ACTION_DIM)], dtype=np.float32
    )


class _FakeLeRobotDataset:
    def __init__(self, episode_lengths=(12, 7), *, task_names=("task_a", "task_b")):
        starts = []
        stops = []
        cursor = 0
        self._episode_for_frame = []
        self._local_for_frame = []
        for episode_index, length in enumerate(episode_lengths):
            starts.append(cursor)
            stops.append(cursor + length)
            for local_index in range(length):
                self._episode_for_frame.append(episode_index)
                self._local_for_frame.append(local_index)
            cursor += length

        self.episode_data_index = {
            "from": np.asarray(starts, dtype=np.int64),
            "to": np.asarray(stops, dtype=np.int64),
        }
        self.episode_task_index = np.arange(len(episode_lengths), dtype=np.int64) % len(task_names)
        self.meta = dataclasses.make_dataclass("FakeMeta", [("tasks", dict)])(
            tasks={index: name for index, name in enumerate(task_names)}
        )

    def __len__(self):
        return len(self._episode_for_frame)

    def __getitem__(self, index):
        frame_index = int(index)
        episode_index = self._episode_for_frame[frame_index]
        local_index = self._local_for_frame[frame_index]
        return {
            "image": {"base_0_rgb": np.full((2, 2, 3), local_index, dtype=np.float32)},
            "image_mask": {"base_0_rgb": np.asarray(True)},
            "state": np.asarray([episode_index, local_index], dtype=np.float32),
            "actions": _frame_action(local_index, episode_index),
            "prompt": f"episode {episode_index}",
        }


class _FakeLeRobotDatasetWithSampleTask(_FakeLeRobotDataset):
    def __init__(self):
        super().__init__((6, 6), task_names=("task from episode zero", "task from episode one"))
        del self.episode_task_index

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        episode_index = self._episode_for_frame[int(index)]
        return {
            **sample,
            "prompt": "stale prompt that must be replaced",
            "task_index": np.asarray(episode_index, dtype=np.int64),
            "task": self.meta.tasks[episode_index],
        }


def _batch_from_fake_episode_lengths(episode_lengths=(12, 7)):
    dataset = _episode_loader.LeRobotEpisodeDataset(
        dataset=_FakeLeRobotDataset(episode_lengths),
        action_horizon=ACTION_HORIZON,
        query_stride=QUERY_STRIDE,
        executed_horizon=QUERY_STRIDE,
    )
    episodes = [dataset.episode_at(i) for i in range(len(dataset.episodes))]
    return _episode_loader.EpisodeCollator()(episodes)


def test_episode_batch_collates_full_episodes_with_query_padding_and_masks():
    batch = _batch_from_fake_episode_lengths()

    assert isinstance(batch, _episode_loader.EpisodeBatch)
    assert batch.actions.shape == (2, 3, ACTION_HORIZON, ACTION_DIM)
    assert batch.action_mask.shape == (2, 3, ACTION_HORIZON)
    assert batch.executed_actions.shape == (2, 3, QUERY_STRIDE, ACTION_DIM)
    assert batch.executed_action_mask.shape == (2, 3, QUERY_STRIDE)
    assert batch.observation.state.shape == (2, 3, 2)
    assert batch.observation.images["base_0_rgb"].shape == (2, 3, 2, 2, 3)

    np.testing.assert_array_equal(batch.query_mask[0], [True, True, True])
    np.testing.assert_array_equal(batch.query_mask[1], [True, True, False])
    np.testing.assert_array_equal(batch.reset_mask[0], [True, False, False])
    np.testing.assert_array_equal(batch.reset_mask[1], [True, False, False])
    np.testing.assert_array_equal(batch.episode_index, [0, 1])
    assert not batch.executed_action_mask[:, 0].any()
    np.testing.assert_allclose(batch.executed_actions[0, 1], batch.actions[0, 0, :QUERY_STRIDE])


def test_action_horizon_and_executed_prefix_never_cross_episode_boundaries():
    batch = _batch_from_fake_episode_lengths()

    np.testing.assert_array_equal(batch.action_mask[0, 0], [True] * 10)
    np.testing.assert_array_equal(batch.action_mask[0, 1], [True] * 7 + [False] * 3)
    np.testing.assert_array_equal(batch.action_mask[0, 2], [True] * 2 + [False] * 8)
    np.testing.assert_array_equal(batch.action_mask[1, 0], [True] * 7 + [False] * 3)
    np.testing.assert_array_equal(batch.action_mask[1, 1], [True] * 2 + [False] * 8)

    assert not batch.actions[0, 1, 7:].any()
    assert not batch.actions[0, 2, 2:].any()
    assert not batch.actions[1, 0, 7:].any()
    assert not batch.actions[1, 1, 2:].any()
    assert not batch.actions[1, 2].any()
    assert not batch.executed_actions[1, 0].any()
    np.testing.assert_allclose(batch.executed_actions[1, 1], batch.actions[1, 0, :QUERY_STRIDE])
    assert batch.executed_actions[1, 1, 0, 0] >= 1000.0
    assert not np.any(batch.executed_actions[1, 1, :, 0] < 1000.0)


def test_collator_pads_only_query_dimension():
    batch = _batch_from_fake_episode_lengths((3, 11))

    assert batch.actions.shape == (2, 3, ACTION_HORIZON, ACTION_DIM)
    assert batch.observation.state.shape == (2, 3, 2)
    np.testing.assert_array_equal(batch.query_mask[0], [True, False, False])
    np.testing.assert_array_equal(batch.query_mask[1], [True, True, True])
    np.testing.assert_array_equal(batch.action_mask[0, 0], [True, True, True, False, False, False, False, False, False, False])
    np.testing.assert_array_equal(batch.action_mask[1, 2], [True, False, False, False, False, False, False, False, False, False])
    assert not batch.actions[0, 1:].any()
    assert not batch.actions[1, 2, 1:].any()


def test_lerobot_episode_dataset_indexes_queries_from_episode_data_index_and_applies_transform_hook():
    calls = []

    def transform(sample):
        calls.append(int(sample["state"][1]))
        return {
            **sample,
            "state": sample["state"] + np.asarray([10.0, 100.0], dtype=np.float32),
            "actions": sample["actions"] + np.asarray([500.0, 500.0, 500.0], dtype=np.float32),
        }

    dataset = _episode_loader.LeRobotEpisodeDataset(
        dataset=_FakeLeRobotDataset((12, 7)),
        action_horizon=ACTION_HORIZON,
        query_stride=QUERY_STRIDE,
        executed_horizon=QUERY_STRIDE,
        transforms=[transform],
    )

    assert len(dataset.episodes) == 2
    assert [query.frame_index for query in dataset.episodes[0].queries] == [0, 5, 10]
    assert [query.frame_index for query in dataset.episodes[1].queries] == [12, 17]
    assert dataset.query_record(1, 1).episode_index == 1

    episode = dataset.episode_at(0)
    assert {0, 5, 10}.issubset(set(calls))
    np.testing.assert_allclose(episode.observation.state[:, 0], [10.0, 10.0, 10.0])
    np.testing.assert_allclose(episode.observation.state[:, 1], [100.0, 105.0, 110.0])
    np.testing.assert_allclose(episode.actions[0, 0], _frame_action(0, 0) + 500.0)
    np.testing.assert_allclose(episode.actions[1, 0], _frame_action(5, 0) + 500.0)


def test_one_dimensional_action_fallback_applies_transforms_to_every_future_frame():
    def transform(sample):
        return {**sample, "actions": sample["actions"] + np.asarray([500.0, 500.0, 500.0], dtype=np.float32)}

    dataset = _episode_loader.LeRobotEpisodeDataset(
        dataset=_FakeLeRobotDataset((12, 7)),
        action_horizon=ACTION_HORIZON,
        query_stride=QUERY_STRIDE,
        executed_horizon=QUERY_STRIDE,
        transforms=[transform],
    )

    episode = dataset.episode_at(0)
    expected_first_query = np.stack([_frame_action(frame, 0) + 500.0 for frame in range(ACTION_HORIZON)], axis=0)
    expected_second_query = np.zeros((ACTION_HORIZON, ACTION_DIM), dtype=np.float32)
    expected_second_query[:7] = np.stack([_frame_action(frame, 0) + 500.0 for frame in range(5, 12)], axis=0)
    np.testing.assert_allclose(episode.actions[0], expected_first_query)
    np.testing.assert_allclose(episode.actions[1], expected_second_query)


def test_prompt_from_task_uses_sample_task_when_episode_task_index_is_absent():
    seen_prompts = []

    def capture_prompt(sample):
        seen_prompts.append(sample["prompt"])
        return sample

    dataset = _episode_loader.LeRobotEpisodeDataset(
        dataset=_FakeLeRobotDatasetWithSampleTask(),
        action_horizon=2,
        query_stride=QUERY_STRIDE,
        executed_horizon=2,
        transforms=[capture_prompt],
        prompt_from_task=True,
    )

    _ = dataset.episode_at(1)
    assert seen_prompts
    assert set(seen_prompts) == {"task from episode one"}


def test_episode_data_config_is_consumed_by_dataset_and_balanced_sampler_factories():
    episode_config = _config.EpisodeDataConfig(
        query_stride=3,
        executed_horizon=2,
        suite_weights={"mem": 1.0, "long": 9.0},
    )
    dataset = _episode_loader.create_lerobot_episode_dataset(
        data_config=_config.DataConfig(repo_id="fake"),
        episode_config=episode_config,
        action_horizon=4,
        dataset=_FakeLeRobotDataset((8,)),
    )

    assert dataset.query_stride == 3
    assert dataset.executed_horizon == 2
    assert [query.frame_index for query in dataset.episodes[0].queries] == [0, 3, 6]
    batch = _episode_loader.EpisodeCollator()([dataset.episode_at(0)])
    assert batch.executed_actions.shape == (1, 3, 2, ACTION_DIM)

    suites = [
        _episode_loader.QuerySuite(
            name="mem",
            weight=1.0,
            tasks=[_episode_loader.QueryTask("mem_task", [_episode_loader.QueryEpisode("mem_ep", 2)])],
        ),
        _episode_loader.QuerySuite(
            name="long",
            weight=1.0,
            tasks=[_episode_loader.QueryTask("long_task", [_episode_loader.QueryEpisode("long_ep", 2)])],
        ),
    ]
    sampler = _episode_loader.create_balanced_query_dataset(suites, episode_config=episode_config, seed=3)
    suite_counts = collections.Counter(sampler.record_at(index).suite_name for index in range(1000))
    assert suite_counts["long"] / sum(suite_counts.values()) > 0.85


def test_balanced_query_dataset_samples_suite_then_task_episode_and_query_uniformly_with_seed():
    suites = [
        _episode_loader.QuerySuite(
            name="mem",
            weight=1.0,
            tasks=[
                _episode_loader.QueryTask(
                    name="mem_task",
                    episodes=[
                        _episode_loader.QueryEpisode(episode_id="mem_short", num_queries=1),
                        _episode_loader.QueryEpisode(episode_id="mem_long", num_queries=6),
                    ],
                )
            ],
        ),
        _episode_loader.QuerySuite(
            name="long",
            weight=3.0,
            tasks=[
                _episode_loader.QueryTask(
                    name="long_a",
                    episodes=[_episode_loader.QueryEpisode(episode_id="long_a_ep", num_queries=2)],
                ),
                _episode_loader.QueryTask(
                    name="long_b",
                    episodes=[_episode_loader.QueryEpisode(episode_id="long_b_ep", num_queries=4)],
                ),
            ],
        ),
    ]

    first = _episode_loader.BalancedQueryDataset(suites, seed=7)
    second = _episode_loader.BalancedQueryDataset(suites, seed=7)
    first_records = [first.record_at(i) for i in range(4000)]
    second_records = [second.record_at(i) for i in range(4000)]

    assert first_records == second_records
    suite_counts = collections.Counter(record.suite_name for record in first_records)
    long_task_counts = collections.Counter(record.task_name for record in first_records if record.suite_name == "long")
    mem_episode_counts = collections.Counter(record.episode_id for record in first_records if record.suite_name == "mem")
    mem_query_counts = collections.Counter(
        record.query_index for record in first_records if record.episode_id == "mem_long"
    )

    assert 0.70 <= suite_counts["long"] / len(first_records) <= 0.80
    assert 0.45 <= long_task_counts["long_a"] / sum(long_task_counts.values()) <= 0.55
    assert 0.45 <= mem_episode_counts["mem_short"] / sum(mem_episode_counts.values()) <= 0.55
    assert max(mem_query_counts.values()) - min(mem_query_counts.values()) < 0.08 * sum(mem_query_counts.values())


def test_balanced_query_dataset_rejects_invalid_sampling_specs():
    with np.testing.assert_raises_regex(ValueError, "weight"):
        _episode_loader.BalancedQueryDataset([_episode_loader.QuerySuite(name="bad", weight=0.0, tasks=[])])
    with np.testing.assert_raises_regex(ValueError, "num_queries"):
        _episode_loader.BalancedQueryDataset(
            [
                _episode_loader.QuerySuite(
                    name="suite",
                    weight=1.0,
                    tasks=[
                        _episode_loader.QueryTask(
                            name="task", episodes=[_episode_loader.QueryEpisode("empty", num_queries=0)]
                        )
                    ],
                )
            ]
        )
