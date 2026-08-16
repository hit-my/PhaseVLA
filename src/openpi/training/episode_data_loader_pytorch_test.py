from __future__ import annotations

import numpy as np
import pytest
import torch

import openpi.models.model as _model
from openpi.training import episode_data_loader as _episode_loader


ACTION_HORIZON = 2
ACTION_DIM = 3
EXECUTED_HORIZON = 1


def _observation(batch_size: int, num_queries: int) -> _model.Observation:
    return _model.Observation(
        images={"cam": np.ones((batch_size, num_queries, 2, 2, 3), dtype=np.float32)},
        image_masks={"cam": np.ones((batch_size, num_queries), dtype=np.bool_)},
        state=np.arange(batch_size * num_queries * 4, dtype=np.float32).reshape(batch_size, num_queries, 4),
        tokenized_prompt=np.zeros((batch_size, num_queries, 5), dtype=np.int32),
        tokenized_prompt_mask=np.ones((batch_size, num_queries, 5), dtype=np.bool_),
    )


def _episode_batch(*, batch_size: int = 2, num_queries: int = 3, query_mask: np.ndarray | None = None) -> _episode_loader.EpisodeBatch:
    if query_mask is None:
        query_mask = np.ones((batch_size, num_queries), dtype=np.bool_)
    return _episode_loader.EpisodeBatch(
        observation=_observation(batch_size, num_queries),
        actions=np.arange(batch_size * num_queries * ACTION_HORIZON * ACTION_DIM, dtype=np.float64).reshape(
            batch_size, num_queries, ACTION_HORIZON, ACTION_DIM
        ),
        action_mask=np.ones((batch_size, num_queries, ACTION_HORIZON), dtype=np.bool_),
        executed_actions=np.ones((batch_size, num_queries, EXECUTED_HORIZON, ACTION_DIM), dtype=np.float64),
        executed_action_mask=np.ones((batch_size, num_queries, EXECUTED_HORIZON), dtype=np.bool_),
        query_mask=query_mask,
        reset_mask=np.pad(np.ones((batch_size, 1), dtype=np.bool_), ((0, 0), (0, max(num_queries - 1, 0)))),
        episode_index=np.arange(batch_size, dtype=np.int32),
    )


def test_episode_batch_to_torch_converts_observation_tree_and_batch_fields_on_device():
    query_mask = np.asarray([[True, True, False], [True, False, False]], dtype=np.bool_)
    batch = _episode_batch(query_mask=query_mask)

    converted = _episode_loader.episode_batch_to_torch(batch, torch.device("cpu"))

    assert isinstance(converted, _episode_loader.TorchEpisodeBatch)
    assert converted.observation.state.shape == (2, 3, 4)
    assert converted.observation.state.device.type == "cpu"
    assert converted.observation.images["cam"].device.type == "cpu"
    assert converted.observation.images["cam"].dtype == torch.float32
    assert converted.observation.image_masks["cam"].dtype is torch.bool
    assert converted.actions.dtype is torch.float32
    assert converted.executed_actions.dtype is torch.float32
    assert converted.action_mask.dtype is torch.bool
    assert converted.executed_action_mask.dtype is torch.bool
    assert converted.query_mask.dtype is torch.bool
    assert converted.reset_mask.dtype is torch.bool
    torch.testing.assert_close(converted.query_mask, torch.as_tensor(query_mask, dtype=torch.bool))


def test_episode_batch_to_torch_rejects_empty_batch():
    batch = _episode_batch(batch_size=0, num_queries=1, query_mask=np.zeros((0, 1), dtype=np.bool_))

    with pytest.raises(ValueError, match="empty"):
        _episode_loader.episode_batch_to_torch(batch, torch.device("cpu"))


def test_episode_batch_to_torch_rejects_zero_query_axis():
    batch = _episode_batch(batch_size=1, num_queries=0, query_mask=np.zeros((1, 0), dtype=np.bool_))

    with pytest.raises(ValueError, match="zero queries"):
        _episode_loader.episode_batch_to_torch(batch, torch.device("cpu"))


def test_episode_batch_to_torch_rejects_non_prefix_query_masks():
    batch = _episode_batch(batch_size=1, num_queries=3, query_mask=np.asarray([[True, False, True]], dtype=np.bool_))

    with pytest.raises(ValueError, match="right-side padded prefix"):
        _episode_loader.episode_batch_to_torch(batch, torch.device("cpu"))
