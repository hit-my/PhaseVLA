import dataclasses

import jax
import numpy as np
import torch

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))

def test_torch_data_loader_preserves_episode_batch_dataclass():
    from openpi.models import model as _model
    from openpi.training import episode_data_loader as _episode_loader

    example = _episode_loader.EpisodeExample(
        observation=_model.Observation(
            images={"cam": np.zeros((1, 2, 2, 3), dtype=np.float32)},
            image_masks={"cam": np.ones((1,), dtype=np.bool_)},
            state=np.zeros((1, 8), dtype=np.float32),
            tokenized_prompt=np.zeros((1, 4), dtype=np.int32),
            tokenized_prompt_mask=np.ones((1, 4), dtype=np.bool_),
        ),
        actions=np.zeros((1, 2, 8), dtype=np.float32),
        action_mask=np.ones((1, 2), dtype=np.bool_),
        executed_actions=np.zeros((1, 1, 8), dtype=np.float32),
        executed_action_mask=np.zeros((1, 1), dtype=np.bool_),
        episode_index=0,
    )
    loader = _data_loader.TorchDataLoader(
        [example],
        local_batch_size=1,
        num_batches=1,
        framework="pytorch",
        collate_fn=_episode_loader.EpisodeCollator(),
    )

    batch = next(iter(loader))

    assert isinstance(batch, _episode_loader.TorchEpisodeBatch)
    assert torch.is_tensor(batch.actions)
    assert torch.is_tensor(batch.observation.state)


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)
