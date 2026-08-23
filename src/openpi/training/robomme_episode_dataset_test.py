from __future__ import annotations

import dataclasses
import pickle

import numpy as np
import pytest

from openpi.training import robomme_episode_dataset as _robomme_episode


ACTION_HORIZON = 20
ACTION_DIM = 3


def _sample(
    epis_idx: int,
    step_idx: int,
    *,
    exec_start_idx: int = 0,
    action_dim: int = ACTION_DIM,
    action_steps: int = ACTION_HORIZON,
    epis_idx_value=None,
    step_idx_value=None,
    exec_start_value=None,
    is_demo: bool = False,
) -> dict:
    actions = np.stack(
        [
            np.asarray(
                [1000.0 * epis_idx + 10.0 * step_idx + float(t) + 0.1 * dim for dim in range(action_dim)],
                dtype=np.float32,
            )
            for t in range(action_steps)
        ],
        axis=0,
    )
    return {
        "epis_idx": np.asarray([epis_idx], dtype=np.int32) if epis_idx_value is None else epis_idx_value,
        "step_idx": np.asarray([step_idx], dtype=np.int32) if step_idx_value is None else step_idx_value,
        "exec_start_idx": np.asarray([exec_start_idx], dtype=np.int32)
        if exec_start_value is None
        else exec_start_value,
        "image": np.full((2, 2, 3), step_idx, dtype=np.uint8),
        "wrist_image": np.full((2, 2, 3), step_idx + 1, dtype=np.uint8),
        "state": np.asarray([epis_idx, step_idx], dtype=np.float32),
        "actions": actions,
        "is_demo": np.asarray([is_demo], dtype=np.bool_),
        "prompt": f"episode {epis_idx} step {step_idx}",
        "simple_subgoal": f"simple {epis_idx}-{step_idx}",
        "grounded_subgoal": f"grounded {epis_idx}-{step_idx}",
        "simple_subgoal_online": f"simple online {epis_idx}-{step_idx}",
        "grounded_subgoal_online": f"grounded online {epis_idx}-{step_idx}",
    }


def _write_pickle(data_dir, file_index: int, sample: dict) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / f"{file_index}.pkl").open("wb") as f:
        pickle.dump(sample, f)


def test_sorts_pickles_by_episode_and_step_and_preserves_payloads(tmp_path):
    data_dir = tmp_path / "data"
    _write_pickle(data_dir, 0, _sample(1, 6, exec_start_idx=5))
    _write_pickle(data_dir, 1, _sample(0, 1))
    _write_pickle(data_dir, 2, _sample(1, 5, exec_start_idx=5))
    _write_pickle(data_dir, 3, _sample(0, 0))

    dataset = _robomme_episode.RoboMMEEpisodeDataset(data_dir, window_queries=2)

    assert [(ref.epis_idx, ref.step_idx) for ref in dataset.sample_refs] == [(0, 0), (0, 1), (1, 5), (1, 6)]
    assert len(dataset) == 4
    assert [(ref.epis_idx, ref.step_idx) for ref in dataset[0].sample_refs] == [(0, 0), (0, 1)]
    assert [(ref.epis_idx, ref.step_idx) for ref in dataset[2].sample_refs] == [(1, 5), (1, 6)]
    assert dataset[2].samples[0]["prompt"] == "episode 1 step 5"


def test_windows_pad_queries_without_crossing_episode_and_mask_only_query_padding(tmp_path):
    data_dir = tmp_path / "data"
    _write_pickle(data_dir, 0, _sample(0, 0))
    _write_pickle(data_dir, 1, _sample(0, 1))
    _write_pickle(data_dir, 2, _sample(1, 0))
    _write_pickle(data_dir, 3, _sample(1, 1))
    _write_pickle(data_dir, 4, _sample(1, 2))

    dataset = _robomme_episode.RoboMMEEpisodeDataset(data_dir, window_queries=3)

    first = dataset[0]
    np.testing.assert_array_equal(first.query_mask, [True, True, False])
    assert [(ref.epis_idx, ref.step_idx) for ref in first.sample_refs] == [(0, 0), (0, 1)]
    np.testing.assert_array_equal(first.action_mask[0], [True] * ACTION_HORIZON)
    np.testing.assert_array_equal(first.action_mask[1], [True] * ACTION_HORIZON)
    assert not first.actions[2].any()
    assert not first.action_mask[2].any()

    second = dataset[1]
    np.testing.assert_array_equal(second.query_mask, [True, False, False])
    assert [(ref.epis_idx, ref.step_idx) for ref in second.sample_refs] == [(0, 1)]
    np.testing.assert_array_equal(second.action_mask[0], [True] * ACTION_HORIZON)
    assert not second.actions[1:].any()
    assert not second.action_mask[1:].any()

    third = dataset[2]
    assert [(ref.epis_idx, ref.step_idx) for ref in third.sample_refs] == [(1, 0), (1, 1), (1, 2)]


@pytest.mark.parametrize(
    "samples, error",
    [
        ([_sample(0, 0), _sample(0, 0)], "duplicate"),
        ([_sample(0, 0), _sample(0, 2)], "continuous"),
        ([_sample(0, 1, exec_start_idx=0), _sample(0, 2, exec_start_idx=0)], "exec_start_idx"),
    ],
)
def test_rejects_duplicate_gapped_or_misaligned_episode_steps(tmp_path, samples, error):
    data_dir = tmp_path / "data"
    for index, sample in enumerate(samples):
        _write_pickle(data_dir, index, sample)

    with pytest.raises(ValueError, match=error):
        _robomme_episode.RoboMMEEpisodeDataset(data_dir, window_queries=2)


def test_middle_windows_expose_burn_in_refs_and_step_indices(tmp_path):
    data_dir = tmp_path / "data"
    for step_idx in range(4):
        _write_pickle(data_dir, step_idx, _sample(0, step_idx))

    dataset = _robomme_episode.RoboMMEEpisodeDataset(data_dir, window_queries=2)

    assert dataset[0].burn_in_refs == ()
    assert dataset[0].burn_in_step_indices == ()

    window = dataset[2]
    assert window.burn_in_step_indices == (0, 1)
    assert [(ref.epis_idx, ref.step_idx) for ref in window.burn_in_refs] == [(0, 0), (0, 1)]
    assert dataset.sample_payload(window.burn_in_refs[0])["prompt"] == "episode 0 step 0"


def test_query_stride_matches_online_execution_cadence(tmp_path):
    data_dir = tmp_path / "data"
    for step_idx in range(50):
        _write_pickle(data_dir, step_idx, _sample(0, step_idx))

    dataset = _robomme_episode.RoboMMEEpisodeDataset(
        data_dir,
        window_queries=2,
        query_stride=16,
    )

    assert [(ref.epis_idx, ref.step_idx) for ref in dataset.sample_refs] == [(0, 0), (0, 16), (0, 32), (0, 48)]
    assert len(dataset) == 4
    assert dataset[1].burn_in_step_indices == (0,)
    assert dataset[1].train_step_indices == (16, 32)


@pytest.mark.parametrize("action_steps", [19, 21])
def test_rejects_non_twenty_step_action_chunks(tmp_path, action_steps):
    data_dir = tmp_path / "data"
    _write_pickle(data_dir, 0, _sample(0, 0, action_steps=action_steps))

    with pytest.raises(ValueError, match="20"):
        _robomme_episode.RoboMMEEpisodeDataset(data_dir)


def test_reset_mask_marks_only_episode_start_query(tmp_path):
    data_dir = tmp_path / "data"
    for step_idx in range(3):
        _write_pickle(data_dir, step_idx, _sample(0, step_idx))

    dataset = _robomme_episode.RoboMMEEpisodeDataset(data_dir, window_queries=2)

    np.testing.assert_array_equal(dataset[0].reset_mask, [True, False])
    np.testing.assert_array_equal(dataset[1].reset_mask, [False, False])
    np.testing.assert_array_equal(dataset[2].reset_mask, [False, False])


def test_returned_payloads_cannot_pollute_overlapping_windows(tmp_path):
    data_dir = tmp_path / "data"
    for step_idx in range(3):
        _write_pickle(data_dir, step_idx, _sample(0, step_idx))

    dataset = _robomme_episode.RoboMMEEpisodeDataset(data_dir, window_queries=2)
    first = dataset[0]
    overlapping = first.samples[1]
    original_action = float(overlapping["actions"][0, 0])

    overlapping["prompt"] = "polluted"
    with pytest.raises(ValueError, match="read-only"):
        overlapping["actions"][0, 0] = -123.0

    payload = dataset.sample_payload(first.sample_refs[1])
    payload["prompt"] = "also polluted"
    with pytest.raises(ValueError, match="read-only"):
        payload["actions"][0, 0] = -456.0

    second = dataset[1]
    assert second.samples[0]["prompt"] == "episode 0 step 1"
    assert float(second.samples[0]["actions"][0, 0]) == original_action


def test_rejects_invalid_constructor_arguments_empty_dirs_and_bad_samples(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with pytest.raises(ValueError, match="window_queries"):
        _robomme_episode.RoboMMEEpisodeDataset(data_dir, window_queries=0)
    with pytest.raises(ValueError, match="action_horizon"):
        _robomme_episode.RoboMMEEpisodeDataset(data_dir, action_horizon=10)
    with pytest.raises(ValueError, match="No RoboMME pickle samples"):
        _robomme_episode.RoboMMEEpisodeDataset(data_dir)

    missing_dir = tmp_path / "missing"
    missing = _sample(0, 0)
    del missing["simple_subgoal"]
    _write_pickle(missing_dir, 0, missing)
    with pytest.raises(ValueError, match="simple_subgoal"):
        _robomme_episode.RoboMMEEpisodeDataset(missing_dir)

    demo_dir = tmp_path / "demo"
    _write_pickle(demo_dir, 0, _sample(0, 0, is_demo=True))
    with pytest.raises(ValueError, match="is_demo"):
        _robomme_episode.RoboMMEEpisodeDataset(demo_dir)


def test_rejects_action_width_mismatch_across_episodes(tmp_path):
    data_dir = tmp_path / "data"
    _write_pickle(data_dir, 0, _sample(0, 0, action_dim=3))
    _write_pickle(data_dir, 1, _sample(1, 0, action_dim=4))

    with pytest.raises(ValueError, match="action width"):
        _robomme_episode.RoboMMEEpisodeDataset(data_dir)


@pytest.mark.parametrize(
    "sample_kwargs",
    [
        {"epis_idx_value": np.asarray([0.0], dtype=np.float32)},
        {"step_idx_value": np.asarray(["0"])},
        {"exec_start_value": "0"},
    ],
)
def test_rejects_non_integer_metadata_scalars(tmp_path, sample_kwargs):
    data_dir = tmp_path / "data"
    _write_pickle(data_dir, 0, _sample(0, 0, **sample_kwargs))

    with pytest.raises(ValueError, match="integer"):
        _robomme_episode.RoboMMEEpisodeDataset(data_dir)

def test_full_episodes_use_variable_length_sequences(tmp_path):
    data_dir = tmp_path / "data"
    for step_idx in range(5):
        _write_pickle(data_dir, step_idx, _sample(0, step_idx))

    dataset = _robomme_episode.RoboMMEEpisodeDataset(
        data_dir,
        window_queries=2,
        query_stride=1,
        full_episodes=True,
    )

    assert len(dataset) == 1
    window = dataset[0]
    assert len(window.samples) == 5
    assert window.burn_in_refs == ()
    assert window.train_step_indices == (0, 1, 2, 3, 4)
    assert window.padding_query_indices == ()
    np.testing.assert_array_equal(window.query_mask, [True, True, True, True, True])
    np.testing.assert_array_equal(window.reset_mask, [True, False, False, False, False])


def test_windows_expose_burn_in_train_padding_and_detach_contract(tmp_path):
    data_dir = tmp_path / "data"
    for step_idx in range(4):
        _write_pickle(data_dir, step_idx, _sample(0, step_idx))

    dataset = _robomme_episode.RoboMMEEpisodeDataset(data_dir, window_queries=3)

    first = dataset[0]
    assert first.burn_in_reset_mask == ()
    assert first.train_step_indices == (0, 1, 2)
    assert first.train_query_slice == slice(0, 3)
    assert first.padding_query_indices == ()
    assert first.reset_before_burn_in is True
    assert first.detach_after is True

    middle = dataset[2]
    assert middle.burn_in_step_indices == (0, 1)
    assert middle.burn_in_reset_mask == (True, False)
    assert middle.train_step_indices == (2, 3, None)
    assert middle.train_query_slice == slice(0, 2)
    assert middle.padding_query_indices == (2,)
    assert middle.reset_before_burn_in is True
    assert middle.detach_after is True

def test_transformed_middle_window_includes_burn_in_and_marks_only_train_queries(tmp_path):
    from types import SimpleNamespace

    from openpi import transforms
    from openpi.policies.robomme_policy import RoboMMEInputs
    from openpi.training import episode_data_loader

    data_dir = tmp_path / "data"
    for step_idx in range(4):
        sample = _sample(0, step_idx, action_dim=8)
        sample["state"] = np.arange(8, dtype=np.float32) + step_idx
        _write_pickle(data_dir, step_idx, sample)

    data_config = SimpleNamespace(
        repo_id="fake",
        norm_stats=None,
        use_quantile_norm=False,
        repack_transforms=transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                    }
                )
            ]
        ),
        data_transforms=transforms.Group(inputs=[RoboMMEInputs()]),
        model_transforms=transforms.Group(inputs=[transforms.PadStatesAndActions(32)]),
    )
    model_config = SimpleNamespace(action_horizon=20, action_dim=32, execution_horizon=16)
    windows = _robomme_episode.RoboMMEEpisodeDataset(data_dir, window_queries=2)
    dataset = _robomme_episode.RoboMMETransformedEpisodeDataset(windows, data_config, model_config)

    example = dataset[2]
    np.testing.assert_array_equal(example.train_query_mask, [False, False, True, True])
    assert example.actions.shape == (4, 20, 32)
    assert example.executed_actions.shape == (4, 16, 32)

    batch = episode_data_loader.EpisodeCollator()([example])
    np.testing.assert_array_equal(batch.query_mask, [[True, True, True, True]])
    np.testing.assert_array_equal(batch.train_query_mask, [[False, False, True, True]])
    np.testing.assert_array_equal(batch.reset_mask, [[True, False, False, False]])

def test_real_robomme_config_accepts_official_minimal_pickle(tmp_path):
    from openpi.models_pytorch.futuremamba_config import FutureMambaPytorchConfig
    from openpi.training import config as training_config

    data_dir = tmp_path / "data"
    sample = _sample(0, 0, action_dim=8)
    sample["state"] = np.arange(8, dtype=np.float32)
    _write_pickle(data_dir, 0, sample)
    model_config = FutureMambaPytorchConfig(
        action_horizon=20,
        discrete_state_input=False,
        execution_horizon=16,
    )
    data_config = training_config.RoboMMEDataConfig(
        repo_id="robomme", episode_data_dir=str(data_dir)
    ).create(tmp_path, model_config)
    data_config = dataclasses.replace(data_config, repo_id="fake")

    dataset = _robomme_episode.create_robomme_episode_dataset(
        data_config,
        model_config,
        training_config.EpisodeDataConfig(window_queries=2, executed_horizon=16),
    )
    example = dataset[0]

    assert example.actions.shape == (1, 20, 32)
    np.testing.assert_array_equal(example.train_query_mask, [True])
