from __future__ import annotations

import copy
import dataclasses
import hashlib
import pathlib

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.models import futuremamba_config as _futuremamba_config
from openpi.models.mamba import MambaConfig
from openpi.shared import normalize as _normalize
from openpi.training import config as _config
from openpi.training import episode_data_loader as _episode_loader
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders as _weight_loaders

from . import train_futuremamba


def _norm_stats() -> dict[str, _normalize.NormStats]:
    return {
        "state": _normalize.NormStats(
            mean=np.asarray([1.0, 2.0], dtype=np.float32),
            std=np.asarray([3.0, 4.0], dtype=np.float32),
            q01=np.asarray([0.0, 1.0], dtype=np.float32),
            q99=np.asarray([2.0, 3.0], dtype=np.float32),
        ),
        "actions": _normalize.NormStats(
            mean=np.asarray([5.0, 6.0], dtype=np.float32),
            std=np.asarray([7.0, 8.0], dtype=np.float32),
            q01=np.asarray([4.0, 5.0], dtype=np.float32),
            q99=np.asarray([6.0, 7.0], dtype=np.float32),
        ),
    }


def _norm_checksum(norm_stats: dict[str, _normalize.NormStats]) -> str:
    digest = hashlib.sha256()
    for key in sorted(norm_stats):
        digest.update(key.encode())
        stats = norm_stats[key]
        for value in (stats.mean, stats.std, stats.q01, stats.q99):
            if value is not None:
                array = np.asarray(value)
                digest.update(str(array.dtype).encode())
                digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
                digest.update(array.tobytes())
    return digest.hexdigest()


def _tree_checksum(tree, *, include_path) -> str:
    digest = hashlib.sha256()
    for path, value in sorted(nnx.state(tree, nnx.Param).flat_state().items(), key=lambda item: item[0]):
        path_str = "/".join(str(part) for part in path)
        if not include_path(path_str):
            continue
        array = np.asarray(value.value)
        digest.update(path_str.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


class _TinyEpisodeModel(nnx.Module):
    action_horizon = 2
    action_dim = 2

    def __init__(self, rngs: nnx.Rngs):
        self.base = nnx.Linear(2, 2, rngs=rngs)
        self.futuremamba = nnx.Linear(2, 2, rngs=rngs)

    def eval(self):
        self.training = False

    def compute_episode_loss(self, rng, batch, *, train: bool = False, add_executed_action_noise: bool = False):
        if train:
            raise AssertionError("FutureMamba episode trainer must keep the model in eval mode")
        if not add_executed_action_noise:
            raise AssertionError("FutureMamba episode trainer must request executed-action noise")
        del rng
        x = jnp.mean(batch.executed_actions, axis=(1, 2))
        target = jnp.mean(batch.actions, axis=(1, 2))
        frozen = jax.lax.stop_gradient(self.base(x))
        pred = self.futuremamba(frozen)
        flow_loss = jnp.mean(jnp.square(pred - target))
        handoff_loss = jnp.asarray(0.25, dtype=jnp.float32) * flow_loss
        handoff_error = flow_loss + jnp.asarray(0.5, dtype=jnp.float32)
        boundary_loss = jnp.asarray(0.125, dtype=jnp.float32) * flow_loss
        boundary_error = flow_loss + jnp.asarray(0.25, dtype=jnp.float32)
        return {
            "loss": flow_loss + handoff_loss + boundary_loss,
            "flow_loss": flow_loss,
            "handoff_loss": handoff_loss,
            "handoff_error": handoff_error,
            "boundary_loss": boundary_loss,
            "boundary_error": boundary_error,
        }


@dataclasses.dataclass(frozen=True)
class _TinyEpisodeModelConfig:
    action_horizon: int = 2
    action_dim: int = 2

    def create(self, rng):
        return _TinyEpisodeModel(nnx.Rngs(rng))

    @property
    def model_type(self):
        return _config.ModelType.PI05

    def get_freeze_filter(self):
        return nnx.All(nnx.Param, nnx.Not(train_futuremamba.FUTUREMAMBA_PATH_FILTER))


@dataclasses.dataclass(frozen=True)
class _FakeEpisodeDataConfig(_config.DataConfigFactory):
    repo_id: str = "fake"

    def create(self, assets_dirs: pathlib.Path, model_config):
        del assets_dirs, model_config
        return _config.DataConfig(repo_id="fake")


def _episode_batch(step_offset: float = 0.0):
    observation = train_futuremamba.episode_loader._model.Observation(
        images={"base_0_rgb": np.zeros((2, 1, 1, 1, 3), dtype=np.float32)},
        image_masks={"base_0_rgb": np.ones((2, 1), dtype=np.bool_)},
        state=np.zeros((2, 1, 2), dtype=np.float32),
    )
    actions = np.asarray(
        [
            [[1.0 + step_offset, -1.0], [0.5, 0.0]],
            [[-0.5, 1.5 + step_offset], [0.0, 0.25]],
        ],
        dtype=np.float32,
    )[:, None, :, :]
    executed_actions = np.asarray(
        [
            [[0.25, -0.25], [0.5 + step_offset, 0.25]],
            [[-0.25, 0.5], [0.75, -0.5 - step_offset]],
        ],
        dtype=np.float32,
    )[:, None, :, :]
    return train_futuremamba.episode_loader.EpisodeBatch(
        observation=observation,
        actions=actions,
        action_mask=np.ones((2, 1, 2), dtype=np.bool_),
        executed_actions=executed_actions,
        executed_action_mask=np.ones((2, 1, 2), dtype=np.bool_),
        query_mask=np.ones((2, 1), dtype=np.bool_),
        reset_mask=np.asarray([[True], [True]], dtype=np.bool_),
        episode_index=np.asarray([0, 1], dtype=np.int32),
    )


class _FiniteEpisodeLoader:
    def __init__(self, data_config):
        self._data_config = data_config
        self._batches = [_episode_batch(0.0), _episode_batch(0.5), _episode_batch(1.0), _episode_batch(1.5)]

    def data_config(self):
        return self._data_config

    def __iter__(self):
        while True:
            yield from self._batches


def _tiny_config(tmp_path: pathlib.Path, *, resume: bool = False, num_train_steps: int = 2) -> _config.TrainConfig:
    model = _TinyEpisodeModelConfig()
    return _config.TrainConfig(
        name="tiny_futuremamba_episode",
        exp_name="smoke",
        model=model,
        data=_FakeEpisodeDataConfig(),
        weight_loader=_weight_loaders.NoOpWeightLoader(),
        freeze_filter=model.get_freeze_filter(),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1, peak_lr=1e-2, decay_steps=10, decay_lr=1e-2),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=None,
        batch_size=2,
        num_train_steps=num_train_steps,
        save_interval=1,
        log_interval=1,
        overwrite=not resume,
        resume=resume,
        wandb_enabled=False,
        checkpoint_base_dir=str(tmp_path / "checkpoints"),
        assets_base_dir=str(tmp_path / "assets"),
        num_workers=0,
    )


def test_futuremamba_configs_share_norm_asset_and_freeze_only_stage_b_plugin():
    base = _config.get_config("pi05_futuremamba_base")
    stage_b = _config.get_config("futuremamba_libero_mem")

    assert base.model.pi05 is True
    assert base.model.discrete_state_input is True
    assert base.model.action_horizon == 10
    assert base.data.assets.asset_id == stage_b.data.assets.asset_id == "futuremamba/libero_mem_long_train"
    assert base.episode_data.suite_weights == {"LIBERO-Mem": 0.5, "LIBERO-Long": 0.5}
    assert stage_b.episode_data.suite_weights == {"LIBERO-Mem": 0.5, "LIBERO-Long": 0.5}

    assert isinstance(stage_b.model, _futuremamba_config.FutureMambaConfig)
    assert stage_b.model.action_horizon == 10
    assert stage_b.model.executed_horizon == 5
    assert stage_b.model.num_denoise_steps == 10
    assert stage_b.model.handoff_ratio == 0.2
    assert stage_b.model.progress_depth == 4
    assert stage_b.freeze_filter == stage_b.model.get_freeze_filter()
    assert isinstance(stage_b.weight_loader, _weight_loaders.LatestCheckpointWeightLoader)


def test_futuremamba_configs_load_identical_norm_stats_checksum(tmp_path):
    asset_id = "futuremamba/libero_mem_long_train"
    _normalize.save(tmp_path / "pi05_futuremamba_base" / asset_id, _norm_stats())
    base = dataclasses.replace(_config.get_config("pi05_futuremamba_base"), assets_base_dir=str(tmp_path))
    stage_b = dataclasses.replace(_config.get_config("futuremamba_libero_mem"), assets_base_dir=str(tmp_path))

    base_stats = base.data.create(base.assets_dirs, base.model).norm_stats
    stage_b_stats = stage_b.data.create(stage_b.assets_dirs, stage_b.model).norm_stats

    assert _norm_checksum(base_stats) == _norm_checksum(stage_b_stats)


def test_balanced_query_sampling_not_biased_by_episode_length_with_equal_suite_weights():
    episode_config = _config.get_config("futuremamba_libero_mem").episode_data
    sampler = _episode_loader.create_balanced_query_dataset(
        [
            _episode_loader.QuerySuite(
                "LIBERO-Mem",
                1.0,
                [_episode_loader.QueryTask("mem", [_episode_loader.QueryEpisode("short", 1)])],
            ),
            _episode_loader.QuerySuite(
                "LIBERO-Long",
                1.0,
                [_episode_loader.QueryTask("long", [_episode_loader.QueryEpisode("long", 100)])],
            ),
        ],
        episode_config=episode_config,
        seed=11,
    )

    counts = {"LIBERO-Mem": 0, "LIBERO-Long": 0}
    for index in range(2000):
        counts[sampler.record_at(index).suite_name] += 1

    assert abs(counts["LIBERO-Mem"] - counts["LIBERO-Long"]) < 140


def test_trainable_path_validation_rejects_non_futuremamba_parameters():
    config = dataclasses.replace(_tiny_config(pathlib.Path("/tmp/unused")), freeze_filter=nnx.Nothing)
    model = config.model.create(jax.random.key(0))

    with pytest.raises(ValueError, match="non-FutureMamba trainable parameter.*base"):
        train_futuremamba.validate_futuremamba_trainables(config, model)


def test_two_step_fake_episode_training_updates_plugin_only_and_resumes_to_step_four(monkeypatch, tmp_path):
    monkeypatch.setattr(train_futuremamba, "init_wandb", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        train_futuremamba.episode_loader,
        "create_episode_data_loader",
        lambda config, sharding=None, shuffle=False, num_batches=None, skip_norm_stats=False: _FiniteEpisodeLoader(
            config.data.create(config.assets_dirs, config.model)
        ),
    )
    config = _tiny_config(tmp_path)
    _, init_rng = jax.random.split(jax.random.key(config.seed))
    _, model_rng = jax.random.split(init_rng)
    initial_model = config.model.create(model_rng)
    base_before = _tree_checksum(initial_model, include_path=lambda path: not path.startswith("futuremamba/"))
    plugin_before = _tree_checksum(initial_model, include_path=lambda path: path.startswith("futuremamba/"))

    train_futuremamba.main(config)
    train_state, _, data_loader = train_futuremamba.restore_train_state_for_test(
        dataclasses.replace(config, overwrite=False, resume=True), jax.random.key(config.seed)
    )
    model_after = nnx.merge(train_state.model_def, train_state.params)

    assert int(train_state.step) == 2
    assert _tree_checksum(model_after, include_path=lambda path: not path.startswith("futuremamba/")) == base_before
    assert _tree_checksum(model_after, include_path=lambda path: path.startswith("futuremamba/")) != plugin_before

    train_futuremamba.main(dataclasses.replace(config, overwrite=False, resume=True, num_train_steps=4))
    resumed_state, _, _ = train_futuremamba.restore_train_state_for_test(
        dataclasses.replace(config, overwrite=False, resume=True, num_train_steps=4), jax.random.key(config.seed)
    )

    assert int(resumed_state.step) == 4
    assert data_loader.data_config().repo_id == "fake"
