from __future__ import annotations

import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
from flax import struct
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as episode_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders

FUTUREMAMBA_PATH_FILTER = nnx_utils.PathRegex("futuremamba/.*")


@struct.dataclass
class TrainEpisodeBatch:
    observation: _model.Observation
    actions: jax.Array
    action_mask: jax.Array
    executed_actions: jax.Array
    executed_action_mask: jax.Array
    query_mask: jax.Array
    reset_mask: jax.Array
    episode_index: jax.Array


def _as_train_episode_batch(batch: episode_loader.EpisodeBatch | TrainEpisodeBatch) -> TrainEpisodeBatch:
    if isinstance(batch, TrainEpisodeBatch):
        return batch
    return TrainEpisodeBatch(
        observation=jax.tree.map(jnp.asarray, batch.observation),
        actions=jnp.asarray(batch.actions),
        action_mask=jnp.asarray(batch.action_mask),
        executed_actions=jnp.asarray(batch.executed_actions),
        executed_action_mask=jnp.asarray(batch.executed_action_mask),
        query_mask=jnp.asarray(batch.query_mask),
        reset_mask=jnp.asarray(batch.reset_mask),
        episode_index=jnp.asarray(batch.episode_index),
    )


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, log_code: bool = False, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return
    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(name=config.exp_name, config=dataclasses.asdict(config), project=config.project_name)
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)
    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def _path_to_str(path: tuple[Any, ...]) -> str:
    return "/".join(str(part) for part in path)


def trainable_param_paths(config: _config.TrainConfig, model: _model.BaseModel) -> tuple[str, ...]:
    return tuple(sorted(_path_to_str(path) for path in nnx.state(model, config.trainable_filter).flat_state()))


def validate_futuremamba_trainables(config: _config.TrainConfig, model: _model.BaseModel) -> tuple[str, ...]:
    paths = trainable_param_paths(config, model)
    bad_paths = [path for path in paths if not path.startswith("futuremamba/")]
    if bad_paths:
        raise ValueError(f"FutureMamba episode training found non-FutureMamba trainable parameter paths: {bad_paths}")
    if not paths:
        raise ValueError("FutureMamba episode training requires at least one trainable parameter under futuremamba/")
    return paths


def _trainable_param_count(params: nnx.State) -> jax.Array:
    leaves = jax.tree.leaves(params)
    return jnp.asarray(sum(int(np.prod(getattr(leaf, "value", leaf).shape)) for leaf in leaves), dtype=jnp.float32)


def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        validate_futuremamba_trainables(config, model)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        trainable_params = params.filter(config.trainable_filter)
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(trainable_params),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)
    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    return train_state, state_sharding


def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: TrainEpisodeBatch,
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.eval()

    def loss_fn(model: _model.BaseModel, rng: at.KeyArrayLike, batch: TrainEpisodeBatch):
        losses = model.compute_episode_loss(rng, batch, train=False, add_executed_action_noise=True)
        return losses["loss"], losses

    train_rng = jax.random.fold_in(rng, state.step)
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, losses), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model, train_rng, batch)
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_trainable_params = optax.apply_updates(params, updates)
    nnx.update(model, new_trainable_params)
    new_params = nnx.state(model)
    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )
    info = {
        "loss": loss,
        "flow_loss": losses["flow_loss"],
        "handoff_loss": losses["handoff_loss"],
        "handoff_error": losses["handoff_error"],
        "boundary_loss": losses["boundary_loss"],
        "boundary_error": losses["boundary_error"],
        "grad_norm": optax.global_norm(grads),
        "trainable_param_count": _trainable_param_count(params),
    }
    return new_state, info


def _create_episode_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None,
    shuffle: bool,
):
    return episode_loader.create_episode_data_loader(config, sharding=sharding, shuffle=shuffle)


def _episode_iterator_at_step(data_loader: episode_loader.DataLoader[episode_loader.EpisodeBatch], start_step: int):
    """Build a deterministic episode iterator aligned to the restored train step.

    checkpoints.restore_state does not restore data-loader cursors; episode loaders are seeded from config.seed,
    so replaying exactly train_state.step batches restores the next batch without consuming before restore.
    """
    if start_step < 0:
        raise ValueError(f"Cannot restore episode loader to negative train step {start_step}.")
    data_iter = iter(data_loader)
    for _ in range(start_step):
        next(data_iter)
    return data_iter


def restore_train_state_for_test(config: _config.TrainConfig, init_rng: at.KeyArrayLike):
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    data_loader = _create_episode_loader(config, sharding=data_sharding, shuffle=False)
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=False,
        resume=True,
    )
    train_state_shape, train_state_sharding = init_train_state(config, init_rng, mesh, resume=True)
    if not resuming:
        raise ValueError(f"No checkpoint found in {config.checkpoint_dir}")
    train_state = _checkpoints.restore_state(checkpoint_manager, train_state_shape, data_loader)
    return train_state, train_state_sharding, data_loader


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    if config.batch_size % jax.device_count() != 0:
        raise ValueError(f"Batch size {config.batch_size} must be divisible by device count {jax.device_count()}.")
    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    data_loader = _create_episode_loader(config, sharding=data_sharding, shuffle=True)
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")
    trainable_paths = trainable_param_paths(config, nnx.merge(train_state.model_def, train_state.params))
    logging.info("FutureMamba trainable params:\n%s", "\n".join(trainable_paths))
    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)
    # checkpoints.restore_state currently restores model/optimizer state but not an episode-loader cursor.
    # The episode loader is constructed from config.seed, so restore the cursor deterministically from the
    # restored train step. The loop saves before any lookahead, keeping train_state.step equal to batches consumed.
    start_step = int(train_state.step)
    data_iter = _episode_iterator_at_step(data_loader, start_step)
    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps), initial=start_step, total=config.num_train_steps, dynamic_ncols=True
    )
    infos = []
    for step in pbar:
        batch = _as_train_episode_batch(next(data_iter))
        if step == start_step:
            logging.info(f"Initialized episode data loader:\n{training_utils.array_tree_to_info(batch)}")
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{key}={value:.4f}" for key, value in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            if config.wandb_enabled:
                wandb.log(reduced_info, step=step)
            infos = []
        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)
    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
