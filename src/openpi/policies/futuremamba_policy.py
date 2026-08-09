from collections.abc import Sequence
import copy
import dataclasses
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from openpi_client import base_policy as _base_policy


class FutureMambaPolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        model: Any,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        _sample_actions_with_memory=None,
    ):
        self._model = model
        self._transforms = tuple(transforms)
        self._output_transforms = tuple(output_transforms)
        self._input_transform = _transforms.compose(self._transforms)
        self._output_transform = _transforms.compose(self._output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._base_rng = rng if rng is not None else jax.random.key(0)
        self._rng = self._base_rng
        self._next_fork_id = 0

        if not hasattr(model, "initial_memory_state"):
            raise AttributeError("FutureMambaPolicy model must define initial_memory_state(batch_size)")
        if _sample_actions_with_memory is not None:
            self._sample_actions_with_memory = _sample_actions_with_memory
        elif hasattr(model, "sample_actions_with_memory"):
            self._sample_actions_with_memory = _jit_if_possible(model.sample_actions_with_memory)
        else:
            raise AttributeError("FutureMambaPolicy model must define sample_actions_with_memory(...)")
        self.reset()

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if "executed_actions" not in inputs or "executed_action_mask" not in inputs:
            raise ValueError("FutureMambaPolicy input transforms must produce executed_actions and executed_action_mask")
        executed_actions = inputs.pop("executed_actions")
        executed_action_mask = inputs.pop("executed_action_mask")

        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        executed_actions = jnp.asarray(executed_actions)[np.newaxis, ...]
        executed_action_mask = jnp.asarray(executed_action_mask)[np.newaxis, ...]

        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        self._rng, sample_rng = jax.random.split(self._rng)
        start_time = time.monotonic()
        actions, next_state, diagnostics = self._sample_actions_with_memory(
            sample_rng,
            observation,
            self._memory_state,
            executed_actions,
            executed_action_mask,
            **sample_kwargs,
        )
        model_time = time.monotonic() - start_time
        self._memory_state = next_state

        outputs = {"state": inputs["state"], "actions": actions}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["handoff_step"] = _handoff_step_diagnostic(diagnostics)
        outputs["memory_state_bytes"] = _tree_nbytes(self._memory_state)
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    @override
    def reset(self) -> None:
        self._memory_state = self._model.initial_memory_state(1)

    @override
    def snapshot_state(self):
        return copy.deepcopy(self._memory_state)

    @override
    def restore_state(self, state) -> None:
        self._memory_state = copy.deepcopy(state)

    @override
    def fork(self) -> "FutureMambaPolicy":
        session_rng = jax.random.fold_in(self._base_rng, self._next_fork_id + 1)
        self._next_fork_id += 1
        return FutureMambaPolicy(
            self._model,
            rng=session_rng,
            transforms=self._transforms,
            output_transforms=self._output_transforms,
            sample_kwargs=self._sample_kwargs,
            metadata=self._metadata,
            _sample_actions_with_memory=self._sample_actions_with_memory,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


def _jit_if_possible(method):
    try:
        return nnx_utils.module_jit(method)
    except ValueError:
        return method


def _handoff_step_diagnostic(diagnostics):
    if not diagnostics:
        raise ValueError("FutureMamba diagnostics must include handoff_step")
    if "handoff_steps" in diagnostics:
        value = diagnostics["handoff_steps"]
    elif "handoff_step" in diagnostics:
        value = diagnostics["handoff_step"]
    else:
        raise ValueError("FutureMamba diagnostics must include handoff_step")
    if value is None:
        raise ValueError("FutureMamba diagnostics must include handoff_step")
    return _diagnostic_scalar(value)


def _diagnostic_scalar(value):
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape == ():
        return array.item()
    return array.reshape(-1)[0].item()


def _tree_nbytes(value) -> int:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return sum(_tree_nbytes(getattr(value, field.name)) for field in dataclasses.fields(value))
    if isinstance(value, dict):
        return sum(_tree_nbytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tree_nbytes(item) for item in value)
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return 0
    if array.dtype == object:
        return 0
    return int(array.nbytes)
