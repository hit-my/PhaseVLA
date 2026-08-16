from __future__ import annotations

import dataclasses

import numpy as np

from openpi import transforms
from openpi.models import model as _model


_ROBOMME_IMAGE_SIZE = 224
_ROBOMME_STATE_DIM = 8
_ROBOMME_ACTION_DIM = 8
_ROBOMME_PREDICTION_HORIZON = 20


def _parse_image(image: object) -> np.ndarray:
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.floating):
        if np.any(~np.isfinite(array)) or np.min(array) < 0.0 or np.max(array) > 1.0:
            raise ValueError("RoboMME image floats must be finite and normalized to [0, 1]")
        array = (255.0 * array).astype(np.uint8)
    if array.ndim != 3:
        raise ValueError(f"RoboMME image must be rank 3, got shape {array.shape}")
    if array.shape[0] == 3 and array.shape[-1] != 3:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] != 3:
        raise ValueError(f"RoboMME image must have 3 channels, got shape {array.shape}")
    return np.ascontiguousarray(array)


@dataclasses.dataclass(frozen=True)
class RoboMMEInputs(transforms.DataTransformFn):
    """Map official RoboMME observations and optional memory fields to OpenPI keys."""
    model_type: _model.ModelType = _model.ModelType.PI05

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["observation/state"])
        if state.shape != (_ROBOMME_STATE_DIM,):
            raise ValueError(f"RoboMME state must have shape ({_ROBOMME_STATE_DIM},), got {state.shape}")
        image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
            "static_image_emb": data.get("static_image_emb"),
            "static_pos_emb": data.get("static_pos_emb"),
            "static_state_emb": data.get("static_state_emb"),
            "static_mask": data.get("static_mask"),
            "recur_image_emb": data.get("recur_image_emb"),
            "recur_pos_emb": data.get("recur_pos_emb"),
            "recur_state_emb": data.get("recur_state_emb"),
            "recur_mask": data.get("recur_mask"),
            "simple_subgoal": data.get("simple_subgoal"),
            "grounded_subgoal": data.get("grounded_subgoal"),
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class RoboMMEOutputs(transforms.DataTransformFn):
    """Map padded OpenPI actions to RoboMME joint-angle actions."""

    prediction_horizon: int = _ROBOMME_PREDICTION_HORIZON

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if actions.ndim != 2 or actions.shape[0] != self.prediction_horizon:
            raise ValueError(
                f"RoboMME actions must have shape ({self.prediction_horizon}, action_dim), got {actions.shape}"
            )
        if actions.shape[1] < _ROBOMME_ACTION_DIM:
            raise ValueError(f"RoboMME actions need at least {_ROBOMME_ACTION_DIM} dimensions, got {actions.shape}")
        return {"actions": np.asarray(actions[:, :_ROBOMME_ACTION_DIM])}
