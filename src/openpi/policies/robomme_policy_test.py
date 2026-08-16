from __future__ import annotations

import numpy as np
import pytest

from openpi.policies.robomme_policy import RoboMMEInputs, RoboMMEOutputs


def test_robomme_input_maps_official_observation_keys():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    wrist = np.ones((3, 224, 224), dtype=np.float32)
    state = np.arange(8, dtype=np.float32)

    result = RoboMMEInputs()({
        "observation/image": image,
        "observation/wrist_image": wrist,
        "observation/state": state,
        "prompt": "pick three objects",
    })

    assert result["state"] is state
    assert result["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert result["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    np.testing.assert_array_equal(result["image"]["right_wrist_0_rgb"], np.zeros_like(image))
    assert result["image_mask"] == {
        "base_0_rgb": np.True_,
        "left_wrist_0_rgb": np.True_,
        "right_wrist_0_rgb": np.False_,
    }
    assert result["prompt"] == "pick three objects"

def test_robomme_input_rejects_negative_float_images():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        RoboMMEInputs()({
            "observation/image": np.full((224, 224, 3), -0.5, dtype=np.float32),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.float32),
            "observation/state": np.zeros(8, dtype=np.float32),
            "prompt": "task",
        })

def test_robomme_input_preserves_official_optional_memory_fields_without_prompt():
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    data = {
        "observation/image": image,
        "observation/wrist_image": image,
        "observation/state": np.zeros(8, dtype=np.float32),
        "static_image_emb": np.ones((2, 3), dtype=np.float32),
        "recur_state_emb": np.ones((4, 8), dtype=np.float32),
        "simple_subgoal": "place the object",
    }

    result = RoboMMEInputs()(data)

    assert "prompt" not in result
    assert result["static_image_emb"] is data["static_image_emb"]
    assert result["recur_state_emb"] is data["recur_state_emb"]
    assert result["simple_subgoal"] == "place the object"
    assert result["static_pos_emb"] is None
    assert result["grounded_subgoal"] is None


def test_robomme_input_rejects_wrong_state_width():
    with pytest.raises(ValueError, match="state.*8"):
        RoboMMEInputs()({
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(7, dtype=np.float32),
            "prompt": "task",
        })


def test_robomme_output_keeps_joint_angle_action_width():
    actions = np.zeros((20, 32), dtype=np.float32)
    result = RoboMMEOutputs()(dict(actions=actions))

    assert result["actions"].shape == (20, 8)
    np.testing.assert_array_equal(result["actions"], actions[:, :8])


def test_robomme_output_rejects_wrong_prediction_horizon():
    with pytest.raises(ValueError, match="20"):
        RoboMMEOutputs()(dict(actions=np.zeros((16, 32), dtype=np.float32)))
