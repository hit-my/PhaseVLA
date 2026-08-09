import numpy as np

from openpi import transforms as _transforms
from openpi.policies import libero_policy
from openpi.shared import normalize as _normalize


def test_libero_inputs_preserves_executed_actions_in_raw_action_space():
    raw_actions = np.arange(14, dtype=np.float32).reshape(2, 7)
    data = libero_policy.make_libero_example()
    data["executed_actions"] = raw_actions

    transformed = libero_policy.LiberoInputs(model_type=libero_policy._model.ModelType.PI05)(data)

    assert "executed_actions" in transformed
    np.testing.assert_array_equal(transformed["executed_actions"], raw_actions)


def test_libero_executed_action_chain_matches_target_action_normalization_and_padding():
    norm_stats = {
        "actions": _normalize.NormStats(
            mean=np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.float32),
            std=np.array([1, 2, 5, 10, 10, 20, 25], dtype=np.float32),
        )
    }
    raw = np.array([[11, 22, 35, 60, 80, 120, 145]], dtype=np.float32)
    data = libero_policy.make_libero_example()
    data["actions"] = raw.copy()
    data["executed_actions"] = raw.copy()

    chain = _transforms.compose(
        [
            libero_policy.LiberoInputs(model_type=libero_policy._model.ModelType.PI05),
            _transforms.Normalize(norm_stats),
            _transforms.NormalizeExecutedActions(norm_stats["actions"]),
            _transforms.PadExecutedActions(executed_horizon=3, action_dim=9),
        ]
    )

    transformed = chain(data)

    np.testing.assert_allclose(transformed["executed_actions"][0, :7], transformed["actions"][0, :7])
    np.testing.assert_allclose(transformed["executed_actions"][:, 7:], 0.0)
    np.testing.assert_array_equal(transformed["executed_action_mask"], [True, False, False])
