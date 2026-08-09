import numpy as np
import pytest

import openpi.models.tokenizer as _tokenizer
import openpi.transforms as _transforms
from openpi.shared import normalize as _normalize


def test_repack_transform():
    transform = _transforms.RepackTransform(
        structure={
            "a": {"b": "b/c"},
            "d": "e/f",
        }
    )
    item = {"b": {"c": 1}, "e": {"f": 2}}
    assert transform(item) == {"a": {"b": 1}, "d": 2}


def test_delta_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.DeltaActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 2, 5], [5, 4, 7]]))


def test_delta_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.DeltaActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.DeltaActions(mask=[True, False])
    assert transform(item) is item


def test_absolute_actions():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    transform = _transforms.AbsoluteActions(mask=[False, True])
    transformed = transform(item)

    assert np.all(transformed["state"] == np.array([1, 2, 3]))
    assert np.all(transformed["actions"] == np.array([[3, 6, 5], [5, 8, 7]]))


def test_absolute_actions_noop():
    item = {"state": np.array([1, 2, 3]), "actions": np.array([[3, 4, 5], [5, 6, 7]])}

    # No-op when the mask is disabled.
    transform = _transforms.AbsoluteActions(mask=None)
    assert transform(item) is item

    # No-op when there are no actions in the input.
    del item["actions"]
    transform = _transforms.AbsoluteActions(mask=[True, False])
    assert transform(item) is item


def test_normalize_executed_actions_uses_action_stats_without_touching_actions():
    raw_actions = np.array([[11, 22, 35, 60, 80, 120, 145]], dtype=np.float32)
    data = {"actions": raw_actions.copy(), "executed_actions": raw_actions.copy()}
    stats = _normalize.NormStats(
        mean=np.array([10, 20, 30, 40, 50, 60, 70], dtype=np.float32),
        std=np.array([1, 2, 5, 10, 10, 20, 25], dtype=np.float32),
    )

    transformed = _transforms.NormalizeExecutedActions(stats)(data)

    np.testing.assert_array_equal(transformed["actions"], raw_actions)
    np.testing.assert_allclose(transformed["executed_actions"], [[1, 1, 1, 2, 3, 3, 3]], rtol=1e-6)


def test_normalize_executed_actions_supports_quantile_stats():
    raw_actions = np.array([[2.0, 5.0]], dtype=np.float32)
    stats = _normalize.NormStats(
        mean=np.zeros(2, dtype=np.float32),
        std=np.ones(2, dtype=np.float32),
        q01=np.array([0.0, 1.0], dtype=np.float32),
        q99=np.array([4.0, 9.0], dtype=np.float32),
    )

    transformed = _transforms.NormalizeExecutedActions(stats, use_quantiles=True)({"executed_actions": raw_actions})

    np.testing.assert_allclose(transformed["executed_actions"], [[0.0, 0.0]], atol=1e-6)


def test_pad_executed_actions_pads_horizon_and_action_dim_with_mask():
    data = {"executed_actions": np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)}

    transformed = _transforms.PadExecutedActions(executed_horizon=4, action_dim=5)(data)

    np.testing.assert_allclose(
        transformed["executed_actions"],
        [[1, 2, 3, 0, 0], [4, 5, 6, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
    )
    np.testing.assert_array_equal(transformed["executed_action_mask"], [True, True, False, False])


def test_pad_executed_actions_rejects_prefix_longer_than_horizon():
    with pytest.raises(ValueError, match="executed_actions.*2"):
        _transforms.PadExecutedActions(executed_horizon=2, action_dim=3)(
            {"executed_actions": np.ones((3, 3), dtype=np.float32)}
        )


def test_make_bool_mask():
    assert _transforms.make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
    assert _transforms.make_bool_mask(2, 0, 2) == (True, True, True, True)


def test_tokenize_prompt():
    tokenizer = _tokenizer.PaligemmaTokenizer(max_len=12)
    transform = _transforms.TokenizePrompt(tokenizer)

    data = transform({"prompt": "Hello, world!"})

    tok_prompt, tok_mask = tokenizer.tokenize("Hello, world!")
    assert np.allclose(tok_prompt, data["tokenized_prompt"])
    assert np.allclose(tok_mask, data["tokenized_prompt_mask"])


def test_tokenize_no_prompt():
    transform = _transforms.TokenizePrompt(_tokenizer.PaligemmaTokenizer())

    with pytest.raises(ValueError, match="Prompt is required"):
        transform({})


def test_transform_dict():
    # Rename and remove keys.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a/b": "a/c", "a/c": None}, input)
    assert output == {"a": {"c": 1}}

    # Raises and error since the renamed key conflicts with an existing key.
    with pytest.raises(ValueError, match="Key 'a/c' already exists in output"):
        _transforms.transform_dict({"a/b": "a/c"}, input)

    # Full match is required and so nothing will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a": None}, input)
    assert output == input

    # The regex matches the entire key and so the entire input will be removed.
    input = {"a": {"b": 1, "c": 2}}
    output = _transforms.transform_dict({"a.+": None}, input)
    assert output == {}

    # Replace keys using backreferences. All leaves named 'c' are replaced with 'd'.
    input = {"a": {"b": 1, "c": 1}, "b": {"c": 2}}
    output = _transforms.transform_dict({"(.+)/c": r"\1/d"}, input)
    assert output == {"a": {"b": 1, "d": 1}, "b": {"d": 2}}


def test_extract_prompt_from_task():
    transform = _transforms.PromptFromLeRobotTask({1: "Hello, world!"})

    data = transform({"task_index": 1})
    assert data["prompt"] == "Hello, world!"

    with pytest.raises(ValueError, match="task_index=2 not found in task mapping"):
        transform({"task_index": 2})
