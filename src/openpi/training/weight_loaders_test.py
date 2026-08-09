import copy

from flax import traverse_util
import jax.numpy as jnp
import numpy as np
import pytest

from openpi.training import weight_loaders


def _params():
    return {
        "base": {
            "kernel": np.ones((2, 3), dtype=np.float32),
            "bias": np.ones((3,), dtype=np.float32),
        },
        "futuremamba": {
            "memory": np.full((2, 2), 7.0, dtype=np.float32),
            "token": np.full((1,), 8.0, dtype=np.float32),
        },
    }


def _loaded_base():
    return {
        "base": {
            "kernel": np.full((2, 3), 2.0, dtype=np.float32),
            "bias": np.full((3,), 3.0, dtype=np.float32),
        }
    }


def _flat(tree):
    return traverse_util.flatten_dict(tree, sep="/")


def test_partial_checkpoint_loader_allows_only_futuremamba_missing(monkeypatch):
    ref = _params()
    loaded = _loaded_base()
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    result = weight_loaders.PartialCheckpointWeightLoader("local/path").load(ref)

    np.testing.assert_array_equal(result["base"]["kernel"], loaded["base"]["kernel"])
    np.testing.assert_array_equal(result["base"]["bias"], loaded["base"]["bias"])
    np.testing.assert_array_equal(result["futuremamba"]["memory"], ref["futuremamba"]["memory"])
    np.testing.assert_array_equal(result["futuremamba"]["token"], ref["futuremamba"]["token"])


def test_partial_checkpoint_loader_rejects_non_plugin_missing_key(monkeypatch):
    loaded = _loaded_base()
    del loaded["base"]["bias"]
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    with pytest.raises(ValueError, match="missing.*base/bias"):
        weight_loaders.PartialCheckpointWeightLoader("local/path").load(_params())


def test_partial_checkpoint_loader_rejects_extra_checkpoint_key(monkeypatch):
    loaded = _loaded_base()
    loaded["extra"] = {"kernel": np.ones((1,), dtype=np.float32)}
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    with pytest.raises(ValueError, match="extra.*extra/kernel"):
        weight_loaders.PartialCheckpointWeightLoader("local/path").load(_params())


def test_partial_checkpoint_loader_rejects_shape_mismatch(monkeypatch):
    loaded = _loaded_base()
    loaded["base"]["kernel"] = np.ones((3, 2), dtype=np.float32)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    with pytest.raises(ValueError, match="shape.*base/kernel"):
        weight_loaders.PartialCheckpointWeightLoader("local/path").load(_params())


def test_partial_checkpoint_loader_rejects_dtype_mismatch(monkeypatch):
    loaded = _loaded_base()
    loaded["base"]["bias"] = np.ones((3,), dtype=np.float16)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    with pytest.raises(ValueError, match="dtype.*base/bias"):
        weight_loaders.PartialCheckpointWeightLoader("local/path").load(_params())


def test_partial_checkpoint_loader_uses_default_missing_regex(monkeypatch):
    ref = _params()
    loaded = _loaded_base()
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    result = weight_loaders.PartialCheckpointWeightLoader("local/path").load(ref)
    assert set(_flat(result)) == set(_flat(ref))


def test_partial_checkpoint_loader_does_not_mutate_reference_or_loaded(monkeypatch):
    ref = _params()
    loaded = _loaded_base()
    ref_before = copy.deepcopy(ref)
    loaded_before = copy.deepcopy(loaded)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    weight_loaders.PartialCheckpointWeightLoader("local/path").load(ref)

    for key, expected in _flat(ref_before).items():
        np.testing.assert_array_equal(_flat(ref)[key], expected)
    for key, expected in _flat(loaded_before).items():
        np.testing.assert_array_equal(_flat(loaded)[key], expected)


def test_partial_checkpoint_loader_accepts_jax_arrays_without_cast(monkeypatch):
    ref = {"base": {"kernel": jnp.ones((2,), dtype=jnp.bfloat16)}, "futuremamba": {"p": jnp.ones((1,))}}
    loaded = {"base": {"kernel": jnp.full((2,), 2, dtype=jnp.bfloat16)}}
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    result = weight_loaders.PartialCheckpointWeightLoader("local/path").load(ref)

    assert result["base"]["kernel"].dtype == ref["base"]["kernel"].dtype
    np.testing.assert_array_equal(np.asarray(result["futuremamba"]["p"]), np.asarray(ref["futuremamba"]["p"]))


def test_partial_checkpoint_loader_rejects_remote_scheme_without_download(monkeypatch):
    def fail_download(path):
        raise AssertionError(f"maybe_download must not be called for PartialCheckpointWeightLoader: {path}")

    monkeypatch.setattr(weight_loaders.download, "maybe_download", fail_download)

    with pytest.raises(ValueError, match="local.*scheme"):
        weight_loaders.PartialCheckpointWeightLoader("gs://bucket/checkpoint/params").load(_params())
