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


def _local_params_path(tmp_path):
    params_path = tmp_path / "params"
    params_path.mkdir()
    return str(params_path)


def test_partial_checkpoint_loader_allows_only_futuremamba_missing(monkeypatch, tmp_path):
    ref = _params()
    loaded = _loaded_base()
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    result = weight_loaders.PartialCheckpointWeightLoader(_local_params_path(tmp_path)).load(ref)

    np.testing.assert_array_equal(result["base"]["kernel"], loaded["base"]["kernel"])
    np.testing.assert_array_equal(result["base"]["bias"], loaded["base"]["bias"])
    np.testing.assert_array_equal(result["futuremamba"]["memory"], ref["futuremamba"]["memory"])
    np.testing.assert_array_equal(result["futuremamba"]["token"], ref["futuremamba"]["token"])


def test_partial_checkpoint_loader_rejects_non_plugin_missing_key(monkeypatch, tmp_path):
    loaded = _loaded_base()
    del loaded["base"]["bias"]
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    with pytest.raises(ValueError, match="missing.*base/bias"):
        weight_loaders.PartialCheckpointWeightLoader(_local_params_path(tmp_path)).load(_params())


def test_partial_checkpoint_loader_rejects_extra_checkpoint_key(monkeypatch, tmp_path):
    loaded = _loaded_base()
    loaded["extra"] = {"kernel": np.ones((1,), dtype=np.float32)}
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    with pytest.raises(ValueError, match="extra.*extra/kernel"):
        weight_loaders.PartialCheckpointWeightLoader(_local_params_path(tmp_path)).load(_params())


def test_partial_checkpoint_loader_rejects_shape_mismatch(monkeypatch, tmp_path):
    loaded = _loaded_base()
    loaded["base"]["kernel"] = np.ones((3, 2), dtype=np.float32)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    with pytest.raises(ValueError, match="shape.*base/kernel"):
        weight_loaders.PartialCheckpointWeightLoader(_local_params_path(tmp_path)).load(_params())


def test_partial_checkpoint_loader_rejects_dtype_mismatch(monkeypatch, tmp_path):
    loaded = _loaded_base()
    loaded["base"]["bias"] = np.ones((3,), dtype=np.float16)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    with pytest.raises(ValueError, match="dtype.*base/bias"):
        weight_loaders.PartialCheckpointWeightLoader(_local_params_path(tmp_path)).load(_params())


def test_partial_checkpoint_loader_uses_default_missing_regex(monkeypatch, tmp_path):
    ref = _params()
    loaded = _loaded_base()
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    result = weight_loaders.PartialCheckpointWeightLoader(_local_params_path(tmp_path)).load(ref)
    assert set(_flat(result)) == set(_flat(ref))


def test_partial_checkpoint_loader_does_not_mutate_reference_or_loaded(monkeypatch, tmp_path):
    ref = _params()
    loaded = _loaded_base()
    ref_before = copy.deepcopy(ref)
    loaded_before = copy.deepcopy(loaded)
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    weight_loaders.PartialCheckpointWeightLoader(_local_params_path(tmp_path)).load(ref)

    for key, expected in _flat(ref_before).items():
        np.testing.assert_array_equal(_flat(ref)[key], expected)
    for key, expected in _flat(loaded_before).items():
        np.testing.assert_array_equal(_flat(loaded)[key], expected)


def test_partial_checkpoint_loader_accepts_jax_arrays_without_cast(monkeypatch, tmp_path):
    ref = {"base": {"kernel": jnp.ones((2,), dtype=jnp.bfloat16)}, "futuremamba": {"p": jnp.ones((1,))}}
    loaded = {"base": {"kernel": jnp.full((2,), 2, dtype=jnp.bfloat16)}}
    monkeypatch.setattr(weight_loaders.download, "maybe_download", lambda path: path)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    result = weight_loaders.PartialCheckpointWeightLoader(_local_params_path(tmp_path)).load(ref)

    assert result["base"]["kernel"].dtype == ref["base"]["kernel"].dtype
    np.testing.assert_array_equal(np.asarray(result["futuremamba"]["p"]), np.asarray(ref["futuremamba"]["p"]))


@pytest.mark.parametrize("params_path", ["gs://bucket/checkpoint/params", "http://example.test/checkpoint/params"])
def test_partial_checkpoint_loader_rejects_remote_scheme_without_download_or_restore(monkeypatch, params_path):
    def fail_download(path):
        raise AssertionError(f"maybe_download must not be called for PartialCheckpointWeightLoader: {path}")

    def fail_restore(path, restore_type):
        raise AssertionError(f"restore_params must not be called for remote PartialCheckpointWeightLoader path: {path}")

    monkeypatch.setattr(weight_loaders.download, "maybe_download", fail_download)
    monkeypatch.setattr(weight_loaders._model, "restore_params", fail_restore)

    with pytest.raises(ValueError, match="existing local file or directory.*URI scheme"):
        weight_loaders.PartialCheckpointWeightLoader(params_path).load(_params())


def test_partial_checkpoint_loader_rejects_missing_local_path_before_restore(monkeypatch, tmp_path):
    def fail_download(path):
        raise AssertionError(f"maybe_download must not be called for PartialCheckpointWeightLoader: {path}")

    def fail_restore(path, restore_type):
        raise AssertionError(f"restore_params must not be called for missing PartialCheckpointWeightLoader path: {path}")

    monkeypatch.setattr(weight_loaders.download, "maybe_download", fail_download)
    monkeypatch.setattr(weight_loaders._model, "restore_params", fail_restore)

    with pytest.raises(FileNotFoundError, match="existing local file or directory"):
        weight_loaders.PartialCheckpointWeightLoader(str(tmp_path / "missing" / "params")).load(_params())


def test_latest_checkpoint_loader_chooses_largest_numeric_step_params(monkeypatch, tmp_path):
    root = tmp_path / "base"
    (root / "1" / "params").mkdir(parents=True)
    (root / "20" / "params").mkdir(parents=True)
    (root / "not-a-step" / "params").mkdir(parents=True)
    loaded = _loaded_base()
    calls = []

    def restore(path, restore_type):
        calls.append(path)
        assert restore_type is np.ndarray
        return loaded

    monkeypatch.setattr(weight_loaders._model, "restore_params", restore)

    result = weight_loaders.LatestCheckpointWeightLoader(str(root)).load(_params())

    assert calls == [root / "20" / "params"]
    np.testing.assert_array_equal(result["base"]["kernel"], loaded["base"]["kernel"])
    np.testing.assert_array_equal(result["futuremamba"]["memory"], _params()["futuremamba"]["memory"])


def test_latest_checkpoint_loader_ignores_numeric_steps_without_params(monkeypatch, tmp_path):
    root = tmp_path / "base"
    (root / "1" / "params").mkdir(parents=True)
    (root / "2").mkdir(parents=True)
    loaded = _loaded_base()
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    result = weight_loaders.LatestCheckpointWeightLoader(str(root)).load(_params())

    np.testing.assert_array_equal(result["base"]["bias"], loaded["base"]["bias"])


def test_latest_checkpoint_loader_rejects_empty_or_non_numeric_root_before_restore(monkeypatch, tmp_path):
    root = tmp_path / "base"
    (root / "latest" / "params").mkdir(parents=True)

    def fail_restore(path, restore_type):
        raise AssertionError(f"restore_params must not be called for empty latest root: {path}")

    monkeypatch.setattr(weight_loaders._model, "restore_params", fail_restore)

    with pytest.raises(ValueError, match="No numeric checkpoint step directories"):
        weight_loaders.LatestCheckpointWeightLoader(str(root)).load(_params())


def test_latest_checkpoint_loader_preserves_strict_partial_merge_errors(monkeypatch, tmp_path):
    root = tmp_path / "base"
    (root / "3" / "params").mkdir(parents=True)
    loaded = _loaded_base()
    loaded["base"]["kernel"] = np.ones((3, 2), dtype=np.float32)
    monkeypatch.setattr(weight_loaders._model, "restore_params", lambda path, restore_type: loaded)

    with pytest.raises(ValueError, match="shape.*base/kernel"):
        weight_loaders.LatestCheckpointWeightLoader(str(root)).load(_params())


def test_latest_checkpoint_loader_requires_local_existing_root(monkeypatch, tmp_path):
    def fail_restore(path, restore_type):
        raise AssertionError(f"restore_params must not be called for invalid latest root: {path}")

    monkeypatch.setattr(weight_loaders._model, "restore_params", fail_restore)

    with pytest.raises(ValueError, match="existing local checkpoint root.*URI scheme"):
        weight_loaders.LatestCheckpointWeightLoader("gs://bucket/run").load(_params())
    with pytest.raises(FileNotFoundError, match="existing local checkpoint root"):
        weight_loaders.LatestCheckpointWeightLoader(str(tmp_path / "missing")).load(_params())
