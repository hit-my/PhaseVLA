import dataclasses
import logging
import pathlib
import re
import urllib.parse
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PartialCheckpointWeightLoader(WeightLoader):
    """Strictly loads a checkpoint while allowing only declared missing reference params."""

    params_path: str
    missing_regex: str = "futuremamba/.*"

    def load(self, params: at.Params) -> at.Params:
        params_path = _require_existing_local_path(self.params_path)
        loaded_params = _model.restore_params(params_path, restore_type=np.ndarray)
        return _strict_merge_params(loaded_params, params, missing_regex=self.missing_regex)



@dataclasses.dataclass(frozen=True)
class LatestCheckpointWeightLoader(WeightLoader):
    """Loads the latest numeric-step checkpoint params from a local Orbax checkpoint root."""

    checkpoint_root: str
    missing_regex: str = "futuremamba/.*"

    def load(self, params: at.Params) -> at.Params:
        checkpoint_root = _require_existing_local_checkpoint_root(self.checkpoint_root)
        params_path = _latest_numeric_step_params_path(checkpoint_root)
        loaded_params = _model.restore_params(params_path, restore_type=np.ndarray)
        return _strict_merge_params(loaded_params, params, missing_regex=self.missing_regex)


def _require_existing_local_checkpoint_root(checkpoint_root: str) -> pathlib.Path:
    parsed = urllib.parse.urlparse(checkpoint_root)
    if parsed.scheme:
        raise ValueError(
            "LatestCheckpointWeightLoader requires checkpoint_root to be an existing local checkpoint root; "
            f"got URI scheme {parsed.scheme!r} for {checkpoint_root!r}"
        )
    local_path = pathlib.Path(checkpoint_root).expanduser()
    if not local_path.exists():
        raise FileNotFoundError(
            "LatestCheckpointWeightLoader requires checkpoint_root to be an existing local checkpoint root; "
            f"got missing path {checkpoint_root!r}"
        )
    if not local_path.is_dir():
        raise ValueError(
            "LatestCheckpointWeightLoader requires checkpoint_root to be an existing local checkpoint root directory; "
            f"got file {checkpoint_root!r}"
        )
    return local_path


def _latest_numeric_step_params_path(checkpoint_root: pathlib.Path) -> pathlib.Path:
    candidates: list[tuple[int, pathlib.Path]] = []
    for child in checkpoint_root.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        params_path = child / "params"
        if params_path.exists():
            candidates.append((int(child.name), params_path))
    if not candidates:
        raise ValueError(
            "No numeric checkpoint step directories containing a params item were found in local checkpoint root "
            f"{checkpoint_root!s}"
        )
    return max(candidates, key=lambda item: item[0])[1]

def _require_existing_local_path(params_path: str) -> pathlib.Path:
    parsed = urllib.parse.urlparse(params_path)
    if parsed.scheme:
        raise ValueError(
            "PartialCheckpointWeightLoader requires params_path to be an existing local file or directory; "
            f"got URI scheme {parsed.scheme!r} for {params_path!r}"
        )
    local_path = pathlib.Path(params_path).expanduser()
    if not local_path.exists():
        raise FileNotFoundError(
            "PartialCheckpointWeightLoader requires params_path to be an existing local file or directory; "
            f"got missing path {params_path!r}"
        )
    return local_path


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")


def _strict_merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    pattern = re.compile(missing_regex)

    ref_keys = set(flat_ref)
    loaded_keys = set(flat_loaded)
    extra_keys = sorted(loaded_keys - ref_keys)
    if extra_keys:
        raise ValueError(f"Checkpoint contains extra parameter keys not present in reference: {extra_keys}")

    disallowed_missing = sorted(key for key in ref_keys - loaded_keys if pattern.fullmatch(key) is None)
    if disallowed_missing:
        raise ValueError(f"Checkpoint is missing non-optional parameter keys: {disallowed_missing}")

    result = {}
    for key in sorted(loaded_keys):
        loaded_value = flat_loaded[key]
        ref_value = flat_ref[key]
        if loaded_value.shape != ref_value.shape:
            raise ValueError(
                f"Checkpoint parameter shape mismatch for {key}: got {loaded_value.shape}, expected {ref_value.shape}"
            )
        if loaded_value.dtype != ref_value.dtype:
            raise ValueError(
                f"Checkpoint parameter dtype mismatch for {key}: got {loaded_value.dtype}, expected {ref_value.dtype}"
            )
        result[key] = loaded_value

    for key in sorted(ref_keys - loaded_keys):
        result[key] = flat_ref[key]

    return flax.traverse_util.unflatten_dict(result, sep="/")
