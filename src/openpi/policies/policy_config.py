import hashlib
import json
import logging
import pathlib
from typing import Any

import jax.numpy as jnp
import safetensors.torch
import torch

import openpi.models.model as _model
import openpi.models.futuremamba_config as _futuremamba_config
from openpi.models_pytorch.futuremamba_config import FutureMambaPytorchConfig
import openpi.policies.policy as _policy
import openpi.policies.futuremamba_policy as _futuremamba_policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
from openpi.training import futuremamba_checkpoint as _futuremamba_checkpoint
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = pathlib.Path(download.maybe_download(str(checkpoint_dir)))
    plugin_path = checkpoint_dir / "plugin.safetensors"
    metadata_path = checkpoint_dir / "metadata.json"
    has_plugin = plugin_path.is_file()
    has_metadata = metadata_path.is_file()
    if has_plugin != has_metadata:
        missing = "metadata.json" if has_plugin else "plugin.safetensors"
        raise ValueError(f"incomplete FutureMamba bundle: missing {missing}")

    is_bundle = has_plugin and has_metadata
    is_pytorch = (checkpoint_dir / "model.safetensors").is_file()
    assets_checkpoint_dir = checkpoint_dir
    policy_metadata = train_config.policy_metadata
    logging.info("Loading model...")
    if is_bundle:
        if not isinstance(train_config.model, FutureMambaPytorchConfig):
            raise TypeError("FutureMamba plugin bundle requires FutureMambaPytorchConfig")
        device = torch.device(pytorch_device or ("cuda" if torch.cuda.is_available() else "cpu"))
        model, bundle_metadata, assets_checkpoint_dir = _load_futuremamba_bundle(train_config.model, checkpoint_dir, device)
        is_pytorch = True
        pytorch_device = str(device)
        policy_metadata = bundle_metadata
    elif is_pytorch:
        weight_path = checkpoint_dir / "model.safetensors"
        model = train_config.model.load_pytorch(train_config, str(weight_path))
        if hasattr(model, "paligemma_with_expert"):
            model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(assets_checkpoint_dir / "assets", data_config.asset_id)

    if is_pytorch and pytorch_device is None:
        pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"

    is_futuremamba = isinstance(
        train_config.model, (_futuremamba_config.FutureMambaConfig, FutureMambaPytorchConfig)
    )
    action_only = False
    if action_only:
        action_stats = norm_stats["actions"]
        input_transforms = [
            transforms.NormalizeExecutedActions(action_stats, use_quantiles=True),
        ]
        output_transforms = [
            transforms.Unnormalize({"actions": action_stats}, use_quantiles=True),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ]
    else:
        input_transforms = [
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            transforms.NormalizeExecutedActions(
                norm_stats["actions"], use_quantiles=data_config.use_quantile_norm
            ),
        ]
        input_transforms.extend(data_config.model_transforms.inputs)
        output_transforms = [
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ]
    if is_futuremamba:
        futuremamba_sample_kwargs = dict(sample_kwargs or {})
        futuremamba_sample_kwargs.setdefault("num_steps", train_config.model.num_denoise_steps)
        if not action_only:
            futuremamba_sample_kwargs.setdefault("handoff_ratio", train_config.model.handoff_ratio)
        return _futuremamba_policy.FutureMambaPolicy(
            model,
            transforms=input_transforms,
            output_transforms=output_transforms,
            sample_kwargs=futuremamba_sample_kwargs,
            pytorch_device=pytorch_device,
            metadata=policy_metadata,
        )

    return _policy.Policy(
        model,
        transforms=input_transforms,
        output_transforms=output_transforms,
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )


def _load_futuremamba_bundle(
    model_config: FutureMambaPytorchConfig,
    bundle_dir: pathlib.Path | str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], pathlib.Path]:
    bundle_dir = pathlib.Path(bundle_dir)
    metadata_path = bundle_dir / "metadata.json"
    plugin_path = bundle_dir / "plugin.safetensors"
    if not metadata_path.is_file() or not plugin_path.is_file():
        raise ValueError("incomplete FutureMamba bundle: plugin.safetensors and metadata.json are required")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid FutureMamba metadata.json: {error}") from error
    identity = _futuremamba_checkpoint._validate_metadata(metadata)
    expected = model_config.checkpoint_metadata()
    for field, expected_value in expected.items():
        if expected_value is not None and identity[field] != expected_value:
            raise ValueError(
                f"FutureMamba bundle identity mismatch for {field}: "
                f"expected {expected_value!r}, got {identity[field]!r}"
            )

    if identity.get("architecture") == "action_history_mamba_progress_expert":
        raise ValueError("standalone action-only FutureMamba bundles are incompatible with PE-to-AE handoff")

    base_uri = identity["base_checkpoint_uri"]
    if not isinstance(base_uri, str) or not base_uri:
        raise ValueError("FutureMamba bundle base_checkpoint_uri must be a non-empty string")
    base_path = pathlib.Path(download.maybe_download(base_uri.removeprefix("file://"))).expanduser()
    base_root = base_path.parent if base_path.name == "model.safetensors" else base_path
    weight_path = base_path if base_path.name == "model.safetensors" else base_path / "model.safetensors"
    if not weight_path.is_file():
        raise FileNotFoundError(f"converted base model.safetensors not found: {weight_path}")

    model = model_config.create_pytorch().to(device)
    try:
        safetensors.torch.load_model(model.base, weight_path, strict=True, device=str(device))
    except Exception as error:
        raise ValueError(f"strict base checkpoint load failed for {weight_path}: {error}") from error
    model.freeze_base()
    actual_base_checksum = model.base_checksum()
    if identity["base_checkpoint_checksum"] != actual_base_checksum:
        raise ValueError(
            "base_checkpoint_checksum mismatch: "
            f"expected {identity['base_checkpoint_checksum']!r}, got {actual_base_checksum!r}"
        )
    expected_assets_checksum = identity["base_assets_checksum"]
    if expected_assets_checksum is not None:
        assets_path = base_root / "assets"
        if not assets_path.is_dir():
            raise ValueError(f"base assets directory not found: {assets_path}")
        actual_assets_checksum = _directory_checksum(assets_path)
        if expected_assets_checksum != actual_assets_checksum:
            raise ValueError(
                f"base_assets_checksum mismatch: expected {expected_assets_checksum!r}, got {actual_assets_checksum!r}"
            )
    try:
        plugin_state = safetensors.torch.load_file(plugin_path, device=str(device))
        _futuremamba_checkpoint._strict_load_plugin(model, plugin_state)
    except Exception as error:
        raise ValueError(f"strict plugin load failed for {plugin_path}: {error}") from error
    model.eval()
    return model, metadata, base_root


def _directory_checksum(root: pathlib.Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
