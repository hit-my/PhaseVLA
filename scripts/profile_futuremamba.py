from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import pickle
import time
from typing import Any

import torch

BATCH_SIZE = 1
DEFAULT_ACTION_CHUNK_STEPS = 20
DEFAULT_SOLVER_STEPS = 10
PROFILE_SCHEMA_VERSION = 1
_PROFILE_TYPE = "futuremamba_task15_profile"
_REQUIRED_METADATA_FIELDS = (
    "base_checkpoint_checksum",
    "base_assets_checksum",
    "robomme_policy_commit",
    "robomme_benchmark_commit",
    "mamba_repo_commit",
    "memory_backend",
    "memory_state_schema_version",
)
_EPISODE_QUERY_MEASUREMENT_SOURCE = "futuremamba_policy_timing"



@dataclasses.dataclass(frozen=True)
class LoadedFutureMambaBundle:
    model: torch.nn.Module
    metadata: dict[str, Any]
    base_root: Path


@dataclasses.dataclass(frozen=True)
class ProfileArtifacts:
    flops: dict[str, Any]
    training_memory: dict[str, Any]
    episode: dict[str, Any]


class TorchPeakMemory:
    def __init__(self, device: torch.device) -> None:
        self._device = torch.device(device)

    def reset(self) -> None:
        if self._device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self._device)

    def read(self) -> int:
        if self._device.type != "cuda":
            raise ValueError("inference peak GPU memory requires a CUDA device")
        return int(torch.cuda.max_memory_allocated(self._device))


class TorchFutureMambaProfileAdapter:
    def __init__(self, model: torch.nn.Module, observation: Any, *, device: torch.device) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.observation = _to_device(observation, device)
        self.device = device
        config = getattr(model, "config", None)
        self.num_steps = int(getattr(config, "num_denoise_steps", DEFAULT_SOLVER_STEPS))
        self.action_chunk_steps = int(getattr(config, "action_horizon", DEFAULT_ACTION_CHUNK_STEPS))
        if self.action_chunk_steps != DEFAULT_ACTION_CHUNK_STEPS:
            raise ValueError(
                f"FutureMamba task-15 profile requires 20-step action chunks, got {self.action_chunk_steps}"
            )
        self.handoff_ratio = float(getattr(config, "handoff_ratio", 0.0))
        self._last_vlm_token: torch.Tensor | None = None

    def parameter_counts(self) -> dict[str, int]:
        total = 0
        trainable = 0
        plugin = 0
        for name, parameter in self.model.named_parameters():
            count = int(parameter.numel())
            total += count
            if parameter.requires_grad:
                trainable += count
            if name.startswith("futuremamba."):
                plugin += count
        if total <= 0:
            raise ValueError("FutureMamba model exposes no parameters")
        if plugin <= 0:
            raise ValueError("FutureMamba model exposes no futuremamba.* plugin parameters")
        return {"total": total, "trainable": trainable, "plugin": plugin}

    def initial_memory_state(self):
        dtype = _module_dtype(getattr(self.model, "futuremamba", self.model))
        return self.model.initial_memory_state(BATCH_SIZE, self.device, dtype)

    def run_memory_step(self, memory_state):
        token = self._cached_last_vlm_token()
        with torch.no_grad():
            _, next_state = self.model.futuremamba.compute_memory_token(token, memory_state)
        return next_state

    def run_action_chunk(self, observation: Any, memory_state: Any):
        del observation
        with torch.no_grad():
            actions, next_state, diagnostics = self.model.sample_actions_with_memory(
                self.observation,
                memory_state,
                num_steps=self.num_steps,
                handoff_ratio=self.handoff_ratio,
            )
        if not torch.is_tensor(actions):
            raise ValueError("FutureMamba action chunk must be a torch.Tensor")
        if actions.ndim != 3 or int(actions.shape[0]) != BATCH_SIZE:
            raise ValueError(f"FutureMamba action chunk must have shape [1, steps, dim], got {tuple(actions.shape)}")
        if int(actions.shape[1]) != self.action_chunk_steps:
            raise ValueError(
                f"FutureMamba action chunk must contain {self.action_chunk_steps} actions, got {int(actions.shape[1])}"
            )
        return actions, next_state, diagnostics

    def _cached_last_vlm_token(self) -> torch.Tensor:
        if self._last_vlm_token is not None:
            return self._last_vlm_token
        if not hasattr(self.model, "base") or not hasattr(self.model, "futuremamba"):
            raise ValueError("FutureMamba profile requires model.base and model.futuremamba")
        with torch.no_grad():
            frozen = self.model.base.encode_frozen_prefix(self.observation, train=False)
            detach = getattr(self.model, "_detached_frozen_prefix", None)
            if callable(detach):
                frozen = detach(frozen)
            token = self.model.base.last_valid_prefix(frozen).detach()
        if token.ndim != 2 or int(token.shape[0]) != BATCH_SIZE:
            raise ValueError(f"last VLM token must have shape [1, width], got {tuple(token.shape)}")
        self._last_vlm_token = token
        return token


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"percentile q must be in [0, 1], got {q}")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def resolve_checkpoint_params(checkpoint_root: str | Path | None) -> Path | None:
    if checkpoint_root is None:
        return None
    root = Path(checkpoint_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"checkpoint root must be an existing local directory: {root}")
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir() or not child.name.isdecimal():
            continue
        params = child / "params"
        if params.exists() and params.is_dir():
            candidates.append((int(child.name), params))
    if not candidates:
        raise FileNotFoundError(f"checkpoint root {root} contains no numeric step '<step>/params' directory")
    return max(candidates, key=lambda item: item[0])[1]


def to_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_json_sha256(value: Any) -> str:
    return hashlib.sha256(to_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_futuremamba_bundle(config_name: str, bundle_dir: str | Path, device: torch.device | str) -> LoadedFutureMambaBundle:
    try:
        training_config = importlib.import_module("openpi.training.config")
        futuremamba_config = importlib.import_module("openpi.models_pytorch.futuremamba_config")
        policy_config = importlib.import_module("openpi.policies.policy_config")
    except ModuleNotFoundError as error:
        raise RuntimeError("Unable to load real PyTorch FutureMamba dependencies") from error

    try:
        train_config = training_config.get_config(config_name)
    except Exception as error:
        raise RuntimeError(f"Unable to resolve training config {config_name!r}") from error
    expected_config_type = futuremamba_config.FutureMambaPytorchConfig
    if not isinstance(train_config.model, expected_config_type):
        raise TypeError(f"training config {config_name!r} must use FutureMambaPytorchConfig")

    strict_loader = getattr(policy_config, "_load_futuremamba_bundle", None)
    if not callable(strict_loader):
        raise RuntimeError("openpi.policies.policy_config._load_futuremamba_bundle is required")
    model, metadata, base_root = strict_loader(train_config.model, Path(bundle_dir), torch.device(device))
    if not isinstance(metadata, Mapping):
        raise ValueError("FutureMamba bundle loader returned non-mapping metadata")
    return LoadedFutureMambaBundle(model=model, metadata=dict(metadata), base_root=Path(base_root))


def load_required_artifacts(
    *,
    flop_artifact: str | Path | None,
    training_memory_artifact: str | Path | None,
    episode_artifact: str | Path | None,
    bundle_metadata: Mapping[str, Any],
    bundle_metadata_sha256: str,
) -> ProfileArtifacts:
    if flop_artifact is None:
        raise ValueError("flop_artifact is required for a formal FutureMamba profile")
    if training_memory_artifact is None:
        raise ValueError("training_memory_artifact is required for a formal FutureMamba profile")
    if episode_artifact is None:
        raise ValueError("episode_artifact is required for a formal FutureMamba profile")
    return ProfileArtifacts(
        flops=load_flop_artifact(flop_artifact, bundle_metadata, bundle_metadata_sha256),
        training_memory=load_training_memory_artifact(
            training_memory_artifact, bundle_metadata, bundle_metadata_sha256
        ),
        episode=load_episode_artifact(episode_artifact, bundle_metadata, bundle_metadata_sha256),
    )


def load_flop_artifact(path: str | Path, bundle_metadata: Mapping[str, Any], bundle_metadata_sha256: str) -> dict[str, Any]:
    artifact_path = Path(path)
    data = _read_json_mapping(artifact_path)
    if data.get("artifact_type") != "futuremamba_flop_analysis":
        raise ValueError("flop artifact_type must be 'futuremamba_flop_analysis'")
    _validate_schema_version(data, "flop")
    _validate_bound_provenance(data, bundle_metadata, bundle_metadata_sha256)
    source = _required_str(data, "measurement_source")
    if source == "parameter_count":
        raise ValueError("flop artifact measurement_source must not be parameter_count")
    if source not in {"tool_analysis", "measured"}:
        raise ValueError("flop artifact measurement_source must be tool_analysis or measured")
    tool = _required_str(data, "tool")
    base_flops = _required_positive_int(data, "base_flops")
    plugin_flops = _required_nonnegative_int(data, "plugin_flops")
    if plugin_flops == 0:
        raise ValueError("plugin_flops must be a positive tool/measured value for FutureMamba")
    contract = data.get("input_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("flop artifact input_contract must be a mapping")
    if int(contract.get("batch_size", -1)) != BATCH_SIZE:
        raise ValueError("flop artifact input_contract.batch_size must be 1")
    if int(contract.get("action_chunk_steps", -1)) != DEFAULT_ACTION_CHUNK_STEPS:
        raise ValueError("flop artifact input_contract.action_chunk_steps must be 20")
    return {
        "base_flops": base_flops,
        "plugin_flops": plugin_flops,
        "futuremamba_total_flops": base_flops + plugin_flops,
        "relative_plugin_over_base": plugin_flops / base_flops,
        "measurement_source": source,
        "tool": tool,
        "artifact_sha256": file_sha256(artifact_path),
    }


def load_training_memory_artifact(
    path: str | Path,
    bundle_metadata: Mapping[str, Any],
    bundle_metadata_sha256: str,
) -> dict[str, Any]:
    artifact_path = Path(path)
    data = _read_json_mapping(artifact_path)
    if data.get("artifact_type") != "futuremamba_training_memory":
        raise ValueError("training memory artifact_type must be 'futuremamba_training_memory'")
    _validate_schema_version(data, "training memory")
    _validate_bound_provenance(data, bundle_metadata, bundle_metadata_sha256)
    if data.get("current_training_run") is not True:
        raise ValueError("training memory artifact current_training_run must be true")
    source = _required_str(data, "measurement_source")
    if source not in {"torch.cuda.max_memory_allocated", "torch.cuda.max_memory_reserved"}:
        raise ValueError("training memory measurement_source must be a PyTorch CUDA allocator peak")
    peak = _required_positive_int(data, "peak_memory_bytes")
    producer = _required_str(data, "producer")
    return {
        "peak_memory_bytes": peak,
        "measurement_source": source,
        "producer": producer,
        "artifact_sha256": file_sha256(artifact_path),
    }


def load_episode_artifact(path: str | Path, bundle_metadata: Mapping[str, Any], bundle_metadata_sha256: str) -> dict[str, Any]:
    artifact_path = Path(path)
    data = _read_json_mapping(artifact_path)
    if data.get("artifact_type") != "futuremamba_episode_timing":
        raise ValueError("episode artifact_type must be 'futuremamba_episode_timing'")
    _validate_schema_version(data, "episode")
    _validate_bound_provenance(data, bundle_metadata, bundle_metadata_sha256)
    times, measurement_source = _episode_query_infer_ms(data)
    episode_count = len(data["episodes"])
    return {
        "average_query_ms": sum(times) / len(times),
        "query_count": len(times),
        "episode_count": episode_count,
        "measurement_source": measurement_source,
        "artifact_sha256": file_sha256(artifact_path),
    }


def profile_model(
    adapter: Any,
    observation: Any,
    *,
    config_name: str,
    method_id: str,
    train_seed: int,
    bundle_path: str | Path,
    bundle_metadata: Mapping[str, Any],
    bundle_metadata_sha256: str,
    episode_metrics: Mapping[str, Any],
    flop_metrics: Mapping[str, Any],
    training_memory_metrics: Mapping[str, Any],
    runtime_info: Mapping[str, Any],
    warmup: int = 3,
    measured: int = 20,
    timer: Callable[[], float] | None = None,
    synchronizer: Callable[[], None] | None = None,
    peak_memory: Any | None = None,
) -> dict[str, Any]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if measured <= 0:
        raise ValueError("measured must be positive")
    timer = time.perf_counter if timer is None else timer
    synchronizer = (lambda: None) if synchronizer is None else synchronizer
    peak_memory = TorchPeakMemory(torch.device(runtime_info["device"])) if peak_memory is None else peak_memory

    metadata = dict(bundle_metadata)
    _validate_bundle_metadata_for_report(metadata)
    counts = _parameter_counts(adapter)
    memory_state = adapter.initial_memory_state()
    memory_bytes = _memory_state_bytes(memory_state)

    synchronizer()
    peak_memory.reset()
    memory_state, memory_latencies = _measure_stateful_operation(
        memory_state,
        lambda state: adapter.run_memory_step(state),
        warmup=warmup,
        measured=measured,
        timer=timer,
        synchronizer=synchronizer,
    )
    memory_state, action_latencies = _measure_stateful_operation(
        memory_state,
        lambda state: adapter.run_action_chunk(observation, state)[1],
        warmup=warmup,
        measured=measured,
        timer=timer,
        synchronizer=synchronizer,
    )
    del memory_state
    synchronizer()
    inference_peak = int(peak_memory.read())
    if inference_peak <= 0:
        raise ValueError("inference peak GPU memory must be a positive measured value")

    plugin_ratio = counts["plugin"] / counts["total"]
    if not method_id:
        raise ValueError("method_id must be a non-empty string")
    if isinstance(train_seed, bool) or not isinstance(train_seed, int):
        raise ValueError("train_seed must be an integer")
    report = {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile_type": _PROFILE_TYPE,
        "method_id": method_id,
        "train_seed": train_seed,
        "config_name": str(config_name),
        "bundle_identity": _bundle_identity(metadata, bundle_path, bundle_metadata_sha256),
        "source_commits": {
            "mamba": _required_metadata_str(metadata, "mamba_repo_commit"),
            "robomme_policy": _required_metadata_str(metadata, "robomme_policy_commit"),
            "robomme_benchmark": _required_metadata_str(metadata, "robomme_benchmark_commit"),
        },
        "parameters": {
            "total": counts["total"],
            "trainable": counts["trainable"],
            "plugin": counts["plugin"],
            "plugin_ratio": plugin_ratio,
        },
        "memory_state_bytes": memory_bytes,
        "latency_ms": {
            "memory_step": {
                "median": percentile(memory_latencies, 0.50),
                "p95": percentile(memory_latencies, 0.95),
            },
            "action_chunk_20": {
                "median": percentile(action_latencies, 0.50),
                "p95": percentile(action_latencies, 0.95),
            },
        },
        "inference_peak_memory_bytes": inference_peak,
        "training_peak_memory_bytes": _required_positive_int(training_memory_metrics, "peak_memory_bytes"),
        "episode_average_query_ms": _required_positive_float(episode_metrics, "average_query_ms"),
        "episode_timing": {
            "query_count": _required_positive_int(episode_metrics, "query_count"),
            "episode_count": _required_positive_int(episode_metrics, "episode_count"),
            "measurement_source": _required_str(episode_metrics, "measurement_source"),
            "artifact_sha256": _required_str(episode_metrics, "artifact_sha256"),
        },
        "flops": {
            "base_flops": _required_positive_int(flop_metrics, "base_flops"),
            "plugin_flops": _required_positive_int(flop_metrics, "plugin_flops"),
            "futuremamba_total_flops": _required_positive_int(flop_metrics, "futuremamba_total_flops"),
            "relative_plugin_over_base": _required_positive_float(flop_metrics, "relative_plugin_over_base"),
            "measurement_source": _required_str(flop_metrics, "measurement_source"),
            "tool": _required_str(flop_metrics, "tool"),
            "artifact_sha256": _required_str(flop_metrics, "artifact_sha256"),
        },
        "training_memory_artifact": {
            "measurement_source": _required_str(training_memory_metrics, "measurement_source"),
            "producer": _required_str(training_memory_metrics, "producer"),
            "artifact_sha256": _required_str(training_memory_metrics, "artifact_sha256"),
        },
        "measurement_counts": {"warmup": int(warmup), "measured": int(measured), "batch_size": BATCH_SIZE},
    }
    _reject_nulls(report)
    return report


def load_profile_observation(path: str | Path | None, device: torch.device) -> Any:
    if path is None:
        raise ValueError("episode_data is required to materialize real PyTorch profile tensors")
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"episode_data must be an existing file: {data_path}")
    if data_path.suffix in {".pt", ".pth"}:
        payload = torch.load(data_path, map_location=device, weights_only=False)
    elif data_path.suffix == ".json":
        payload = json.loads(data_path.read_text(encoding="utf-8"))
    elif data_path.suffix in {".pkl", ".pickle"}:
        with data_path.open("rb") as handle:
            payload = pickle.load(handle)
    else:
        raise ValueError("episode_data must be a .pt, .pth, .json, .pkl, or .pickle file")
    if isinstance(payload, Mapping) and "observation" in payload:
        payload = payload["observation"]
    if hasattr(payload, "to_dict") and hasattr(payload, "state"):
        return _to_device(payload, device)
    if not isinstance(payload, Mapping):
        raise ValueError("episode_data must contain an Observation or observation mapping")
    observation_dict = _prepare_observation_mapping(payload, device)
    observation_module = importlib.import_module("openpi.models.model")
    return observation_module.Observation.from_dict(observation_dict)


def runtime_info(device: torch.device) -> dict[str, str]:
    info = {
        "device": str(device),
        "device_type": device.type,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "not_available",
    }
    try:
        info["triton_version"] = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        info["triton_version"] = "not_installed"
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
        major, minor = torch.cuda.get_device_capability(device)
        info["compute_capability"] = f"{major}.{minor}"
    else:
        info["gpu_name"] = "not_cuda_device"
        info["compute_capability"] = "not_cuda"
    return info


def make_synchronizer(device: torch.device) -> Callable[[], None]:
    if device.type != "cuda":
        return lambda: None

    def synchronize() -> None:
        torch.cuda.synchronize(device)

    return synchronize


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.bundle is None:
        raise ValueError("--bundle is required; strict PyTorch FutureMamba bundles are the only supported input")
    bundle = load_futuremamba_bundle(args.config, args.bundle, device)
    metadata_sha = stable_json_sha256(bundle.metadata)
    artifacts = load_required_artifacts(
        flop_artifact=args.flop_artifact,
        training_memory_artifact=args.training_memory_artifact,
        episode_artifact=args.episode_artifact,
        bundle_metadata=bundle.metadata,
        bundle_metadata_sha256=metadata_sha,
    )
    observation = load_profile_observation(args.episode_data, device)
    adapter = TorchFutureMambaProfileAdapter(bundle.model, observation, device=device)
    report = profile_model(
        adapter,
        observation,
        config_name=args.config,
        method_id=args.method_id,
        train_seed=args.train_seed,
        bundle_path=args.bundle,
        bundle_metadata=bundle.metadata,
        bundle_metadata_sha256=metadata_sha,
        episode_metrics=artifacts.episode,
        flop_metrics=artifacts.flops,
        training_memory_metrics=artifacts.training_memory,
        runtime_info=runtime_info(device),
        warmup=args.warmup,
        measured=args.measured,
        synchronizer=make_synchronizer(device),
        peak_memory=TorchPeakMemory(device),
    )
    output = to_json(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile a strict PyTorch FutureMamba bundle with task-15 evidence.")
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--train-seed", required=True, type=int)
    parser.add_argument("--config", default="futuremamba_robomme_mamba2")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--episode-artifact", type=Path, required=True)
    parser.add_argument("--episode-data", type=Path, required=True)
    parser.add_argument("--flop-artifact", type=Path, required=True)
    parser.add_argument("--training-memory-artifact", type=Path, required=True)
    parser.add_argument("--device")
    parser.add_argument("--warmup", "--warmup-queries", dest="warmup", type=int, default=3)
    parser.add_argument("--measured", "--measured-queries", dest="measured", type=int, default=20)
    parser.add_argument("--output", type=Path)
    return parser


def _measure_stateful_operation(
    initial_state: Any,
    operation: Callable[[Any], Any],
    *,
    warmup: int,
    measured: int,
    timer: Callable[[], float],
    synchronizer: Callable[[], None],
) -> tuple[Any, list[float]]:
    state = initial_state
    for _ in range(warmup):
        synchronizer()
        before = timer()
        state = operation(state)
        synchronizer()
        after = timer()
        if after < before:
            raise RuntimeError("profile timer must be monotonic")
    latencies: list[float] = []
    for _ in range(measured):
        synchronizer()
        before = timer()
        state = operation(state)
        synchronizer()
        after = timer()
        if after < before:
            raise RuntimeError("profile timer must be monotonic")
        latencies.append((after - before) * 1000.0)
    return state, latencies


def _parameter_counts(adapter: Any) -> dict[str, int]:
    counts = adapter.parameter_counts()
    normalized = {
        "total": _required_positive_int(counts, "total"),
        "trainable": _required_nonnegative_int(counts, "trainable"),
        "plugin": _required_positive_int(counts, "plugin"),
    }
    if normalized["trainable"] < normalized["plugin"]:
        raise ValueError("trainable parameters must be greater than or equal to plugin parameters")
    if normalized["plugin"] > normalized["total"]:
        raise ValueError("plugin parameters must not exceed total parameters")
    return normalized


def _memory_state_bytes(memory_state: Any) -> dict[str, Any]:
    layers = getattr(memory_state, "layers", None)
    if layers is None:
        raise ValueError("memory state must expose per-layer tensors via .layers")
    layer_rows = []
    total = 0
    for index, layer in enumerate(layers):
        layer_bytes = sum(_tensor_nbytes(tensor) for tensor in layer)
        layer_rows.append({"layer": index, "bytes": layer_bytes})
        total += layer_bytes
    if not layer_rows:
        raise ValueError("memory state must contain at least one layer")
    return {"layers": layer_rows, "total": total}


def _tensor_nbytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    raise ValueError(f"memory state leaf is not tensor-like: {type(value).__name__}")


def _read_json_mapping(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"JSON artifact {path} must contain an object")
    return data


def _validate_schema_version(data: Mapping[str, Any], name: str) -> None:
    version = data.get("schema_version")
    if not isinstance(version, int) or isinstance(version, bool) or version <= 0:
        raise ValueError(f"{name} artifact schema_version must be a positive integer")


def _validate_bound_provenance(
    data: Mapping[str, Any], bundle_metadata: Mapping[str, Any], bundle_metadata_sha256: str
) -> None:
    if data.get("bundle_metadata_sha256") != bundle_metadata_sha256:
        raise ValueError("artifact bundle_metadata_sha256 does not match loaded bundle metadata")
    for field in _REQUIRED_METADATA_FIELDS:
        expected = bundle_metadata.get(field)
        if expected is None:
            raise ValueError(f"bundle metadata missing provenance field {field}")
        if data.get(field) != expected:
            raise ValueError(f"artifact provenance field {field} does not match loaded bundle metadata")


def _episode_query_infer_ms(data: Mapping[str, Any]) -> tuple[list[float], str]:
    if "query_infer_ms" in data:
        raise ValueError("episode artifact must store timings per query, not top-level query_infer_ms")
    episodes = data.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("episode artifact episodes must be a non-empty list")
    values: list[float] = []
    for episode_index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise ValueError("episode artifact episodes entries must be mappings")
        if "query_infer_ms" in episode:
            raise ValueError("episode artifact must store timings per query, not episode query_infer_ms")
        queries = episode.get("queries")
        if not isinstance(queries, list) or not queries:
            raise ValueError(f"episode artifact episode {episode_index} queries must be a non-empty list")
        for query_index, query in enumerate(queries):
            if not isinstance(query, Mapping):
                raise ValueError("episode queries entries must be mappings")
            source = query.get("measurement_source")
            if source != _EPISODE_QUERY_MEASUREMENT_SOURCE:
                raise ValueError(
                    "episode query measurement_source must be "
                    f"{_EPISODE_QUERY_MEASUREMENT_SOURCE!r}"
                )
            timing = query.get("policy_timing")
            if not isinstance(timing, Mapping) or "infer_ms" not in timing:
                raise ValueError(
                    f"episode {episode_index} query {query_index} is missing policy_timing.infer_ms"
                )
            values.append(_positive_float_value(timing["infer_ms"], "policy_timing.infer_ms"))
    return values, _EPISODE_QUERY_MEASUREMENT_SOURCE


def _numeric_list(values: list[Any], field: str) -> list[float]:
    return [_positive_float_value(value, field) for value in values]


def _prepare_observation_mapping(payload: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    data = _to_device(dict(payload), device)
    if "image" not in data or "state" not in data:
        raise ValueError("episode_data observation mapping must include image and state")
    if "image_mask" not in data:
        images = data["image"]
        if not isinstance(images, Mapping):
            raise ValueError("observation image must be a mapping when image_mask is omitted")
        data["image_mask"] = {key: torch.ones((BATCH_SIZE,), dtype=torch.bool, device=device) for key in images}
    state = data["state"]
    if torch.is_tensor(state) and state.ndim == 1:
        data["state"] = state[None, :]
    images = data["image"]
    if isinstance(images, Mapping):
        fixed_images = {}
        for key, value in images.items():
            tensor = value
            if torch.is_tensor(tensor) and tensor.ndim == 3:
                tensor = tensor[None, ...]
            fixed_images[key] = tensor
        data["image"] = fixed_images
    return data


def _to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(value, **{field.name: _to_device(getattr(value, field.name), device) for field in dataclasses.fields(value)})
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        try:
            tensor = torch.as_tensor(value, device=device)
        except (TypeError, ValueError):
            return [_to_device(item, device) for item in value]
        return tensor.to(dtype=torch.float32) if tensor.is_floating_point() else tensor
    return value


def _module_dtype(module: torch.nn.Module) -> torch.dtype:
    parameter = next(module.parameters(), None)
    return torch.float32 if parameter is None else parameter.dtype


def _runtime_report(info: Mapping[str, Any]) -> dict[str, str]:
    return {
        "device": _required_str(info, "device"),
        "device_type": _required_str(info, "device_type"),
        "gpu_name": _required_str(info, "gpu_name"),
        "compute_capability": _required_str(info, "compute_capability"),
        "torch_version": _required_str(info, "torch_version"),
        "triton_version": _required_str(info, "triton_version"),
        "cuda_version": _required_str(info, "cuda_version"),
    }


def _bundle_identity(metadata: Mapping[str, Any], bundle_path: str | Path, metadata_sha: str) -> dict[str, Any]:
    return {
        "bundle_path": str(bundle_path),
        "bundle_metadata_sha256": metadata_sha,
        "schema_version": _required_positive_int(metadata, "schema_version"),
        "base_checkpoint_uri": _required_metadata_str(metadata, "base_checkpoint_uri"),
        "base_checkpoint_checksum": _required_metadata_str(metadata, "base_checkpoint_checksum"),
        "base_assets_checksum": _required_metadata_str(metadata, "base_assets_checksum"),
        "memory_backend": _required_metadata_str(metadata, "memory_backend"),
        "memory_state_schema_version": _required_positive_int(metadata, "memory_state_schema_version"),
        "progress_depth": _required_positive_int(metadata, "progress_depth"),
        "progress_layer_mapping": _required_list(metadata, "progress_layer_mapping"),
        "handoff_ratio": _required_nonnegative_float(metadata, "handoff_ratio"),
        "num_denoise_steps": _required_positive_int(metadata, "num_denoise_steps"),
        "prediction_horizon": _required_positive_int(metadata, "prediction_horizon"),
        "execution_horizon": _required_positive_int(metadata, "execution_horizon"),
        "kernel_mode": _required_metadata_str(metadata, "kernel_mode"),
    }


def _validate_bundle_metadata_for_report(metadata: Mapping[str, Any]) -> None:
    for field in (
        "schema_version",
        "base_checkpoint_uri",
        "base_checkpoint_checksum",
        "base_assets_checksum",
        "robomme_policy_commit",
        "robomme_benchmark_commit",
        "mamba_repo_commit",
        "memory_backend",
        "memory_state_schema_version",
        "progress_depth",
        "progress_layer_mapping",
        "handoff_ratio",
        "num_denoise_steps",
        "prediction_horizon",
        "execution_horizon",
        "kernel_mode",
    ):
        if field not in metadata or metadata[field] is None:
            raise ValueError(f"FutureMamba bundle metadata missing required profile field {field}")


def _reject_nulls(value: Any, path: str = "report") -> None:
    if value is None:
        raise ValueError(f"{path} must not be null")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_nulls(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_nulls(item, f"{path}[{index}]")


def _required_metadata_str(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"FutureMamba bundle metadata field {field} must be a non-empty string")
    return value


def _required_str(data: Mapping[str, Any], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _required_list(data: Mapping[str, Any], field: str) -> list[Any]:
    value = data.get(field)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    return list(value)


def _required_positive_int(data: Mapping[str, Any], field: str) -> int:
    value = data.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return int(value)


def _required_nonnegative_int(data: Mapping[str, Any], field: str) -> int:
    value = data.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return int(value)


def _required_positive_float(data: Mapping[str, Any], field: str) -> float:
    return _positive_float_value(data.get(field), field)


def _required_nonnegative_float(data: Mapping[str, Any], field: str) -> float:
    value = data.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return number


def _positive_float_value(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field} must be a positive number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be a positive finite number")
    return number


if __name__ == "__main__":
    raise SystemExit(main())
