from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable, Mapping, Sequence

SOLVER_STEPS = 10
BATCH_SIZE = 1


def _tree_leaves(value: Any):
    if isinstance(value, Mapping):
        for key in sorted(value):
            yield from _tree_leaves(value[key])
    elif dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from _tree_leaves(getattr(value, field.name))
    elif isinstance(value, (tuple, list)) and not isinstance(value, (bytes, bytearray)):
        for item in value:
            yield from _tree_leaves(item)
    else:
        yield value


def _leaf_nbytes(value: Any) -> int:
    if hasattr(value, "nbytes"):
        return int(value.nbytes)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, bool):
        return 1
    if isinstance(value, int | float):
        return 8
    return 0


def tree_nbytes(value: Any) -> int:
    return sum(_leaf_nbytes(leaf) for leaf in _tree_leaves(value))


def _leaf_size(value: Any) -> int:
    if hasattr(value, "size"):
        return int(value.size)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, bool):
        return 1
    if isinstance(value, int | float):
        return 1
    return 0


def count_parameters(params: Any) -> int:
    return sum(_leaf_size(leaf) for leaf in _tree_leaves(params))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
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


def _model_parameter_counts(model: Any) -> dict[str, int]:
    if hasattr(model, "parameter_counts"):
        counts = model.parameter_counts()
        return {
            "total": int(counts["total"]),
            "trainable": int(counts["trainable"]),
            "plugin": int(counts.get("plugin", counts["trainable"])),
        }
    if hasattr(model, "params"):
        total = count_parameters(model.params)
    else:
        raise RuntimeError("model must expose parameter_counts() or params for profiling")
    trainable = count_parameters(getattr(model, "trainable_params", getattr(model, "futuremamba_params", {})))
    plugin = count_parameters(getattr(model, "futuremamba_params", getattr(model, "trainable_params", {})))
    return {"total": total, "trainable": trainable, "plugin": plugin}


def _empty_batch(model: Any) -> tuple[Any, Any, Any, Any]:
    if hasattr(model, "profile_batch"):
        batch = model.profile_batch(batch_size=BATCH_SIZE)
        return batch["observation"], batch.get("executed_actions"), batch.get("executed_action_mask"), batch.get("rng", 0)
    action_dim = int(getattr(model, "action_dim", 7))
    executed_horizon = int(getattr(getattr(model, "config", object()), "executed_horizon", 5))
    observation = getattr(model, "fake_observation", {"batch_size": BATCH_SIZE})
    executed_actions = [[[0.0 for _ in range(action_dim)] for _ in range(executed_horizon)] for _ in range(BATCH_SIZE)]
    executed_action_mask = [[False for _ in range(executed_horizon)] for _ in range(BATCH_SIZE)]
    return observation, executed_actions, executed_action_mask, 0


def _progress_flops(model: Any) -> int | None:
    value = getattr(model, "progress_flops", None)
    if callable(value):
        value = value(batch_size=BATCH_SIZE, solver_steps=SOLVER_STEPS)
    return None if value is None else int(value)


def _backend_parameter_errors(model: Any, diagnostics: Mapping[str, Any] | None = None) -> dict[str, float]:
    errors = dict(getattr(model, "backend_parameter_errors", {}) or {})
    if diagnostics and "backend_parameter_errors" in diagnostics:
        errors.update(diagnostics["backend_parameter_errors"])
    return {str(key): float(value) for key, value in errors.items()}


def profile_model(
    model: Any,
    *,
    timer: Callable[[], float] | None = None,
    warmup_queries: int = 3,
    measured_queries: int = 20,
    gpu_memory_reader: Callable[[], int | None] | None = None,
) -> dict[str, Any]:
    timer = time.perf_counter if timer is None else timer
    counts = _model_parameter_counts(model)
    observation, executed_actions, executed_action_mask, rng = _empty_batch(model)
    memory_state = model.initial_memory_state(BATCH_SIZE)
    state_bytes = tree_nbytes(memory_state)
    diagnostics: Mapping[str, Any] | None = None

    for _ in range(warmup_queries):
        before = timer()
        _, memory_state, diagnostics = model.sample_actions_with_memory(
            rng,
            observation,
            memory_state,
            executed_actions,
            executed_action_mask,
            num_steps=SOLVER_STEPS,
        )
        after = timer()
        if after < before:
            raise RuntimeError("profile timer must be monotonic")

    latencies_ms: list[float] = []
    peak = gpu_memory_reader() if gpu_memory_reader is not None else None
    for _ in range(measured_queries):
        before = timer()
        _, memory_state, diagnostics = model.sample_actions_with_memory(
            rng,
            observation,
            memory_state,
            executed_actions,
            executed_action_mask,
            num_steps=SOLVER_STEPS,
        )
        after = timer()
        latencies_ms.append((after - before) * 1000.0)
        if gpu_memory_reader is not None:
            current = gpu_memory_reader()
            if current is not None:
                peak = current if peak is None else max(peak, current)

    plugin_ratio = counts["plugin"] / counts["total"] if counts["total"] else 0.0
    return {
        "batch_size": BATCH_SIZE,
        "solver_steps": SOLVER_STEPS,
        "total_parameters": counts["total"],
        "trainable_parameters": counts["trainable"],
        "plugin_parameters": counts["plugin"],
        "plugin_parameter_ratio": plugin_ratio,
        "lightweight_claim": plugin_ratio <= 0.10,
        "progress_flops": _progress_flops(model),
        "state_bytes": state_bytes,
        "latency_p50_ms": percentile(latencies_ms, 0.50),
        "latency_p95_ms": percentile(latencies_ms, 0.95),
        "gpu_peak_bytes": peak,
        "backend_parameter_errors": _backend_parameter_errors(model, diagnostics),
    }


def to_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class FakeFutureMamba:
    progress_flops = 12_345
    backend_parameter_errors = {"gru": 0.03, "lstm": 0.04, "frame_stack": 0.02}
    action_dim = 7

    def __init__(self):
        self.calls = 0

    def parameter_counts(self) -> dict[str, int]:
        return {"total": 1_000_000, "trainable": 80_000, "plugin": 80_000}

    def initial_memory_state(self, batch_size: int):
        return {"ssm": bytearray(batch_size * 64), "conv": bytearray(batch_size * 32)}

    def sample_actions_with_memory(
        self,
        rng,
        observation,
        memory_state,
        executed_actions,
        executed_action_mask,
        *,
        num_steps: int,
    ):
        if num_steps != SOLVER_STEPS:
            raise ValueError(f"profile smoke expects {SOLVER_STEPS} solver steps, got {num_steps}")
        self.calls += 1
        return [[[0.0] * self.action_dim]], memory_state, {"backend_parameter_errors": self.backend_parameter_errors}


def _gpu_peak_reader_from_jax(jax_module: Any) -> Callable[[], int | None]:
    def read_peak() -> int | None:
        try:
            devices = jax_module.devices("gpu")
        except Exception:
            return None
        peaks: list[int] = []
        for device in devices:
            stats = getattr(device, "memory_stats", lambda: None)()
            if not stats:
                continue
            for key in ("peak_bytes_in_use", "bytes_in_use"):
                if key in stats:
                    peaks.append(int(stats[key]))
                    break
        return max(peaks) if peaks else None

    return read_peak


def load_real_model(config_name: str, checkpoint_root: str | Path | None):
    try:
        jax = importlib.import_module("jax")
        nnx = importlib.import_module("flax.nnx")
        training_config = importlib.import_module("openpi.training.config")
        weight_loaders = importlib.import_module("openpi.training.weight_loaders")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Unable to load real FutureMamba dependencies. Install the project environment before running "
            "without --fake-smoke."
        ) from error

    try:
        config = training_config.get_config(config_name)
    except Exception as error:
        raise RuntimeError(f"Unable to resolve training config {config_name!r}") from error

    params_path = resolve_checkpoint_params(checkpoint_root)
    if params_path is not None:
        config = dataclasses.replace(config, weight_loader=weight_loaders.CheckpointWeightLoader(str(params_path)))

    try:
        model = config.model.create(jax.random.key(0))
        params = nnx.state(model, nnx.Param)
        loaded = config.weight_loader.load(params)
        nnx.update(model, loaded)
    except Exception as error:
        raise RuntimeError(f"Unable to instantiate FutureMamba model for {config_name!r}") from error
    return model


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile FutureMamba batch=1 10-step query efficiency.")
    parser.add_argument("--config", default="futuremamba_libero_mem")
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument("--fake-smoke", action="store_true", help="Run deterministic offline smoke without real JAX/OpenPI deps.")
    parser.add_argument("--warmup-queries", type=int, default=3)
    parser.add_argument("--measured-queries", type=int, default=20)
    args = parser.parse_args(argv)

    if args.fake_smoke:
        report = profile_model(
            FakeFutureMamba(),
            warmup_queries=args.warmup_queries,
            measured_queries=args.measured_queries,
            gpu_memory_reader=lambda: 0,
        )
    else:
        model = load_real_model(args.config, args.checkpoint_root)
        try:
            jax = importlib.import_module("jax")
        except ModuleNotFoundError:
            gpu_reader = lambda: None
        else:
            gpu_reader = _gpu_peak_reader_from_jax(jax)
        report = profile_model(
            model,
            warmup_queries=args.warmup_queries,
            measured_queries=args.measured_queries,
            gpu_memory_reader=gpu_reader,
        )
    print(to_json(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
