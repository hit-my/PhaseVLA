#!/usr/bin/env python3
"""Run the fixed Mamba-3 SISO RTX 5090 gate suite.

The runner is intentionally independent from OpenPI training code.  It probes the
official ``mamba_ssm.modules.mamba3.Mamba3`` implementation and writes a stable
machine-readable JSON report even when a gate raises.  Real OpenPI checkpoint and
RoboMME deployment evidence must be supplied as JSON; this runner never marks
those integration gates as passed from a kernel smoke alone.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from collections.abc import Mapping
import datetime as _datetime
from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any


OFFICIAL_MAMBA_COMMIT = "77069de5cdb55cbe98b670889c80df211e031039"
EXPECTED_TORCH = "2.9.1"
EXPECTED_CUDA = "12.8"
EXPECTED_TRITON = "3.5.1"
EXPECTED_MAMBA_VERSION = "2.3.1"
EXPECTED_COMPUTE_CAPABILITY = (12, 0)
EXPECTED_GPU_SUBSTRING = "RTX 5090"
SUPPORTED = "supported"
UNSUPPORTED = "unsupported_on_current_stack"
GATE_STATUS_VALUES = ("passed", "failed", "not_run")
GATE_NAMES = (
    "dependencies",
    "device_256_steps",
    "forward_10_seeds",
    "backward_10_seeds",
    "sequence_step_parity",
    "causality",
    "reset",
    "fixed_state_bytes",
    "openpi_checkpoint_2_to_4",
    "robomme_deployment",
)
FORWARD_SEEDS = tuple(range(10))
BACKWARD_SEEDS = tuple(range(10))
PARITY_NRMSE_MAX = 1e-3
PARITY_COSINE_MIN = 0.9999
CAUSAL_PREFIX = 5
PROTOCOL = ("reset", "add_buffer", "infer", "execute_16_steps", "add_buffer", "infer")
ROBOMME_POLICY_COMMIT = "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
ROBOMME_BENCHMARK_COMMIT = "856bc3a189d4172f3f47dbee4424d585f8d78db3"



class GateContractError(RuntimeError):
    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type

@dataclass(frozen=True)
class GateResult:
    status: str
    evidence: dict[str, Any]
    error: dict[str, str] | None = None

    @staticmethod
    def passed(evidence: Mapping[str, Any] | None = None) -> "GateResult":
        return GateResult("passed", dict(evidence or {}), None)

    @staticmethod
    def failed(error: BaseException | Mapping[str, Any] | str, evidence: Mapping[str, Any] | None = None) -> "GateResult":
        return GateResult("failed", dict(evidence or {}), _serialize_error(error))

    @staticmethod
    def not_run(message: str, evidence: Mapping[str, Any] | None = None) -> "GateResult":
        return GateResult("not_run", dict(evidence or {}), {"type": "NotRun", "message": message})


@dataclass
class GateContext:
    import_module: Callable[[str], Any] = importlib.import_module
    metadata_version: Callable[[str], str] = importlib.metadata.version
    environ: Mapping[str, str] | None = None
    openpi_evidence_path: Path | None = None
    robomme_evidence_path: Path | None = None
    objects: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.environ is None:
            self.environ = os.environ
        if self.objects is None:
            self.objects = {}


def _utc_now() -> str:
    return _datetime.datetime.now(_datetime.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _serialize_error(error: BaseException | Mapping[str, Any] | str) -> dict[str, str]:
    if isinstance(error, GateContractError):
        return {"type": error.error_type[:120], "message": str(error)[:500]}
    if isinstance(error, BaseException):
        message = str(error)
        return {"type": error.__class__.__name__, "message": message[:500]}
    if isinstance(error, Mapping):
        error_type = str(error.get("type", "GateFailure"))
        message = str(error.get("message", "gate failed"))
        return {"type": error_type[:120], "message": message[:500]}
    return {"type": "GateFailure", "message": str(error)[:500]}


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _version_matches(actual: str | None, expected: str) -> bool:
    return actual is not None and actual.split("+")[0] == expected


def _safe_version(ctx: GateContext, package: str) -> str | None:
    try:
        return ctx.metadata_version(package)
    except BaseException:
        return None


def collect_runtime(ctx: GateContext) -> dict[str, Any]:
    torch_version = None
    cuda_version = None
    triton_version = None
    mamba_version = _safe_version(ctx, "mamba-ssm")
    gpu = None
    compute_capability = None

    try:
        torch = ctx.import_module("torch")
        torch_version = getattr(torch, "__version__", None)
        cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
        try:
            if torch.cuda.is_available():
                gpu = torch.cuda.get_device_name(0)
                compute_capability = list(torch.cuda.get_device_capability(0))
        except BaseException:
            gpu = None
            compute_capability = None
    except BaseException:
        pass

    try:
        triton = ctx.import_module("triton")
        triton_version = getattr(triton, "__version__", None)
    except BaseException:
        pass

    return {
        "official_mamba_commit": OFFICIAL_MAMBA_COMMIT,
        "expected": {
            "torch": EXPECTED_TORCH,
            "cuda": EXPECTED_CUDA,
            "triton": EXPECTED_TRITON,
            "mamba": EXPECTED_MAMBA_VERSION,
            "compute_capability": list(EXPECTED_COMPUTE_CAPABILITY),
            "gpu_contains": EXPECTED_GPU_SUBSTRING,
        },
        "torch": torch_version,
        "cuda": cuda_version,
        "triton": triton_version,
        "mamba": mamba_version,
        "gpu": gpu,
        "compute_capability": compute_capability,
    }


def _require(condition: bool, error_type: str, message: str) -> None:
    if not condition:
        raise GateContractError(error_type, message)


def _import_attr(ctx: GateContext, module_name: str, attr_name: str) -> Any:
    module = ctx.import_module(module_name)
    return getattr(module, attr_name)


def _class_identity(cls: Any) -> str:
    return f"{getattr(cls, '__module__', '')}.{getattr(cls, '__qualname__', getattr(cls, '__name__', ''))}"


def _dependencies_gate(ctx: GateContext) -> GateResult:
    evidence: dict[str, Any] = {}
    try:
        torch = ctx.import_module("torch")
        triton = ctx.import_module("triton")
        Mamba3 = _import_attr(ctx, "mamba_ssm.modules.mamba3", "Mamba3")
        siso_kernel = _import_attr(
            ctx,
            "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined",
            "mamba3_siso_combined",
        )
        rotary_step = _import_attr(
            ctx,
            "mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step",
            "apply_rotary_qk_inference_fwd",
        )
        cute_step = _import_attr(ctx, "mamba_ssm.ops.cute.mamba3.mamba3_step_fn", "mamba3_step_fn")
        mamba_version = _safe_version(ctx, "mamba-ssm")

        class_identity = _class_identity(Mamba3)
        evidence.update(
            {
                "torch": getattr(torch, "__version__", None),
                "cuda": getattr(getattr(torch, "version", None), "cuda", None),
                "triton": getattr(triton, "__version__", None),
                "mamba": mamba_version,
                "mamba3_class": class_identity,
                "siso_kernel": callable(siso_kernel),
                "rotary_step_kernel": callable(rotary_step),
                "cute_step_kernel": callable(cute_step),
            }
        )
        _require(_version_matches(evidence["torch"], EXPECTED_TORCH), "TorchVersion", f"expected {EXPECTED_TORCH}, got {evidence['torch']!r}")
        _require(_version_matches(evidence["cuda"], EXPECTED_CUDA), "CudaVersion", f"expected {EXPECTED_CUDA}, got {evidence['cuda']!r}")
        _require(_version_matches(evidence["triton"], EXPECTED_TRITON), "TritonVersion", f"expected {EXPECTED_TRITON}, got {evidence['triton']!r}")
        _require(_version_matches(mamba_version, EXPECTED_MAMBA_VERSION), "MambaVersion", f"expected {EXPECTED_MAMBA_VERSION}, got {mamba_version!r}")
        _require(class_identity == "mamba_ssm.modules.mamba3.Mamba3", "Mamba3Identity", f"official Mamba3 required, got {class_identity}")
        _require(callable(siso_kernel), "Mamba3Kernel", "SISO sequence kernel is not callable")
        _require(callable(rotary_step), "Mamba3Kernel", "rotary step kernel is not callable")
        _require(callable(cute_step), "Mamba3Kernel", "CuTe step kernel is not callable")
        ctx.objects["torch"] = torch
        ctx.objects["Mamba3"] = Mamba3
        return GateResult.passed(evidence)
    except BaseException as exc:
        return GateResult.failed(exc, evidence)


def _torch_and_mamba(ctx: GateContext) -> tuple[Any, Any]:
    torch = ctx.objects.get("torch")
    Mamba3 = ctx.objects.get("Mamba3")
    if torch is None:
        torch = ctx.import_module("torch")
        ctx.objects["torch"] = torch
    if Mamba3 is None:
        Mamba3 = _import_attr(ctx, "mamba_ssm.modules.mamba3", "Mamba3")
        ctx.objects["Mamba3"] = Mamba3
    return torch, Mamba3


def _new_model(ctx: GateContext, *, seed: int, dtype: Any | None = None, batch_size: int = 2) -> Any:
    torch, Mamba3 = _torch_and_mamba(ctx)
    torch.manual_seed(seed)
    device = torch.device("cuda")
    dtype = dtype or torch.float32
    model = Mamba3(
        d_model=64,
        d_state=32,
        expand=2,
        headdim=64,
        ngroups=1,
        rope_fraction=0.5,
        is_mimo=False,
        is_outproj_norm=False,
        chunk_size=64,
        layer_idx=0,
        device=device,
        dtype=dtype,
    )
    model.train()
    states = model.allocate_inference_cache(batch_size, 2048, device=device, dtype=dtype)
    return model, states


def _finite_tensor(torch: Any, tensor: Any) -> bool:
    return bool(torch.isfinite(tensor).all().item())


def _state_summary(states: tuple[Any, ...]) -> list[dict[str, Any]]:
    return [
        {
            "shape": list(state.shape),
            "dtype": str(state.dtype).replace("torch.", ""),
            "bytes": int(state.numel() * state.element_size()),
        }
        for state in states
    ]


def _state_total_bytes(states: tuple[Any, ...]) -> int:
    return int(sum(state.numel() * state.element_size() for state in states))


def _clone_states(states: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(state.clone() for state in states)


def _zero_states(states: tuple[Any, ...]) -> None:
    for state in states:
        state.zero_()


def _step_tokens(torch: Any, model: Any, x: Any, states: tuple[Any, ...]) -> tuple[Any, tuple[Any, ...]]:
    outputs = []
    current = states
    for index in range(x.shape[1]):
        y, angle, ssm, k, v = model.step(x[:, index], *current)
        current = (angle, ssm, k, v)
        outputs.append(y.unsqueeze(1))
    if not outputs:
        return torch.empty(x.shape[0], 0, x.shape[2], device=x.device, dtype=x.dtype), current
    return torch.cat(outputs, dim=1), current


def _device_gate(ctx: GateContext) -> GateResult:
    evidence: dict[str, Any] = {}
    try:
        torch, _ = _torch_and_mamba(ctx)
        _require(torch.cuda.is_available(), "CudaUnavailable", "torch.cuda.is_available() is false")
        gpu = torch.cuda.get_device_name(0)
        capability = tuple(torch.cuda.get_device_capability(0))
        evidence["gpu"] = gpu
        evidence["compute_capability"] = list(capability)
        _require(EXPECTED_GPU_SUBSTRING in gpu, "GpuMismatch", f"expected {EXPECTED_GPU_SUBSTRING}, got {gpu}")
        _require(capability == EXPECTED_COMPUTE_CAPABILITY, "ComputeCapability", f"expected {EXPECTED_COMPUTE_CAPABILITY}, got {capability}")
        model, states = _new_model(ctx, seed=0, dtype=torch.float32, batch_size=2)
        x = torch.randn(2, 256, 64, device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            output, final_states = _step_tokens(torch, model.eval(), x, states)
        evidence["steps"] = 256
        evidence["output_shape"] = list(output.shape)
        evidence["state"] = _state_summary(final_states)
        evidence["state_bytes"] = _state_total_bytes(final_states)
        _require(tuple(output.shape) == (2, 256, 64), "OutputShape", f"unexpected output shape {tuple(output.shape)}")
        _require(_finite_tensor(torch, output), "NonFiniteOutput", "256-step output contains NaN or Inf")
        return GateResult.passed(evidence)
    except BaseException as exc:
        return GateResult.failed(exc, evidence)


def _forward_gate(ctx: GateContext) -> GateResult:
    evidence: dict[str, Any] = {"seeds": list(FORWARD_SEEDS)}
    try:
        torch, _ = _torch_and_mamba(ctx)
        outputs = []
        for seed in FORWARD_SEEDS:
            model, _states = _new_model(ctx, seed=seed, dtype=torch.float32, batch_size=2)
            x = torch.randn(2, 17, 64, device="cuda", dtype=torch.float32)
            y = model.eval()(x)
            _require(tuple(y.shape) == (2, 17, 64), "OutputShape", f"seed {seed} output shape {tuple(y.shape)}")
            _require(_finite_tensor(torch, y), "NonFiniteOutput", f"seed {seed} output contains NaN or Inf")
            outputs.append(float(y.detach().float().norm().item()))
        evidence["output_norms"] = outputs
        return GateResult.passed(evidence)
    except BaseException as exc:
        return GateResult.failed(exc, evidence)


def _backward_gate(ctx: GateContext) -> GateResult:
    evidence: dict[str, Any] = {"seeds": list(BACKWARD_SEEDS)}
    try:
        torch, _ = _torch_and_mamba(ctx)
        trainable_counts = []
        changed_counts = []
        for seed in BACKWARD_SEEDS:
            model, _states = _new_model(ctx, seed=seed, dtype=torch.float32, batch_size=2)
            x = torch.randn(2, 9, 64, device="cuda", dtype=torch.float32, requires_grad=True)
            trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
            before = {name: parameter.detach().clone() for name, parameter in trainable.items()}
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            y = model.train()(x)
            loss = y.float().square().mean()
            loss.backward()
            _require(x.grad is not None and _finite_tensor(torch, x.grad), "InputGradient", f"seed {seed} input gradient missing or non-finite")
            for name, parameter in trainable.items():
                _require(parameter.grad is not None, "ParameterGradient", f"seed {seed} missing grad for {name}")
                _require(_finite_tensor(torch, parameter.grad), "ParameterGradient", f"seed {seed} non-finite grad for {name}")
            optimizer.step()
            unchanged = [name for name, parameter in trainable.items() if torch.equal(before[name], parameter.detach())]
            trainable_counts.append(len(trainable))
            changed_counts.append(len(trainable) - len(unchanged))
            evidence["trainable_parameter_counts"] = trainable_counts
            evidence["changed_parameter_counts"] = changed_counts
            _require(not unchanged, "OptimizerStep", f"seed {seed} optimizer step left trainable parameters unchanged: {unchanged}")
        return GateResult.passed(evidence)
    except BaseException as exc:
        return GateResult.failed(exc, evidence)


def _tensor_nrmse(torch: Any, lhs: Any, rhs: Any) -> float:
    lhs_f = lhs.detach().float()
    rhs_f = rhs.detach().float()
    denom = torch.sqrt(torch.mean(rhs_f.square())).clamp_min(1e-12)
    return float((torch.sqrt(torch.mean((lhs_f - rhs_f).square())) / denom).item())


def _tensor_cosine(torch: Any, lhs: Any, rhs: Any) -> float:
    lhs_f = lhs.detach().float().reshape(-1)
    rhs_f = rhs.detach().float().reshape(-1)
    lhs_norm = torch.linalg.vector_norm(lhs_f)
    rhs_norm = torch.linalg.vector_norm(rhs_f)
    if float(lhs_norm.item()) <= 1e-12 and float(rhs_norm.item()) <= 1e-12:
        return 1.0
    if float(lhs_norm.item()) <= 1e-12 or float(rhs_norm.item()) <= 1e-12:
        return 0.0
    return float(torch.nn.functional.cosine_similarity(lhs_f, rhs_f, dim=0, eps=1e-12).item())


def _state_metrics(torch: Any, lhs: tuple[Any, ...], rhs: tuple[Any, ...]) -> list[dict[str, float]]:
    return [
        {"nrmse": _tensor_nrmse(torch, left, right), "cosine": _tensor_cosine(torch, left, right)}
        for left, right in zip(lhs, rhs, strict=True)
    ]


def _parity_gate(ctx: GateContext) -> GateResult:
    evidence: dict[str, Any] = {"nrmse_max": PARITY_NRMSE_MAX, "cosine_min": PARITY_COSINE_MIN}
    try:
        torch, _ = _torch_and_mamba(ctx)
        model, sequence_states = _new_model(ctx, seed=1234, dtype=torch.float32, batch_size=2)
        step_states = tuple(torch.zeros_like(state) for state in sequence_states)
        x = torch.randn(2, 33, 64, device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            cache = SimpleNamespace(seqlen_offset=0, key_value_memory_dict={})
            sequence_output = model.eval()(x, inference_params=cache)
            sequence_final_states = cache.key_value_memory_dict[0]
            step_output, final_step_states = _step_tokens(torch, model.eval(), x, step_states)
        output_nrmse = _tensor_nrmse(torch, step_output, sequence_output)
        output_cosine = _tensor_cosine(torch, step_output, sequence_output)
        state_metrics = _state_metrics(torch, final_step_states, sequence_final_states)
        _require(math.isfinite(output_nrmse), "ParityMetric", "output NRMSE is non-finite")
        _require(math.isfinite(output_cosine), "ParityMetric", "output cosine is non-finite")
        for index, metrics in enumerate(state_metrics):
            _require(math.isfinite(metrics["nrmse"]), "ParityMetric", f"state {index} NRMSE is non-finite")
            _require(math.isfinite(metrics["cosine"]), "ParityMetric", f"state {index} cosine is non-finite")
        evidence.update(
            {
                "output_nrmse": output_nrmse,
                "output_cosine": output_cosine,
                "state_metrics": state_metrics,
            }
        )
        _require(output_nrmse <= PARITY_NRMSE_MAX, "ParityNRMSE", f"output NRMSE {output_nrmse} exceeds {PARITY_NRMSE_MAX}")
        _require(output_cosine >= PARITY_COSINE_MIN, "ParityCosine", f"output cosine {output_cosine} below {PARITY_COSINE_MIN}")
        for index, metrics in enumerate(state_metrics):
            _require(metrics["nrmse"] <= PARITY_NRMSE_MAX, "StateParityNRMSE", f"state {index} NRMSE {metrics['nrmse']} exceeds {PARITY_NRMSE_MAX}")
            _require(metrics["cosine"] >= PARITY_COSINE_MIN, "StateParityCosine", f"state {index} cosine {metrics['cosine']} below {PARITY_COSINE_MIN}")
        return GateResult.passed(evidence)
    except BaseException as exc:
        return GateResult.failed(exc, evidence)


def _causality_gate(ctx: GateContext) -> GateResult:
    evidence: dict[str, Any] = {"prefix_length": CAUSAL_PREFIX}
    try:
        torch, _ = _torch_and_mamba(ctx)
        model, _states = _new_model(ctx, seed=77, dtype=torch.float32, batch_size=2)
        base = torch.randn(2, 13, 64, device="cuda", dtype=torch.float32)
        changed = base.clone()
        changed[:, CAUSAL_PREFIX:] = torch.randn_like(changed[:, CAUSAL_PREFIX:]) * 7.0
        with torch.inference_mode():
            y_base = model.eval()(base)
            y_changed = model.eval()(changed)
        diff = float((y_base[:, :CAUSAL_PREFIX] - y_changed[:, :CAUSAL_PREFIX]).abs().max().item())
        evidence["max_prefix_abs_diff"] = diff
        _require(diff <= 1e-5, "Causality", f"future-token edit changed prefix by {diff}")
        return GateResult.passed(evidence)
    except BaseException as exc:
        return GateResult.failed(exc, evidence)


def _reset_gate(ctx: GateContext) -> GateResult:
    evidence: dict[str, Any] = {}
    try:
        torch, _ = _torch_and_mamba(ctx)
        model, warm_states = _new_model(ctx, seed=99, dtype=torch.float32, batch_size=2)
        warm = torch.randn(2, 4, 64, device="cuda", dtype=torch.float32)
        probe = torch.randn(2, 6, 64, device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            _, warm_states = _step_tokens(torch, model.eval(), warm, warm_states)
            partial = _clone_states(warm_states)
            for state in partial:
                state[1].zero_()
            continued_out, _ = _step_tokens(torch, model.eval(), probe, _clone_states(warm_states))
            partial_out, _ = _step_tokens(torch, model.eval(), probe, partial)
            zero_states = model.allocate_inference_cache(2, 6, device="cuda", dtype=torch.float32)
            zero_out, _ = _step_tokens(torch, model.eval(), probe, zero_states)
            full_states = _clone_states(warm_states)
            _zero_states(full_states)
            full_out, _ = _step_tokens(torch, model.eval(), probe, full_states)
        reset_row_diff = float((partial_out[1] - zero_out[1]).abs().max().item())
        kept_row_diff = float((partial_out[0] - continued_out[0]).abs().max().item())
        kept_row_vs_zero_diff = float((partial_out[0] - zero_out[0]).abs().max().item())
        full_diff = float((full_out - zero_out).abs().max().item())
        evidence.update(
            {
                "partial_reset_row_max_abs_diff": reset_row_diff,
                "kept_row_vs_continued_max_abs_diff": kept_row_diff,
                "kept_row_vs_zero_max_abs_diff": kept_row_vs_zero_diff,
                "full_reset_max_abs_diff": full_diff,
            }
        )
        _require(reset_row_diff <= 1e-5, "PartialReset", f"reset row differs from zero state by {reset_row_diff}")
        _require(kept_row_diff <= 1e-5, "PartialReset", f"non-reset row changed by {kept_row_diff}")
        _require(kept_row_vs_zero_diff > 1e-7, "PartialReset", "non-reset row matched zero state; partial reset affected too much")
        _require(full_diff <= 1e-5, "FullReset", f"full reset differs from zero state by {full_diff}")
        return GateResult.passed(evidence)
    except BaseException as exc:
        return GateResult.failed(exc, evidence)


def _fixed_state_gate(ctx: GateContext) -> GateResult:
    evidence: dict[str, Any] = {}
    try:
        torch, _ = _torch_and_mamba(ctx)
        model, _unused_states = _new_model(ctx, seed=55, dtype=torch.float32, batch_size=2)
        short_states = model.allocate_inference_cache(2, 2, device="cuda", dtype=torch.float32)
        long_states = model.allocate_inference_cache(2, 2048, device="cuda", dtype=torch.float32)
        x_short = torch.randn(2, 2, 64, device="cuda", dtype=torch.float32)
        x_long = torch.randn(2, 2048, 64, device="cuda", dtype=torch.float32)
        with torch.inference_mode():
            _, final_short = _step_tokens(torch, model.eval(), x_short, short_states)
            _, final_long = _step_tokens(torch, model.eval(), x_long, long_states)
        short_summary = _state_summary(final_short)
        long_summary = _state_summary(final_long)
        evidence.update(
            {
                "length_2_state": short_summary,
                "length_2048_state": long_summary,
                "length_2_bytes": _state_total_bytes(final_short),
                "length_2048_bytes": _state_total_bytes(final_long),
            }
        )
        _require(short_summary == long_summary, "StateShapeBytes", "state tree/shape/bytes differ between length 2 and 2048")
        return GateResult.passed(evidence)
    except BaseException as exc:
        return GateResult.failed(exc, evidence)


def _read_evidence(path: Path | None, gate_name: str) -> tuple[dict[str, Any] | None, GateResult | None]:
    if path is None:
        return None, GateResult.failed({"type": "MissingEvidence", "message": f"{gate_name} evidence JSON path was not provided"})
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except BaseException as exc:
        return None, GateResult.failed(exc, {"path": str(path)})
    if not isinstance(data, dict):
        return None, GateResult.failed({"type": "EvidenceFormat", "message": "evidence JSON must be an object"}, {"path": str(path)})
    return data, None


def _identity_evidence(data: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "gate",
        "backend",
        "mamba_repo_commit",
        "checkpoint_identity",
        "config_identity",
        "robomme_policy_commit",
        "robomme_benchmark_commit",
    )
    return {key: data.get(key) for key in keys if key in data}


def _validate_common_identity(data: Mapping[str, Any], gate_name: str) -> list[str]:
    errors: list[str] = []
    if data.get("status") != "passed":
        errors.append("status must be 'passed'")
    checks = {
        "gate": gate_name,
        "backend": "mamba3_siso",
        "mamba_repo_commit": OFFICIAL_MAMBA_COMMIT,
        "checkpoint_identity": None,
        "config_identity": None,
    }
    for key, expected in checks.items():
        value = data.get(key)
        if expected is None:
            if not isinstance(value, str) or not value:
                errors.append(f"{key} is required")
        elif value != expected:
            errors.append(f"{key} must be {expected!r}")
    return errors

def _load_bound_json_artifacts(
    evidence_path: Path,
    data: Mapping[str, Any],
    required_names: tuple[str, ...],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str], list[str]]:
    artifacts = data.get("artifacts")
    expected_hashes = data.get("artifact_sha256")
    errors: list[str] = []
    if not isinstance(artifacts, Mapping):
        return {}, {}, ["artifacts object is required"]
    if not isinstance(expected_hashes, Mapping):
        return {}, {}, ["artifact_sha256 object is required"]

    bundle_root = evidence_path.parent.resolve()
    loaded: dict[str, Mapping[str, Any]] = {}
    verified_hashes: dict[str, str] = {}
    for name in required_names:
        raw_path = artifacts.get(name)
        expected_hash = expected_hashes.get(name)
        if not isinstance(raw_path, str) or not raw_path:
            errors.append(f"artifacts.{name} path is required")
            continue
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or not expected_hash.isascii():
            errors.append(f"artifact_sha256.{name} must be a 64-character SHA-256 digest")
            continue
        relative_path = Path(raw_path)
        if relative_path.is_absolute():
            errors.append(f"artifacts.{name} must be bundle-relative")
            continue
        artifact_path = (bundle_root / relative_path).resolve()
        try:
            artifact_path.relative_to(bundle_root)
        except ValueError:
            errors.append(f"artifact {name} escapes the evidence bundle")
            continue
        if not artifact_path.is_file():
            errors.append(f"artifact {name} is not a regular file: {artifact_path}")
            continue
        actual_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        if expected_hash.lower() != actual_hash:
            errors.append(f"artifact {name} SHA-256 mismatch")
            continue
        try:
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        except BaseException as exc:
            errors.append(f"artifact {name} is not valid UTF-8 JSON: {exc}")
            continue
        if not isinstance(payload, Mapping):
            errors.append(f"artifact {name} JSON must be an object")
            continue
        loaded[name] = payload
        verified_hashes[name] = actual_hash
    return loaded, verified_hashes, errors


def _identity_matches_hash(data: Mapping[str, Any], identity_name: str, digest: str, errors: list[str]) -> None:
    if data.get(identity_name) != f"sha256:{digest}":
        errors.append(f"{identity_name} must bind to the verified artifact SHA-256")


def _is_json_int(value: Any, expected: int | None = None) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and (expected is None or value == expected)


def _valid_action_record(record: Any, expected_step: int) -> bool:
    if not isinstance(record, Mapping) or not _is_json_int(record.get("step"), expected_step):
        return False
    action = record.get("action")
    return isinstance(action, list) and len(action) == 8 and all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) for value in action
    )


def validate_openpi_checkpoint_evidence(path: Path | str | None) -> GateResult:
    resolved = Path(path) if path is not None else None
    data, error = _read_evidence(resolved, "openpi_checkpoint_2_to_4")
    if error is not None:
        return error
    assert data is not None and resolved is not None
    errors = _validate_common_identity(data, "openpi_checkpoint_2_to_4")
    required_steps = {
        "first_end_step": 2,
        "resume_start_step": 2,
        "resumed_end_step": 4,
        "checkpoint_step": 4,
        "data_iterator_step": 4,
    }
    for key, expected in required_steps.items():
        if not _is_json_int(data.get(key), expected):
            errors.append(f"{key} must be integer {expected}")
    if data.get("only_futuremamba_parameters_updated") is not True:
        errors.append("only_futuremamba_parameters_updated must be true")

    artifacts, hashes, artifact_errors = _load_bound_json_artifacts(
        resolved,
        data,
        ("checkpoint_metadata", "config", "parameter_diff", "trace"),
    )
    errors.extend(artifact_errors)
    if "checkpoint_metadata" in artifacts:
        metadata = artifacts["checkpoint_metadata"]
        checkpoint_entries = metadata.get("checkpoints")
        valid_checkpoint_steps = (
            len(checkpoint_entries) == 2
            and _is_json_int(checkpoint_entries[0].get("step"), 2)
            and _is_json_int(checkpoint_entries[1].get("step"), 4)
            if isinstance(checkpoint_entries, list) and all(isinstance(entry, Mapping) for entry in checkpoint_entries)
            else False
        )
        if not valid_checkpoint_steps or not _is_json_int(metadata.get("checkpoint_step"), 4) or not _is_json_int(metadata.get("data_iterator_step"), 4):
            errors.append("checkpoint_metadata must record integer checkpoints 2 and 4 with iterator step 4")
        _identity_matches_hash(data, "checkpoint_identity", hashes["checkpoint_metadata"], errors)
    if "config" in artifacts:
        config = artifacts["config"]
        if config.get("memory_backend") != "mamba3_siso":
            errors.append("config artifact memory_backend must be 'mamba3_siso'")
        if config.get("mamba_repo_commit") != OFFICIAL_MAMBA_COMMIT:
            errors.append("config artifact mamba_repo_commit must match the official commit")
        _identity_matches_hash(data, "config_identity", hashes["config"], errors)
    if "parameter_diff" in artifacts:
        parameter_diff = artifacts["parameter_diff"]
        updated = parameter_diff.get("updated_parameters")
        if not isinstance(updated, list) or not updated or not all(
            isinstance(name, str) and name.startswith("futuremamba/") for name in updated
        ):
            errors.append("parameter_diff updated_parameters must be non-empty and restricted to futuremamba/")
        if parameter_diff.get("unchanged_non_futuremamba") is not True:
            errors.append("parameter_diff must prove unchanged_non_futuremamba")
    if "trace" in artifacts:
        trace_events = artifacts["trace"].get("events")
        observed = (
            [(entry.get("event"), entry.get("step")) for entry in trace_events]
            if isinstance(trace_events, list) and all(isinstance(entry, Mapping) for entry in trace_events)
            else []
        )
        expected = [
            ("train_end", 2),
            ("checkpoint_save", 2),
            ("checkpoint_restore", 2),
            ("train_end", 4),
            ("checkpoint_save", 4),
        ]
        if observed != expected or not all(_is_json_int(step) for _, step in observed):
            errors.append("trace must record the exact train/save/restore sequence for steps 2 to 4")
    if errors:
        return GateResult.failed({"type": "EvidenceContract", "message": "; ".join(errors)}, {"path": str(resolved), **_identity_evidence(data)})
    return GateResult.passed(
        {
            "path": str(resolved),
            **_identity_evidence(data),
            "first_end_step": 2,
            "resumed_end_step": 4,
            "verified_artifact_sha256": hashes,
        }
    )


def validate_robomme_deployment_evidence(path: Path | str | None) -> GateResult:
    resolved = Path(path) if path is not None else None
    data, error = _read_evidence(resolved, "robomme_deployment")
    if error is not None:
        return error
    assert data is not None and resolved is not None
    errors = _validate_common_identity(data, "robomme_deployment")
    if data.get("robomme_policy_commit") != ROBOMME_POLICY_COMMIT:
        errors.append(f"robomme_policy_commit must be {ROBOMME_POLICY_COMMIT!r}")
    if data.get("robomme_benchmark_commit") != ROBOMME_BENCHMARK_COMMIT:
        errors.append(f"robomme_benchmark_commit must be {ROBOMME_BENCHMARK_COMMIT!r}")
    episodes = data.get("episodes")
    if not _is_json_int(episodes) or episodes < 1:
        errors.append("episodes must be at least 1")
    if data.get("protocol") != list(PROTOCOL):
        errors.append(f"protocol must be {list(PROTOCOL)!r}")
    if not _is_json_int(data.get("executed_steps"), 16):
        errors.append("executed_steps must be integer 16")
    if data.get("state_isolation_ok") is not True:
        errors.append("state_isolation_ok must be true")
    if data.get("timeout") is not False:
        errors.append("timeout must be false")
    if data.get("protocol_errors") != []:
        errors.append("protocol_errors must be []")

    artifacts, hashes, artifact_errors = _load_bound_json_artifacts(
        resolved,
        data,
        ("checkpoint_metadata", "config", "policy_source", "benchmark_source", "trace", "actions"),
    )
    errors.extend(artifact_errors)
    if "checkpoint_metadata" in artifacts:
        if not _is_json_int(artifacts["checkpoint_metadata"].get("checkpoint_step"), 4):
            errors.append("checkpoint_metadata checkpoint_step must be integer 4")
        _identity_matches_hash(data, "checkpoint_identity", hashes["checkpoint_metadata"], errors)
    if "config" in artifacts:
        config = artifacts["config"]
        if config.get("memory_backend") != "mamba3_siso":
            errors.append("config artifact memory_backend must be 'mamba3_siso'")
        if config.get("mamba_repo_commit") != OFFICIAL_MAMBA_COMMIT:
            errors.append("config artifact mamba_repo_commit must match the official commit")
        _identity_matches_hash(data, "config_identity", hashes["config"], errors)
    if "policy_source" in artifacts and artifacts["policy_source"].get("commit") != ROBOMME_POLICY_COMMIT:
        errors.append("policy_source commit does not match the fixed RoboMME policy commit")
    if "benchmark_source" in artifacts and artifacts["benchmark_source"].get("commit") != ROBOMME_BENCHMARK_COMMIT:
        errors.append("benchmark_source commit does not match the fixed RoboMME benchmark commit")
    if "trace" in artifacts:
        trace = artifacts["trace"]
        trace_events = trace.get("events")
        if not isinstance(trace_events, list) or len(trace_events) != len(PROTOCOL) or not all(
            isinstance(entry, Mapping) for entry in trace_events
        ):
            errors.append("trace events must contain exactly six JSON objects")
        else:
            observed_protocol = [entry.get("event") for entry in trace_events]
            timestamps = [entry.get("timestamp_ns") for entry in trace_events]
            if observed_protocol != list(PROTOCOL):
                errors.append("trace must record the exact RoboMME protocol")
            if not all(_is_json_int(timestamp) for timestamp in timestamps) or any(
                left >= right for left, right in zip(timestamps, timestamps[1:])
            ):
                errors.append("trace timestamp_ns values must be strictly increasing integers")
        if not _is_json_int(trace.get("episodes"), episodes):
            errors.append("trace episodes must be the same JSON integer as the evidence summary")
        if trace.get("state_isolation_ok") is not True or trace.get("timeout") is not False or trace.get("protocol_errors") != []:
            errors.append("trace must prove state isolation with no timeout or protocol errors")
    if "actions" in artifacts:
        records = artifacts["actions"].get("records")
        if not isinstance(records, list) or len(records) != 16 or not all(
            _valid_action_record(record, index) for index, record in enumerate(records)
        ):
            errors.append("actions artifact must contain 16 ordered finite 8-D action records")
    if errors:
        return GateResult.failed({"type": "EvidenceContract", "message": "; ".join(errors)}, {"path": str(resolved), **_identity_evidence(data)})
    return GateResult.passed(
        {
            "path": str(resolved),
            **_identity_evidence(data),
            "episodes": episodes,
            "protocol": list(PROTOCOL),
            "executed_steps": 16,
            "verified_artifact_sha256": hashes,
        }
    )


def _openpi_gate(ctx: GateContext) -> GateResult:
    return validate_openpi_checkpoint_evidence(ctx.openpi_evidence_path)


def _robomme_gate(ctx: GateContext) -> GateResult:
    return validate_robomme_deployment_evidence(ctx.robomme_evidence_path)


def default_gate_impls() -> dict[str, Callable[[GateContext], GateResult]]:
    return {
        "dependencies": _dependencies_gate,
        "device_256_steps": _device_gate,
        "forward_10_seeds": _forward_gate,
        "backward_10_seeds": _backward_gate,
        "sequence_step_parity": _parity_gate,
        "causality": _causality_gate,
        "reset": _reset_gate,
        "fixed_state_bytes": _fixed_state_gate,
        "openpi_checkpoint_2_to_4": _openpi_gate,
        "robomme_deployment": _robomme_gate,
    }


def _gate_to_json(name: str, result: GateResult) -> dict[str, Any]:
    payload: dict[str, Any] = {"name": name, "status": result.status, "evidence": _json_safe(result.evidence)}
    if result.error is not None:
        payload["error"] = _json_safe(result.error)
    return payload


def run_mamba3_gate(
    *,
    ctx: GateContext | None = None,
    gate_impls: Mapping[str, Callable[[GateContext], GateResult]] | None = None,
    runtime_collector: Callable[[GateContext], dict[str, Any]] = collect_runtime,
    generated_at: str | None = None,
) -> dict[str, Any]:
    ctx = ctx or GateContext()
    gate_impls = gate_impls or default_gate_impls()
    gates: list[dict[str, Any]] = []
    failed = False

    runtime = runtime_collector(ctx)
    for name in GATE_NAMES:
        if failed:
            gates.append(_gate_to_json(name, GateResult.not_run("previous gate failed")))
            continue
        impl = gate_impls[name]
        try:
            result = impl(ctx)
        except BaseException as exc:
            result = GateResult.failed(exc)
        if result.status not in GATE_STATUS_VALUES:
            result = GateResult.failed({"type": "InvalidGateStatus", "message": f"{name} returned {result.status!r}"}, result.evidence)
        gates.append(_gate_to_json(name, result))
        if result.status != "passed":
            failed = True

    return {
        "schema_version": 1,
        "generated_at": generated_at or _utc_now(),
        "official_mamba_commit": OFFICIAL_MAMBA_COMMIT,
        "status": UNSUPPORTED if failed else SUPPORTED,
        "runtime": _json_safe(runtime),
        "gates": gates,
    }


def write_json_atomic(path: Path | str, payload: Mapping[str, Any], *, replace: Callable[[str, str], None] = os.replace) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(_json_safe(payload), handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        replace(str(temporary_path), str(destination))
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Mamba-3 SISO RTX 5090 hard gate suite.")
    parser.add_argument("--output", type=Path, default=Path("mamba3_gate.json"), help="JSON output path")
    parser.add_argument("--openpi-evidence", type=Path, default=None, help="External OpenPI 2-to-4 checkpoint evidence JSON")
    parser.add_argument("--robomme-evidence", type=Path, default=None, help="External RoboMME deployment evidence JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    ctx = GateContext(openpi_evidence_path=args.openpi_evidence, robomme_evidence_path=args.robomme_evidence)
    try:
        payload = run_mamba3_gate(ctx=ctx)
    except BaseException as exc:
        runtime = collect_runtime(ctx)
        gates = [_gate_to_json(name, GateResult.not_run("runner failed before gate execution")) for name in GATE_NAMES]
        gates[0] = _gate_to_json("dependencies", GateResult.failed(exc))
        payload = {
            "schema_version": 1,
            "generated_at": _utc_now(),
            "official_mamba_commit": OFFICIAL_MAMBA_COMMIT,
            "status": UNSUPPORTED,
            "runtime": runtime,
            "gates": gates,
        }
    write_json_atomic(args.output, payload)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 0 if payload["status"] == SUPPORTED else 1


if __name__ == "__main__":
    raise SystemExit(main())
