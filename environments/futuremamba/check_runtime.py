#!/usr/bin/env python3
"""Probe the isolated FutureMamba runtime.

The probe is intentionally dependency-gate shaped: it always prints one JSON
object on stdout, never relies on tracebacks for expected missing-dependency
states, and exits non-zero only for the Mamba-2 hard gate or the Torch/CUDA
foundation required by the migration plan. Mamba-3 imports remain diagnostics.
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from typing import Any

EXPECTED_TORCH = "2.9.1"
EXPECTED_TRITON = "3.5.1"
OFFICIAL_MAMBA_COMMIT = "77069de5cdb55cbe98b670889c80df211e031039"
REQUIRED_CUDA_HOME = "/usr/local/cuda-12.8"
REQUIRED_MAMBA_FORCE_BUILD = "TRUE"
REQUIRED_TORCH_CUDA_ARCH_LIST = "12.0"


def _error_text(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def _import_attr(module_name: str, attr_name: str) -> tuple[Any | None, str | None]:
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attr_name), None
    except BaseException as exc:  # noqa: BLE001 - probe must report import crashes as JSON.
        return None, _error_text(exc)


def _import_module(module_name: str) -> tuple[Any | None, str | None]:
    try:
        return importlib.import_module(module_name), None
    except BaseException as exc:  # noqa: BLE001 - probe must report import crashes as JSON.
        return None, _error_text(exc)


def _version_matches(actual: str | None, expected: str) -> bool:
    return actual is not None and actual.split("+")[0] == expected


def _nvcc_version(cuda_home: str) -> str | None:
    candidates = []
    if cuda_home:
        candidates.append(os.path.join(cuda_home, "bin", "nvcc"))
    path_nvcc = shutil.which("nvcc")
    if path_nvcc is not None:
        candidates.append(path_nvcc)

    seen = set()
    for candidate in candidates:
        if candidate in seen or not os.path.exists(candidate):
            continue
        seen.add(candidate)
        try:
            completed = subprocess.run(
                [candidate, "--version"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=10,
            )
        except BaseException:  # noqa: BLE001 - nvcc is diagnostic only.
            continue
        output = " ".join(completed.stdout.split())
        if output:
            return output
    return None


def _record_step(
    payload: dict[str, Any],
    key: str,
    errors: dict[str, str],
    func: Callable[[], bool],
) -> bool:
    try:
        ok = bool(func())
    except BaseException as exc:  # noqa: BLE001 - expected failed probe step.
        payload[key] = False
        errors[key] = _error_text(exc)
        return False
    payload[key] = ok
    return ok


def _run_mamba2_forward(Mamba2: Any, torch: Any) -> bool:  # noqa: N803 - imported class name.
    if Mamba2 is None or torch is None or not torch.cuda.is_available():
        return False
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16
    model = Mamba2(
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
        headdim=32,
        chunk_size=16,
        device=device,
        dtype=dtype,
    ).eval()
    inputs = torch.randn(1, 8, 64, device=device, dtype=dtype)
    with torch.inference_mode():
        output = model(inputs)
    return tuple(output.shape) == (1, 8, 64) and bool(torch.isfinite(output).all().item())


def main() -> int:
    errors: dict[str, str] = {}
    hard_failures: list[str] = []
    cuda_home = os.environ.get("CUDA_HOME")

    payload: dict[str, Any] = {
        "official_mamba_commit": OFFICIAL_MAMBA_COMMIT,
        "cuda_home_required": REQUIRED_CUDA_HOME,
        "cuda_home": cuda_home,
        "cuda_home_ok": cuda_home == REQUIRED_CUDA_HOME,
        "mamba_force_build_required": REQUIRED_MAMBA_FORCE_BUILD,
        "mamba_force_build": os.environ.get("MAMBA_FORCE_BUILD"),
        "torch_cuda_arch_list_required": REQUIRED_TORCH_CUDA_ARCH_LIST,
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "nvcc": _nvcc_version(cuda_home or REQUIRED_CUDA_HOME),
        "torch": None,
        "triton": None,
        "cuda": None,
        "gpu": None,
        "compute_capability": None,
        "mamba2": False,
        "mamba2_forward": False,
        "mamba3": False,
        "mamba3_sequence": False,
        "mamba3_rotary_step": False,
        "mamba3_cute_step": False,
    }

    torch, torch_error = _import_module("torch")
    if torch_error is not None:
        errors["torch"] = torch_error
        hard_failures.append("torch_import")
    else:
        payload["torch"] = getattr(torch, "__version__", None)
        version_cuda = getattr(torch, "version", None)
        payload["cuda"] = getattr(version_cuda, "cuda", None)
        if not _version_matches(payload["torch"], EXPECTED_TORCH):
            hard_failures.append("torch_version")
        try:
            if torch.cuda.is_available():
                payload["gpu"] = torch.cuda.get_device_name(0)
                payload["compute_capability"] = list(torch.cuda.get_device_capability(0))
            else:
                hard_failures.append("cuda_unavailable")
        except BaseException as exc:  # noqa: BLE001 - probe must degrade to JSON.
            errors["cuda"] = _error_text(exc)
            hard_failures.append("cuda_query")

    triton, triton_error = _import_module("triton")
    if triton_error is not None:
        errors["triton"] = triton_error
        hard_failures.append("triton_import")
    else:
        payload["triton"] = getattr(triton, "__version__", None)
        if not _version_matches(payload["triton"], EXPECTED_TRITON):
            hard_failures.append("triton_version")

    Mamba2, mamba2_error = _import_attr("mamba_ssm.modules.mamba2", "Mamba2")
    if mamba2_error is not None:
        errors["mamba2"] = mamba2_error
        hard_failures.append("mamba2_import")
    else:
        payload["mamba2"] = True
        if not _record_step(payload, "mamba2_forward", errors, lambda: _run_mamba2_forward(Mamba2, torch)):
            hard_failures.append("mamba2_forward")

    Mamba3, mamba3_error = _import_attr("mamba_ssm.modules.mamba3", "Mamba3")
    payload["mamba3"] = mamba3_error is None and Mamba3 is not None
    if mamba3_error is not None:
        errors["mamba3"] = mamba3_error

    mamba3_sequence, mamba3_sequence_error = _import_attr(
        "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined",
        "mamba3_siso_combined",
    )
    payload["mamba3_sequence"] = mamba3_sequence_error is None and mamba3_sequence is not None
    if mamba3_sequence_error is not None:
        errors["mamba3_sequence"] = mamba3_sequence_error

    mamba3_rotary_step, mamba3_rotary_step_error = _import_attr(
        "mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step",
        "apply_rotary_qk_inference_fwd",
    )
    payload["mamba3_rotary_step"] = mamba3_rotary_step_error is None and mamba3_rotary_step is not None
    if mamba3_rotary_step_error is not None:
        errors["mamba3_rotary_step"] = mamba3_rotary_step_error

    mamba3_cute_step, mamba3_cute_step_error = _import_attr(
        "mamba_ssm.ops.cute.mamba3.mamba3_step_fn",
        "mamba3_step_fn",
    )
    payload["mamba3_cute_step"] = mamba3_cute_step_error is None and mamba3_cute_step is not None
    if mamba3_cute_step_error is not None:
        errors["mamba3_cute_step"] = mamba3_cute_step_error

    payload["hard_failures"] = sorted(set(hard_failures))
    payload["errors"] = errors
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 1 if payload["hard_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
