#!/usr/bin/env python3
"""Probe the isolated FutureMamba runtime.

The probe always emits one JSON object.  Torch/CUDA/Triton and Mamba-2 form
the hard dependency gate; Mamba-3 support remains diagnostic until its
dedicated hardware gate is satisfied.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from typing import Any


EXPECTED_TORCH = "2.9.1"
EXPECTED_TRITON = "3.5.1"
EXPECTED_CUDA = "12.8"
EXPECTED_COMPUTE_CAPABILITY = (12, 0)
OFFICIAL_MAMBA_COMMIT = "77069de5cdb55cbe98b670889c80df211e031039"
REQUIRED_CUDA_HOME = "/usr/local/cuda-12.8"
REQUIRED_MAMBA_FORCE_BUILD = "TRUE"
REQUIRED_TORCH_CUDA_ARCH_LIST = "12.0"
REQUIRED_TRITON_LIBCUDA_PATH = "/usr/local/cuda-12.8/targets/x86_64-linux/lib/stubs"


def _error_text(exc: BaseException) -> str:
    return f"{exc.__class__.__name__}: {exc}"


def _version_matches(actual: str | None, expected: str) -> bool:
    return actual is not None and actual.split("+")[0] == expected


def _nvcc_version(cuda_home: str | None) -> str | None:
    candidates: list[str] = []
    if cuda_home:
        candidates.append(os.path.join(cuda_home, "bin", "nvcc"))
    path_nvcc = shutil.which("nvcc")
    if path_nvcc is not None:
        candidates.append(path_nvcc)

    seen: set[str] = set()
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
        except BaseException:
            continue
        output = " ".join(completed.stdout.split())
        if output:
            return output
    return None


def _nvcc_matches(version: str | None) -> bool:
    if not version:
        return False
    return re.search(r"(?:release\s+|\bV)12\.8(?:\D|$)", version, flags=re.IGNORECASE) is not None


def _import_module(import_module: Callable[[str], Any], module_name: str) -> tuple[Any | None, str | None]:
    try:
        return import_module(module_name), None
    except BaseException as exc:  # noqa: BLE001 - expected probe failures become JSON.
        return None, _error_text(exc)


def _import_attr(
    import_module: Callable[[str], Any], module_name: str, attr_name: str
) -> tuple[Any | None, str | None]:
    module, error = _import_module(import_module, module_name)
    if error is not None:
        return None, error
    try:
        return getattr(module, attr_name), None
    except BaseException as exc:  # noqa: BLE001 - expected probe failures become JSON.
        return None, _error_text(exc)


def _record_step(
    payload: dict[str, Any],
    key: str,
    errors: dict[str, str],
    func: Callable[[], bool],
) -> bool:
    try:
        ok = bool(func())
    except BaseException as exc:  # noqa: BLE001 - expected probe step failures become JSON.
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
        use_mem_eff_path=False,
        device=device,
        dtype=dtype,
    ).eval()
    inputs = torch.randn(1, 8, 64, device=device, dtype=dtype)
    with torch.inference_mode():
        output = model(inputs)
    return tuple(output.shape) == (1, 8, 64) and bool(torch.isfinite(output).all().item())


def probe_runtime(
    *,
    import_module: Callable[[str], Any] = importlib.import_module,
    environ: Mapping[str, str] | None = None,
    nvcc_version: Callable[[str | None], str | None] = _nvcc_version,
    mamba2_forward: Callable[[Any, Any], bool] = _run_mamba2_forward,
) -> dict[str, Any]:
    """Collect and validate runtime facts using injectable dependencies."""
    environ = os.environ if environ is None else environ
    errors: dict[str, str] = {}
    hard_failures: list[str] = []
    cuda_home = environ.get("CUDA_HOME")
    mamba_force_build = environ.get("MAMBA_FORCE_BUILD")
    torch_cuda_arch_list = environ.get("TORCH_CUDA_ARCH_LIST")
    triton_libcuda_path = environ.get("TRITON_LIBCUDA_PATH")
    nvcc = nvcc_version(cuda_home or REQUIRED_CUDA_HOME)

    payload: dict[str, Any] = {
        "official_mamba_commit": OFFICIAL_MAMBA_COMMIT,
        "cuda_home_required": REQUIRED_CUDA_HOME,
        "cuda_home": cuda_home,
        "cuda_home_ok": cuda_home == REQUIRED_CUDA_HOME,
        "mamba_force_build_required": REQUIRED_MAMBA_FORCE_BUILD,
        "mamba_force_build": mamba_force_build,
        "mamba_force_build_ok": mamba_force_build == REQUIRED_MAMBA_FORCE_BUILD,
        "torch_cuda_arch_list_required": REQUIRED_TORCH_CUDA_ARCH_LIST,
        "torch_cuda_arch_list": torch_cuda_arch_list,
        "torch_cuda_arch_list_ok": torch_cuda_arch_list == REQUIRED_TORCH_CUDA_ARCH_LIST,
        "triton_libcuda_path_required": REQUIRED_TRITON_LIBCUDA_PATH,
        "triton_libcuda_path": triton_libcuda_path,
        "triton_libcuda_path_ok": triton_libcuda_path == REQUIRED_TRITON_LIBCUDA_PATH,
        "nvcc": nvcc,
        "nvcc_ok": _nvcc_matches(nvcc),
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

    for key, ok in (
        ("cuda_home", payload["cuda_home_ok"]),
        ("mamba_force_build", payload["mamba_force_build_ok"]),
        ("torch_cuda_arch_list", payload["torch_cuda_arch_list_ok"]),
        ("triton_libcuda_path", payload["triton_libcuda_path_ok"]),
        ("nvcc", payload["nvcc_ok"]),
    ):
        if not ok:
            if key == "nvcc" and nvcc is None:
                errors["nvcc"] = "nvcc version query returned no output"
                hard_failures.append("nvcc")
            elif key == "nvcc":
                hard_failures.append("nvcc_version")
            else:
                hard_failures.append(key)

    torch, torch_error = _import_module(import_module, "torch")
    if torch_error is not None:
        errors["torch"] = torch_error
        hard_failures.append("torch_import")
    else:
        payload["torch"] = getattr(torch, "__version__", None)
        version_cuda = getattr(torch, "version", None)
        payload["cuda"] = getattr(version_cuda, "cuda", None)
        if not _version_matches(payload["torch"], EXPECTED_TORCH):
            hard_failures.append("torch_version")
        if not _version_matches(payload["cuda"], EXPECTED_CUDA):
            hard_failures.append("cuda_version")
        try:
            if torch.cuda.is_available():
                payload["gpu"] = torch.cuda.get_device_name(0)
                payload["compute_capability"] = list(torch.cuda.get_device_capability(0))
                if tuple(payload["compute_capability"]) != EXPECTED_COMPUTE_CAPABILITY:
                    hard_failures.append("compute_capability")
            else:
                hard_failures.append("cuda_unavailable")
        except BaseException as exc:  # noqa: BLE001 - probe must degrade to JSON.
            errors["cuda"] = _error_text(exc)
            hard_failures.append("cuda_query")

    triton, triton_error = _import_module(import_module, "triton")
    if triton_error is not None:
        errors["triton"] = triton_error
        hard_failures.append("triton_import")
    else:
        payload["triton"] = getattr(triton, "__version__", None)
        if not _version_matches(payload["triton"], EXPECTED_TRITON):
            hard_failures.append("triton_version")

    Mamba2, mamba2_error = _import_attr(import_module, "mamba_ssm.modules.mamba2", "Mamba2")
    if mamba2_error is not None:
        errors["mamba2"] = mamba2_error
        hard_failures.append("mamba2_import")
    else:
        payload["mamba2"] = True
        if not _record_step(
            payload,
            "mamba2_forward",
            errors,
            lambda: mamba2_forward(Mamba2, torch),
        ):
            hard_failures.append("mamba2_forward")

    for key, module_name, attr_name in (
        ("mamba3", "mamba_ssm.modules.mamba3", "Mamba3"),
        (
            "mamba3_sequence",
            "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined",
            "mamba3_siso_combined",
        ),
        (
            "mamba3_rotary_step",
            "mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step",
            "apply_rotary_qk_inference_fwd",
        ),
        ("mamba3_cute_step", "mamba_ssm.ops.cute.mamba3.mamba3_step_fn", "mamba3_step_fn"),
    ):
        value, error = _import_attr(import_module, module_name, attr_name)
        payload[key] = error is None and value is not None
        if error is not None:
            errors[key] = error

    payload["hard_failures"] = sorted(set(hard_failures))
    payload["errors"] = errors
    return payload


def exit_code(payload: Mapping[str, Any]) -> int:
    return 1 if payload.get("hard_failures") else 0


def main() -> int:
    payload = probe_runtime()
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return exit_code(payload)


if __name__ == "__main__":
    raise SystemExit(main())
