import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PROBE_PATH = Path(__file__).with_name("check_runtime.py")
SPEC = importlib.util.spec_from_file_location("futuremamba_check_runtime", PROBE_PATH)
assert SPEC is not None and SPEC.loader is not None
check_runtime = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_runtime)


REQUIRED_KEYS = {
    "torch",
    "triton",
    "cuda",
    "gpu",
    "compute_capability",
    "mamba2",
    "mamba2_forward",
    "mamba3",
    "mamba3_sequence",
    "mamba3_rotary_step",
    "mamba3_cute_step",
    "official_mamba_commit",
    "cuda_home",
    "cuda_home_ok",
    "cuda_home_required",
    "mamba_force_build",
    "mamba_force_build_required",
    "torch_cuda_arch_list",
    "torch_cuda_arch_list_required",
    "nvcc",
    "nvcc_ok",
    "hard_failures",
    "errors",
}


class FakeCuda:
    def __init__(self, *, available=True, capability=(12, 0)):
        self._available = available
        self._capability = capability

    def is_available(self):
        return self._available

    def get_device_name(self, index):
        assert index == 0
        return "NVIDIA GeForce RTX 5090"

    def get_device_capability(self, index):
        assert index == 0
        return self._capability


class FakeTorch:
    __version__ = "2.9.1"

    def __init__(self, *, cuda="12.8", available=True, capability=(12, 0)):
        self.version = SimpleNamespace(cuda=cuda)
        self.cuda = FakeCuda(available=available, capability=capability)


class FakeMamba2:
    pass


def _fake_importer(
    torch, *, triton_version="3.5.1", mamba2_available=True, mamba3_available=False
):
    modules = {
        "torch": torch,
        "triton": SimpleNamespace(__version__=triton_version),
    }
    if mamba2_available:
        modules["mamba_ssm.modules.mamba2"] = SimpleNamespace(Mamba2=FakeMamba2)
    if mamba3_available:
        modules.update(
            {
                "mamba_ssm.modules.mamba3": SimpleNamespace(Mamba3=object),
                "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined": SimpleNamespace(
                    mamba3_siso_combined=object
                ),
                "mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step": SimpleNamespace(
                    apply_rotary_qk_inference_fwd=object
                ),
                "mamba_ssm.ops.cute.mamba3.mamba3_step_fn": SimpleNamespace(mamba3_step_fn=object),
            }
        )

    def importer(name):
        if name not in modules:
            raise ModuleNotFoundError(name)
        return modules[name]

    return importer


def _environment(**overrides):
    values = {
        "CUDA_HOME": "/usr/local/cuda-12.8",
        "MAMBA_FORCE_BUILD": "TRUE",
        "TORCH_CUDA_ARCH_LIST": "12.0",
    }
    values.update(overrides)
    return values


def _probe(
    *,
    torch=None,
    triton_version="3.5.1",
    environment=None,
    nvcc="Cuda compilation tools, release 12.8, V12.8.93",
    mamba2_available=True,
    mamba2_forward=True,
    mamba3=False,
):
    torch = torch or FakeTorch()
    return check_runtime.probe_runtime(
        import_module=_fake_importer(
            torch,
            triton_version=triton_version,
            mamba2_available=mamba2_available,
            mamba3_available=mamba3,
        ),
        environ=environment or _environment(),
        nvcc_version=lambda cuda_home: nvcc,
        mamba2_forward=lambda mamba2, runtime_torch: mamba2_forward,
    )


def test_complete_mamba2_runtime_passes_without_mamba3():
    payload = _probe()

    assert payload["hard_failures"] == []
    assert check_runtime.exit_code(payload) == 0
    assert payload["mamba2"] is True
    assert payload["mamba2_forward"] is True
    assert payload["mamba3"] is False
    assert payload["nvcc_ok"] is True


@pytest.mark.parametrize(
    ("change", "failure"),
    [
        (lambda: {"triton_version": "3.4.0"}, "triton_version"),
        (lambda: {"environment": _environment(CUDA_HOME="/usr/local/cuda-11.5")}, "cuda_home"),
        (lambda: {"environment": _environment(MAMBA_FORCE_BUILD="FALSE")}, "mamba_force_build"),
        (lambda: {"environment": _environment(TORCH_CUDA_ARCH_LIST="8.0")}, "torch_cuda_arch_list"),
        (lambda: {"torch": SimpleNamespace(**FakeTorch().__dict__, __version__="2.8.0")}, "torch_version"),
        (lambda: {"torch": FakeTorch(cuda="11.8")}, "cuda_version"),
        (lambda: {"torch": FakeTorch(capability=(8, 0))}, "compute_capability"),
        (lambda: {"nvcc": "Cuda compilation tools, release 11.5, V11.5.119"}, "nvcc_version"),
        (lambda: {"nvcc": None}, "nvcc"),
        (lambda: {"torch": FakeTorch(available=False)}, "cuda_unavailable"),
    ],
)
def test_any_runtime_gate_mismatch_is_a_hard_failure(change, failure):
    kwargs = change()
    payload = _probe(**kwargs)

    assert failure in payload["hard_failures"]
    assert check_runtime.exit_code(payload) != 0


def test_mamba3_import_failure_is_diagnostic_only():
    payload = _probe(mamba3=False)

    assert payload["mamba2"] is True
    assert payload["mamba2_forward"] is True
    assert payload["mamba3"] is False
    assert "mamba3" in payload["errors"]
    assert payload["hard_failures"] == []


def test_missing_mamba2_import_is_a_hard_failure():
    payload = _probe(mamba2_available=False)

    assert payload["mamba2"] is False
    assert "mamba2_import" in payload["hard_failures"]
    assert check_runtime.exit_code(payload) != 0


def test_failed_mamba2_forward_is_a_hard_failure():
    payload = _probe(mamba2_forward=False)

    assert payload["mamba2"] is True
    assert payload["mamba2_forward"] is False
    assert "mamba2_forward" in payload["hard_failures"]
    assert check_runtime.exit_code(payload) != 0


def test_probe_contract_contains_actual_diagnostics():
    payload = _probe()

    assert REQUIRED_KEYS <= payload.keys()
    assert payload["official_mamba_commit"] == "77069de5cdb55cbe98b670889c80df211e031039"
    assert payload["cuda_home_required"] == "/usr/local/cuda-12.8"
    assert payload["mamba_force_build_required"] == "TRUE"
    assert payload["torch_cuda_arch_list_required"] == "12.0"
    assert isinstance(payload["hard_failures"], list)
    assert isinstance(payload["errors"], dict)


def test_main_emits_one_json_line(monkeypatch, capsys):
    payload = _probe()
    monkeypatch.setattr(check_runtime, "probe_runtime", lambda: payload)

    assert check_runtime.main() == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert len(captured.out.strip().splitlines()) == 1
    assert json.loads(captured.out) == payload
