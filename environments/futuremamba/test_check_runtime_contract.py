import json
import subprocess
import sys
from pathlib import Path


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
    "cuda_home_required",
    "mamba_force_build_required",
    "torch_cuda_arch_list_required",
    "hard_failures",
    "errors",
}


def test_probe_emits_single_json_line_and_hard_fails_when_runtime_is_missing():
    probe = Path(__file__).with_name("check_runtime.py")

    completed = subprocess.run(
        [sys.executable, str(probe)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode != 0
    assert completed.stderr == ""
    output_lines = completed.stdout.strip().splitlines()
    assert len(output_lines) == 1

    payload = json.loads(output_lines[0])
    assert REQUIRED_KEYS <= payload.keys()
    assert payload["official_mamba_commit"] == "77069de5cdb55cbe98b670889c80df211e031039"
    assert payload["cuda_home_required"] == "/usr/local/cuda-12.8"
    assert payload["mamba_force_build_required"] == "TRUE"
    assert payload["torch_cuda_arch_list_required"] == "12.0"
    assert payload["mamba3"] is False or payload["mamba3_sequence"] is False or payload["mamba3_rotary_step"] is False or payload["mamba3_cute_step"] is False or payload["mamba2"] is False or payload["mamba2_forward"] is False
    assert isinstance(payload["hard_failures"], list)
    assert isinstance(payload["errors"], dict)
