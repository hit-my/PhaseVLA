from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).with_name("check_mamba3_gate.py")
SPEC = importlib.util.spec_from_file_location("check_mamba3_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
check_gate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = check_gate
SPEC.loader.exec_module(check_gate)


EXPECTED_GATES = (
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


def test_fixed_mamba_source_version_matches_pinned_commit():
    assert check_gate.OFFICIAL_MAMBA_COMMIT == "77069de5cdb55cbe98b670889c80df211e031039"
    assert check_gate.EXPECTED_MAMBA_VERSION == "2.3.1"


class _FakeCuda:
    def is_available(self):
        return True

    def get_device_name(self, index):
        assert index == 0
        return "NVIDIA GeForce RTX 5090"

    def get_device_capability(self, index):
        assert index == 0
        return (12, 0)


class _FakeTorch:
    __version__ = "2.9.1"
    version = SimpleNamespace(cuda="12.8")
    cuda = _FakeCuda()


class _FakeMamba2:
    pass


def test_gate_names_and_status_values_are_fixed():
    assert check_gate.GATE_NAMES == EXPECTED_GATES
    assert check_gate.GATE_STATUS_VALUES == ("passed", "failed", "not_run")


def test_failed_gate_marks_following_gates_not_run():
    calls: list[str] = []

    def passed(name: str):
        def impl(ctx):
            del ctx
            calls.append(name)
            return check_gate.GateResult.passed({"called": name})

        return impl

    def failed(name: str):
        def impl(ctx):
            del ctx
            calls.append(name)
            return check_gate.GateResult.failed({"type": "ContractFailure", "message": "boom"})

        return impl

    impls = {name: passed(name) for name in check_gate.GATE_NAMES}
    impls["forward_10_seeds"] = failed("forward_10_seeds")

    payload = check_gate.run_mamba3_gate(
        gate_impls=impls,
        runtime_collector=lambda ctx: {"torch": "fake"},
        generated_at="2026-08-12T00:00:00Z",
    )

    assert calls == ["dependencies", "device_256_steps", "forward_10_seeds"]
    statuses = {gate["name"]: gate["status"] for gate in payload["gates"]}
    assert statuses["dependencies"] == "passed"
    assert statuses["device_256_steps"] == "passed"
    assert statuses["forward_10_seeds"] == "failed"
    for name in check_gate.GATE_NAMES[3:]:
        assert statuses[name] == "not_run"
    assert payload["status"] == "unsupported_on_current_stack"


def test_gate_exceptions_are_serialized_and_do_not_prevent_json_payload():
    def exploding(ctx):
        del ctx
        raise RuntimeError("kernel launch failed")

    impls = {name: (lambda ctx: check_gate.GateResult.passed({})) for name in check_gate.GATE_NAMES}
    impls["device_256_steps"] = exploding

    payload = check_gate.run_mamba3_gate(
        gate_impls=impls,
        runtime_collector=lambda ctx: {},
        generated_at="2026-08-12T00:00:00Z",
    )

    device_gate = payload["gates"][1]
    assert device_gate["name"] == "device_256_steps"
    assert device_gate["status"] == "failed"
    assert device_gate["error"] == {"type": "RuntimeError", "message": "kernel launch failed"}
    assert json.loads(json.dumps(payload))["gates"][1]["error"]["type"] == "RuntimeError"
    assert payload["gates"][2]["status"] == "not_run"

def test_non_finite_gate_evidence_is_sanitized_for_strict_json():
    impls = {
        name: (lambda ctx: check_gate.GateResult.passed({}))
        for name in check_gate.GATE_NAMES
    }
    impls["dependencies"] = lambda ctx: check_gate.GateResult.passed(
        {"metric": float("nan"), "nested": [float("inf"), -float("inf")]}
    )

    payload = check_gate.run_mamba3_gate(
        gate_impls=impls,
        runtime_collector=lambda ctx: {},
        generated_at="2026-08-12T00:00:00Z",
    )

    assert payload["gates"][0]["evidence"] == {"metric": None, "nested": [None, None]}
    json.dumps(payload, allow_nan=False)


def test_dependencies_gate_does_not_import_or_accept_mamba2_fallback():
    imported: list[str] = []

    def import_module(name: str):
        imported.append(name)
        if name == "torch":
            return _FakeTorch
        if name == "triton":
            return SimpleNamespace(__version__="3.5.1")
        if name == "mamba_ssm.modules.mamba3":
            return SimpleNamespace(Mamba3=_FakeMamba2)
        if name == "mamba_ssm.ops.triton.mamba3.mamba3_siso_combined":
            return SimpleNamespace(mamba3_siso_combined=object())
        if name == "mamba_ssm.ops.triton.mamba3.mamba3_mimo_rotary_step":
            return SimpleNamespace(apply_rotary_qk_inference_fwd=object())
        if name == "mamba_ssm.ops.cute.mamba3.mamba3_step_fn":
            return SimpleNamespace(mamba3_step_fn=object())
        if name == "mamba_ssm.modules.mamba2":
            raise AssertionError("Mamba2 fallback must never be imported")
        raise ModuleNotFoundError(name)

    ctx = check_gate.GateContext(
        import_module=import_module,
        metadata_version=lambda package: "2.3.1" if package == "mamba-ssm" else None,
        environ={},
        openpi_evidence_path=None,
        robomme_evidence_path=None,
    )

    result = check_gate.default_gate_impls()["dependencies"](ctx)

    assert result.status == "failed"
    assert "Mamba3" in result.error["message"]
    assert all("mamba2" not in name for name in imported)


def test_atomic_json_write_uses_replace_and_preserves_existing_file_on_failure(tmp_path: Path):
    output = tmp_path / "mamba3_gate.json"
    output.write_text("old", encoding="utf-8")
    payload = {"status": "unsupported_on_current_stack", "gates": []}
    captured = {}

    def failing_replace(src, dst):
        captured["src"] = Path(src)
        captured["dst"] = Path(dst)
        assert json.loads(Path(src).read_text(encoding="utf-8")) == payload
        raise RuntimeError("replace failed")

    try:
        check_gate.write_json_atomic(output, payload, replace=failing_replace)
    except RuntimeError as error:
        assert "replace failed" in str(error)
    else:
        raise AssertionError("replace failure was swallowed")

    assert captured["dst"] == output
    assert output.read_text(encoding="utf-8") == "old"
    assert not captured["src"].exists()


class _FakeParameter:
    def __init__(self, value: int) -> None:
        self.value = value
        self.requires_grad = True
        self.grad = None

    def detach(self):
        return self

    def clone(self):
        return _FakeParameter(self.value)


class _FakeLoss:
    def __init__(self, model, input_tensor) -> None:
        self.model = model
        self.input_tensor = input_tensor

    def float(self):
        return self

    def square(self):
        return self

    def mean(self):
        return self

    def backward(self):
        self.input_tensor.grad = object()
        for parameter in self.model.parameters():
            parameter.grad = object()


class _FakeTrainModel:
    def __init__(self, *, change_all: bool) -> None:
        self.named = [("futuremamba/first", _FakeParameter(1)), ("futuremamba/second", _FakeParameter(2))]
        self.change_all = change_all

    def named_parameters(self):
        return iter(self.named)

    def parameters(self):
        return [parameter for _, parameter in self.named]

    def train(self):
        return self

    def __call__(self, input_tensor):
        return _FakeLoss(self, input_tensor)


class _FakeOptimizer:
    def __init__(self, parameters, *, change_all: bool) -> None:
        self.parameters = list(parameters)
        self.change_all = change_all

    def step(self):
        changed = self.parameters if self.change_all else self.parameters[:1]
        for parameter in changed:
            parameter.value += 1


def test_backward_gate_requires_every_trainable_parameter_to_change(monkeypatch):
    model = _FakeTrainModel(change_all=False)
    input_tensor = SimpleNamespace(grad=None)
    fake_torch = SimpleNamespace(
        float32="float32",
        randn=lambda *args, **kwargs: input_tensor,
        equal=lambda before, after: before.value == after.value,
        optim=SimpleNamespace(
            AdamW=lambda parameters, lr: _FakeOptimizer(parameters, change_all=model.change_all)
        ),
    )
    ctx = check_gate.GateContext(objects={"torch": fake_torch, "Mamba3": object()})
    monkeypatch.setattr(check_gate, "BACKWARD_SEEDS", (0,))
    monkeypatch.setattr(check_gate, "_new_model", lambda *args, **kwargs: (model, ()))
    monkeypatch.setattr(check_gate, "_finite_tensor", lambda *args: True)

    failed = check_gate.default_gate_impls()["backward_10_seeds"](ctx)

    assert failed.status == "failed"
    assert "futuremamba/second" in failed.error["message"]

    model = _FakeTrainModel(change_all=True)
    passed = check_gate.default_gate_impls()["backward_10_seeds"](ctx)
    assert passed.status == "passed"
    assert passed.evidence["trainable_parameter_counts"] == [2]
    assert passed.evidence["changed_parameter_counts"] == [2]


def test_fixed_state_gate_allocates_each_tested_sequence_length(monkeypatch):
    allocations = []

    class FakeModel:
        def allocate_inference_cache(self, batch_size, max_seqlen, **kwargs):
            allocations.append((batch_size, max_seqlen))
            return (SimpleNamespace(length=max_seqlen),)

        def eval(self):
            return self

    fake_torch = SimpleNamespace(
        float32="float32",
        randn=lambda *args, **kwargs: SimpleNamespace(shape=args),
        inference_mode=lambda: __import__("contextlib").nullcontext(),
    )
    ctx = check_gate.GateContext(objects={"torch": fake_torch, "Mamba3": object()})
    monkeypatch.setattr(check_gate, "_new_model", lambda *args, **kwargs: (FakeModel(), (object(),)))
    monkeypatch.setattr(check_gate, "_step_tokens", lambda torch, model, x, states: (object(), states))
    monkeypatch.setattr(check_gate, "_state_summary", lambda states: [{"shape": [2, 64], "bytes": 512}])
    monkeypatch.setattr(check_gate, "_state_total_bytes", lambda states: 512)

    result = check_gate.default_gate_impls()["fixed_state_bytes"](ctx)

    assert result.status == "passed"
    assert allocations == [(2, 2), (2, 2048)]


def _write_json_artifact(path: Path, payload) -> str:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_fields(paths: dict[str, Path], digests: dict[str, str]) -> dict:
    return {
        "artifacts": {name: path.name for name, path in paths.items()},
        "artifact_sha256": digests,
    }


def test_openpi_checkpoint_evidence_requires_bound_runtime_artifacts(tmp_path: Path):
    missing = check_gate.validate_openpi_checkpoint_evidence(None)
    assert missing.status == "failed"

    summary_only = tmp_path / "summary-only-openpi.json"
    summary_only.write_text(
        json.dumps(
            {
                "status": "passed",
                "gate": "openpi_checkpoint_2_to_4",
                "backend": "mamba3_siso",
                "mamba_repo_commit": check_gate.OFFICIAL_MAMBA_COMMIT,
                "first_end_step": 2,
                "resume_start_step": 2,
                "resumed_end_step": 4,
                "checkpoint_step": 4,
                "data_iterator_step": 4,
                "only_futuremamba_parameters_updated": True,
                "checkpoint_identity": "sha256:handwritten",
                "config_identity": "sha256:handwritten",
                "config": {
                    "memory_backend": "mamba3_siso",
                    "mamba_repo_commit": check_gate.OFFICIAL_MAMBA_COMMIT,
                },
            }
        ),
        encoding="utf-8",
    )
    assert check_gate.validate_openpi_checkpoint_evidence(summary_only).status == "failed"

    paths = {
        "checkpoint_metadata": tmp_path / "openpi-checkpoint-metadata.json",
        "config": tmp_path / "openpi-config.json",
        "parameter_diff": tmp_path / "openpi-parameter-diff.json",
        "trace": tmp_path / "openpi-trace.json",
    }
    digests = {
        "checkpoint_metadata": _write_json_artifact(
            paths["checkpoint_metadata"],
            {"checkpoints": [{"step": 2}, {"step": 4}], "checkpoint_step": 4, "data_iterator_step": 4},
        ),
        "config": _write_json_artifact(
            paths["config"],
            {"memory_backend": "mamba3_siso", "mamba_repo_commit": check_gate.OFFICIAL_MAMBA_COMMIT},
        ),
        "parameter_diff": _write_json_artifact(
            paths["parameter_diff"],
            {"updated_parameters": ["futuremamba/memory.in_proj.weight"], "unchanged_non_futuremamba": True},
        ),
        "trace": _write_json_artifact(
            paths["trace"],
            {
                "events": [
                    {"event": "train_end", "step": 2},
                    {"event": "checkpoint_save", "step": 2},
                    {"event": "checkpoint_restore", "step": 2},
                    {"event": "train_end", "step": 4},
                    {"event": "checkpoint_save", "step": 4},
                ]
            },
        ),
    }
    evidence_path = tmp_path / "real-openpi.json"
    evidence_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "gate": "openpi_checkpoint_2_to_4",
                "backend": "mamba3_siso",
                "mamba_repo_commit": check_gate.OFFICIAL_MAMBA_COMMIT,
                "first_end_step": 2,
                "resume_start_step": 2,
                "resumed_end_step": 4,
                "checkpoint_step": 4,
                "data_iterator_step": 4,
                "only_futuremamba_parameters_updated": True,
                "checkpoint_identity": f"sha256:{digests['checkpoint_metadata']}",
                "config_identity": f"sha256:{digests['config']}",
                **_artifact_fields(paths, digests),
            }
        ),
        encoding="utf-8",
    )

    assert check_gate.validate_openpi_checkpoint_evidence(evidence_path).status == "passed"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    checkpoint_metadata = json.loads(paths["checkpoint_metadata"].read_text(encoding="utf-8"))
    checkpoint_metadata["checkpoints"] = [{"step": 2.0}, {"step": 4.0}]
    digests["checkpoint_metadata"] = _write_json_artifact(paths["checkpoint_metadata"], checkpoint_metadata)
    evidence["artifact_sha256"]["checkpoint_metadata"] = digests["checkpoint_metadata"]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert check_gate.validate_openpi_checkpoint_evidence(evidence_path).status == "failed"

    checkpoint_metadata["checkpoints"] = [{"step": 2}, {"step": 4}]
    digests["checkpoint_metadata"] = _write_json_artifact(paths["checkpoint_metadata"], checkpoint_metadata)
    evidence["artifact_sha256"]["checkpoint_metadata"] = digests["checkpoint_metadata"]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    evidence["only_futuremamba_parameters_updated"] = 1
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert check_gate.validate_openpi_checkpoint_evidence(evidence_path).status == "failed"

    evidence["only_futuremamba_parameters_updated"] = True
    evidence["artifacts"]["trace"] = str(paths["trace"].resolve())
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert check_gate.validate_openpi_checkpoint_evidence(evidence_path).status == "failed"

    evidence["artifacts"]["trace"] = paths["trace"].name
    trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
    trace["events"].append("unexpected-entry")
    digests["trace"] = _write_json_artifact(paths["trace"], trace)
    evidence["artifact_sha256"]["trace"] = digests["trace"]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert check_gate.validate_openpi_checkpoint_evidence(evidence_path).status == "failed"

    trace["events"].pop()
    digests["trace"] = _write_json_artifact(paths["trace"], trace)
    evidence["artifact_sha256"]["trace"] = digests["trace"]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert check_gate.validate_openpi_checkpoint_evidence(evidence_path).status == "passed"

    paths["trace"].write_text("{}", encoding="utf-8")
    assert check_gate.validate_openpi_checkpoint_evidence(evidence_path).status == "failed"


def test_robomme_evidence_requires_bound_protocol_artifacts(tmp_path: Path):
    missing = check_gate.validate_robomme_deployment_evidence(None)
    assert missing.status == "failed"

    summary_only = tmp_path / "summary-only-robomme.json"
    summary_only.write_text(
        json.dumps(
            {
                "status": "passed",
                "gate": "robomme_deployment",
                "backend": "mamba3_siso",
                "mamba_repo_commit": check_gate.OFFICIAL_MAMBA_COMMIT,
                "checkpoint_identity": "sha256:handwritten",
                "config_identity": "sha256:handwritten",
                "robomme_policy_commit": check_gate.ROBOMME_POLICY_COMMIT,
                "robomme_benchmark_commit": check_gate.ROBOMME_BENCHMARK_COMMIT,
                "episodes": 1,
                "protocol": list(check_gate.PROTOCOL),
                "executed_steps": 16,
                "state_isolation_ok": True,
                "timeout": False,
                "protocol_errors": [],
            }
        ),
        encoding="utf-8",
    )
    assert check_gate.validate_robomme_deployment_evidence(summary_only).status == "failed"

    paths = {
        "checkpoint_metadata": tmp_path / "robomme-checkpoint-metadata.json",
        "config": tmp_path / "robomme-config.json",
        "policy_source": tmp_path / "robomme-policy-source.json",
        "benchmark_source": tmp_path / "robomme-benchmark-source.json",
        "trace": tmp_path / "robomme-trace.json",
        "actions": tmp_path / "robomme-actions.json",
    }
    digests = {
        "checkpoint_metadata": _write_json_artifact(paths["checkpoint_metadata"], {"checkpoint_step": 4}),
        "config": _write_json_artifact(
            paths["config"],
            {"memory_backend": "mamba3_siso", "mamba_repo_commit": check_gate.OFFICIAL_MAMBA_COMMIT},
        ),
        "policy_source": _write_json_artifact(
            paths["policy_source"], {"commit": check_gate.ROBOMME_POLICY_COMMIT}
        ),
        "benchmark_source": _write_json_artifact(
            paths["benchmark_source"], {"commit": check_gate.ROBOMME_BENCHMARK_COMMIT}
        ),
        "trace": _write_json_artifact(
            paths["trace"],
            {
                "events": [
                    {"event": event, "timestamp_ns": index + 1}
                    for index, event in enumerate(check_gate.PROTOCOL)
                ],
                "episodes": 1,
                "state_isolation_ok": True,
                "timeout": False,
                "protocol_errors": [],
            },
        ),
        "actions": _write_json_artifact(
            paths["actions"],
            {"records": [{"step": index, "action": [0.0] * 8} for index in range(16)]},
        ),
    }
    evidence_path = tmp_path / "real-robomme.json"
    evidence_path.write_text(
        json.dumps(
            {
                "status": "passed",
                "gate": "robomme_deployment",
                "backend": "mamba3_siso",
                "mamba_repo_commit": check_gate.OFFICIAL_MAMBA_COMMIT,
                "checkpoint_identity": f"sha256:{digests['checkpoint_metadata']}",
                "config_identity": f"sha256:{digests['config']}",
                "robomme_policy_commit": check_gate.ROBOMME_POLICY_COMMIT,
                "robomme_benchmark_commit": check_gate.ROBOMME_BENCHMARK_COMMIT,
                "episodes": 1,
                "protocol": list(check_gate.PROTOCOL),
                "executed_steps": 16,
                "state_isolation_ok": True,
                "timeout": False,
                "protocol_errors": [],
                **_artifact_fields(paths, digests),
            }
        ),
        encoding="utf-8",
    )

    assert check_gate.validate_robomme_deployment_evidence(evidence_path).status == "passed"

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
    trace["episodes"] = True
    digests["trace"] = _write_json_artifact(paths["trace"], trace)
    evidence["artifact_sha256"]["trace"] = digests["trace"]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert check_gate.validate_robomme_deployment_evidence(evidence_path).status == "failed"

    trace["episodes"] = 1.0
    digests["trace"] = _write_json_artifact(paths["trace"], trace)
    evidence["artifact_sha256"]["trace"] = digests["trace"]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert check_gate.validate_robomme_deployment_evidence(evidence_path).status == "failed"

    trace["episodes"] = 1
    digests["trace"] = _write_json_artifact(paths["trace"], trace)
    evidence["artifact_sha256"]["trace"] = digests["trace"]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    actions = json.loads(paths["actions"].read_text(encoding="utf-8"))
    actions["records"][0]["step"] = False
    digests["actions"] = _write_json_artifact(paths["actions"], actions)
    evidence["artifact_sha256"]["actions"] = digests["actions"]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert check_gate.validate_robomme_deployment_evidence(evidence_path).status == "failed"

    actions["records"][0]["step"] = 0
    digests["actions"] = _write_json_artifact(paths["actions"], actions)
    evidence["artifact_sha256"]["actions"] = digests["actions"]
    trace = json.loads(paths["trace"].read_text(encoding="utf-8"))
    trace["events"].append("unexpected-entry")
    digests["trace"] = _write_json_artifact(paths["trace"], trace)
    evidence["artifact_sha256"]["trace"] = digests["trace"]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert check_gate.validate_robomme_deployment_evidence(evidence_path).status == "failed"

    trace["events"].pop()
    digests["trace"] = _write_json_artifact(paths["trace"], trace)
    evidence["artifact_sha256"]["trace"] = digests["trace"]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    assert check_gate.validate_robomme_deployment_evidence(evidence_path).status == "passed"

    paths["actions"].write_text(json.dumps({"records": []}), encoding="utf-8")
    assert check_gate.validate_robomme_deployment_evidence(evidence_path).status == "failed"
