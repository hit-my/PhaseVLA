from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from openpi.models import model as _model
from openpi.models_pytorch.mamba_memory import MemorySnapshot

SCRIPT = Path(__file__).with_name("eval_robomme_memory_swap.py")
spec = importlib.util.spec_from_file_location("eval_robomme_memory_swap", SCRIPT)
assert spec is not None and spec.loader is not None
memory_swap = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = memory_swap
spec.loader.exec_module(memory_swap)


class _FakeMemoryBackend:
    backend_id = "fake_backend"
    state_schema_version = 4

    def initial_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> MemorySnapshot:
        return MemorySnapshot(
            self.backend_id,
            self.state_schema_version,
            batch_size,
            ((torch.zeros(batch_size, 2, device=device, dtype=dtype),),),
        )


class _FakePlugin(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.memory_backend = _FakeMemoryBackend()

    def compute_memory_token(self, last_vlm_token: torch.Tensor, memory_state: MemorySnapshot):
        value = memory_state.layers[0][0]
        token = (last_vlm_token[:, :1] + value.sum(dim=1, keepdim=True)).unsqueeze(-1)
        next_memory = MemorySnapshot(
            memory_state.backend_id,
            memory_state.state_schema_version,
            memory_state.batch_size,
            ((value + last_vlm_token[:, :2],),),
        )
        return token, next_memory

    def forward_progress(self, prefix_cache, prefix_mask, memory_token, noisy_actions, timestep):
        del prefix_cache, prefix_mask, timestep
        return noisy_actions * 0.25 + memory_token.reshape(noisy_actions.shape[0], 1, 1)


class _FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.progress_calls = []
        self.action_calls = []

    def _preprocess_observation(self, observation, *, train: bool = False):
        del train
        return None, None, None, None, observation.state + 10.0

    def encode_frozen_prefix(self, observation, *, train: bool = False):
        del train
        batch = observation.state.shape[0]
        hidden = torch.stack(
            [observation.state, observation.state + 1.0, observation.state + 2.0], dim=1
        )
        pad_mask = torch.ones(batch, 3, dtype=torch.bool, device=observation.state.device)
        cache = tuple(
            (
                torch.full((batch, 1, 3, 1), float(layer), dtype=observation.state.dtype),
                torch.full((batch, 1, 3, 1), float(layer + 10), dtype=observation.state.dtype),
            )
            for layer in range(2)
        )
        return SimpleNamespace(hidden=hidden, pad_mask=pad_mask, kv_cache=cache)

    def last_valid_prefix(self, frozen):
        return frozen.hidden[:, -1]

    def denoise_step(self, processed_state, pad_mask, kv_cache, x_t, timestep):
        del pad_mask, kv_cache
        self.action_calls.append({"x": x_t.detach().clone(), "t": timestep.detach().clone()})
        return -0.5 * x_t + processed_state.sum(dim=1)[:, None, None] * 0.01


class _FakeFutureMambaModel(nn.Module):
    def __init__(self, *, action_horizon: int = 3, action_dim: int = 2):
        super().__init__()
        self.config = SimpleNamespace(
            action_horizon=action_horizon,
            action_dim=action_dim,
            num_denoise_steps=4,
            handoff_ratio=0.5,
        )
        self.base = _FakeBase()
        self.futuremamba = _FakePlugin()

    def initial_memory_state(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        return self.futuremamba.memory_backend.initial_state(batch_size, device=device, dtype=dtype)


def _snapshot(values, *, backend="fake_backend", schema=4, dtype=torch.float32, batch_size=1):
    tensor = torch.tensor(values, dtype=dtype).reshape(batch_size, 2)
    return MemorySnapshot(backend, schema, batch_size, ((tensor,),))


def _artifact_dict(*, memory_a=None, memory_b=None, include_labels=False):
    memory_a = memory_a or memory_swap.serialize_memory_snapshot(_snapshot([[1.0, 2.0]]))
    memory_b = memory_b or memory_swap.serialize_memory_snapshot(_snapshot([[3.0, 4.0]]))
    artifact = {
        "artifact_schema_version": 1,
        "episode_id": "ep-7",
        "query_a_id": "q-a",
        "query_b_id": "q-b",
        "current_query_id": "q-current",
        "observation": {
            "state": [0.25, -0.5],
            "image": {"base_0_rgb": [[[[0, 1, 2]]]]},
            "image_mask": {"base_0_rgb": [True]},
            "tokenized_prompt": [[1, 2, 3]],
            "tokenized_prompt_mask": [[True, True, True]],
        },
        "physical_state": {"joint_position": [0.1, 0.2], "gripper": [1.0]},
        "prompt": "pick the blue cube",
        "noise": [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]],
        "num_steps": 4,
        "handoff_step": 2,
        "memory_a": memory_a,
        "memory_b": memory_b,
        "provenance": {"artifact": "unit-test"},
    }
    if include_labels:
        artifact["labels"] = {
            "progress_probe": [
                {"predicted": "start", "label": "start"},
                {"predicted": "middle", "label": "done"},
            ],
            "branch": "left",

            "stop": False,
            "episode_success": True,
            "provenance": {"labels": "human-eval"},
        }
    return artifact


def _write_artifact(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_checksum_contract_accepts_only_memory_difference(tmp_path: Path):
    artifact = memory_swap.load_intervention_artifact(_write_artifact(tmp_path / "artifact.json", _artifact_dict()))

    identity = memory_swap.intervention_identity(artifact)

    assert identity["only_memory_differs"] is True
    assert identity["inputs"]["observation_checksum"] == memory_swap.tree_checksum(artifact.observation)
    assert identity["inputs"]["physical_state_checksum"] == memory_swap.tree_checksum(artifact.physical_state)
    assert identity["inputs"]["prompt_checksum"] == memory_swap.tree_checksum(artifact.prompt)
    assert identity["inputs"]["noise_checksum"] == memory_swap.tensor_checksum(artifact.noise)
    assert identity["inputs"]["solver_schedule_checksum"] == memory_swap.tree_checksum(
        {"num_steps": 4, "handoff_step": 2}
    )
    assert identity["memory_a"]["checksum"] != identity["memory_b"]["checksum"]


def test_artifact_rejects_missing_noise_and_identical_memory(tmp_path: Path):
    payload = _artifact_dict()
    payload.pop("noise")
    with pytest.raises(ValueError, match="noise"):
        memory_swap.load_intervention_artifact(_write_artifact(tmp_path / "missing_noise.json", payload))

    same = memory_swap.serialize_memory_snapshot(_snapshot([[1.0, 2.0]]))
    payload = _artifact_dict(memory_a=same, memory_b=same)
    with pytest.raises(ValueError, match="memory checksums must differ"):
        memory_swap.load_intervention_artifact(_write_artifact(tmp_path / "same_memory.json", payload))


def test_validate_memory_rejects_backend_schema_shape_dtype_and_batch():
    reference = _snapshot([[1.0, 2.0]])
    expected = _snapshot([[0.0, 0.0]])

    for bad, match in [
        (_snapshot([[1.0, 2.0]], backend="other"), "backend"),
        (_snapshot([[1.0, 2.0]], schema=99), "schema"),
        (MemorySnapshot("fake_backend", 4, 1, ((torch.zeros(1, 3),),)), "shape"),
        (_snapshot([[1.0, 2.0]], dtype=torch.float64), "dtype"),
        (_snapshot([[1.0, 2.0], [3.0, 4.0]], batch_size=2), "batch"),
    ]:
        with pytest.raises(ValueError, match=match):
            memory_swap.validate_memory_snapshot(bad, expected)

    memory_swap.validate_memory_snapshot(reference, expected)


def test_instrumented_rollout_records_true_solver_boundaries_and_memory_only_swap(tmp_path: Path):
    artifact = memory_swap.load_intervention_artifact(_write_artifact(tmp_path / "artifact.json", _artifact_dict()))
    model = _FakeFutureMambaModel()

    result = memory_swap.evaluate_memory_swap(model, artifact, device=torch.device("cpu"))

    assert result["schema_version"] == memory_swap.OUTPUT_SCHEMA_VERSION
    assert result["backend"] == "fake_backend"
    assert result["state_schema_version"] == 4
    assert result["handoff_step"] == 2
    assert result["input_identity"]["only_memory_differs"] is True
    assert len(result["steps"]) == 4
    assert [step["branch"] for step in result["steps"]] == ["progress", "progress", "action", "action"]
    assert result["steps"][0]["velocity_distance_l2"] > 0.0
    assert result["handoff"]["pre_error_l2"] == result["steps"][1]["state_distance_l2"]
    assert result["handoff"]["post_error_l2"] == result["steps"][2]["state_distance_l2"]
    assert result["final_action_chunk_distance_l2"] == result["steps"][-1]["state_distance_l2"]
    assert result["trajectory_a"][0] != result["trajectory_b"][0]
    assert result["unsupported_reports"] == [
        "progress_probe_accuracy requires artifact.labels.progress_probe with provenance",
        "branch_stop_behavior requires artifact.labels.branch/stop with provenance",
        "episode_success requires artifact.labels.episode_success with provenance",
    ]


def test_k_zero_and_k_num_steps_call_boundaries(tmp_path: Path):
    base_payload = _artifact_dict()

    zero_payload = dict(base_payload, handoff_step=0)
    zero_artifact = memory_swap.load_intervention_artifact(_write_artifact(tmp_path / "k0.json", zero_payload))
    zero_result = memory_swap.evaluate_memory_swap(_FakeFutureMambaModel(), zero_artifact, device=torch.device("cpu"))
    assert [step["branch"] for step in zero_result["steps"]] == ["action", "action", "action", "action"]
    assert zero_result["handoff"]["pre_error_l2"] == 0.0
    assert zero_result["handoff"]["post_error_l2"] == zero_result["steps"][0]["state_distance_l2"]

    full_payload = dict(base_payload, handoff_step=4)
    full_artifact = memory_swap.load_intervention_artifact(_write_artifact(tmp_path / "kfull.json", full_payload))
    full_result = memory_swap.evaluate_memory_swap(_FakeFutureMambaModel(), full_artifact, device=torch.device("cpu"))
    assert [step["branch"] for step in full_result["steps"]] == ["progress", "progress", "progress", "progress"]
    assert full_result["handoff"]["pre_error_l2"] == full_result["steps"][-1]["state_distance_l2"]
    assert full_result["handoff"]["post_error_l2"] == full_result["steps"][-1]["state_distance_l2"]


def test_reports_with_labels_require_provenance_and_are_computed(tmp_path: Path):
    artifact = memory_swap.load_intervention_artifact(
        _write_artifact(tmp_path / "labels.json", _artifact_dict(include_labels=True))
    )

    result = memory_swap.evaluate_memory_swap(_FakeFutureMambaModel(), artifact, device=torch.device("cpu"))

    assert result["provenance"]["labels"] == {"labels": "human-eval"}
    assert result["reports"]["progress_probe_accuracy"] == 0.5
    assert result["reports"]["branch_stop_behavior"] == {"branch": "left", "stop": False}
    assert result["reports"]["episode_success"] is True
    assert result["unsupported_reports"] == []


def test_labels_without_provenance_fail_closed(tmp_path: Path):
    payload = _artifact_dict(include_labels=True)
    payload["labels"].pop("provenance")

    with pytest.raises(ValueError, match="label provenance"):
        memory_swap.load_intervention_artifact(_write_artifact(tmp_path / "bad_labels.json", payload))


def test_cli_loads_bundle_artifact_and_writes_deterministic_json(monkeypatch, tmp_path: Path):
    artifact_path = _write_artifact(tmp_path / "artifact.json", _artifact_dict())
    output_path = tmp_path / "out.json"
    captured = {}

    def fake_loader(config, bundle_dir, device):
        captured["config"] = config
        captured["bundle_dir"] = bundle_dir
        captured["device"] = device
        return _FakeFutureMambaModel(), {"bundle": "metadata"}, tmp_path / "base"

    monkeypatch.setattr(memory_swap, "_load_futuremamba_bundle", fake_loader)

    args = memory_swap.Args(bundle=str(tmp_path / "bundle"), artifact=str(artifact_path), output=str(output_path), device="cpu")
    memory_swap.main(args)
    first = output_path.read_text(encoding="utf-8")
    memory_swap.main(args)
    second = output_path.read_text(encoding="utf-8")

    assert first == second
    written = json.loads(first)
    assert written["bundle_metadata"] == {"bundle": "metadata"}
    assert written["base_checkpoint_root"] == str(tmp_path / "base")
    assert captured["bundle_dir"] == Path(tmp_path / "bundle")
    assert captured["device"] == torch.device("cpu")
