from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

OFFICIAL_TASK_SUITES: dict[str, tuple[str, ...]] = {
    "Counting": ("BinFill", "PickXtimes", "SwingXtimes"),
}
OFFICIAL_TASKS: tuple[str, ...] = ("BinFill", "PickXtimes", "SwingXtimes")
OFFICIAL_SPLIT_EPISODE_COUNTS: dict[str, int] = {"train": 100, "validation": 50, "test": 50}
OFFICIAL_DATASET_BY_SPLIT = {"train": "train", "validation": "val", "test": "test"}
ROBOMME_POLICY_LEARNING_COMMIT = "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
ROBOMME_BENCHMARK_COMMIT = "856bc3a189d4172f3f47dbee4424d585f8d78db3"
MAMBA3_REQUIRED_GATES = (
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
DEFAULT_EVAL_SEED = 7
DEFAULT_SERVER_PORT = 8000
MAMBA2_TRAIN_SEEDS = (0, 42, 7)
_MAIN_AXES = {
    "backend": "mamba2",
    "layer_selection": "Uniform",
    "progress_depth": 6,
    "handoff_ratio": 0.2,
    "handoff_steps": "0<K<N",
    "memory": "full",
    "initialization": "Action Expert",
    "loss": "flow+terminal",
    "progress_expert": "memory",
}


@dataclasses.dataclass(frozen=True)
class EpisodeSelection:
    task: str
    category: str
    split: str
    episode_ids: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class MethodSpec:
    method: str
    variant: str
    method_id: str
    policy_name: str
    config_name: str
    backend: str
    checkpoint_key: str
    train_seeds: tuple[int, ...]
    axes: dict[str, Any]
    requires_gate: bool = False
    use_history: bool = False
    subgoal_type: str | None = None
    use_memer: bool = False
    use_oracle: bool = False
    use_qwenvl: bool = False
    use_gemini: bool = False


@dataclasses.dataclass(frozen=True)
class CheckpointEntry:
    path: Path
    checkpoint_id: int
    config: str
    backend: str
    provenance: dict[str, Any]
    policy_name: str
    train_seed: int


@dataclasses.dataclass(frozen=True)
class ExperimentRecord:
    method: str
    variant: str
    method_id: str
    task: str
    category: str
    split: str
    episode_id: int
    train_seed: int
    eval_seed: int
    checkpoint: dict[str, Any]
    config: dict[str, Any]
    backend: str
    experiment_id: str
    provenance: dict[str, Any]
    server: dict[str, Any]
    eval: dict[str, Any]
    axis: dict[str, Any]
    mamba3_gate: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["axis"] = dict(sorted(self.axis.items()))
        data["checkpoint"] = dict(sorted(self.checkpoint.items()))
        data["config"] = dict(sorted(self.config.items()))
        data["eval"] = dict(sorted(self.eval.items()))
        data["provenance"] = dict(sorted(self.provenance.items()))
        data["server"] = dict(sorted(self.server.items()))
        if self.mamba3_gate is not None:
            data["mamba3_gate"] = dict(sorted(self.mamba3_gate.items()))
        return data


@dataclasses.dataclass(frozen=True)
class Manifest:
    schema: str
    schema_version: int
    generated_by: str
    provenance: dict[str, Any]
    phases: dict[str, list[dict[str, Any]]]
    experiments: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiments": self.experiments,
            "generated_by": self.generated_by,
            "phases": {
                phase: [dict(sorted(item.items())) for item in items]
                for phase, items in sorted(self.phases.items())
            },
            "provenance": dict(sorted(self.provenance.items())),
            "schema": self.schema,
            "schema_version": self.schema_version,
        }


def _task_to_category(task: str) -> str:
    for category, tasks in OFFICIAL_TASK_SUITES.items():
        if task in tasks:
            return category
    raise ValueError(f"unknown RoboMME task: {task!r}")


def _validate_episode_ids(split: str, episode_ids: Sequence[int]) -> tuple[int, ...]:
    if split not in OFFICIAL_SPLIT_EPISODE_COUNTS:
        raise ValueError(f"unknown split: {split!r}")
    episode_count = OFFICIAL_SPLIT_EPISODE_COUNTS[split]
    normalized = tuple(int(episode_id) for episode_id in episode_ids)
    if not normalized:
        raise ValueError("episode_ids must not be empty")
    if len(set(normalized)) != len(normalized) or tuple(sorted(normalized)) != normalized:
        raise ValueError("episode_ids must be sorted and unique")
    if any(episode_id < 0 or episode_id >= episode_count for episode_id in normalized):
        raise ValueError(f"episode_id outside available range [0, {episode_count}) for split {split!r}")
    return normalized


def phase_episode_plan(phase: str) -> list[EpisodeSelection]:
    if phase == "minimal":
        selections = (
            ("PickXtimes", "validation", range(10)),
            ("BinFill", "validation", range(10)),
        )
    elif phase == "counting":
        selections = tuple(
            (task, "validation", range(50))
            for task in ("BinFill", "PickXtimes", "SwingXtimes")
        )
    else:
        raise ValueError(f"unknown phase: {phase!r}")
    return [
        EpisodeSelection(
            task=task,
            category=_task_to_category(task),
            split=split,
            episode_ids=_validate_episode_ids(split, tuple(episode_ids)),
        )
        for task, split, episode_ids in selections
    ]


def _seed_tuple(values: Sequence[int]) -> tuple[int, ...]:
    seeds = tuple(int(value) for value in values)
    if not seeds:
        raise ValueError("train_seeds must not be empty")
    if len(set(seeds)) != len(seeds):
        raise ValueError("train_seeds must be unique")
    return seeds


def _future_axes(**overrides: Any) -> dict[str, Any]:
    axes = dict(_MAIN_AXES)
    axes.update(overrides)
    return axes


def _future_spec(
    method_id: str,
    variant: str,
    axes: dict[str, Any],
    *,
    config_name: str = "futuremamba_robomme_mamba2",
    seeds: Sequence[int] = (7,),
) -> MethodSpec:
    return MethodSpec(
        method="futuremamba",
        variant=variant,
        method_id=method_id,
        policy_name="futuremamba_mamba2",
        config_name=config_name,
        backend="mamba2",
        checkpoint_key=method_id,
        train_seeds=_seed_tuple(seeds),
        axes=axes,
        use_history=True,
    )


def method_specs(*, include_mamba3: bool = False) -> list[MethodSpec]:
    specs: list[MethodSpec] = [
        MethodSpec(
            method="pi05_baseline",
            variant="official baseline",
            method_id="pi05_baseline",
            policy_name="pi05_baseline",
            config_name="pi05_baseline",
            backend="baseline",
            checkpoint_key="pi05_baseline",
            train_seeds=(7,),
            axes={"family": "baseline", "history": False},
        ),
        MethodSpec(
            method="past-actions",
            variant="past-actions",
            method_id="past-actions",
            policy_name="futuremamba_past_actions",
            config_name="futuremamba_robomme_mamba2",
            backend="mamba2",
            checkpoint_key="past-actions",
            train_seeds=(7,),
            axes={"family": "baseline", "memory": "past-actions"},
            use_history=True,
        ),
        MethodSpec(
            method="MemER",
            variant="grounded memory",
            method_id="MemER",
            policy_name="symbolic-grounded-subgoal",
            config_name="mme_vla_suite",
            backend="symbolic",
            checkpoint_key="MemER",
            train_seeds=(7,),
            axes={"family": "official baseline", "memory": "MemER"},
            use_history=True,
            use_memer=True,
            subgoal_type="grounded_subgoal",
        ),
        MethodSpec(
            method="FrameSamp+Modul",
            variant="perceptual memory",
            method_id="perceptual-framesamp-modul",
            policy_name="perceptual-framesamp-modul",
            config_name="mme_vla_suite",
            backend="symbolic",
            checkpoint_key="perceptual-framesamp-modul",
            train_seeds=(7,),
            axes={"architecture": "FrameSamp", "family": "perceptual", "head": "Modul"},
            use_history=True,
        ),
        MethodSpec(
            method="FrameSamp+Expert",
            variant="perceptual memory",
            method_id="perceptual-framesamp-expert",
            policy_name="perceptual-framesamp-expert",
            config_name="mme_vla_suite",
            backend="symbolic",
            checkpoint_key="perceptual-framesamp-expert",
            train_seeds=(7,),
            axes={"architecture": "FrameSamp", "family": "perceptual", "head": "Expert"},
            use_history=True,
        ),
        MethodSpec(
            method="recurrent-rmt-expert",
            variant="recurrent memory",
            method_id="recurrent-rmt-expert",
            policy_name="recurrent-rmt-expert",
            config_name="mme_vla_suite",
            backend="symbolic",
            checkpoint_key="recurrent-rmt-expert",
            train_seeds=(7,),
            axes={"architecture": "RMT", "family": "recurrent", "head": "Expert"},
            use_history=True,
        ),
        _future_spec("futuremamba_mamba2", "mamba2", _future_axes(), seeds=MAMBA2_TRAIN_SEEDS),
        _future_spec("futuremamba_layer_first", "layer_selection=First", _future_axes(layer_selection="First")),
        _future_spec(
            "futuremamba_layer_sensitivity",
            "layer_selection=Sensitivity",
            _future_axes(layer_selection="Sensitivity"),
        ),
        _future_spec("futuremamba_layer_random", "layer_selection=Random", _future_axes(layer_selection="Random")),
        _future_spec(
            "futuremamba_depth_4",
            "progress_depth=4",
            _future_axes(progress_depth=4),
            config_name="futuremamba_robomme_mamba2_depth4",
        ),
        _future_spec(
            "futuremamba_depth_9",
            "progress_depth=9",
            _future_axes(progress_depth=9),
            config_name="futuremamba_robomme_mamba2_depth9",
        ),
        _future_spec(
            "futuremamba_handoff_0p4",
            "handoff_ratio=0.4",
            _future_axes(handoff_ratio=0.4),
            config_name="futuremamba_robomme_mamba2_handoff_0p4",
        ),
        _future_spec(
            "futuremamba_handoff_0p6",
            "handoff_ratio=0.6",
            _future_axes(handoff_ratio=0.6),
            config_name="futuremamba_robomme_mamba2_handoff_0p6",
        ),
        _future_spec(
            "futuremamba_handoff_k0",
            "handoff_ratio=0.0;K=0",
            _future_axes(handoff_ratio=0.0, handoff_steps="K=0"),
            config_name="futuremamba_robomme_mamba2_handoff_k0",
        ),
        _future_spec(
            "futuremamba_handoff_kn",
            "handoff_ratio=1.0;K=N",
            _future_axes(handoff_ratio=1.0, handoff_steps="K=N"),
            config_name="futuremamba_robomme_mamba2_handoff_kn",
        ),
        _future_spec("futuremamba_memory_none", "memory=none", _future_axes(memory="none")),
        _future_spec("futuremamba_memory_shuffled", "memory=shuffled", _future_axes(memory="shuffled")),
        _future_spec(
            "futuremamba_init_random",
            "initialization=random",
            _future_axes(initialization="random"),
        ),
        _future_spec(
            "futuremamba_loss_flow_only",
            "loss=flow-only",
            _future_axes(loss="flow-only"),
            config_name="futuremamba_robomme_mamba2_flow_only",
        ),
        _future_spec(
            "futuremamba_progress_no_memory",
            "Progress Expert=no-memory",
            _future_axes(progress_expert="no-memory"),
        ),
    ]
    if include_mamba3:
        specs.append(
            MethodSpec(
                method="futuremamba",
                variant="mamba3_siso",
                method_id="futuremamba_mamba3_siso",
                policy_name="futuremamba_mamba3_siso",
                config_name="futuremamba_robomme_mamba3_siso",
                backend="mamba3_siso",
                checkpoint_key="futuremamba_mamba3_siso",
                train_seeds=_seed_tuple(MAMBA2_TRAIN_SEEDS),
                axes=_future_axes(backend="mamba3_siso"),
                requires_gate=True,
                use_history=True,
            )
        )
    ids = [spec.method_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("method_id values must be unique")
    return specs


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _load_gate(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    gate = _load_json(path)
    status = gate.get("status")
    if status not in {"passed", "unsupported_on_current_stack"}:
        raise ValueError(f"Mamba-3 gate {path} has invalid status: {status!r}")
    gates = gate.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError(f"Mamba-3 gate {path} is missing gate details")
    status_by_name = {entry.get("name"): entry.get("status") for entry in gates if isinstance(entry, dict)}
    missing = [name for name in MAMBA3_REQUIRED_GATES if name not in status_by_name]
    if missing:
        raise ValueError(f"Mamba-3 gate {path} is missing required gates: {missing}")
    invalid_statuses = {
        name: gate_status
        for name, gate_status in status_by_name.items()
        if name in MAMBA3_REQUIRED_GATES and gate_status not in {"passed", "failed", "not_run"}
    }
    if invalid_statuses:
        raise ValueError(f"Mamba-3 gate {path} has invalid gate statuses: {invalid_statuses}")
    all_passed = all(status_by_name[name] == "passed" for name in MAMBA3_REQUIRED_GATES)
    if status == "passed" and not all_passed:
        raise ValueError(f"Mamba-3 gate {path} is marked passed but not all required gates passed")
    if status != "passed" and all_passed:
        raise ValueError(f"Mamba-3 gate {path} has all gates passed but top-level status is {status!r}")
    return gate


def _gate_passed(gate: Mapping[str, Any] | None) -> bool:
    if gate is None or gate.get("status") != "passed":
        return False
    gates = gate.get("gates", [])
    status_by_name = {entry.get("name"): entry.get("status") for entry in gates if isinstance(entry, dict)}
    return all(status_by_name.get(name) == "passed" for name in MAMBA3_REQUIRED_GATES)


def gate_allows_mamba3(path: Path | str | None) -> bool:
    return _gate_passed(_load_gate(Path(path)) if path is not None else None)


def _load_checkpoint_mapping(path: Path) -> dict[str, Any]:
    mapping = _load_json(path)
    checkpoints = mapping.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise ValueError("checkpoint mapping must define a checkpoints object")
    return mapping


def _require_string(obj: Mapping[str, Any], key: str, *, context: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must define non-empty string {key!r}")
    return value


def _require_int(obj: Mapping[str, Any], key: str, *, context: str) -> int:
    value = obj.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{context} must define integer {key!r}")
    return value


def _checkpoint_entry_from_mapping(
    checkpoints: Mapping[str, Any],
    spec: MethodSpec,
    train_seed: int,
    *,
    base_dir: Path,
) -> CheckpointEntry:
    if spec.checkpoint_key not in checkpoints:
        raise ValueError(f"missing checkpoint mapping for {spec.checkpoint_key}")
    entry_obj = checkpoints[spec.checkpoint_key]
    if isinstance(entry_obj, Mapping) and str(train_seed) in entry_obj and isinstance(entry_obj[str(train_seed)], Mapping):
        entry_obj = entry_obj[str(train_seed)]
    elif isinstance(entry_obj, Mapping) and train_seed in entry_obj and isinstance(entry_obj[train_seed], Mapping):
        entry_obj = entry_obj[train_seed]
    elif isinstance(entry_obj, Mapping) and "checkpoint" in entry_obj:
        if len(spec.train_seeds) != 1:
            raise ValueError(
                f"checkpoint mapping for {spec.checkpoint_key} must define per-seed entries for multi-seed spec"
            )
        entry_obj = entry_obj
    else:
        raise ValueError(f"checkpoint mapping for {spec.checkpoint_key} is missing identity for seed {train_seed}")
    if not isinstance(entry_obj, Mapping):
        raise ValueError(f"checkpoint mapping for {spec.checkpoint_key} must be an object")
    context = f"checkpoint mapping for {spec.checkpoint_key} seed {train_seed}"
    checkpoint = Path(_require_string(entry_obj, "checkpoint", context=context))
    if not checkpoint.is_absolute():
        checkpoint = base_dir / checkpoint
    if "placeholder" in str(checkpoint).lower():
        raise ValueError(f"checkpoint path must not be a placeholder: {checkpoint}")
    if not checkpoint.exists():
        raise ValueError(f"checkpoint path does not exist: {checkpoint}")
    checkpoint_id = _require_int(entry_obj, "checkpoint_id", context=context)
    config = _require_string(entry_obj, "config", context=context)
    backend = _require_string(entry_obj, "backend", context=context)
    policy_name = _require_string(entry_obj, "policy_name", context=context)
    provenance = entry_obj.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError(f"checkpoint mapping for {spec.checkpoint_key} seed {train_seed} must include provenance")
    provenance_seed = provenance.get("train_seed")
    if provenance_seed is not None and int(provenance_seed) != int(train_seed):
        raise ValueError(
            f"checkpoint mapping for {spec.checkpoint_key} provenance train_seed {provenance_seed!r} "
            f"does not match requested seed {train_seed}"
        )
    return CheckpointEntry(
        path=checkpoint,
        checkpoint_id=checkpoint_id,
        config=config,
        backend=backend,
        provenance=dict(provenance),
        policy_name=policy_name,
        train_seed=train_seed,
    )


def _validate_checkpoint_matches_spec(spec: MethodSpec, checkpoint: CheckpointEntry) -> None:
    if checkpoint.backend != spec.backend:
        raise ValueError(f"checkpoint backend {checkpoint.backend!r} does not match method backend {spec.backend!r}")
    if checkpoint.config != spec.config_name:
        raise ValueError(f"checkpoint config {checkpoint.config!r} does not match method config {spec.config_name!r}")
    if checkpoint.policy_name != spec.policy_name:
        raise ValueError(
            f"checkpoint policy_name {checkpoint.policy_name!r} does not match method policy_name {spec.policy_name!r}"
        )
    if spec.backend == "mamba3_siso" and (checkpoint.config.endswith("mamba2") or checkpoint.backend == "mamba2"):
        raise ValueError("Mamba-3 experiments require a distinct Mamba-3 config and checkpoint identity")


def _safe_id(value: str) -> str:
    safe = []
    for char in value:
        if char.isalnum() or char in {"-", "_", "="}:
            safe.append(char)
        elif char in {" ", "+", "/"}:
            safe.append("-")
        else:
            safe.append("-")
    return "".join(safe).strip("-")


def _uses_official_policy_server(spec: MethodSpec) -> bool:
    return spec.config_name in {"pi05_baseline", "mme_vla_suite"}


def _server_argv(entry: CheckpointEntry, spec: MethodSpec, port: int) -> list[str]:
    if _uses_official_policy_server(spec):
        return [
            "uv",
            "run",
            "scripts/serve_policy.py",
            "--seed",
            str(entry.train_seed),
            "--port",
            str(port),
            "policy:checkpoint",
            "--policy.dir",
            str(entry.path),
            "--policy.config",
            entry.config,
        ]
    return [
        "uv",
        "run",
        "scripts/serve_policy.py",
        "--port",
        str(port),
        "policy:checkpoint",
        "--policy.dir",
        str(entry.path),
        "--policy.config",
        entry.config,
    ]


def _server_payload(entry: CheckpointEntry, spec: MethodSpec, port: int) -> dict[str, Any]:
    return {
        "argv": _server_argv(entry, spec, port),
        "cwd": "third_party/robomme_policy_learning" if _uses_official_policy_server(spec) else ".",
        "env": {"CUDA_VISIBLE_DEVICES": "0"},
        "launcher": "scripts/serve_policy.py",
        "params": {
            "checkpoint_dir": str(entry.path),
            "config": entry.config,
            "policy_name": spec.policy_name,
            "port": port,
            "seed": entry.train_seed,
        },
    }


def _official_dataset_for_split(split: str) -> str:
    try:
        return OFFICIAL_DATASET_BY_SPLIT[split]
    except KeyError as exc:
        raise ValueError(f"unknown split: {split!r}") from exc


def _eval_argv(
    entry: CheckpointEntry,
    spec: MethodSpec,
    selection: EpisodeSelection,
    episode_id: int,
    port: int,
    result_identity: Mapping[str, Any],
) -> list[str]:
    argv = [
        "python",
        "scripts/run_robomme_single_episode.py",
        "--task-name",
        selection.task,
        "--episode-id",
        str(episode_id),
        "--split",
        selection.split,
        "--dataset",
        _official_dataset_for_split(selection.split),
        "--port",
        str(port),
        "--model-seed",
        str(entry.train_seed),
        "--model-ckpt-id",
        str(entry.checkpoint_id),
        "--policy-name",
        spec.policy_name,
        "--save-dir",
        f"runs/evaluation/{spec.policy_name}/ckpt{entry.checkpoint_id}/seed{entry.train_seed}",
        "--result-identity-json",
        json.dumps(result_identity, sort_keys=True, separators=(",", ":")),
    ]
    if spec.use_history:
        argv.append("--use-history")
    if spec.use_oracle:
        argv.append("--use-oracle")
    if spec.use_qwenvl:
        argv.append("--use-qwenvl")
    if spec.use_memer:
        argv.append("--use-memer")
    if spec.use_gemini:
        argv.append("--use-gemini")
    if spec.subgoal_type is not None:
        argv.extend(["--subgoal-type", spec.subgoal_type])
    return argv


def _eval_payload(
    entry: CheckpointEntry,
    spec: MethodSpec,
    selection: EpisodeSelection,
    episode_id: int,
    port: int,
    result_identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "argv": _eval_argv(entry, spec, selection, episode_id, port, result_identity),
        "cwd": ".",
        "env": {"CUDA_VISIBLE_DEVICES": "1"},
        "launcher": "scripts/run_robomme_single_episode.py",
        "params": {
            "episode_id": episode_id,
            "model_ckpt_id": entry.checkpoint_id,
            "model_seed": entry.train_seed,
            "policy_name": spec.policy_name,
            "port": port,
            "split": selection.split,
            "subgoal_type": spec.subgoal_type,
            "task_name": selection.task,
            "official_single_episode_launcher": {
                "argv": [
                    "python",
                    "third_party/robomme_policy_learning/third_party/robomme_benchmark/scripts/run_example.py",
                    "--dataset",
                    _official_dataset_for_split(selection.split),
                    "--task-id",
                    selection.task,
                    "--episode-idx",
                    str(episode_id),
                    "--action-space-type",
                    "joint_angle",
                ],
                "params": {
                    "action_space_type": "joint_angle",
                    "dataset": _official_dataset_for_split(selection.split),
                    "episode_idx": episode_id,
                    "task_id": selection.task,
                },
            },
            "use_history": spec.use_history,
            "use_memer": spec.use_memer,
        },
    }


def _experiment_id(spec: MethodSpec, task: str, split: str, episode_id: int, train_seed: int, eval_seed: int) -> str:
    return _safe_id(f"{spec.method_id}__{task}__{split}__ep{episode_id:03d}__train{train_seed}__eval{eval_seed}")


def _checkpoint_identity(checkpoint: CheckpointEntry) -> dict[str, Any]:
    return {
        "backend": checkpoint.backend,
        "checkpoint_id": checkpoint.checkpoint_id,
        "config": checkpoint.config,
        "path": str(checkpoint.path),
        "policy_name": checkpoint.policy_name,
        "provenance": dict(sorted(checkpoint.provenance.items())),
        "train_seed": checkpoint.train_seed,
    }


def _config_identity(spec: MethodSpec, checkpoint: CheckpointEntry) -> dict[str, Any]:
    return {
        "axes": dict(sorted(spec.axes.items())),
        "backend": checkpoint.backend,
        "checkpoint_key": spec.checkpoint_key,
        "method": spec.method,
        "method_id": spec.method_id,
        "name": checkpoint.config,
        "policy_name": spec.policy_name,
        "train_seed": checkpoint.train_seed,
        "variant": spec.variant,
    }


def _manifest_records(
    *,
    phase: str,
    checkpoint_mapping: Mapping[str, Any],
    checkpoint_base_dir: Path,
    include_mamba3: bool,
    port: int,
    eval_seed: int,
    gate: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    records: list[ExperimentRecord] = []
    for selection in phase_episode_plan(phase):
        for spec in method_specs(include_mamba3=include_mamba3):
            for train_seed in spec.train_seeds:
                checkpoint = _checkpoint_entry_from_mapping(
                    checkpoint_mapping,
                    spec,
                    train_seed,
                    base_dir=checkpoint_base_dir,
                )
                _validate_checkpoint_matches_spec(spec, checkpoint)
                for episode_id in selection.episode_ids:
                    experiment_id = _experiment_id(
                        spec,
                        selection.task,
                        selection.split,
                        episode_id,
                        train_seed,
                        eval_seed,
                    )
                    checkpoint_identity = _checkpoint_identity(checkpoint)
                    config_identity = _config_identity(spec, checkpoint)
                    provenance = {
                        "benchmark_commit": ROBOMME_BENCHMARK_COMMIT,
                        "checkpoint_identity": str(checkpoint.path),
                        "checkpoint_source": spec.checkpoint_key,
                        "config_identity": checkpoint.config,
                        "mamba3_gate": gate["status"] if gate is not None else None,
                        "policy_commit": ROBOMME_POLICY_LEARNING_COMMIT,
                        "suite_source": "third_party/robomme_policy_learning/third_party/robomme_benchmark/readme.md",
                        "task_source": "third_party/robomme_policy_learning/examples/robomme/utils.py:TASK_NAME_LIST",
                    }
                    result_identity = {
                        "experiment_id": experiment_id,
                        "method_id": spec.method_id,
                        "train_seed": train_seed,
                        "task_name": selection.task,
                        "episode_id": episode_id,
                        "split": selection.split,
                        "checkpoint": checkpoint_identity,
                        "config": config_identity,
                        "provenance": provenance,
                    }
                    records.append(
                        ExperimentRecord(
                            axis=dict(sorted(spec.axes.items())),
                            backend=spec.backend,
                            category=selection.category,
                            checkpoint=checkpoint_identity,
                            config=config_identity,
                            eval=_eval_payload(checkpoint, spec, selection, episode_id, port, result_identity),
                            eval_seed=eval_seed,
                            experiment_id=experiment_id,
                            episode_id=episode_id,
                            mamba3_gate=gate if spec.requires_gate else None,
                            method=spec.method,
                            method_id=spec.method_id,
                            provenance=provenance,
                            server=_server_payload(checkpoint, spec, port),
                            split=selection.split,
                            task=selection.task,
                            train_seed=train_seed,
                            variant=spec.variant,
                        )
                    )
    serialized = [record.to_dict() for record in records]
    experiment_ids = [item["experiment_id"] for item in serialized]
    if len(experiment_ids) != len(set(experiment_ids)):
        raise ValueError("experiment_id must be unique across manifest records")
    if any(not item["checkpoint"]["path"] or not item["config"]["name"] or not item["provenance"] for item in serialized):
        raise ValueError("manifest contains incomplete identity fields")
    return sorted(serialized, key=lambda item: item["experiment_id"])


def build_manifest(
    *,
    checkpoint_mapping_path: Path,
    stage: str,
    output_path: Path | None = None,
    port: int = DEFAULT_SERVER_PORT,
    eval_seed: int = DEFAULT_EVAL_SEED,
    mamba3_gate_path: Path | None = None,
) -> dict[str, Any]:
    mapping = _load_checkpoint_mapping(checkpoint_mapping_path)
    checkpoints = mapping["checkpoints"]
    if not isinstance(checkpoints, dict):
        raise ValueError("checkpoint mapping must define a checkpoints object")
    gate = _load_gate(mamba3_gate_path)
    experiments = _manifest_records(
        phase=stage,
        checkpoint_mapping=checkpoints,
        checkpoint_base_dir=checkpoint_mapping_path.parent,
        include_mamba3=_gate_passed(gate),
        port=port,
        eval_seed=eval_seed,
        gate=gate,
    )
    phases = {
        stage: [
            {"episode_count": len(item.episode_ids), "split": item.split, "task": item.task}
            for item in phase_episode_plan(stage)
        ]
    }
    manifest = Manifest(
        schema="openpi.robomme_experiment_matrix",
        schema_version=1,
        generated_by="scripts/run_robomme_experiment_matrix.py",
        provenance={
            "benchmark_commit": ROBOMME_BENCHMARK_COMMIT,
            "checkpoint_mapping_source": str(checkpoint_mapping_path),
            "eval_seed": eval_seed,
            "mamba3_gate_source": str(mamba3_gate_path) if mamba3_gate_path is not None else None,
            "policy_commit": ROBOMME_POLICY_LEARNING_COMMIT,
            "server_port": port,
            "suite_source": "third_party/robomme_policy_learning/third_party/robomme_benchmark/readme.md",
            "task_source": "third_party/robomme_policy_learning/examples/robomme/utils.py:TASK_NAME_LIST",
        },
        phases=phases,
        experiments=experiments,
    )
    result = manifest.to_dict()
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a deterministic RoboMME experiment matrix")
    parser.add_argument("--checkpoint-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("minimal", "counting"), required=True)
    parser.add_argument("--port", type=int, default=DEFAULT_SERVER_PORT)
    parser.add_argument("--eval-seed", type=int, default=DEFAULT_EVAL_SEED)
    parser.add_argument("--mamba3-gate", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _parse_args(argv)
    return build_manifest(
        checkpoint_mapping_path=args.checkpoint_mapping,
        stage=args.stage,
        output_path=args.output,
        port=args.port,
        eval_seed=args.eval_seed,
        mamba3_gate_path=args.mamba3_gate,
    )


if __name__ == "__main__":
    main()
