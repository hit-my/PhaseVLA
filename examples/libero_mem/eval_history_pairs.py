from __future__ import annotations

import argparse
from collections.abc import Mapping
import copy
import dataclasses
import importlib.util
import inspect
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

_BUILD_PATH = Path(__file__).with_name("build_history_pairs.py")
_BUILD_SPEC = importlib.util.spec_from_file_location("libero_mem_build_history_pairs_for_eval", _BUILD_PATH)
build_history_pairs = importlib.util.module_from_spec(_BUILD_SPEC)
sys.modules[_BUILD_SPEC.name] = build_history_pairs
assert _BUILD_SPEC.loader is not None
_BUILD_SPEC.loader.exec_module(build_history_pairs)
HistoryPair = build_history_pairs.HistoryPair
SyntheticCandidate = build_history_pairs.SyntheticCandidate
stable_checksum = build_history_pairs.stable_checksum


CONDITIONS = ("frozen_baseline", "correct", "zero", "truncated", "shuffled", "swapped")


def evaluate_history_pair(
    pair: HistoryPair,
    *,
    policy: Any,
    environment: Any,
    history_replayer: Any,
    branch_scorer: Any,
    truncated_k: int,
    shuffle_seed: int = 0,
) -> list[dict[str, Any]]:
    """Evaluate a pair under injected state interventions.

    The policy only receives the canonical current observation and an explicit
    policy-state snapshot. Hidden progress state is restored on the evaluator
    environment, never passed into policy input.
    """

    build_history_pairs.validate_pair_manifest(pair.manifest)
    _validate_pair_state_checksums(pair)
    noise = {"seed": int(pair.noise_seed)}
    snapshots = _snapshots_for_pair(
        pair, history_replayer=history_replayer, truncated_k=truncated_k, shuffle_seed=shuffle_seed
    )
    rows: list[dict[str, Any]] = []
    for trial_branch in ("a", "b"):
        candidate = getattr(pair, trial_branch)
        trial_manifest = pair.manifest[trial_branch]
        evaluator_progress_state = copy.deepcopy(candidate.evaluator_progress_state)
        for condition, state in _states_for_trial(trial_branch, snapshots=snapshots, truncated_k=truncated_k):
            environment.restore(
                observation=copy.deepcopy(pair.current_observation),
                physical_state=copy.deepcopy(pair.canonical_physical_state),
                evaluator_progress_state=copy.deepcopy(evaluator_progress_state),
                noise=copy.deepcopy(noise),
            )
            action = _policy_action(
                policy,
                observation=copy.deepcopy(pair.current_observation),
                state=copy.deepcopy(state),
                noise=copy.deepcopy(noise),
                task=pair.manifest["task_id"],
            )
            semantic = _validate_branch_semantics(
                _call_branch_scorer(
                    branch_scorer,
                    action,
                    target_branch=trial_manifest["expert_branch"],
                    target_predicate=trial_manifest["target_predicate"],
                    task=pair.manifest["task_id"],
                    progress_label=trial_manifest["progress_label"],
                    environment=environment,
                ),
                target_branch=trial_manifest["expert_branch"],
            )
            observed_branch = _observed_branch_from_semantics(semantic)
            branch_correct = observed_branch == str(trial_manifest["expert_branch"])
            row = {
                "pair_id": pair.pair_id,
                "trial_branch": trial_branch,
                "condition": condition,
                "episode_id": trial_manifest["episode_id"],
                "query_id": trial_manifest["query_id"],
                "target_branch": trial_manifest["expert_branch"],
                "target_predicate": trial_manifest["target_predicate"],
                "evaluator_progress_label": trial_manifest["progress_label"],
                "observation_checksum": stable_checksum(pair.current_observation),
                "physical_state_checksum": stable_checksum(pair.canonical_physical_state),
                "evaluator_progress_checksum": stable_checksum(evaluator_progress_state),
                "noise_checksum": stable_checksum(noise),
                "state_checksum": stable_checksum(state),
                "observed_branch": observed_branch,
                "event_trace": copy.deepcopy(semantic["event_trace"]),
                "branch_correct": branch_correct,
                "semantic_branch": semantic,
                "action_prefix_distance": action_prefix_distance(action, trial_manifest["expert_branch"]),
            }
            rows.append(row)
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def branch_accuracy(rows: list[dict[str, Any]], *, condition: str | None = None) -> float:
    filtered = [row for row in rows if condition is None or row.get("condition") == condition]
    if not filtered:
        raise ValueError("branch accuracy is undefined for zero rows")
    return sum(1 for row in filtered if branch_correct_from_result(row)) / len(filtered)


def branch_correct_from_result(row: dict[str, Any]) -> bool:
    target_branch = row.get("target_branch")
    if target_branch is None:
        raise ValueError("result row missing target_branch")
    return _observed_branch_from_result(row) == str(target_branch)


def _observed_branch_from_result(row: dict[str, Any]) -> str:
    if "event_trace" not in row:
        raise ValueError("result row missing event_trace")
    observed_from_trace = _observed_branch_from_event_trace(row["event_trace"])
    observed_branch = row.get("observed_branch")
    if observed_branch is not None and str(observed_branch) != observed_from_trace:
        raise ValueError("result row observed_branch conflicts with event_trace")
    return observed_from_trace


def _validate_branch_semantics(raw_semantic: Any, *, target_branch: str) -> dict[str, Any]:
    if not isinstance(raw_semantic, dict):
        raise ValueError("branch_scorer must return an object")
    semantic = dict(raw_semantic)
    if "event_trace" not in semantic:
        raise ValueError("branch_scorer result missing event_trace")
    observed_from_trace = _observed_branch_from_event_trace(semantic["event_trace"])
    observed_branch = semantic.get("observed_branch", semantic.get("entered_branch", observed_from_trace))
    if str(observed_branch) != observed_from_trace:
        raise ValueError("branch_scorer observed branch conflicts with event_trace")
    if "target_branch" in semantic and str(semantic["target_branch"]) != str(target_branch):
        raise ValueError("branch_scorer target_branch conflicts with trial target")
    semantic["observed_branch"] = observed_from_trace
    semantic["target_branch"] = str(target_branch)
    return semantic


def _observed_branch_from_semantics(semantic: dict[str, Any]) -> str:
    return str(semantic["observed_branch"])


def _observed_branch_from_event_trace(event_trace: Any) -> str:
    if not isinstance(event_trace, list) or not event_trace:
        raise ValueError("event_trace must be a non-empty list")
    for event in event_trace:
        if not isinstance(event, dict):
            continue
        if "branch" in event:
            return str(event["branch"])
        if "observed_branch" in event:
            return str(event["observed_branch"])
        if "entered_branch" in event:
            return str(event["entered_branch"])
        if "target_predicate" in event and event.get("satisfied") is True:
            return str(event["target_predicate"])
    raise ValueError("event_trace does not contain an entered branch label")


def _validate_pair_state_checksums(pair: HistoryPair) -> None:
    if stable_checksum(pair.canonical_physical_state) != pair.manifest["canonical_physical_state_checksum"]:
        raise ValueError("canonical_physical_state_checksum does not match pair state")
    if stable_checksum(pair.current_observation) != pair.manifest["current_observation_checksum"]:
        raise ValueError("current_observation_checksum does not match pair observation")
    for branch_name in ("a", "b"):
        candidate = getattr(pair, branch_name)
        expected_checksum = pair.manifest[branch_name]["evaluator_progress_checksum"]
        if stable_checksum(candidate.evaluator_progress_state) != expected_checksum:
            raise ValueError(f"{branch_name} evaluator_progress_checksum does not match pair state")


def load_policy(policy_path: str | Path | None = None, *, policy_config: str = "futuremamba_libero_mem") -> Any:
    """Late-import and adapt the real FutureMamba checkpoint policy for history intervention eval."""

    if policy_path is None:
        raise ValueError(
            "Real policy loading requires --policy-path/--checkpoint-dir pointing at an OpenPI checkpoint. "
            "For offline smoke tests call main(..., policy=...) with an injected synthetic policy."
        )
    try:
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config
    except Exception as exc:  # pragma: no cover - exercised only with real dependencies.
        raise RuntimeError(
            "Unable to load real OpenPI policy dependencies. Install OpenPI runtime dependencies, or call "
            "main(..., policy=...) with an injected policy."
        ) from exc

    train_config = _config.get_config(policy_config)
    policy = _policy_config.create_trained_policy(train_config, Path(policy_path))
    return _PolicyActionAdapter(policy)


def evaluate_history_pairs(
    pairs: list[HistoryPair],
    *,
    policy: Any,
    environment_factory: Any,
    history_replayer: Any,
    branch_scorer: Any,
    truncated_k: int,
    shuffle_seed: int = 0,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    creates_environment = callable(environment_factory)
    for pair in pairs:
        environment = environment_factory(pair) if creates_environment else environment_factory
        try:
            rows.extend(
                evaluate_history_pair(
                    pair,
                    policy=policy,
                    environment=environment,
                    history_replayer=history_replayer,
                    branch_scorer=branch_scorer,
                    truncated_k=truncated_k,
                    shuffle_seed=shuffle_seed,
                )
            )
        finally:
            if creates_environment and hasattr(environment, "close"):
                environment.close()
    return rows


def _load_real_environment_factory(
    *, args: argparse.Namespace | None = None, pairs: list[HistoryPair] | None = None
) -> Any:
    imports = _load_real_libero_dependencies()
    suite_name = str(getattr(args, "task_suite_name", None) or os.environ.get("LIBERO_MEM_TASK_SUITE", "libero_mem"))
    try:
        suite = imports["benchmark"].get_benchmark_dict()[suite_name]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown LIBERO task suite {suite_name!r}; pass --task-suite-name with a known suite"
        ) from exc
    factory = _LiberoMemEnvironmentFactory(
        task_suite=suite,
        task_suite_name=suite_name,
        get_libero_path=imports["get_libero_path"],
        offscreen_env=imports["OffScreenRenderEnv"],
        env_adapter_module=imports["env_adapter"],
    )
    if pairs is not None:
        factory.validate_pairs(pairs)
    return factory


def _load_real_history_replayer(
    *, args: argparse.Namespace | None = None, pairs: list[HistoryPair] | None = None, policy: Any | None = None
) -> Any:
    if pairs and _manifest_embeds_policy_states(pairs):
        return _ManifestPolicyStateReplayer()
    repo_id = getattr(args, "history_repo_id", None) if args is not None else None
    dataset_root = getattr(args, "history_dataset_root", None) if args is not None else None
    repo_id = repo_id or os.environ.get("LIBERO_MEM_HISTORY_REPO_ID")
    dataset_root = dataset_root or os.environ.get("LIBERO_MEM_HISTORY_DATASET_ROOT")
    if repo_id is None and dataset_root is None:
        raise ValueError(
            "Real history replay requires replayable policy-memory history for per-condition interventions. Pass "
            "--history-repo-id and optional --history-dataset-root for a LeRobot dataset, set "
            "LIBERO_MEM_HISTORY_REPO_ID, or include per-condition policy_state_snapshots/history_state_snapshots "
            "in the pair manifest. Branch-only snapshots cannot distinguish correct/truncated/shuffled histories."
        )
    if policy is None:
        raise ValueError("Real history replay requires a loaded policy; pass --policy-path/--checkpoint-dir")
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover - exercised only with real dependencies.
        raise RuntimeError(
            "LeRobotDataset is unavailable. Install LeRobot dependencies or inject history_replayer=... for offline smoke."
        ) from exc
    resolved_repo_id = repo_id or "local/libero_mem_history"
    dataset_kwargs = {"repo_id": resolved_repo_id}
    if dataset_root is not None:
        dataset_kwargs["root"] = Path(dataset_root)
    return _LeRobotHistoryReplayer(LeRobotDataset(**dataset_kwargs), policy=policy)


def _load_real_branch_scorer(*, args: argparse.Namespace | None = None) -> Any:
    max_steps = int(getattr(args, "branch_score_steps", 10) if args is not None else 10)
    if max_steps < 1:
        raise ValueError("--branch-score-steps must be positive")
    return _LiberoBranchScorer(max_steps=max_steps)


def _load_real_libero_dependencies() -> dict[str, Any]:
    try:
        from libero.libero import benchmark
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except Exception as exc:  # pragma: no cover - exercised only with real dependencies.
        raise RuntimeError(
            "Unable to load real LIBERO-Mem dependencies. Install the third_party/libero fork/runtime, or call "
            "main(..., environment_factory=...) with an injected factory."
        ) from exc
    return {
        "benchmark": benchmark,
        "get_libero_path": get_libero_path,
        "OffScreenRenderEnv": OffScreenRenderEnv,
        "env_adapter": _load_local_module("env_adapter.py", "libero_mem_env_adapter_for_history_eval"),
    }


def _load_local_module(filename: str, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _PolicyActionAdapter:
    def __init__(self, policy: Any):
        self._policy = policy

    def act(self, observation: Any, *, state: Any, noise: dict[str, int], task: str, progress_state: Any = None) -> Any:
        del progress_state
        self._restore_or_reset(state)
        payload = _policy_observation_payload(observation, task=task)
        noise_array = _policy_noise_array(self._policy, noise)
        if hasattr(self._policy, "infer"):
            if noise_array is None:
                return self._policy.infer(payload)
            try:
                return self._policy.infer(payload, noise=noise_array)
            except TypeError:
                return self._policy.infer(payload)
        if callable(self._policy):
            return self._policy(payload)
        raise TypeError("policy must provide infer(...) or be callable")

    def infer(self, observation: dict[str, Any], *, noise: Any | None = None) -> Any:
        if hasattr(self._policy, "infer"):
            if noise is None:
                return self._policy.infer(observation)
            try:
                return self._policy.infer(observation, noise=noise)
            except TypeError:
                return self._policy.infer(observation)
        if callable(self._policy):
            return self._policy(observation)
        raise TypeError("policy must provide infer(...) or be callable")

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()

    def snapshot_state(self) -> Any:
        if not hasattr(self._policy, "snapshot_state"):
            raise TypeError("policy must provide snapshot_state() for real history replay")
        return self._policy.snapshot_state()

    def restore_state(self, state: Any) -> None:
        if not hasattr(self._policy, "restore_state"):
            raise TypeError("policy must provide restore_state(state) for state intervention")
        self._policy.restore_state(state)

    def fork(self) -> _PolicyActionAdapter:
        if hasattr(self._policy, "fork"):
            return _PolicyActionAdapter(self._policy.fork())
        return _PolicyActionAdapter(copy.deepcopy(self._policy))

    def _restore_or_reset(self, state: Any) -> None:
        if state in (None, {}):
            self.reset()
            return
        self.restore_state(state)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._policy, name)


class _LiberoMemEnvironmentFactory:
    def __init__(
        self,
        *,
        task_suite: Any,
        task_suite_name: str,
        get_libero_path: Any,
        offscreen_env: Any,
        env_adapter_module: Any,
    ):
        self._task_suite = task_suite
        self._task_suite_name = task_suite_name
        self._get_libero_path = get_libero_path
        self._offscreen_env = offscreen_env
        self._env_adapter_module = env_adapter_module

    def validate_pairs(self, pairs: list[HistoryPair]) -> None:
        for pair in pairs:
            self._resolve_task(pair.manifest["task_id"])

    def __call__(self, pair: HistoryPair) -> Any:
        _task_index, task_object = self._resolve_task(pair.manifest["task_id"])
        task_text = str(getattr(task_object, "language", pair.manifest["task_id"]))
        try:
            task_bddl = Path(self._get_libero_path("bddl_files")) / task_object.problem_folder / task_object.bddl_file
        except AttributeError as exc:
            raise ValueError(
                "LIBERO task object must expose problem_folder and bddl_file for environment construction"
            ) from exc
        env = self._offscreen_env(bddl_file_name=task_bddl, camera_heights=256, camera_widths=256)
        return _RestorableLiberoMemEnvironment(env, task_text=task_text, env_adapter_module=self._env_adapter_module)

    def _resolve_task(self, manifest_task_id: Any) -> tuple[int, Any]:
        wanted = str(manifest_task_id)
        task_count = int(self._task_suite.n_tasks)
        numeric = _numeric_task_id(wanted)
        if numeric is not None and 0 <= numeric < task_count:
            return numeric, self._task_suite.get_task(numeric)
        for task_index in range(task_count):
            task_object = self._task_suite.get_task(task_index)
            aliases = {
                str(task_index),
                str(getattr(task_object, "language", "")),
                str(getattr(task_object, "bddl_file", "")),
                Path(str(getattr(task_object, "bddl_file", ""))).stem,
            }
            if wanted in aliases:
                return task_index, task_object
        raise ValueError(
            f"Cannot map pair manifest task_id={wanted!r} to suite {self._task_suite_name!r}. "
            "Use numeric LIBERO task ids in the manifest or pass --task-suite-name matching the manifest."
        )


class _RestorableLiberoMemEnvironment:
    def __init__(self, env: Any, *, task_text: str, env_adapter_module: Any):
        self._env = env
        self._task_text = task_text
        self._adapter = env_adapter_module.LiberoMemEnvAdapter(env, task_text=task_text)
        self._has_reset = False
        self._last_observation = None
        self._closed = False

    @property
    def env(self) -> Any:
        return self._env

    def restore(self, *, observation: Any, physical_state: Any, evaluator_progress_state: Any, noise: Any) -> None:
        del noise
        if not self._has_reset:
            self._env.reset()
            self._has_reset = True
        restored_observation = _restore_physical_state(self._env, physical_state)
        _restore_libero_progress_state(self._env, evaluator_progress_state)
        self._last_observation = copy.deepcopy(observation) if observation is not None else restored_observation

    def step(self, action: Any) -> Any:
        return self._adapter.step(action)

    def advance(self, action: Any) -> Any:
        return self.step(action)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        seen: set[int] = set()
        for target in (self._adapter, self._env, _raw_libero_env(self._env)):
            if id(target) in seen:
                continue
            seen.add(id(target))
            close = getattr(target, "close", None)
            if callable(close):
                close()

    def score_action(
        self,
        action: Any,
        *,
        target_branch: str,
        target_predicate: str,
        task: str,
        progress_label: str,
        max_steps: int,
    ) -> dict[str, Any]:
        del task
        actions = _action_sequence(action)
        before = _snapshot_satisfied(self._adapter, self._last_observation)
        event_trace: list[dict[str, Any]] = []
        for step_index, single_action in enumerate(actions[:max_steps], start=1):
            snapshot = self.step(single_action)
            self._last_observation = getattr(snapshot, "observation", None)
            after = list(getattr(snapshot, "satisfied_subgoals", []) or [])
            new_subgoals = [subgoal for subgoal in after if subgoal not in before]
            for subgoal in new_subgoals:
                branch = _branch_label_from_subgoal(
                    subgoal, target_branch=target_branch, target_predicate=target_predicate
                )
                event_trace.append(
                    {
                        "event": "entered_branch",
                        "branch": branch,
                        "target_predicate": branch,
                        "satisfied": True,
                        "progress_label": progress_label,
                        "step": step_index,
                        "subgoal": str(subgoal),
                    }
                )
                return {"observed_branch": branch, "target_branch": str(target_branch), "event_trace": event_trace}
            atomic_predicates = getattr(snapshot, "atomic_predicates", {}) or {}
            for predicate, satisfied in atomic_predicates.items():
                if satisfied and _same_label(predicate, target_predicate):
                    event_trace.append(
                        {
                            "event": "entered_branch",
                            "branch": str(target_branch),
                            "target_predicate": str(target_predicate),
                            "satisfied": True,
                            "progress_label": progress_label,
                            "step": step_index,
                        }
                    )
                    return {
                        "observed_branch": str(target_branch),
                        "target_branch": str(target_branch),
                        "event_trace": event_trace,
                    }
            before = after
        event_trace.append(
            {
                "event": "no_branch_entered",
                "branch": "unresolved",
                "target_branch": str(target_branch),
                "progress_label": progress_label,
                "steps": min(len(actions), max_steps),
            }
        )
        return {"observed_branch": "unresolved", "target_branch": str(target_branch), "event_trace": event_trace}


class _LeRobotHistoryReplayer:
    def __init__(self, dataset: Any, *, policy: Any):
        self._dataset = dataset
        self._policy = policy

    def __call__(
        self, history_indices: list[int], *, branch: str, pair: HistoryPair | None = None, candidate: Any = None
    ) -> Any:
        del branch, pair, candidate
        session = self._policy.fork() if hasattr(self._policy, "fork") else copy.deepcopy(self._policy)
        if hasattr(session, "reset"):
            session.reset()
        for history_index in history_indices:
            row = _to_plain_data(self._dataset[int(history_index)])
            payload = _dataset_row_policy_payload(row)
            if hasattr(session, "infer"):
                session.infer(payload)
            else:
                session(payload)
        if not hasattr(session, "snapshot_state"):
            raise TypeError("policy must provide snapshot_state() after history replay")
        return session.snapshot_state()


class _ManifestPolicyStateReplayer:
    def __call__(
        self,
        history_indices: list[int],
        *,
        branch: str,
        pair: HistoryPair | None = None,
        candidate: Any = None,
        intervention: str | None = None,
    ) -> Any:
        manifest_branch = _manifest_branch_for_candidate(pair, candidate) if pair is not None else None
        if manifest_branch is None:
            raise ValueError("manifest policy-state replay requires pair and candidate context")
        if intervention is None:
            raise ValueError("manifest policy-state replay requires a per-condition intervention label")
        snapshots = _manifest_policy_state_snapshots(manifest_branch)
        entry = snapshots.get(intervention) or snapshots.get(f"{intervention}_{len(history_indices)}")
        if entry is None:
            raise ValueError(
                f"pair manifest branch missing policy/history state snapshot for intervention {intervention!r}"
            )
        return _manifest_snapshot_state(entry, history_indices=history_indices, branch=branch)


class _LiberoBranchScorer:
    def __init__(self, *, max_steps: int):
        self._max_steps = max_steps

    def __call__(
        self,
        action: Any,
        *,
        target_branch: str,
        task: str,
        progress_label: str,
        environment: Any | None = None,
        target_predicate: str | None = None,
    ) -> dict[str, Any]:
        if isinstance(action, Mapping) and "branch" in action and "actions" not in action:
            observed_branch = str(action["branch"])
            return {
                "observed_branch": observed_branch,
                "target_branch": str(target_branch),
                "event_trace": [
                    {
                        "event": "entered_branch",
                        "branch": observed_branch,
                        "task": task,
                        "progress_label": progress_label,
                    }
                ],
            }
        if environment is None or not hasattr(environment, "score_action"):
            raise ValueError(
                "Real branch scoring requires a restorable LIBERO environment; inject branch_scorer for smoke tests"
            )
        return environment.score_action(
            action,
            target_branch=str(target_branch),
            target_predicate=str(target_predicate or target_branch),
            task=str(task),
            progress_label=str(progress_label),
            max_steps=self._max_steps,
        )


def _snapshots_for_pair(
    pair: HistoryPair, *, history_replayer: Any, truncated_k: int, shuffle_seed: int
) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for branch_name in ("a", "b"):
        candidate = getattr(pair, branch_name)
        full_history = list(candidate.history_indices)
        snapshots[f"{branch_name}:correct"] = _call_history_replayer(
            history_replayer,
            full_history,
            branch=candidate.target_branch,
            pair=pair,
            candidate=candidate,
            intervention="correct",
        )
        snapshots[f"{branch_name}:truncated"] = _call_history_replayer(
            history_replayer,
            full_history[-truncated_k:],
            branch=candidate.target_branch,
            pair=pair,
            candidate=candidate,
            intervention="truncated",
        )
        snapshots[f"{branch_name}:shuffled"] = _call_history_replayer(
            history_replayer,
            _deterministic_shuffle(full_history, seed=shuffle_seed),
            branch=candidate.target_branch,
            pair=pair,
            candidate=candidate,
            intervention="shuffled",
        )
    return snapshots


def _states_for_trial(
    trial_branch: str, *, snapshots: dict[str, dict[str, Any]], truncated_k: int
) -> list[tuple[str, Any]]:
    other_branch = "b" if trial_branch == "a" else "a"
    return [
        ("frozen_baseline", None),
        ("correct", snapshots[f"{trial_branch}:correct"]),
        ("zero", {}),
        (f"truncated_{truncated_k}", snapshots[f"{trial_branch}:truncated"]),
        ("shuffled", snapshots[f"{trial_branch}:shuffled"]),
        ("swapped", snapshots[f"{other_branch}:correct"]),
    ]


def _policy_action(policy: Any, *, observation: Any, state: Any, noise: dict[str, int], task: str) -> Any:
    if hasattr(policy, "act"):
        return policy.act(observation, state=state, noise=noise, task=task, progress_state=None)
    if hasattr(policy, "infer"):
        return _PolicyActionAdapter(policy).act(observation, state=state, noise=noise, task=task, progress_state=None)
    return policy(observation, state=state, noise=noise, task=task, progress_state=None)


def _call_history_replayer(
    history_replayer: Any,
    history_indices: list[int],
    *,
    branch: str,
    pair: HistoryPair,
    candidate: Any,
    intervention: str,
) -> Any:
    kwargs = {"branch": branch}
    if _callable_accepts(history_replayer, "pair"):
        kwargs["pair"] = pair
    if _callable_accepts(history_replayer, "candidate"):
        kwargs["candidate"] = candidate
    if _callable_accepts(history_replayer, "intervention"):
        kwargs["intervention"] = intervention
    return history_replayer(history_indices, **kwargs)


def _call_branch_scorer(
    branch_scorer: Any,
    action: Any,
    *,
    target_branch: str,
    target_predicate: str,
    task: str,
    progress_label: str,
    environment: Any,
) -> Any:
    kwargs = {"target_branch": target_branch, "task": task, "progress_label": progress_label}
    if _callable_accepts(branch_scorer, "target_predicate"):
        kwargs["target_predicate"] = target_predicate
    if _callable_accepts(branch_scorer, "environment"):
        kwargs["environment"] = environment
    return branch_scorer(action, **kwargs)


def _callable_accepts(callable_obj: Any, parameter: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    parameters = signature.parameters
    return parameter in parameters or any(value.kind == inspect.Parameter.VAR_KEYWORD for value in parameters.values())


def _policy_observation_payload(observation: Any, *, task: str) -> dict[str, Any]:
    if not isinstance(observation, Mapping):
        raise TypeError("policy observation must be a mapping")
    payload = copy.deepcopy(dict(observation))
    if {
        "agentview_image",
        "robot0_eye_in_hand_image",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
    } <= set(payload):
        main_module = _load_local_module("main.py", "libero_mem_main_for_history_eval")
        executed_prefix = payload.get("executed_actions", [])
        return main_module._libero_infer_payload(
            observation=payload, task=str(task), executed_prefix=list(executed_prefix)
        )
    payload.setdefault("prompt", str(task))
    payload.setdefault("executed_actions", [])
    return payload


def _policy_noise_array(policy: Any, noise: Any) -> Any | None:
    if not isinstance(noise, Mapping) or "seed" not in noise:
        return None
    try:
        import numpy as np
    except Exception:
        return None
    model = getattr(policy, "_model", None)
    config = getattr(model, "config", None)
    action_dim = int(getattr(model, "action_dim", getattr(config, "action_dim", 7)))
    action_horizon = int(getattr(model, "action_horizon", getattr(config, "action_horizon", 10)))
    rng = np.random.default_rng(int(noise["seed"]))
    return rng.standard_normal((action_horizon, action_dim), dtype=np.float32)


def _numeric_task_id(value: str) -> int | None:
    if value.isdigit():
        return int(value)
    match = re.search(r"(?:^|[/_-])(?:task)?(\d+)(?:$|[/_-])", value)
    return int(match.group(1)) if match else None


def _restore_physical_state(env: Any, physical_state: Any) -> Any:
    state = _extract_physical_state(physical_state)
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - runtime dependency.
        raise RuntimeError("Restoring LIBERO physical state requires numpy") from exc
    state_array = np.asarray(state)
    if state_array.ndim != 1:
        raise ValueError(
            "canonical_physical_state for real LIBERO restore must be a flat MuJoCo state, or a mapping with "
            "mujoco_state/sim_state/physics_state/state"
        )
    if hasattr(env, "set_init_state"):
        return env.set_init_state(state_array)
    if hasattr(env, "set_state"):
        env.set_state(state_array)
        return None
    raw_env = _raw_libero_env(env)
    if hasattr(raw_env, "sim") and hasattr(raw_env.sim, "set_state_from_flattened"):
        raw_env.sim.set_state_from_flattened(state_array)
        raw_env.sim.forward()
        return raw_env._get_observations() if hasattr(raw_env, "_get_observations") else None
    raise ValueError("LIBERO environment does not expose set_init_state, set_state, or sim.set_state_from_flattened")


def _extract_physical_state(physical_state: Any) -> Any:
    if isinstance(physical_state, Mapping):
        for key in ("mujoco_state", "sim_state", "physics_state", "state"):
            if key in physical_state:
                return physical_state[key]
    return physical_state


def _restore_libero_progress_state(env: Any, evaluator_progress_state: Any) -> None:
    raw_env = _raw_libero_env(env)
    if hasattr(raw_env, "restore_subgoal_progress"):
        raw_env.restore_subgoal_progress(copy.deepcopy(evaluator_progress_state))
        return
    if isinstance(evaluator_progress_state, Mapping):
        nested = evaluator_progress_state.get("evaluator_progress_state")
        if nested is not None:
            evaluator_progress_state = nested
    if isinstance(evaluator_progress_state, Mapping):
        satisfied = _first_present(
            evaluator_progress_state, "_satisfied_subgoals", "satisfied_subgoals", "raw_satisfied_subgoals"
        )
        if satisfied is None:
            raise ValueError(
                "evaluator_progress_state for real LIBERO restore must include _satisfied_subgoals/satisfied_subgoals"
            )
        raw_env._satisfied_subgoals = copy.deepcopy(satisfied)
        raw_env._sub_goal_live_time = int(
            _first_present(evaluator_progress_state, "_sub_goal_live_time", "sub_goal_live_time", default=0)
        )
        raw_env._sub_goal_nonlive_time = int(
            _first_present(evaluator_progress_state, "_sub_goal_nonlive_time", "sub_goal_nonlive_time", default=0)
        )
        if "_overshot" in evaluator_progress_state or "overshot" in evaluator_progress_state:
            raw_env._overshot = bool(_first_present(evaluator_progress_state, "_overshot", "overshot", default=False))
        return
    if isinstance(evaluator_progress_state, list):
        raw_env._satisfied_subgoals = copy.deepcopy(evaluator_progress_state)
        raw_env._sub_goal_live_time = 0
        raw_env._sub_goal_nonlive_time = 0
        return
    raise ValueError("unsupported evaluator_progress_state for real LIBERO restore")


def _raw_libero_env(env: Any) -> Any:
    return getattr(env, "env", env)


def _first_present(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _snapshot_satisfied(adapter: Any, observation: Any) -> list[Any]:
    if observation is None:
        return []
    try:
        return list(adapter.snapshot(observation, success=False).satisfied_subgoals)
    except Exception as exc:
        raise RuntimeError(f"snapshot_satisfied failed: {exc}") from exc


def _action_sequence(action: Any) -> list[Any]:
    if isinstance(action, Mapping):
        if "actions" in action:
            action = action["actions"]
        elif "action" in action:
            action = action["action"]
        else:
            return []
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - runtime dependency.
        raise RuntimeError("Scoring LIBERO actions requires numpy") from exc
    array = np.asarray(action, dtype=np.float32)
    if array.ndim == 1:
        return [array]
    if array.ndim == 2:
        return [array[index] for index in range(array.shape[0])]
    raise ValueError(f"action must be rank 1 or 2 for LIBERO branch scoring, got shape {array.shape}")


def _branch_label_from_subgoal(subgoal: Any, *, target_branch: str, target_predicate: str) -> str:
    text = str(subgoal)
    if _same_label(text, target_branch) or _same_label(text, target_predicate):
        return str(target_branch)
    return text


def _same_label(left: Any, right: Any) -> bool:
    return _normalize_label(left) == _normalize_label(right) or _normalize_label(right) in _normalize_label(left)


def _normalize_label(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _dataset_row_policy_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    if "observation" in row and isinstance(row["observation"], Mapping):
        payload = copy.deepcopy(dict(row["observation"]))
    else:
        payload = {}
        key_map = {
            "image": "observation/image",
            "wrist_image": "observation/wrist_image",
            "state": "observation/state",
            "observation.image": "observation/image",
            "observation.wrist_image": "observation/wrist_image",
            "observation.state": "observation/state",
        }
        for source_key, target_key in key_map.items():
            if source_key in row:
                payload[target_key] = row[source_key]
    prompt = row.get("task", row.get("prompt", row.get("language_instruction", "")))
    payload.setdefault("prompt", str(prompt))
    payload.setdefault("executed_actions", row.get("executed_actions", []))
    return payload


def _to_plain_data(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _to_plain_data(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return value.detach().cpu().numpy()
    if hasattr(value, "numpy"):
        try:
            return value.numpy()
        except TypeError:
            pass
    return value


def _manifest_embeds_policy_states(pairs: list[HistoryPair]) -> bool:
    return all(_manifest_policy_state_snapshots(pair.manifest[branch]) for pair in pairs for branch in ("a", "b"))


def _manifest_branch_for_candidate(pair: HistoryPair | None, candidate: Any) -> dict[str, Any] | None:
    if pair is None or candidate is None:
        return None
    for branch_name in ("a", "b"):
        if getattr(pair, branch_name) is candidate:
            return pair.manifest[branch_name]
    return None


def _manifest_policy_state_snapshots(manifest_branch: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("policy_state_snapshots", "history_state_snapshots"):
        value = manifest_branch.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _manifest_snapshot_state(entry: Any, *, history_indices: list[int], branch: str) -> Any:
    if isinstance(entry, Mapping):
        expected_history = entry.get("history_indices")
        if expected_history is not None and list(expected_history) != list(history_indices):
            raise ValueError("manifest policy-state snapshot history_indices do not match requested intervention")
        expected_branch = entry.get("branch")
        if expected_branch is not None and str(expected_branch) != str(branch):
            raise ValueError("manifest policy-state snapshot branch does not match requested intervention")
        for key in ("state", "policy_state", "history_state", "snapshot"):
            if key in entry:
                return copy.deepcopy(entry[key])
    return copy.deepcopy(entry)


def _deterministic_shuffle(values: list[int], *, seed: int) -> list[int]:
    decorated = [
        (stable_checksum({"seed": int(seed), "position": index, "value": value}), index, value)
        for index, value in enumerate(values)
    ]
    return [value for _checksum, _index, value in sorted(decorated)]


def action_prefix_distance(action: Any, target_branch: str) -> float:
    predicted = ""
    if isinstance(action, dict):
        predicted = str(action.get("branch", ""))
    else:
        predicted = str(action)
    target = str(target_branch)
    common = 0
    for left, right in zip(predicted, target, strict=False):
        if left != right:
            break
        common += 1
    return float(max(len(predicted), len(target)) - common)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate LIBERO-Mem history-pair interventions.")
    parser.add_argument("--pairs", "--manifest-jsonl", dest="pairs", type=Path, required=True)
    parser.add_argument("--results", "--output-jsonl", dest="results", type=Path, required=True)
    parser.add_argument("--policy-path", "--checkpoint-dir", dest="policy_path", type=Path)
    parser.add_argument("--policy-config", default="futuremamba_libero_mem")
    parser.add_argument("--task-suite-name", default="libero_mem")
    parser.add_argument("--history-repo-id")
    parser.add_argument("--history-dataset-root", type=Path)
    parser.add_argument("--branch-score-steps", type=int, default=10)
    parser.add_argument("--truncated-k", type=int, default=2)
    parser.add_argument("--shuffle-seed", type=int, default=0)
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    policy: Any | None = None,
    environment_factory: Any | None = None,
    history_replayer: Any | None = None,
    branch_scorer: Any | None = None,
) -> list[dict[str, Any]]:
    args = _parse_args(argv)
    pairs = build_history_pairs.load_history_pairs(args.pairs)
    resolved_policy = policy if policy is not None else load_policy(args.policy_path, policy_config=args.policy_config)
    resolved_environment_factory = (
        environment_factory
        if environment_factory is not None
        else _load_real_environment_factory(args=args, pairs=pairs)
    )
    resolved_history_replayer = (
        history_replayer
        if history_replayer is not None
        else _load_real_history_replayer(args=args, pairs=pairs, policy=resolved_policy)
    )
    resolved_branch_scorer = branch_scorer if branch_scorer is not None else _load_real_branch_scorer(args=args)
    rows = evaluate_history_pairs(
        pairs,
        policy=resolved_policy,
        environment_factory=resolved_environment_factory,
        history_replayer=resolved_history_replayer,
        branch_scorer=resolved_branch_scorer,
        truncated_k=args.truncated_k,
        shuffle_seed=args.shuffle_seed,
    )
    write_jsonl(args.results, rows)
    return rows


if __name__ == "__main__":
    main()
