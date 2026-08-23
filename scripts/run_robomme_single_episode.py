#!/usr/bin/env python3
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any
from collections.abc import Mapping


def _official_runtime() -> SimpleNamespace:
    examples_dir = (
        Path(__file__).resolve().parents[1]
        / "third_party"
        / "robomme_policy_learning"
        / "examples"
        / "robomme"
    )
    sys.path.insert(0, str(examples_dir))
    import eval as official_eval

    return SimpleNamespace(
        TASK_NAME_LIST=official_eval.TASK_NAME_LIST,
        EnvRunner=official_eval.EnvRunner,
        EpisodeEvaluator=official_eval.EpisodeEvaluator,
        build_subgoal_predictor=official_eval.build_subgoal_predictor,
        setup_save_directory=official_eval.setup_save_directory,
        check_args=official_eval.check_args,
        Args=official_eval.Args,
    )


_DATASET_BY_SPLIT = {"train": "train", "validation": "val", "val": "val", "test": "test"}


def _canonical_split(split: str | None) -> str:
    if split is None or split == "":
        return "test"
    normalized = str(split).lower()
    if normalized == "val":
        return "validation"
    if normalized not in {"train", "validation", "test"}:
        raise ValueError(f"split must be one of train, validation, or test; got {split!r}")
    return normalized


def _official_dataset(split: str | None, dataset: str | None = None) -> str:
    canonical_split = _canonical_split(split)
    expected_dataset = _DATASET_BY_SPLIT[canonical_split]
    if dataset is None or dataset == "":
        return expected_dataset
    normalized_dataset = str(dataset).lower()
    normalized_dataset = _DATASET_BY_SPLIT.get(normalized_dataset, normalized_dataset)
    if normalized_dataset not in {"train", "val", "test"}:
        raise ValueError(f"dataset must be one of train, val, or test; got {dataset!r}")
    if normalized_dataset != expected_dataset:
        raise ValueError(
            f"dataset {dataset!r} does not match split {canonical_split!r}; expected {expected_dataset!r}"
        )
    return normalized_dataset


_RESULT_IDENTITY_FIELDS = (
    "experiment_id",
    "method_id",
    "train_seed",
    "task_name",
    "episode_id",
    "split",
    "checkpoint",
    "config",
    "provenance",
)


def _validate_result_identity(
    value: Mapping[str, Any] | None,
    *,
    task_name: str,
    episode_id: int,
    split: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    identity = dict(value)
    missing = [field for field in _RESULT_IDENTITY_FIELDS if field not in identity]
    if missing:
        raise ValueError(f"result identity missing required field {missing[0]!r}")
    if identity["task_name"] != task_name:
        raise ValueError("result identity task_name does not match selected task")
    if identity["episode_id"] != episode_id:
        raise ValueError("result identity episode_id does not match selected episode")
    if identity["split"] != split:
        raise ValueError("result identity split does not match selected split")
    if isinstance(identity["train_seed"], bool) or not isinstance(identity["train_seed"], int):
        raise ValueError("result identity train_seed must be an integer")
    for field in ("checkpoint", "config", "provenance"):
        if not isinstance(identity[field], Mapping) or not identity[field]:
            raise ValueError(f"result identity {field} must be a non-empty object")
    return identity


def _write_results(save_dir: Path, task_name: str, episode_id: int, outcome: str) -> None:
    success = outcome == "success"
    save_dir.mkdir(parents=True, exist_ok=True)
    progress_path = save_dir / "progress.json"
    progress = json.loads(progress_path.read_text(encoding="utf-8")) if progress_path.exists() else {}
    task_progress = progress.setdefault(task_name, {})
    task_progress[str(episode_id)] = success if outcome != "unknown" else "error"

    success_rate = {}
    for existing_task, episodes in progress.items():
        completed = [value for value in episodes.values() if isinstance(value, bool)]
        if completed:
            success_rate[existing_task] = sum(completed) / len(completed)
    final = {
        "success_rate": success_rate,
        "total_success_rate": sum(success_rate.values()) / len(success_rate) if success_rate else 0.0,
    }
    progress_path.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
    (save_dir / "log.json").write_text(json.dumps(final, indent=2) + "\n", encoding="utf-8")


def run_one_episode(
    args: Any,
    *,
    task_name: str,
    episode_id: int,
    runtime: Any | None = None,
    result_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = _official_runtime() if runtime is None else runtime
    if hasattr(runtime, "check_args"):
        runtime.check_args(args)
    if task_name not in runtime.TASK_NAME_LIST:
        raise ValueError(f"unknown RoboMME task: {task_name}")
    if episode_id < 0:
        raise ValueError("episode_id must be non-negative")

    split = _canonical_split(getattr(args, "split", "test"))
    dataset = _official_dataset(split, getattr(args, "dataset", None))
    identity = _validate_result_identity(
        result_identity,
        task_name=task_name,
        episode_id=episode_id,
        split=split,
    )
    save_dir = Path(runtime.setup_save_directory(args))
    video_dir = save_dir / "videos"
    runner = runtime.EnvRunner(task_name, video_dir, max_steps=args.max_steps, split=split)
    try:
        if episode_id >= runner.num_episodes:
            raise ValueError(
                f"episode_id {episode_id} outside available range [0, {runner.num_episodes})"
            )
        predictor = runtime.build_subgoal_predictor(args, save_dir)
        evaluator = runtime.EpisodeEvaluator(args, save_dir)
        runner.make_env(episode_id)
        outcome = evaluator.eval_each_episode(runner, predictor, video_dir)
        _write_results(save_dir, task_name, episode_id, outcome)
        return {
            **identity,
            "task_name": task_name,
            "episode_id": episode_id,
            "split": split,
            "dataset": dataset,
            "outcome": outcome,
            "success": outcome == "success",
        }
    finally:
        runner.close_env()


@dataclasses.dataclass
class LauncherArgs:
    task_name: str = "PickXtimes"
    episode_id: int = 0
    host: str = "0.0.0.0"
    port: int = 8001
    obs_horizon: int = 16
    max_steps: int = 1300
    memory_update_stride: int | None = None
    save_dir: str = "runs/evaluation"
    overwrite: bool = False
    use_history: bool = False
    policy_name: str = "pi05_baseline"
    model_seed: int = 7
    model_ckpt_id: int = 79999
    use_oracle: bool = False
    use_qwenvl: bool = False
    use_memer: bool = False
    use_gemini: bool = False
    subgoal_type: str | None = None
    gemini_model_name: str = "gemini-2.5-pro"
    qwenvl_simpleSG_adapter_path: str = (
        "runs/ckpts/vlm_subgoal_predictor/qwenvl/simple_subgoal/checkpoint-1400"
    )
    qwenvl_groundSG_adapter_path: str = (
        "runs/ckpts/vlm_subgoal_predictor/qwenvl/grounded_subgoal/checkpoint-1200"
    )
    memer_adapter_path: str = (
        "runs/ckpts/vlm_subgoal_predictor/memer/grounded_subgoal/checkpoint-1300"
    )
    subgoal_keep_period: int = 1
    split: str = "test"
    dataset: str | None = None
    result_identity_json: str | None = None


def main(args: LauncherArgs) -> None:
    runtime = _official_runtime()
    official_kwargs = dict(
        host=args.host,
        port=args.port,
        obs_horizon=args.obs_horizon,
        max_steps=args.max_steps,
        memory_update_stride=args.memory_update_stride,
        save_dir=args.save_dir,
        overwrite=args.overwrite,
        use_history=args.use_history,
        policy_name=args.policy_name,
        model_seed=args.model_seed,
        model_ckpt_id=args.model_ckpt_id,
        only_tasks=args.task_name,
        use_oracle=args.use_oracle,
        use_qwenvl=args.use_qwenvl,
        use_memer=args.use_memer,
        use_gemini=args.use_gemini,
        subgoal_type=args.subgoal_type,
        gemini_model_name=args.gemini_model_name,
        qwenvl_simpleSG_adapter_path=args.qwenvl_simpleSG_adapter_path,
        qwenvl_groundSG_adapter_path=args.qwenvl_groundSG_adapter_path,
        memer_adapter_path=args.memer_adapter_path,
        subgoal_keep_period=args.subgoal_keep_period,
    )
    try:
        official_args = runtime.Args(**official_kwargs, split=_canonical_split(args.split), dataset=_official_dataset(args.split, args.dataset))
    except TypeError:
        official_args = runtime.Args(**official_kwargs)
        official_args.split = _canonical_split(args.split)
        official_args.dataset = _official_dataset(args.split, args.dataset)
    result_identity = None
    if args.result_identity_json:
        try:
            parsed_identity = json.loads(args.result_identity_json)
        except json.JSONDecodeError as error:
            raise ValueError(f"result_identity_json must be valid JSON: {error}") from error
        if not isinstance(parsed_identity, Mapping):
            raise ValueError("result_identity_json must contain a JSON object")
        result_identity = parsed_identity
    result = run_one_episode(
        official_args,
        task_name=args.task_name,
        episode_id=args.episode_id,
        result_identity=result_identity,
        runtime=runtime,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    import tyro

    main(tyro.cli(LauncherArgs))
