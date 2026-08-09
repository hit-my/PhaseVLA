from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Iterable
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import tyro

DEFAULT_FORK_COMMIT = "0eee0defa4024e51511b6e15a79f30e9620209de"
DEFAULT_TRAIN_REPO_ID = "futuremamba/libero_mem_long_train"
DEFAULT_VAL_REPO_ID = "futuremamba/libero_mem_long_val"


@dataclasses.dataclass(frozen=True)
class OfficialEpisode:
    suite_id: str
    task_id: int
    episode_id: int
    task_family: str
    memory_length: int
    instruction: str
    image_reasoning: str | None
    steps: Iterable[dict[str, Any]]


@dataclasses.dataclass(frozen=True)
class ConversionResult:
    train_episode_ids: list[int]
    val_episode_ids: list[int]
    metadata: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Args:
    libero_mem_data_dir: Path | None = None
    libero_long_data_dir: Path | None = None
    output_root: Path | None = None
    train_repo_id: str = DEFAULT_TRAIN_REPO_ID
    val_repo_id: str = DEFAULT_VAL_REPO_ID
    max_episodes_per_suite: int | None = None
    fps: int = 10
    libero_mem_train_count: int = 100
    libero_mem_val_count: int = 20
    libero_long_train_count: int = 40
    libero_long_val_count: int = 10


def create_lerobot_dataset(*, repo_id: str, root: str | Path | None = None, image_shape: tuple[int, int, int] = (256, 256, 3), fps: int = 10):
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    dataset_root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
    if dataset_root.exists():
        import shutil

        shutil.rmtree(dataset_root)
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        robot_type="panda",
        fps=fps,
        features=lerobot_features(image_shape=image_shape),
        use_videos=False,
    )


def lerobot_features(*, image_shape: tuple[int, int, int] = (256, 256, 3)) -> dict[str, dict[str, Any]]:
    return {
        "image": {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channel"]},
        "wrist_image": {"dtype": "image", "shape": image_shape, "names": ["height", "width", "channel"]},
        "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
        "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        "is_first": {"dtype": "bool", "shape": (1,), "names": None},
        "is_last": {"dtype": "bool", "shape": (1,), "names": None},
    }


def write_episode_to_lerobot(writer: Any, episode: OfficialEpisode) -> dict[str, Any]:
    saw_step = False
    for step_index, step in enumerate(episode.steps):
        saw_step = True
        frame = _frame_from_step(step, step_index=step_index)
        _add_frame(writer, frame, episode.instruction)
    if not saw_step:
        raise ValueError(f"episode {episode.episode_id} contains no steps")
    writer.save_episode()
    return episode_metadata(episode)


def split_episodes(
    episodes: Iterable[OfficialEpisode],
    *,
    train_count: int,
    val_count: int,
    per_suite_counts: dict[str, tuple[int, int]] | None = None,
    group_by_task: bool = True,
) -> tuple[list[OfficialEpisode], list[OfficialEpisode]]:
    episodes_by_group: dict[tuple[str, int], list[OfficialEpisode]] = {}
    for episode in episodes:
        group = (episode.suite_id, int(episode.task_id)) if group_by_task else ("__all__", 0)
        episodes_by_group.setdefault(group, []).append(episode)

    train: list[OfficialEpisode] = []
    val: list[OfficialEpisode] = []
    for (suite_id, task_id), task_episodes in sorted(episodes_by_group.items()):
        sorted_group_episodes = sorted(task_episodes, key=lambda episode: int(episode.episode_id))
        counts_suite_id = sorted_group_episodes[0].suite_id
        group_train_count, group_val_count = (
            per_suite_counts.get(counts_suite_id, (train_count, val_count))
            if per_suite_counts
            else (train_count, val_count)
        )
        required = group_train_count + group_val_count
        if len(sorted_group_episodes) < required:
            prefix = f"suite={suite_id} task={task_id}" if group_by_task else "all episodes"
            raise ValueError(
                f"{prefix} requires {required} episodes "
                f"(train={group_train_count}, val={group_val_count}), found {len(sorted_group_episodes)}"
            )
        train.extend(sorted_group_episodes[:group_train_count])
        val.extend(sorted_group_episodes[group_train_count:required])

    return train, val


def convert_suite(
    episodes: Iterable[OfficialEpisode],
    *,
    train_writer: Any,
    val_writer: Any,
    train_count: int,
    val_count: int,
    output_root: str | Path,
    source_checksum: str,
    fork_commit: str = DEFAULT_FORK_COMMIT,
    per_suite_counts: dict[str, tuple[int, int]] | None = None,
    group_by_task: bool = True,
) -> ConversionResult:
    train_episodes, val_episodes = split_episodes(
        episodes,
        train_count=train_count,
        val_count=val_count,
        per_suite_counts=per_suite_counts,
        group_by_task=group_by_task,
    )
    episode_metadata_by_key: dict[str, dict[str, Any]] = {}
    for episode in train_episodes:
        episode_metadata_by_key[_episode_key(episode)] = write_episode_to_lerobot(train_writer, episode)
    for episode in val_episodes:
        episode_metadata_by_key[_episode_key(episode)] = write_episode_to_lerobot(val_writer, episode)

    train_episode_ids = [episode.episode_id for episode in train_episodes]
    val_episode_ids = [episode.episode_id for episode in val_episodes]
    metadata = {
        "episodes": episode_metadata_by_key,
        "tasks": _task_mapping([*train_episodes, *val_episodes]),
    }
    output_path = Path(output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "train_episode_ids": train_episode_ids,
        "val_episode_ids": val_episode_ids,
        "train_episodes": [_episode_identity(episode) for episode in train_episodes],
        "val_episodes": [_episode_identity(episode) for episode in val_episodes],
        "source_checksum": source_checksum,
        "fork_commit": fork_commit,
    }
    _write_json(output_path / "split_manifest.json", manifest)
    _write_json(output_path / "metadata.json", metadata)
    _write_repo_sidecar(train_writer, "split_manifest.json", manifest)
    _write_repo_sidecar(val_writer, "split_manifest.json", manifest)
    _write_repo_sidecar(train_writer, "metadata.json", metadata)
    _write_repo_sidecar(val_writer, "metadata.json", metadata)
    return ConversionResult(train_episode_ids=train_episode_ids, val_episode_ids=val_episode_ids, metadata=metadata)



def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_repo_sidecar(writer: Any, filename: str, payload: dict[str, Any]) -> None:
    root = getattr(writer, "root", None)
    if root is None:
        return
    _write_json(Path(root) / "meta" / filename, payload)

def iter_rlds_episodes(
    data_dir: str | Path,
    dataset_name: str,
    *,
    suite_id: str,
    task_family: str,
    max_episodes: int | None = None,
) -> Iterator[OfficialEpisode]:
    import tensorflow_datasets as tfds

    dataset = tfds.load(dataset_name, data_dir=str(data_dir), split="train", download=False)
    for index, raw_episode in enumerate(dataset):
        if max_episodes is not None and index >= max_episodes:
            break
        episode = _episode_from_rlds(raw_episode, default_suite_id=suite_id, default_family=task_family, fallback_id=index)
        yield episode


def default_episode_iterator(
    *, libero_mem_data_dir: Path | None, libero_long_data_dir: Path | None, max_episodes_per_suite: int | None = None
) -> Iterator[OfficialEpisode]:
    if libero_mem_data_dir is not None:
        yield from iter_rlds_episodes(
            libero_mem_data_dir,
            "libero_mem",
            suite_id="libero_mem",
            task_family="libero_mem",
            max_episodes=max_episodes_per_suite,
        )
    if libero_long_data_dir is not None:
        yield from iter_rlds_episodes(
            libero_long_data_dir,
            "libero_10",
            suite_id="libero_10",
            task_family="libero_long",
            max_episodes=max_episodes_per_suite,
        )


def main(args: Args) -> None:
    episodes = list(
        default_episode_iterator(
            libero_mem_data_dir=args.libero_mem_data_dir,
            libero_long_data_dir=args.libero_long_data_dir,
            max_episodes_per_suite=args.max_episodes_per_suite,
        )
    )
    if not episodes:
        raise ValueError("No local LIBERO episodes found; pass --libero-mem-data-dir or --libero-long-data-dir")
    output_root = args.output_root or Path.cwd() / "data" / "libero_mem_lerobot"
    train_writer = create_lerobot_dataset(repo_id=args.train_repo_id, root=output_root / args.train_repo_id, fps=args.fps)
    val_writer = create_lerobot_dataset(repo_id=args.val_repo_id, root=output_root / args.val_repo_id, fps=args.fps)
    smoke = args.max_episodes_per_suite is not None
    train_count = 1 if smoke else args.libero_mem_train_count
    val_count = 1 if smoke else args.libero_mem_val_count
    per_suite_counts = None
    if not smoke:
        per_suite_counts = {
            "libero_mem": (args.libero_mem_train_count, args.libero_mem_val_count),
            "libero_10": (args.libero_long_train_count, args.libero_long_val_count),
            "libero_long": (args.libero_long_train_count, args.libero_long_val_count),
        }
    source_checksum = checksum_paths(path for path in [args.libero_mem_data_dir, args.libero_long_data_dir] if path is not None)
    convert_suite(
        episodes,
        train_writer=train_writer,
        val_writer=val_writer,
        train_count=train_count,
        val_count=val_count,
        output_root=output_root,
        source_checksum=source_checksum,
        fork_commit=DEFAULT_FORK_COMMIT,
        per_suite_counts=per_suite_counts,
        group_by_task=not smoke,
    )


def cli() -> None:
    main(tyro.cli(Args))


def checksum_paths(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((Path(p) for p in paths), key=lambda p: str(p)):
        digest.update(str(path).encode())
        if path.is_file():
            digest.update(path.read_bytes())
        elif path.is_dir():
            for child in sorted((p for p in path.rglob("*") if p.is_file()), key=lambda p: str(p.relative_to(path))):
                digest.update(str(child.relative_to(path)).encode())
                digest.update(child.read_bytes())
        else:
            raise FileNotFoundError(path)
    return digest.hexdigest()


def episode_metadata(episode: OfficialEpisode) -> dict[str, Any]:
    return {
        "suite_id": episode.suite_id,
        "task_id": int(episode.task_id),
        "episode_id": int(episode.episode_id),
        "task_family": episode.task_family,
        "memory_length": int(episode.memory_length),
        "instruction": episode.instruction,
        "image_reasoning": episode.image_reasoning,
    }


def _episode_key(episode: OfficialEpisode) -> str:
    return f"{episode.suite_id}/{int(episode.task_id)}/{int(episode.episode_id)}"


def _episode_identity(episode: OfficialEpisode) -> dict[str, Any]:
    return {
        "suite_id": episode.suite_id,
        "task_id": int(episode.task_id),
        "episode_id": int(episode.episode_id),
    }


def _frame_from_step(step: dict[str, Any], *, step_index: int) -> dict[str, Any]:
    observation = step["observation"]
    return {
        "image": _as_image(observation["image"]),
        "wrist_image": _as_image(observation["wrist_image"]),
        "state": _as_vector(observation["state"], length=8, name="state"),
        "actions": _as_vector(step["action"], length=7, name="action"),
        "is_first": np.asarray([step.get("is_first", step_index == 0)], dtype=np.bool_),
        "is_last": np.asarray([step.get("is_last", False)], dtype=np.bool_),
    }


def _add_frame(writer: Any, frame: dict[str, Any], instruction: str) -> None:
    try:
        writer.add_frame(frame, task=instruction)
    except TypeError as exc:
        if "task" not in str(exc):
            raise
        writer.add_frame({**frame, "task": instruction})


def _as_vector(value: Any, *, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    return array


def _as_image(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"image must be HxWx3 uint8, got {array.shape}")
    return array


def _task_mapping(episodes: Iterable[OfficialEpisode]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for episode in episodes:
        mapping[f"{episode.suite_id}/{int(episode.task_id)}"] = episode.instruction
    return mapping


def _episode_from_rlds(raw_episode: Any, *, default_suite_id: str, default_family: str, fallback_id: int) -> OfficialEpisode:
    raw = _to_numpy(raw_episode)
    metadata = raw.get("metadata", raw)
    steps = list(_steps_from_raw(raw))
    instruction = _decode_first_present(raw, metadata, "language_instruction", "instruction", default="")
    if not instruction and steps:
        instruction = _decode(steps[0].get("language_instruction", b""))
    episode_id = int(_first_present(metadata, raw, "episode_id", "demo_id", default=fallback_id))
    return OfficialEpisode(
        suite_id=_decode_first_present(metadata, raw, "suite_id", "suite", default=default_suite_id),
        task_id=int(_first_present(metadata, raw, "task_id", default=0)),
        episode_id=episode_id,
        task_family=_decode_first_present(metadata, raw, "task_family", "family", default=default_family),
        memory_length=int(_first_present(metadata, raw, "memory_length", default=0)),
        instruction=instruction,
        image_reasoning=_optional_decode(_first_present(metadata, raw, "image_reasoning", default=None)),
        steps=steps,
    )


def _steps_from_raw(raw: dict[str, Any]) -> Iterable[dict[str, Any]]:
    steps = raw["steps"]
    if hasattr(steps, "as_numpy_iterator"):
        return steps.as_numpy_iterator()
    return steps


def _to_numpy(value: Any) -> Any:
    if hasattr(value, "numpy"):
        return value.numpy()
    if isinstance(value, dict):
        return {key: _to_numpy(item) for key, item in value.items()}
    return value


def _decode_first_present(primary: dict[str, Any], secondary: dict[str, Any], *keys: str, default: str) -> str:
    return _decode(_first_present(primary, secondary, *keys, default=default))


def _first_present(primary: dict[str, Any], secondary: dict[str, Any], *keys: str, default: Any) -> Any:
    for source in (primary, secondary):
        for key in keys:
            if key in source:
                return source[key]
    return default


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value.item())
    return str(value)


def _optional_decode(value: Any) -> str | None:
    if value is None:
        return None
    return _decode(value)


def _default_train_count(episodes: list[OfficialEpisode]) -> int:
    suite_ids = {episode.suite_id for episode in episodes}
    if suite_ids == {"libero_10"}:
        return 40
    return 100


def _default_val_count(episodes: list[OfficialEpisode]) -> int:
    suite_ids = {episode.suite_id for episode in episodes}
    if suite_ids == {"libero_10"}:
        return 10
    return 20


if __name__ == "__main__":
    cli()
