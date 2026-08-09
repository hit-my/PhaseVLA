import importlib.util
import json
import pathlib
import subprocess
import sys

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import pytest

_CONVERTER_SPEC = importlib.util.spec_from_file_location(
    "convert_to_lerobot", pathlib.Path(__file__).with_name("convert_to_lerobot.py")
)
convert_to_lerobot = importlib.util.module_from_spec(_CONVERTER_SPEC)
sys.modules[_CONVERTER_SPEC.name] = convert_to_lerobot
_CONVERTER_SPEC.loader.exec_module(convert_to_lerobot)
from openpi import transforms as _transforms


class RecordingWriter:
    def __init__(self, root=None):
        self.root = root
        self.frames = []
        self.saved_episodes = 0

    def add_frame(self, frame, *, task):
        self.frames.append((dict(frame), task))

    def save_episode(self):
        self.saved_episodes += 1


def make_episode(
    episode_id,
    *,
    suite_id="libero_mem",
    task_id=4,
    instruction="put the mug on the tray",
    image_reasoning="offline visual notes",
):
    base = np.arange(8, dtype=np.float32) + float(episode_id)
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3) + episode_id
    wrist_image = np.arange(12, 24, dtype=np.uint8).reshape(2, 2, 3) + episode_id
    steps = []
    for step_index in range(3):
        steps.append(
            {
                "observation": {
                    "state": base + step_index,
                    "image": image + step_index,
                    "wrist_image": wrist_image + step_index,
                },
                "action": np.arange(7, dtype=np.float32) + step_index,
                "is_first": step_index == 0,
                "is_last": step_index == 2,
            }
        )
    return convert_to_lerobot.OfficialEpisode(
        suite_id=suite_id,
        task_id=task_id,
        episode_id=episode_id,
        task_family="spatial",
        memory_length=2,
        instruction=instruction,
        image_reasoning=image_reasoning,
        steps=steps,
    )


def test_write_episode_maps_frames_and_keeps_reasoning_out_of_prompt():
    writer = RecordingWriter()
    episode = make_episode(7)

    metadata = convert_to_lerobot.write_episode_to_lerobot(writer, episode)

    assert writer.saved_episodes == 1
    assert [task for _, task in writer.frames] == ["put the mug on the tray"] * 3
    assert "offline visual notes" not in [task for _, task in writer.frames]
    assert metadata == {
        "suite_id": "libero_mem",
        "task_id": 4,
        "episode_id": 7,
        "task_family": "spatial",
        "memory_length": 2,
        "instruction": "put the mug on the tray",
        "image_reasoning": "offline visual notes",
    }
    first_frame = writer.frames[0][0]
    assert first_frame["state"].shape == (8,)
    assert first_frame["state"].dtype == np.float32
    assert first_frame["actions"].shape == (7,)
    assert first_frame["actions"].dtype == np.float32
    np.testing.assert_array_equal(first_frame["image"], episode.steps[0]["observation"]["image"])
    np.testing.assert_array_equal(first_frame["wrist_image"], episode.steps[0]["observation"]["wrist_image"])
    assert [frame["is_first"].tolist() for frame, _ in writer.frames] == [[True], [False], [False]]
    assert [frame["is_last"].tolist() for frame, _ in writer.frames] == [[False], [False], [True]]


def test_split_episodes_sorts_by_numeric_demo_id_and_refuses_short_counts():
    episodes = [make_episode(10), make_episode(2), make_episode(1), make_episode(20)]

    train, val = convert_to_lerobot.split_episodes(episodes, train_count=2, val_count=1)

    assert [episode.episode_id for episode in train] == [1, 2]
    assert [episode.episode_id for episode in val] == [10]

    with pytest.raises(ValueError, match="requires 5 episodes.*found 4"):
        convert_to_lerobot.split_episodes(episodes, train_count=3, val_count=2)


def test_split_episodes_applies_counts_per_task_after_numeric_sort():
    episodes = [
        make_episode(20, task_id=2),
        make_episode(10, task_id=1),
        make_episode(2, task_id=2),
        make_episode(1, task_id=1),
    ]

    train, val = convert_to_lerobot.split_episodes(episodes, train_count=1, val_count=1)

    assert [(episode.task_id, episode.episode_id) for episode in train] == [(1, 1), (2, 2)]
    assert [(episode.task_id, episode.episode_id) for episode in val] == [(1, 10), (2, 20)]

    with pytest.raises(ValueError, match="suite=libero_mem task=1 requires 3 episodes.*found 2"):
        convert_to_lerobot.split_episodes(episodes, train_count=2, val_count=1)


def test_convert_suite_writes_split_manifest_and_metadata(tmp_path):
    train_writer = RecordingWriter(root=tmp_path / "train_repo")
    val_writer = RecordingWriter(root=tmp_path / "val_repo")
    episodes = [make_episode(3), make_episode(1), make_episode(2)]

    result = convert_to_lerobot.convert_suite(
        episodes,
        train_writer=train_writer,
        val_writer=val_writer,
        train_count=1,
        val_count=1,
        output_root=tmp_path,
        source_checksum="abc123",
        fork_commit="0eee0defa4024e51511b6e15a79f30e9620209de",
    )

    assert [frame["state"][0] for frame, _ in train_writer.frames] == [1.0, 2.0, 3.0]
    assert [frame["state"][0] for frame, _ in val_writer.frames] == [2.0, 3.0, 4.0]
    assert train_writer.saved_episodes == 1
    assert val_writer.saved_episodes == 1
    assert result.train_episode_ids == [1]
    assert result.val_episode_ids == [2]
    manifest = json.loads((tmp_path / "split_manifest.json").read_text())
    assert manifest["train_episode_ids"] == [1]
    assert manifest["val_episode_ids"] == [2]
    assert manifest["source_checksum"] == "abc123"
    assert manifest["fork_commit"] == "0eee0defa4024e51511b6e15a79f30e9620209de"
    assert json.loads((train_writer.root / "meta" / "split_manifest.json").read_text())["train_episode_ids"] == [1]
    assert json.loads((val_writer.root / "meta" / "split_manifest.json").read_text())["val_episode_ids"] == [2]
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["episodes"]["libero_mem/4/1"]["suite_id"] == "libero_mem"
    assert metadata["episodes"]["libero_mem/4/2"]["memory_length"] == 2


def test_lerobot_roundtrip_preserves_task_index_episode_boundaries_and_images(tmp_path):
    repo_id = "futuremamba/test_libero_mem"
    root = tmp_path / "repo"
    dataset = convert_to_lerobot.create_lerobot_dataset(repo_id=repo_id, root=root, image_shape=(2, 2, 3))
    episode = make_episode(5)

    convert_to_lerobot.write_episode_to_lerobot(dataset, episode)
    reloaded = LeRobotDataset(repo_id=repo_id, root=root)

    assert len(reloaded) == 3
    assert reloaded.meta.tasks == {0: "put the mug on the tray"}
    prompt = _transforms.PromptFromLeRobotTask(reloaded.meta.tasks)({"task_index": reloaded[0]["task_index"]})["prompt"]
    assert prompt == "put the mug on the tray"
    assert "offline visual notes" not in prompt
    assert [int(reloaded[i]["episode_index"]) for i in range(3)] == [0, 0, 0]
    assert [int(reloaded[i]["frame_index"]) for i in range(3)] == [0, 1, 2]
    assert [bool(reloaded[i]["is_first"]) for i in range(3)] == [True, False, False]
    assert [bool(reloaded[i]["is_last"]) for i in range(3)] == [False, False, True]
    np.testing.assert_array_equal(
        (np.asarray(reloaded[0]["image"]).transpose(1, 2, 0) * 255).astype(np.uint8),
        episode.steps[0]["observation"]["image"],
    )
    np.testing.assert_array_equal(
        (np.asarray(reloaded[0]["wrist_image"]).transpose(1, 2, 0) * 255).astype(np.uint8),
        episode.steps[0]["observation"]["wrist_image"],
    )
    assert np.asarray(reloaded[0]["state"]).shape == (8,)
    assert np.asarray(reloaded[0]["actions"]).shape == (7,)


def test_cli_help_exposes_top_level_flags_without_args_prefix():
    result = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).with_name("convert_to_lerobot.py")), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--libero-mem-data-dir" in result.stdout
    assert "--libero-long-data-dir" in result.stdout
    assert "--max-episodes-per-suite" in result.stdout
    assert "--args." not in result.stdout


def test_cli_parses_top_level_flags_without_fetching_data(monkeypatch, tmp_path):
    captured_args = []

    def fake_main(args):
        captured_args.append(args)

    monkeypatch.setattr(convert_to_lerobot, "main", fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "convert_to_lerobot.py",
            "--libero-mem-data-dir",
            str(tmp_path / "mem"),
            "--libero-long-data-dir",
            str(tmp_path / "long"),
            "--max-episodes-per-suite",
            "3",
        ],
    )

    convert_to_lerobot.cli()

    assert captured_args == [
        convert_to_lerobot.Args(
            libero_mem_data_dir=tmp_path / "mem",
            libero_long_data_dir=tmp_path / "long",
            max_episodes_per_suite=3,
        )
    ]
