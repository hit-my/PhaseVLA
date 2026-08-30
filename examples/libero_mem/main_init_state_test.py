from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from examples.libero_mem import main


@dataclasses.dataclass
class Snapshot:
    observation: dict[str, np.ndarray]
    success: bool = False
    satisfied_subgoals: list[Any] = dataclasses.field(default_factory=list)
    overshot: bool = False
    atomic_predicates: dict[str, bool] = dataclasses.field(default_factory=dict)


class BaseAdapter:
    def __init__(self):
        self.reset_calls = 0
        self.snapshot_observations = []

    def reset(self):
        self.reset_calls += 1
        return Snapshot({"value": np.asarray([0])})

    def snapshot(self, observation, *, success=False):
        self.snapshot_observations.append(observation)
        return Snapshot(observation, success=success)

    def step(self, action):
        return Snapshot({"value": np.asarray([99])})


class EnvWithoutInit:
    def step(self, action):
        return {"value": np.asarray([1])}, 0.0, False, {}


class EnvWithInit:
    def __init__(self):
        self.init_calls = []
        self.steps = 0

    def set_init_state(self, init_state):
        self.init_calls.append(init_state)
        return {"value": np.asarray(init_state)}

    def step(self, action):
        self.steps += 1
        return {"value": np.asarray([self.steps])}, 0.0, False, {}


def test_episode_setup_refuses_environment_without_init_state():
    adapter = main.EpisodeSetupAdapter(
        BaseAdapter(), env=EnvWithoutInit(), init_state=np.asarray([1, 2]), num_steps_wait=10
    )
    with pytest.raises(TypeError, match="set_init_state"):
        adapter.reset()


def test_episode_setup_applies_init_state_before_waiting():
    base = BaseAdapter()
    env = EnvWithInit()
    init_state = np.asarray([4, 5, 6])
    adapter = main.EpisodeSetupAdapter(base, env=env, init_state=init_state, num_steps_wait=3)

    snapshot = adapter.reset()

    assert base.reset_calls == 1
    assert len(env.init_calls) == 1
    np.testing.assert_array_equal(env.init_calls[0], init_state)
    assert env.steps == 3
    np.testing.assert_array_equal(snapshot.observation["value"], np.asarray([3]))


class FakeModel:
    nbody = 0
    njnt = 2

    def joint_id2name(self, index):
        return (main.LIBERO_MEM_BOTTLE_JOINT, "plate_2_joint0")[index]


class FakeData:
    def __init__(self):
        self.qpos = {
            main.LIBERO_MEM_BOTTLE_JOINT: np.asarray([0.1, 0.2, 1.08, 0.0, 0.0, 0.0, 1.0]),
            "plate_2_joint0": np.asarray([0.1, 0.2, 0.97, 1.0, 0.0, 0.0, 0.0]),
        }
        self.qvel = {}

    def get_joint_qpos(self, name):
        return self.qpos[name]

    def set_joint_qpos(self, name, value):
        self.qpos[name] = np.asarray(value).copy()

    def set_joint_qvel(self, name, value):
        self.qvel[name] = np.asarray(value).copy()


class FakeSim:
    def __init__(self):
        self.model = FakeModel()
        self.data = FakeData()
        self.forward_calls = 0

    def forward(self):
        self.forward_calls += 1


class FallingBottleEnv(EnvWithInit):
    def __init__(self):
        super().__init__()
        self.sim = FakeSim()

    def step(self, action):
        self.steps += 1
        if self.steps == 1:
            self.sim.data.qpos[main.LIBERO_MEM_BOTTLE_JOINT][3:7] = np.asarray(
                [2**-0.5, 2**-0.5, 0.0, 0.0]
            )
        return {"value": np.asarray([self.steps])}, 0.0, False, {}


def test_episode_setup_reseats_fallen_bottle_and_waits_again():
    base = BaseAdapter()
    env = FallingBottleEnv()
    adapter = main.EpisodeSetupAdapter(
        base,
        env=env,
        init_state=np.asarray([4, 5, 6]),
        num_steps_wait=2,
        stabilize_libero_mem_objects=True,
    )

    snapshot = adapter.reset()

    bottle = env.sim.data.qpos[main.LIBERO_MEM_BOTTLE_JOINT]
    plate = env.sim.data.qpos["plate_2_joint0"]
    assert env.steps == 4
    assert env.sim.forward_calls == 1
    assert main._object_tilt_degrees(bottle[3:7]) == pytest.approx(0.0)
    np.testing.assert_allclose(bottle[:2], plate[:2])
    assert bottle[2] == pytest.approx(plate[2] + main.LIBERO_MEM_BOTTLE_PLACEMENT_HEIGHT)
    np.testing.assert_array_equal(env.sim.data.qvel[main.LIBERO_MEM_BOTTLE_JOINT], np.zeros(6))
    np.testing.assert_array_equal(snapshot.observation["value"], np.asarray([4]))


def test_stabilizer_leaves_upright_bottle_untouched():
    env = FallingBottleEnv()
    plan = main._capture_libero_mem_stabilization_plan(env)

    changed = main._stabilize_libero_mem_object(env, plan)

    assert changed is False
    assert env.sim.forward_calls == 0

def test_infer_payload_matches_training_image_orientation():
    height = width = main.LIBERO_INFERENCE_IMAGE_SIZE
    row = np.arange(height, dtype=np.uint8)[:, None, None]
    column = np.arange(width, dtype=np.uint8)[None, :, None]
    base_image = np.concatenate(
        (
            np.broadcast_to(row, (height, width, 1)),
            np.broadcast_to(column, (height, width, 1)),
            np.broadcast_to(row ^ column, (height, width, 1)),
        ),
        axis=-1,
    )
    wrist_image = np.ascontiguousarray(255 - base_image)
    observation = {
        "agentview_image": base_image,
        "robot0_eye_in_hand_image": wrist_image,
        "robot0_eef_pos": np.asarray([1.0, 2.0, 3.0]),
        "robot0_eef_quat": np.asarray([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.asarray([0.1, -0.1]),
    }

    payload = main._libero_infer_payload(
        observation=observation,
        task="task",
        executed_prefix=[[1.0] * 7],
    )

    np.testing.assert_array_equal(payload["observation/image"], base_image[::-1, :])
    np.testing.assert_array_equal(payload["observation/wrist_image"], wrist_image[::-1, :])
    assert not np.array_equal(payload["observation/image"], base_image[::-1, ::-1])
    np.testing.assert_allclose(
        payload["observation/state"],
        np.asarray([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.1, -0.1]),
    )
    assert payload["executed_actions"].shape == (1, 7)

class Suite:
    n_tasks = 1

    def get_task(self, index):
        assert index == 0
        return dataclasses.make_dataclass(
            "Task", [("language", str), ("task_family", str), ("memory_length", int), ("goals", dict), ("problem_folder", str), ("bddl_file", str)]
        )("task", "libero_mem", 0, {"Sequence": []}, "libero_mem", "task.bddl")

    def get_task_init_states(self, index):
        assert index == 0
        return [np.asarray([7, 8])]


class Benchmark:
    def get_benchmark_dict(self):
        return {"libero_mem": Suite}


class Client:
    def __init__(self, host, port):
        self.host = host
        self.port = port


class ClientModule:
    WebsocketClientPolicy = Client


class BadEnv:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def seed(self, value):
        self.seed_value = value


def test_eval_rejects_factory_that_returns_raw_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(
        main,
        "_load_real_dependencies",
        lambda: {
            "env_adapter": object(),
            "websocket_client_policy": ClientModule,
            "benchmark": Benchmark(),
            "get_libero_path": lambda key: tmp_path,
            "OffScreenRenderEnv": BadEnv,
        },
    )
    (tmp_path / "libero_mem").mkdir()
    (tmp_path / "libero_mem/task.bddl").write_text("fixture")
    args = main.Args(task_suite_name="libero_mem", task_ids=(0,), num_trials_per_task=1)

    with pytest.raises(TypeError, match="OffScreenRenderEnv must expose set_init_state"):
        main.eval_libero_mem(args, episode_runner=lambda **kwargs: kwargs)
