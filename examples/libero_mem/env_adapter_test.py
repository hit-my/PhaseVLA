import importlib.util
import pathlib
import sys

import numpy as np

_ADAPTER_SPEC = importlib.util.spec_from_file_location("env_adapter", pathlib.Path(__file__).with_name("env_adapter.py"))
env_adapter = importlib.util.module_from_spec(_ADAPTER_SPEC)
sys.modules[_ADAPTER_SPEC.name] = env_adapter
_ADAPTER_SPEC.loader.exec_module(env_adapter)


class FakeLiberoMemEnv:
    def __init__(self):
        self.reset_calls = 0
        self.reset_subgoal_progress_calls = 0
        self.step_calls = 0
        self.check_success_calls = []
        self.satisfied_queries = []
        self._overshot = True
        self._atomic_states = {"on(cube, plate)": False, "in(cube, bowl)": True}

    def reset(self):
        self.reset_calls += 1
        self._overshot = True
        return {"reset": self.reset_calls}

    def reset_subgoal_progress(self):
        self.reset_subgoal_progress_calls += 1

    def step(self, action):
        self.step_calls += 1
        self._atomic_states["on(cube, plate)"] = True
        return {"step": self.step_calls}, 0.25, False, {"internal_success_probe": False}

    def _check_success(self, *, inc=False):
        self.check_success_calls.append(inc)
        return self.step_calls >= 2

    def get_satisfied_subgoals(self, task_text):
        self.satisfied_queries.append(task_text)
        return [f"{task_text}:subgoal-{self.step_calls}"]

    def get_atomic_predicate_states(self):
        return dict(self._atomic_states)


def test_reset_resets_official_progress_and_clears_overshot():
    fake_env = FakeLiberoMemEnv()
    adapter = env_adapter.LiberoMemEnvAdapter(fake_env, task_text="pick and place")

    result = adapter.reset()

    assert result.observation == {"reset": 1}
    assert fake_env.reset_subgoal_progress_calls == 1
    assert fake_env._overshot is False
    assert result.overshot is False
    assert result.success is False
    assert result.satisfied_subgoals == ["pick and place:subgoal-0"]
    assert fake_env.check_success_calls == []


def test_step_calls_incremental_success_check_exactly_once_per_physical_step():
    fake_env = FakeLiberoMemEnv()
    adapter = env_adapter.LiberoMemEnvAdapter(fake_env, task_text="pick and place")
    adapter.reset()

    first = adapter.step(np.arange(7, dtype=np.float32))
    second = adapter.step(np.arange(7, dtype=np.float32))

    assert fake_env.step_calls == 2
    assert fake_env.check_success_calls == [True, True]
    assert first.success is False
    assert second.success is True
    assert second.reward == 0.25
    assert second.done is False
    assert second.info == {"internal_success_probe": False}
    assert second.satisfied_subgoals == ["pick and place:subgoal-2"]
    assert second.atomic_predicates == {"on(cube, plate)": True, "in(cube, bowl)": True}


def test_reset_does_not_leak_overshot_between_episodes():
    fake_env = FakeLiberoMemEnv()
    adapter = env_adapter.LiberoMemEnvAdapter(fake_env, task_text="pick and place")

    adapter.reset()
    fake_env._overshot = True
    leaked = adapter.snapshot({"stale": True})
    reset = adapter.reset()

    assert leaked.overshot is True
    assert reset.overshot is False
    assert fake_env._overshot is False
