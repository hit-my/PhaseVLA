from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any


@dataclasses.dataclass(frozen=True)
class EnvSnapshot:
    observation: Any
    success: bool
    satisfied_subgoals: list[Any]
    overshot: bool
    atomic_predicates: dict[str, bool]


@dataclasses.dataclass(frozen=True)
class EnvStepResult(EnvSnapshot):
    reward: float
    done: bool
    info: dict[str, Any]


class LiberoMemEnvAdapter:
    """Thin adapter around LIBERO-Mem's ordered subgoal progress API."""

    def __init__(self, env: Any, *, task_text: str):
        self._env = env
        self._task_text = task_text

    @property
    def env(self) -> Any:
        return self._env

    @property
    def task_text(self) -> str:
        return self._task_text

    def reset(self) -> EnvSnapshot:
        observation = self._env.reset()
        self._env.reset_subgoal_progress()
        setattr(self._env, "_overshot", False)
        return self.snapshot(observation, success=False)

    def step(self, action: Any) -> EnvStepResult:
        observation, reward, done, info = self._env.step(action)
        success = bool(self._env._check_success(inc=True))
        snapshot = self.snapshot(observation, success=success)
        return EnvStepResult(
            observation=snapshot.observation,
            success=snapshot.success,
            satisfied_subgoals=snapshot.satisfied_subgoals,
            overshot=snapshot.overshot,
            atomic_predicates=snapshot.atomic_predicates,
            reward=reward,
            done=bool(done),
            info=dict(info),
        )

    def snapshot(self, observation: Any, *, success: bool = False) -> EnvSnapshot:
        return EnvSnapshot(
            observation=observation,
            success=bool(success),
            satisfied_subgoals=list(self._env.get_satisfied_subgoals(self._task_text)),
            overshot=bool(getattr(self._env, "_overshot", False)),
            atomic_predicates=self._atomic_predicates(),
        )

    def _atomic_predicates(self) -> dict[str, bool]:
        if hasattr(self._env, "get_atomic_predicate_states"):
            return _bool_dict(self._env.get_atomic_predicate_states())
        if hasattr(self._env, "get_atomic_predicates"):
            return _bool_dict(self._env.get_atomic_predicates())
        if hasattr(self._env, "atomic_predicates"):
            value = self._env.atomic_predicates
            if callable(value):
                value = value()
            return _bool_dict(value)
        if hasattr(self._env, "_atomic_predicates"):
            return _bool_dict(getattr(self._env, "_atomic_predicates"))
        return {}


def _bool_dict(value: Any) -> dict[str, bool]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): bool(predicate_value) for key, predicate_value in value.items()}
    return {str(key): True for key in value}
