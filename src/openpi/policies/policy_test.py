import numpy as np

from openpi_client import base_policy
from openpi.policies import policy as _policy


class _StatefulPolicy(base_policy.BasePolicy):
    def __init__(self):
        self.value = 0
        self.reset_count = 0
        self.restored = None

    def infer(self, obs):
        self.value += int(obs.get("delta", 1))
        return {"value": self.value}

    def reset(self):
        self.reset_count += 1
        self.value = 0

    def snapshot_state(self):
        return {"value": self.value}

    def restore_state(self, state):
        self.restored = state
        self.value = state["value"]

    def fork(self):
        return _StatefulPolicy()


def test_base_policy_fork_defaults_to_self_for_stateless_compatibility():
    class Stateless(base_policy.BasePolicy):
        def infer(self, obs):
            return obs

    policy = Stateless()

    assert policy.fork() is policy


def test_policy_recorder_forwards_lifecycle_calls(tmp_path):
    wrapped = _StatefulPolicy()
    recorder = _policy.PolicyRecorder(wrapped, tmp_path)

    recorder.infer({"delta": 3})
    snapshot = recorder.snapshot_state()
    recorder.reset()
    recorder.restore_state(snapshot)

    assert wrapped.reset_count == 1
    assert wrapped.restored == snapshot
    assert wrapped.value == 3


def test_policy_recorder_fork_wraps_independent_session_policy(tmp_path):
    wrapped = _StatefulPolicy()
    recorder = _policy.PolicyRecorder(wrapped, tmp_path)

    fork = recorder.fork()

    assert fork is not recorder
    assert fork.infer({"delta": 2}) == {"value": 2}
    assert wrapped.value == 0
    assert (tmp_path / "fork_0").is_dir()
