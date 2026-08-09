import abc
from typing import Dict


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        """Infer actions from observations."""

    def reset(self) -> None:
        """Reset the policy to its initial state."""
        pass

    def snapshot_state(self):
        """Return an opaque policy state snapshot."""
        raise NotImplementedError("Policy does not support state snapshots")

    def restore_state(self, state) -> None:
        """Restore a state produced by snapshot_state."""
        raise NotImplementedError("Policy does not support state restore")

    def fork(self) -> "BasePolicy":
        """Return a policy session for an independent connection."""
        return self
