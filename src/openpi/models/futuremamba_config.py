import dataclasses
from typing import Literal

import flax.nnx as nnx
from typing_extensions import override

import openpi.models.gemma as _gemma
from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.models.mamba import MambaConfig
from openpi.models.progress_expert import make_layer_mapping
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils


MemoryInput = Literal["token_action", "token_only"]
ConditioningPool = Literal["last_valid", "attention", "tokens4", "tokens8"]
MemoryBackend = Literal["mamba", "gru", "lstm", "frame_stack", "none"]
DecoderMode = Literal["handoff", "action_memory_full"]
Coupling = Literal["hard", "convex", "residual"]


@dataclasses.dataclass(frozen=True)
class FutureMambaConfig(pi0_config.Pi0Config):
    pi05: bool = True
    discrete_state_input: bool = True

    progress_depth: int = 4
    progress_prefix_layer_indices: tuple[int, ...] | None = None
    handoff_ratio: float = 0.2
    num_denoise_steps: int = 10
    executed_horizon: int = 5
    executed_action_noise_std: float = 0.01
    handoff_loss_weight: float = 1.0
    boundary_loss_weight: float = 0.1
    memory: MambaConfig = dataclasses.field(default_factory=lambda: MambaConfig(d_model=1024))
    memory_input: MemoryInput = "token_action"
    conditioning_pool: ConditioningPool = "last_valid"
    memory_backend: MemoryBackend = "mamba"
    frame_stack_window: int = 4
    bptt_window_queries: int | None = None
    decoder_mode: DecoderMode = "handoff"
    coupling: Coupling = "hard"
    use_prefix_cache: bool = True
    reset_memory_every_query: bool = False

    def __post_init__(self):
        if self.pi05 is not True:
            raise ValueError("FutureMambaConfig requires pi05=True")
        if self.discrete_state_input is not True:
            raise ValueError("FutureMambaConfig requires discrete_state_input=True")
        super().__post_init__()
        self._validate_config()

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    @property
    def resolved_progress_layer_indices(self) -> tuple[int, ...]:
        action_depth = _gemma.get_config(self.action_expert_variant).depth
        if self.progress_prefix_layer_indices is None:
            return make_layer_mapping(action_depth, self.progress_depth)
        return tuple(int(index) for index in self.progress_prefix_layer_indices)

    @override
    def create(self, rng: at.KeyArrayLike):
        if self.memory_backend != "mamba":
            raise NotImplementedError(f"FutureMamba memory_backend={self.memory_backend!r} is not implemented")
        if self.decoder_mode != "handoff":
            raise NotImplementedError(f"FutureMamba decoder_mode={self.decoder_mode!r} is not implemented")
        from openpi.models.futuremamba import FutureMamba

        return FutureMamba(self, rngs=nnx.Rngs(rng))

    def get_trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx_utils.PathRegex("futuremamba/.*"))

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex("futuremamba/.*")))

    def _validate_config(self) -> None:
        if not 0 <= self.handoff_ratio <= 1:
            raise ValueError(f"handoff_ratio must be in [0, 1], got {self.handoff_ratio}")
        if self.num_denoise_steps <= 0:
            raise ValueError(f"num_denoise_steps must be positive, got {self.num_denoise_steps}")
        if self.action_horizon <= 0:
            raise ValueError(f"action_horizon must be positive, got {self.action_horizon}")
        if self.executed_horizon <= 0 or self.executed_horizon > self.action_horizon:
            raise ValueError(
                f"executed_horizon must be positive and <= action_horizon ({self.action_horizon}), "
                f"got {self.executed_horizon}"
            )
        if self.executed_action_noise_std < 0:
            raise ValueError(
                f"executed_action_noise_std must be non-negative, got {self.executed_action_noise_std}"
            )
        if self.handoff_loss_weight < 0:
            raise ValueError(f"handoff_loss_weight must be non-negative, got {self.handoff_loss_weight}")
        if self.boundary_loss_weight < 0:
            raise ValueError(f"boundary_loss_weight must be non-negative, got {self.boundary_loss_weight}")
        if self.frame_stack_window <= 0:
            raise ValueError(f"frame_stack_window must be positive, got {self.frame_stack_window}")
        if self.bptt_window_queries is not None and self.bptt_window_queries <= 0:
            raise ValueError(f"bptt_window_queries must be positive when set, got {self.bptt_window_queries}")

        action_config = _gemma.get_config(self.action_expert_variant)
        if self.progress_depth <= 0 or self.progress_depth > action_config.depth:
            raise ValueError(
                f"progress_depth must be in [1, {action_config.depth}], got {self.progress_depth}"
            )
        if self.memory.d_model != action_config.width:
            raise ValueError(
                f"memory d_model ({self.memory.d_model}) must match action expert width ({action_config.width})"
            )

        self._validate_progress_layer_indices(action_config.depth)
        self._validate_enums()

    def _validate_progress_layer_indices(self, action_depth: int) -> None:
        if self.progress_prefix_layer_indices is None:
            return
        indices = tuple(int(index) for index in self.progress_prefix_layer_indices)
        if len(indices) != self.progress_depth:
            raise ValueError(
                f"progress_prefix_layer_indices length ({len(indices)}) must equal progress_depth ({self.progress_depth})"
            )
        if len(indices) > 1 and (indices[0] != 0 or indices[-1] != action_depth - 1):
            raise ValueError(
                "progress_prefix_layer_indices must cover the first and last layers when progress_depth > 1, "
                f"got {indices} for action depth {action_depth}"
            )
        if any(index < 0 or index >= action_depth for index in indices):
            raise ValueError(f"progress_prefix_layer_indices entries must be in range [0, {action_depth}), got {indices}")
        if any(left >= right for left, right in zip(indices, indices[1:])):
            raise ValueError(f"progress_prefix_layer_indices must be strictly increasing, got {indices}")

    def _validate_enums(self) -> None:
        if self.memory_input not in ("token_action", "token_only"):
            raise ValueError(f"Unknown memory_input: {self.memory_input!r}")
        if self.conditioning_pool not in ("last_valid", "attention", "tokens4", "tokens8"):
            raise ValueError(f"Unknown conditioning_pool: {self.conditioning_pool!r}")
        if self.memory_backend not in ("mamba", "gru", "lstm", "frame_stack", "none"):
            raise ValueError(f"Unknown memory_backend: {self.memory_backend!r}")
        if self.decoder_mode not in ("handoff", "action_memory_full"):
            raise ValueError(f"Unknown decoder_mode: {self.decoder_mode!r}")
        if self.coupling not in ("hard", "convex", "residual"):
            raise ValueError(f"Unknown coupling: {self.coupling!r}")
