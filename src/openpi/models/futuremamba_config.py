import dataclasses
import math
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


_PARAMETER_MATCH_TOLERANCE = 0.05


def _mamba_parameter_count(config: MambaConfig) -> int:
    d_model = int(config.d_model)
    d_inner = d_model * int(config.expand)
    d_state = int(config.d_state)
    d_conv = int(config.d_conv)
    dt_rank = int(config.dt_rank)
    per_layer = 0
    per_layer += d_model
    per_layer += d_model * (2 * d_inner) + (2 * d_inner)
    per_layer += d_inner * (dt_rank + 2 * d_state)
    per_layer += dt_rank * d_inner + d_inner
    per_layer += d_inner * d_model + d_model
    per_layer += d_conv * d_inner + d_inner
    per_layer += d_inner * d_state
    per_layer += d_inner
    return per_layer * int(config.depth)


def _gru_parameter_count(d_model: int, hidden_width: int) -> int:
    return (
        d_model * hidden_width
        + hidden_width
        + hidden_width * (3 * hidden_width)
        + 3 * hidden_width
        + hidden_width * (3 * hidden_width)
        + hidden_width * d_model
        + d_model
    )


def _lstm_parameter_count(d_model: int, hidden_width: int) -> int:
    return (
        d_model * hidden_width
        + hidden_width
        + hidden_width * (4 * hidden_width)
        + 4 * hidden_width
        + hidden_width * (4 * hidden_width)
        + hidden_width * d_model
        + d_model
    )


def _frame_stack_parameter_count(d_model: int, frame_stack_window: int, hidden_width: int) -> int:
    return frame_stack_window * d_model * hidden_width + hidden_width + hidden_width * d_model + d_model


def _nearest_hidden_width(target_count: int, counter) -> tuple[int, int]:
    max_width = max(4096, int(math.ceil(math.sqrt(max(target_count, 1)))) * 4)
    best_width = 1
    best_count = counter(1)
    best_error = abs(best_count - target_count)
    for width in range(2, max_width + 1):
        count = counter(width)
        error = abs(count - target_count)
        if error < best_error:
            best_width = width
            best_count = count
            best_error = error
    return best_width, best_count


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
        from openpi.models.futuremamba import FutureMamba

        return FutureMamba(self, rngs=nnx.Rngs(rng))

    def get_trainable_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx_utils.PathRegex("futuremamba/.*"))

    @override
    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        return nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex("futuremamba/.*")))

    @property
    def mamba_memory_parameter_count(self) -> int:
        return _mamba_parameter_count(self.memory)

    def _memory_backend_parameter_info(self) -> tuple[int, int, float, bool]:
        target = self.mamba_memory_parameter_count
        d_model = int(self.memory.d_model)
        if self.memory_backend == "mamba":
            width = d_model * int(self.memory.expand)
            count = target
        elif self.memory_backend == "gru":
            width, count = _nearest_hidden_width(target, lambda hidden: _gru_parameter_count(d_model, hidden))
        elif self.memory_backend == "lstm":
            width, count = _nearest_hidden_width(target, lambda hidden: _lstm_parameter_count(d_model, hidden))
        elif self.memory_backend == "frame_stack":
            width, count = _nearest_hidden_width(
                target, lambda hidden: _frame_stack_parameter_count(d_model, int(self.frame_stack_window), hidden)
            )
        elif self.memory_backend == "none":
            width, count = 0, 0
        else:
            raise ValueError(f"Unknown memory_backend: {self.memory_backend!r}")
        error = 0.0 if target == 0 else abs(float(count) - float(target)) / float(target)
        matched = error <= _PARAMETER_MATCH_TOLERANCE
        return width, count, error, matched

    @property
    def memory_backend_hidden_width(self) -> int:
        return self._memory_backend_parameter_info()[0]

    @property
    def memory_backend_parameter_count(self) -> int:
        return self._memory_backend_parameter_info()[1]

    @property
    def memory_backend_parameter_error(self) -> float:
        return self._memory_backend_parameter_info()[2]

    @property
    def memory_backend_parameter_matched(self) -> bool:
        return self._memory_backend_parameter_info()[3]

    def checkpoint_metadata(self) -> dict[str, object]:
        switches = {
            "rho": self.handoff_ratio,
            "handoff_ratio": self.handoff_ratio,
            "depth": self.progress_depth,
            "progress_depth": self.progress_depth,
            "progress_prefix_layer_indices": self.resolved_progress_layer_indices,
            "handoff_loss_weight": self.handoff_loss_weight,
            "boundary_loss_weight": self.boundary_loss_weight,
            "loss_weights": {
                "handoff": self.handoff_loss_weight,
                "boundary": self.boundary_loss_weight,
            },
            "memory_input": self.memory_input,
            "token_only": self.memory_input == "token_only",
            "pooling": self.conditioning_pool,
            "conditioning_pool": self.conditioning_pool,
            "prefix_cache": self.use_prefix_cache,
            "use_prefix_cache": self.use_prefix_cache,
            "reset_every_query": self.reset_memory_every_query,
            "reset_memory_every_query": self.reset_memory_every_query,
            "backend": self.memory_backend,
            "memory_backend": self.memory_backend,
            "decoder": self.decoder_mode,
            "decoder_mode": self.decoder_mode,
            "coupling": self.coupling,
            "frame_stack_window": self.frame_stack_window,
            "bptt_window_queries": self.bptt_window_queries,
            "bptt": self.bptt_window_queries,
            "memory": dataclasses.asdict(self.memory),
        }
        width, count, error, matched = self._memory_backend_parameter_info()
        return {
            **switches,
            "memory_backend_hidden_width": width,
            "memory_parameter_count": count,
            "mamba_memory_parameter_count": self.mamba_memory_parameter_count,
            "memory_parameter_error": error,
            "parameter_matched": matched,
            "parameter_match_tolerance": _PARAMETER_MATCH_TOLERANCE,
            "all_ablation_switches": switches,
        }

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
        _ = self.memory_backend_parameter_error

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
