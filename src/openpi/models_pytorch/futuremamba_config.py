import dataclasses
from typing import Literal

from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma


MemoryBackend = Literal["mamba2", "mamba3_siso", "gru", "lstm", "frame_stack", "none"]
KernelMode = Literal["fallback", "triton", "cute"]

MAMBA_REPO_COMMIT = "77069de5cdb55cbe98b670889c80df211e031039"
_MEMORY_BACKENDS: frozenset[str] = frozenset({"mamba2", "mamba3_siso", "gru", "lstm", "frame_stack", "none"})
_KERNEL_MODES: frozenset[str] = frozenset({"fallback", "triton", "cute"})


@dataclasses.dataclass(frozen=True)
class MambaMemoryConfig:
    d_model: int = 1024
    depth: int = 2
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    ngroups: int = 1
    d_conv: int = 4
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = False
    use_mem_eff_path: bool = False


@dataclasses.dataclass(frozen=True)
class FutureMambaPytorchConfig(pi0_config.Pi0Config):
    """PyTorch-only FutureMamba configuration with an explicit backend contract."""

    pi05: bool = True
    discrete_state_input: bool = True
    memory: MambaMemoryConfig = dataclasses.field(default_factory=MambaMemoryConfig)
    memory_backend: MemoryBackend = "mamba2"
    progress_depth: int = 6
    handoff_ratio: float = 0.2
    num_denoise_steps: int = 10
    execution_horizon: int = 16
    terminal_loss_weight: float = 1.0
    handoff_loss_weight: float = 0.0
    boundary_loss_weight: float = 0.0
    action_expert_gradient_checkpointing: bool = False
    terminal_loss_batch_fraction: float = 1.0
    terminal_loss_queries_per_episode: int | None = None
    frozen_prefix_microbatch_size: int = 2

    # Training checkpoint identity and runtime provenance. These remain explicit so
    schema_version: int = 3
    base_checkpoint_uri: str | None = None
    base_checkpoint_checksum: str | None = None
    base_assets_checksum: str | None = None
    robomme_policy_commit: str | None = None
    robomme_benchmark_commit: str | None = None
    robomme_dataset_checksum: str | None = None
    robomme_task_suite: str | None = None
    train_seed: int | None = None
    mamba_repo_commit: str = MAMBA_REPO_COMMIT
    memory_state_schema_version: int = 1
    state_dtypes: dict[str, str] = dataclasses.field(default_factory=dict)
    kernel_mode: KernelMode = "fallback"
    torch_version: str | None = None
    triton_version: str | None = None
    cuda_version: str | None = None
    gpu_name: str | None = None
    compute_capability: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.pi05 is not True:
            raise ValueError("FutureMambaPytorchConfig requires pi05=True")
        if not isinstance(self.discrete_state_input, bool):
            raise ValueError("discrete_state_input must be bool")
        if self.memory_backend not in _MEMORY_BACKENDS:
            raise ValueError(
                f"memory_backend must be one of {sorted(_MEMORY_BACKENDS)}, got {self.memory_backend!r}"
            )
        if self.kernel_mode not in _KERNEL_MODES:
            raise ValueError(f"kernel_mode must be one of {sorted(_KERNEL_MODES)}, got {self.kernel_mode!r}")
        if not 0 <= self.handoff_ratio <= 1:
            raise ValueError(f"handoff_ratio must be in [0, 1], got {self.handoff_ratio}")
        if self.num_denoise_steps <= 0:
            raise ValueError(f"num_denoise_steps must be positive, got {self.num_denoise_steps}")
        if self.execution_horizon <= 0 or self.execution_horizon > self.action_horizon:
            raise ValueError(
                f"execution_horizon must be positive and <= action_horizon ({self.action_horizon}), "
                f"got {self.execution_horizon}"
            )
        if self.terminal_loss_weight < 0:
            raise ValueError(f"terminal_loss_weight must be non-negative, got {self.terminal_loss_weight}")
        if self.handoff_loss_weight < 0:
            raise ValueError(f"handoff_loss_weight must be non-negative, got {self.handoff_loss_weight}")
        if self.boundary_loss_weight < 0:
            raise ValueError(f"boundary_loss_weight must be non-negative, got {self.boundary_loss_weight}")
        if not isinstance(self.action_expert_gradient_checkpointing, bool):
            raise ValueError("action_expert_gradient_checkpointing must be bool")
        if not 0 < self.terminal_loss_batch_fraction <= 1:
            raise ValueError(
                "terminal_loss_batch_fraction must be in (0, 1], "
                f"got {self.terminal_loss_batch_fraction}"
            )
        if self.terminal_loss_queries_per_episode is not None and self.terminal_loss_queries_per_episode <= 0:
            raise ValueError(
                "terminal_loss_queries_per_episode must be positive or None, "
                f"got {self.terminal_loss_queries_per_episode}"
            )
        if self.frozen_prefix_microbatch_size <= 0:
            raise ValueError(
                "frozen_prefix_microbatch_size must be positive, "
                f"got {self.frozen_prefix_microbatch_size}"
            )
        action_expert_config = _gemma.get_config(self.action_expert_variant)
        if self.progress_depth <= 0 or self.progress_depth > action_expert_config.depth:
            raise ValueError(
                f"progress_depth must be in [1, {action_expert_config.depth}], got {self.progress_depth}"
            )
        if self.memory.d_model != action_expert_config.width:
            raise ValueError(
                f"memory d_model ({self.memory.d_model}) must match action expert width ({action_expert_config.width})"
            )

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05

    @override
    def create(self, rng):
        del rng
        raise RuntimeError("FutureMambaPytorchConfig is PyTorch-only; call create_pytorch()")

    def create_pytorch(self) -> "FutureMambaPytorch":
        from openpi.models_pytorch.futuremamba import FutureMambaPytorch

        return FutureMambaPytorch(self)

    @property
    def resolved_progress_layer_indices(self) -> tuple[int, ...]:
        action_depth = _gemma.get_config(self.action_expert_variant).depth
        if self.progress_depth == 1:
            return (0,)
        if self.progress_depth == action_depth:
            return tuple(range(action_depth))
        return tuple(round(index * (action_depth - 1) / (self.progress_depth - 1)) for index in range(self.progress_depth))

    def checkpoint_metadata(self) -> dict[str, object]:
        """Return the complete Stage B metadata identity contract."""
        return {
            "schema_version": self.schema_version,
            "base_checkpoint_uri": self.base_checkpoint_uri,
            "base_checkpoint_checksum": self.base_checkpoint_checksum,
            "base_assets_checksum": self.base_assets_checksum,
            "robomme_policy_commit": self.robomme_policy_commit,
            "robomme_benchmark_commit": self.robomme_benchmark_commit,
            "robomme_dataset_checksum": self.robomme_dataset_checksum,
            "robomme_task_suite": self.robomme_task_suite,
            "train_seed": self.train_seed,
            "mamba_repo_commit": self.mamba_repo_commit,
            "memory_backend": self.memory_backend,
            "memory_state_schema_version": self.memory_state_schema_version,
            "memory_config": dataclasses.asdict(self.memory),
            "progress_depth": self.progress_depth,
            "progress_layer_mapping": list(self.resolved_progress_layer_indices),
            "handoff_ratio": self.handoff_ratio,
            "num_denoise_steps": self.num_denoise_steps,
            "prediction_horizon": self.action_horizon,
            "execution_horizon": self.execution_horizon,
            "loss_weights": {
                "terminal": self.terminal_loss_weight,
                "handoff": self.handoff_loss_weight,
                "boundary": self.boundary_loss_weight,
            },
            "action_expert_gradient_checkpointing": self.action_expert_gradient_checkpointing,
            "terminal_loss_batch_fraction": self.terminal_loss_batch_fraction,
            "terminal_loss_queries_per_episode": self.terminal_loss_queries_per_episode,
            "frozen_prefix_microbatch_size": self.frozen_prefix_microbatch_size,
            "training_dtype": self.dtype,
            "state_dtypes": dict(self.state_dtypes),
            "kernel_mode": self.kernel_mode,
            "torch_version": self.torch_version,
            "triton_version": self.triton_version,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "compute_capability": self.compute_capability,
        }
