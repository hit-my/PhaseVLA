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
    progress_depth: int = 4
    handoff_ratio: float = 0.2
    num_denoise_steps: int = 10
    executed_horizon: int = 5

    # Training checkpoint identity and runtime provenance. These remain explicit so
    # metadata is stable even before a checkpoint is attached to the config.
    schema_version: int = 1
    base_checkpoint_uri: str | None = None
    base_checkpoint_checksum: str | None = None
    base_assets_checksum: str | None = None
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
        if self.discrete_state_input is not True:
            raise ValueError("FutureMambaPytorchConfig requires discrete_state_input=True")
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
        if self.executed_horizon <= 0 or self.executed_horizon > self.action_horizon:
            raise ValueError(
                f"executed_horizon must be positive and <= action_horizon ({self.action_horizon}), "
                f"got {self.executed_horizon}"
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
            "mamba_repo_commit": self.mamba_repo_commit,
            "memory_backend": self.memory_backend,
            "memory_state_schema_version": self.memory_state_schema_version,
            "memory_config": dataclasses.asdict(self.memory),
            "progress_depth": self.progress_depth,
            "progress_layer_mapping": list(self.resolved_progress_layer_indices),
            "handoff_ratio": self.handoff_ratio,
            "num_denoise_steps": self.num_denoise_steps,
            "executed_horizon": self.executed_horizon,
            "loss_weights": {},
            "training_dtype": self.dtype,
            "state_dtypes": dict(self.state_dtypes),
            "kernel_mode": self.kernel_mode,
            "torch_version": self.torch_version,
            "triton_version": self.triton_version,
            "cuda_version": self.cuda_version,
            "gpu_name": self.gpu_name,
            "compute_capability": self.compute_capability,
        }
