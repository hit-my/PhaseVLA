import dataclasses
import math
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
    progress_layer_mapping: tuple[int, ...] | None = None
    handoff_ratio: float = 0.4
    num_denoise_steps: int = 10
    execution_horizon: int = 20
    action_history_chunk_size: int = 20
    memory_input_source: Literal["executed_actions"] = "executed_actions"
    action_history_encoding: Literal["masked_flat_projection"] = "masked_flat_projection"
    terminal_loss_weight: float = 0.0
    handoff_loss_weight: float = 0.0
    boundary_loss_weight: float = 0.0
    action_expert_gradient_checkpointing: bool = False
    terminal_loss_batch_fraction: float = 1.0
    terminal_loss_queries_per_episode: int | None = None
    frozen_prefix_microbatch_size: int = 2
    progress_use_prefix_kv: bool = True
    progress_prefix_kv_dropout: float = 0.0
    progress_memory_tokens: int = 1
    schema_version: int = 5
    architecture: Literal["action_history_mamba_pe_ae_handoff"] = "action_history_mamba_pe_ae_handoff"
    base_checkpoint_uri: str | None = None
    base_checkpoint_checksum: str | None = None
    base_assets_checksum: str | None = None
    assets_uri: str | None = None
    dataset_uri: str | None = None
    dataset_checksum: str | None = None
    task_name: str | None = None
    robomme_policy_commit: str | None = None
    robomme_benchmark_commit: str | None = None
    robomme_dataset_checksum: str | None = None
    robomme_task_suite: str | None = None
    train_seed: int | None = None
    mamba_repo_commit: str = MAMBA_REPO_COMMIT
    memory_state_schema_version: int = 2
    history_state_schema_version: int = 1
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
        if self.memory_input_source != "executed_actions":
            raise ValueError(f"memory_input_source must be 'executed_actions', got {self.memory_input_source!r}")
        if self.action_history_encoding != "masked_flat_projection":
            raise ValueError(
                "action_history_encoding must be 'masked_flat_projection', "
                f"got {self.action_history_encoding!r}"
            )
        if self.action_history_chunk_size <= 0:
            raise ValueError(
                f"action_history_chunk_size must be positive, got {self.action_history_chunk_size}"
            )
        if self.num_denoise_steps <= 0:
            raise ValueError(f"num_denoise_steps must be positive, got {self.num_denoise_steps}")
        if self.frozen_prefix_microbatch_size <= 0:
            raise ValueError(
                f"frozen_prefix_microbatch_size must be positive, got {self.frozen_prefix_microbatch_size}"
            )
        if not isinstance(self.progress_use_prefix_kv, bool):
            raise ValueError("progress_use_prefix_kv must be bool")
        if not 0.0 <= self.progress_prefix_kv_dropout <= 1.0:
            raise ValueError(
                "progress_prefix_kv_dropout must be in [0, 1], "
                f"got {self.progress_prefix_kv_dropout}"
            )
        if self.progress_memory_tokens <= 0:
            raise ValueError(
                f"progress_memory_tokens must be positive, got {self.progress_memory_tokens}"
            )
        action_expert_config = _gemma.get_config(self.action_expert_variant)
        if self.progress_depth <= 0 or self.progress_depth > action_expert_config.depth:
            raise ValueError(
                f"progress_depth must be in [1, {action_expert_config.depth}], got {self.progress_depth}"
            )
        if self.progress_layer_mapping is not None:
            mapping = tuple(int(layer) for layer in self.progress_layer_mapping)
            if len(mapping) != self.progress_depth:
                raise ValueError("progress_layer_mapping length must equal progress_depth")
            if mapping[0] != 0 or mapping[-1] != action_expert_config.depth - 1:
                raise ValueError("progress_layer_mapping must cover first and last Action Expert layers")
            if any(layer < 0 or layer >= action_expert_config.depth for layer in mapping):
                raise ValueError("progress_layer_mapping contains an out-of-range layer")
            if any(left >= right for left, right in zip(mapping[:-1], mapping[1:], strict=True)):
                raise ValueError("progress_layer_mapping must be strictly increasing")
        if self.memory.d_model <= 0:
            raise ValueError(f"memory d_model must be positive, got {self.memory.d_model}")
        if self.memory.depth <= 0 or self.memory.d_state <= 0:
            raise ValueError("memory depth and d_state must be positive")
        if self.memory.expand <= 0 or self.memory.headdim <= 0:
            raise ValueError("memory expand and headdim must be positive")
        if (self.memory.d_model * self.memory.expand) % self.memory.headdim != 0:
            raise ValueError(
                "memory d_model * expand must be divisible by headdim, "
                f"got {self.memory.d_model} * {self.memory.expand} and {self.memory.headdim}"
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
        if self.progress_layer_mapping is not None:
            return tuple(self.progress_layer_mapping)
        if self.progress_depth == 1:
            return (0,)
        if self.progress_depth == action_depth:
            return tuple(range(action_depth))
        return tuple(round(index * (action_depth - 1) / (self.progress_depth - 1)) for index in range(self.progress_depth))

    def checkpoint_metadata(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "architecture": self.architecture,
            "base_checkpoint_uri": self.base_checkpoint_uri,
            "base_checkpoint_checksum": self.base_checkpoint_checksum,
            "base_assets_checksum": self.base_assets_checksum,
            "assets_uri": self.assets_uri,
            "dataset_uri": self.dataset_uri,
            "dataset_checksum": self.dataset_checksum,
            "task_name": self.task_name,
            "train_seed": self.train_seed,
            "mamba_repo_commit": self.mamba_repo_commit,
            "memory_backend": self.memory_backend,
            "memory_state_schema_version": self.memory_state_schema_version,
            "history_state_schema_version": self.history_state_schema_version,
            "memory_config": dataclasses.asdict(self.memory),
            "progress_depth": self.progress_depth,
            "progress_memory_tokens": self.progress_memory_tokens,
            "progress_layer_mapping": list(self.resolved_progress_layer_indices),
            "handoff_ratio": self.handoff_ratio,
            "denoising_order": "progress_expert_then_action_expert",
            "num_denoise_steps": self.num_denoise_steps,
            "progress_denoise_steps": math.ceil(self.handoff_ratio * self.num_denoise_steps),
            "prediction_horizon": self.action_horizon,
            "execution_horizon": self.execution_horizon,
            "memory_input_source": self.memory_input_source,
            "action_history_encoding": self.action_history_encoding,
            "action_history_chunk_size": self.action_history_chunk_size,
            "training_query_stride": 1,
            "memory_update_timing": "append_previous_executed_actions_before_current_query",
            "partial_chunk_behavior": "right_pad_and_mask_without_commit",
            "empty_history_behavior": "learned_token_without_state_update",
            "uses_vlm_hidden_for_memory": False,
            "uses_prefix_kv_for_progress": self.progress_use_prefix_kv,
            "loss_weights": {
                "terminal": self.terminal_loss_weight,
                "handoff": self.handoff_loss_weight,
                "boundary": self.boundary_loss_weight,
            },
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
