from __future__ import annotations

import dataclasses
from typing import Literal

import openpi.models.gemma as gemma
from openpi.models_pytorch.futuremamba_config import FutureMambaPytorchConfig


@dataclasses.dataclass(frozen=True)
class MemoryAEConfig(FutureMambaPytorchConfig):
    """Frozen VLM with one trainable memory-conditioned original action expert."""

    architecture: Literal["action_history_mamba_memory_ae"] = "action_history_mamba_memory_ae"
    handoff_ratio: float = 0.0
    progress_depth: int = 18
    progress_layer_mapping: tuple[int, ...] | None = None
    frozen_prefix_microbatch_size: int = 4
    discrete_state_input: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.handoff_ratio != 0.0:
            raise ValueError("MemoryAE has no PE handoff; handoff_ratio must be zero")
        if self.memory_backend != "mamba2":
            raise ValueError("MemoryAE ablation requires the original Mamba-2 backend")
        if self.progress_depth != gemma.get_config(self.action_expert_variant).depth:
            raise ValueError("MemoryAE needs prefix KV from every action-expert layer")
        if self.discrete_state_input:
            raise ValueError("MemoryAE uses the mainline continuous-state input protocol")

    def create_pytorch(self):
        from openpi.models_pytorch.memory_ae import MemoryAEPytorch

        return MemoryAEPytorch(self)

    def checkpoint_metadata(self) -> dict[str, object]:
        metadata = super().checkpoint_metadata()
        metadata.update(
            architecture=self.architecture,
            progress_depth=0,
            progress_layer_mapping=[],
            progress_denoise_steps=0,
            denoising_order="memory_conditioned_action_expert_only",
            uses_prefix_kv_for_progress=False,
            action_expert_depth=gemma.get_config(self.action_expert_variant).depth,
            action_expert_denoise_steps=self.num_denoise_steps,
            memory_injection="per_layer_action_expert_kv",
            train_time_distribution="pi05_full_time_beta",
            vlm_frozen=True,
            action_expert_trainable=True,
        )
        return metadata
