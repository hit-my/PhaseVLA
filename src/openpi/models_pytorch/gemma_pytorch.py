import copy
import dataclasses
from typing import Literal

import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma

try:
    from transformers.cache_utils import Cache
except ImportError:  # pragma: no cover - transformers is a required runtime dependency.
    Cache = ()


def _is_transformers_cache(cache: object) -> bool:
    return isinstance(cache, Cache)


def _is_legacy_cache_sequence(cache: object) -> bool:
    return isinstance(cache, (tuple, list))


def _map_cache_tensors(cache: object, transform):
    if torch.is_tensor(cache):
        return transform(cache)
    if _is_legacy_cache_sequence(cache):
        mapped = [_map_cache_tensors(item, transform) for item in cache]
        return tuple(mapped) if isinstance(cache, tuple) else mapped
    if _is_transformers_cache(cache):
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            mapped_cache = copy.copy(cache)
            mapped_cache.key_cache = [_map_cache_tensors(tensor, transform) for tensor in cache.key_cache]
            mapped_cache.value_cache = [_map_cache_tensors(tensor, transform) for tensor in cache.value_cache]
            return mapped_cache
        if hasattr(cache, "to_legacy_cache") and callable(cache.to_legacy_cache):
            return _map_cache_tensors(cache.to_legacy_cache(), transform)
        raise TypeError(f"Unsupported Transformers cache structure: {type(cache).__name__}")
    raise TypeError(f"Unsupported cache structure: {type(cache).__name__}")


def iter_cache_tensors(cache: object):
    if torch.is_tensor(cache):
        yield cache
        return
    if _is_legacy_cache_sequence(cache):
        for item in cache:
            yield from iter_cache_tensors(item)
        return
    if _is_transformers_cache(cache):
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            for tensor in cache.key_cache:
                yield from iter_cache_tensors(tensor)
            for tensor in cache.value_cache:
                yield from iter_cache_tensors(tensor)
            return
        if hasattr(cache, "to_legacy_cache") and callable(cache.to_legacy_cache):
            yield from iter_cache_tensors(cache.to_legacy_cache())
            return
        raise TypeError(f"Unsupported Transformers cache structure: {type(cache).__name__}")
    raise TypeError(f"Unsupported cache structure: {type(cache).__name__}")


def detach_cache(cache: object):
    return _map_cache_tensors(cache, torch.Tensor.detach)


def slice_cache_batch(cache: object, index: int):
    if index < 0:
        raise ValueError("cache batch index must be non-negative")
    return _map_cache_tensors(cache, lambda tensor: tensor[index : index + 1])


def _clone_detached_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone()

def clone_cache(cache: object):
    if _is_legacy_cache_sequence(cache):
        return _map_cache_tensors(cache, _clone_detached_tensor)
    if _is_transformers_cache(cache):
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            cloned_cache = copy.copy(cache)
            cloned_cache.key_cache = [_clone_detached_tensor(tensor) for tensor in cache.key_cache]
            cloned_cache.value_cache = [_clone_detached_tensor(tensor) for tensor in cache.value_cache]
            return cloned_cache
        raise TypeError(f"Unsupported Transformers cache structure: {type(cache).__name__}")
    raise TypeError(f"Unsupported cache structure: {type(cache).__name__}")


def clone_selected_prefix_cache(cache: object, layer_indices):
    if _is_legacy_cache_sequence(cache):
        selected = [_map_cache_tensors(cache[layer_idx], _clone_detached_tensor) for layer_idx in layer_indices]
        return tuple(selected) if isinstance(cache, tuple) else selected
    if _is_transformers_cache(cache):
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            cloned_cache = copy.copy(cache)
            cloned_cache.key_cache = [_clone_detached_tensor(cache.key_cache[layer_idx]) for layer_idx in layer_indices]
            cloned_cache.value_cache = [_clone_detached_tensor(cache.value_cache[layer_idx]) for layer_idx in layer_indices]
            if hasattr(cloned_cache, "_seen_tokens"):
                cloned_cache._seen_tokens = cloned_cache.get_seq_length(0) if cloned_cache.key_cache else 0
            return cloned_cache
        raise TypeError(f"Unsupported Transformers cache structure: {type(cache).__name__}")
    raise TypeError(f"Unsupported cache structure: {type(cache).__name__}")


@dataclasses.dataclass(frozen=True)
class PrefixKVView:
    """Read-only detached prefix K/V layers selected for suffix-only attention."""

    layers: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    valid_lengths: torch.Tensor

    @classmethod
    def from_cache(
        cls,
        cache: object,
        layer_indices,
        prefix_mask: torch.BoolTensor,
    ) -> "PrefixKVView":
        if prefix_mask.ndim != 2:
            raise ValueError(f"prefix_mask must have shape [batch, prefix], got {tuple(prefix_mask.shape)}")
        layer_indices = tuple(layer_indices)
        selected = clone_selected_prefix_cache(cache, layer_indices)
        raw_layers = tuple(_cache_layers_as_key_values(selected))
        if len(raw_layers) != len(layer_indices):
            raise ValueError(f"selected {len(raw_layers)} cache layers for {len(layer_indices)} layer indices")
        batch_size, source_prefix_len = prefix_mask.shape
        valid_lengths = prefix_mask.long().sum(dim=-1)
        max_valid_length = int(valid_lengths.max().item()) if valid_lengths.numel() else 0
        layers = []
        for layer_idx, (key, value) in enumerate(raw_layers):
            _validate_kv_tensor(key, "key", layer_idx, batch_size, source_prefix_len)
            _validate_kv_tensor(value, "value", layer_idx, batch_size, source_prefix_len)
            if key.shape != value.shape:
                raise ValueError(
                    f"prefix cache layer {layer_idx} key/value shapes must match, got {tuple(key.shape)} and {tuple(value.shape)}"
                )
            layers.append(
                (
                    _pack_valid_prefix_tensor(key, prefix_mask, max_valid_length),
                    _pack_valid_prefix_tensor(value, prefix_mask, max_valid_length),
                )
            )
        return cls(layers=tuple(layers), valid_lengths=valid_lengths.detach().clone())

    def layer(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        key, value = self.layers[index]
        return key.detach(), value.detach()

    def batch_slice(self, index: int) -> "PrefixKVView":
        if index < 0 or index >= self.valid_lengths.shape[0]:
            raise IndexError(f"prefix cache batch index {index} is out of range")
        return PrefixKVView(
            layers=tuple((key[index : index + 1], value[index : index + 1]) for key, value in self.layers),
            valid_lengths=self.valid_lengths[index : index + 1],
        )


def _pack_valid_prefix_tensor(
    tensor: torch.Tensor,
    prefix_mask: torch.BoolTensor,
    max_valid_length: int,
) -> torch.Tensor:
    packed = tensor.new_zeros(tensor.shape[0], tensor.shape[1], max_valid_length, tensor.shape[3])
    for batch_idx in range(tensor.shape[0]):
        valid_positions = torch.nonzero(prefix_mask[batch_idx], as_tuple=False).flatten()
        if valid_positions.numel() > 0:
            packed[batch_idx, :, : valid_positions.numel(), :] = tensor[batch_idx, :, valid_positions, :]
    return packed.contiguous()


def _cache_layers_as_key_values(cache: object) -> list[tuple[torch.Tensor, torch.Tensor]]:
    if _is_transformers_cache(cache):
        if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
            return list(zip(cache.key_cache, cache.value_cache, strict=True))
        if hasattr(cache, "to_legacy_cache") and callable(cache.to_legacy_cache):
            return _cache_layers_as_key_values(cache.to_legacy_cache())
    if _is_legacy_cache_sequence(cache):
        layers = []
        for layer in cache:
            if not _is_legacy_cache_sequence(layer) or len(layer) < 2:
                raise TypeError(f"Unsupported cache layer structure: {type(layer).__name__}")
            key, value = layer[0], layer[1]
            if not torch.is_tensor(key) or not torch.is_tensor(value):
                raise TypeError("prefix cache layers must contain tensor key/value pairs")
            layers.append((key, value))
        return layers
    raise TypeError(f"Unsupported cache structure: {type(cache).__name__}")


def _validate_kv_tensor(
    tensor: torch.Tensor,
    name: str,
    layer_idx: int,
    batch_size: int,
    prefix_len: int,
) -> None:
    if tensor.ndim != 4:
        raise ValueError(f"prefix cache layer {layer_idx} {name} must have shape [batch, heads, prefix, dim]")
    if tensor.shape[0] != batch_size:
        raise ValueError(
            f"prefix cache layer {layer_idx} {name} batch {tensor.shape[0]} does not match prefix_mask batch {batch_size}"
        )
    if tensor.shape[2] != prefix_len:
        raise ValueError(
            f"prefix cache layer {layer_idx} {name} length {tensor.shape[2]} does not match prefix_mask length {prefix_len}"
        )


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )

        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None
        self.gemma_expert.lm_head = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None]
        if inputs_embeds[1] is None:
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
        elif inputs_embeds[0] is None:
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
        else:
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None

        return [prefix_output, suffix_output], prefix_past_key_values
