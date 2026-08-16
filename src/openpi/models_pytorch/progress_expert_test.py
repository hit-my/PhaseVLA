from __future__ import annotations

import copy
import dataclasses

import pytest
import torch
from torch import nn
from transformers.cache_utils import DynamicCache
from transformers.models.gemma import modeling_gemma

from openpi.models_pytorch.gemma_pytorch import PrefixKVView, iter_cache_tensors
from openpi.models_pytorch.progress_expert import (
    ProgressExpertPytorch,
    default_progress_depth,
    make_layer_mapping,
)


@dataclasses.dataclass
class _TinyConfig:
    action_horizon: int = 3
    action_dim: int = 2
    action_expert_variant: str = "dummy"
    progress_depth: int = 2
    dtype: str = "float32"


def _legacy_cache(*, batch: int = 2, layers: int = 4, prefix_len: int = 5, requires_grad: bool = False):
    cache = []
    for layer_idx in range(layers):
        base = torch.arange(batch * 1 * prefix_len * 16, dtype=torch.float32).reshape(batch, 1, prefix_len, 16)
        key = (base + layer_idx * 1000).requires_grad_(requires_grad)
        value = (base + layer_idx * 1000 + 500).requires_grad_(requires_grad)
        cache.append((key, value))
    return tuple(cache)


def _dynamic_cache(*, batch: int = 2, layers: int = 4, prefix_len: int = 5):
    return DynamicCache.from_legacy_cache(_legacy_cache(batch=batch, layers=layers, prefix_len=prefix_len))



def _pack_cache_tensor(tensor: torch.Tensor, prefix_mask: torch.BoolTensor) -> torch.Tensor:
    max_valid = int(prefix_mask.long().sum(dim=-1).max().item())
    packed = tensor.new_zeros(tensor.shape[0], tensor.shape[1], max_valid, tensor.shape[3])
    for batch_idx in range(tensor.shape[0]):
        valid_positions = torch.nonzero(prefix_mask[batch_idx], as_tuple=False).flatten()
        packed[batch_idx, :, : valid_positions.numel(), :] = tensor[batch_idx, :, valid_positions, :]
    return packed


def _packed_cache(cache, prefix_mask: torch.BoolTensor):
    return tuple((_pack_cache_tensor(key, prefix_mask), _pack_cache_tensor(value, prefix_mask)) for key, value in cache)

def _tiny_model(**overrides) -> ProgressExpertPytorch:
    torch.manual_seed(7)
    config = _TinyConfig(**overrides)
    return ProgressExpertPytorch(config)


def test_initialize_from_action_expert_copies_uniform_layers_without_sharing_storage():
    model = _tiny_model()
    source_layers = nn.ModuleList(
        modeling_gemma.GemmaDecoderLayer(model.hf_config, layer_idx=index) for index in range(4)
    )
    source_norm = modeling_gemma.GemmaRMSNorm(
        model.width,
        eps=model.hf_config.rms_norm_eps,
        cond_dim=model.hf_config.adarms_cond_dim,
    )
    source_action_in = copy.deepcopy(model.action_in_proj)
    source_time_in = copy.deepcopy(model.time_mlp_in)
    source_time_out = copy.deepcopy(model.time_mlp_out)
    source_action_out = copy.deepcopy(model.action_out_proj)
    with torch.no_grad():
        for source_index, source in enumerate(source_layers):
            for parameter_index, parameter in enumerate(source.parameters()):
                parameter.fill_(source_index + parameter_index / 100.0)

    model.initialize_from_action_expert(
        action_layers=source_layers,
        action_norm=source_norm,
        action_in_proj=source_action_in,
        time_mlp_in=source_time_in,
        time_mlp_out=source_time_out,
        action_out_proj=source_action_out,
    )

    assert model.layer_mapping == (0, 3)
    for target, source in zip(model.layers, (source_layers[0], source_layers[3]), strict=True):
        for target_module, source_module in (
            (target.self_attn, source.self_attn),
            (target.mlp, source.mlp),
            (target.input_layernorm.dense, source.input_layernorm.dense),
            (target.post_attention_layernorm.dense, source.post_attention_layernorm.dense),
        ):
            for name, parameter in target_module.state_dict().items():
                source_parameter = source_module.state_dict()[name]
                torch.testing.assert_close(parameter, source_parameter)
                assert parameter.data_ptr() != source_parameter.data_ptr()
    for target, source in (
        (model.norm.dense, source_norm.dense),
        (model.action_in_proj, source_action_in),
        (model.time_mlp_in, source_time_in),
        (model.time_mlp_out, source_time_out),
        (model.action_out_proj, source_action_out),
    ):
        for name, parameter in target.state_dict().items():
            source_parameter = source.state_dict()[name]
            torch.testing.assert_close(parameter, source_parameter)
            assert parameter.data_ptr() != source_parameter.data_ptr()


def test_layer_mapping_covers_endpoints_and_rejects_invalid_depth():
    assert default_progress_depth(18) == 6
    assert make_layer_mapping(18, 6) == (0, 3, 7, 10, 14, 17)
    assert make_layer_mapping(4, 4) == (0, 1, 2, 3)
    assert make_layer_mapping(1, 1) == (0,)
    with pytest.raises(ValueError, match="cover first and last"):
        make_layer_mapping(4, 1)

    with pytest.raises(ValueError, match="progress_depth"):
        make_layer_mapping(3, 4)
    with pytest.raises(ValueError, match="strictly increasing"):
        make_layer_mapping(3, 0)


def test_prefix_view_selects_detached_readonly_layers_without_sharing_storage():
    source = _legacy_cache(requires_grad=True)
    before = [tensor.detach().clone() for tensor in iter_cache_tensors(source)]

    view = PrefixKVView.from_cache(source, layer_indices=(0, 3), prefix_mask=torch.ones(2, 5, dtype=torch.bool))

    assert len(view.layers) == 2
    assert view.valid_lengths.tolist() == [5, 5]
    assert all(tensor.requires_grad is False for layer in view.layers for tensor in layer)
    for actual, expected in zip(iter_cache_tensors(source), before, strict=True):
        torch.testing.assert_close(actual, expected)
        assert actual.requires_grad is True
    for view_tensor, source_tensor in zip((view.layers[0][0], view.layers[0][1]), source[0], strict=True):
        assert view_tensor.data_ptr() != source_tensor.data_ptr()


def test_prefix_view_packs_non_left_aligned_valid_tokens():
    prefix_mask = torch.tensor([[True, False, True, False, True], [False, True, True, False, False]])
    source = _legacy_cache(requires_grad=True)

    view = PrefixKVView.from_cache(source, layer_indices=(0,), prefix_mask=prefix_mask)

    expected_key, expected_value = _packed_cache(source[:1], prefix_mask)[0]
    torch.testing.assert_close(view.layers[0][0], expected_key)
    torch.testing.assert_close(view.layers[0][1], expected_value)
    assert all(tensor.requires_grad is False for tensor in view.layers[0])


def test_non_left_aligned_prefix_matches_equivalent_packed_prefix():
    model = _tiny_model()
    prefix_mask = torch.tensor([[True, False, True, False, True], [False, True, True, False, False]])
    source = _legacy_cache()
    packed_source = _packed_cache(source, prefix_mask)
    packed_mask = torch.arange(3)[None, :] < prefix_mask.long().sum(dim=-1)[:, None]
    noisy_actions = torch.randn(2, 3, 2)
    memory_token = torch.randn(2, 1, model.width)
    timestep = torch.tensor([0.25, 0.75])

    non_left_output = model(
        PrefixKVView.from_cache(source, layer_indices=model.layer_mapping, prefix_mask=prefix_mask),
        prefix_mask,
        memory_token,
        noisy_actions,
        timestep,
    )
    packed_output = model(
        PrefixKVView.from_cache(packed_source, layer_indices=model.layer_mapping, prefix_mask=packed_mask),
        packed_mask,
        memory_token,
        noisy_actions,
        timestep,
    )

    torch.testing.assert_close(non_left_output, packed_output, rtol=0, atol=1e-6)


def test_progress_attention_projections_are_bias_free():
    model = _tiny_model()

    for layer in model.layers:
        assert layer.self_attn.config.attention_bias is False
        assert layer.self_attn.q_proj.bias is None
        assert layer.self_attn.k_proj.bias is None
        assert layer.self_attn.v_proj.bias is None
        assert layer.self_attn.o_proj.bias is None


def test_rope_matches_gemma_rotary_embedding_for_multiple_positions():
    model = _tiny_model()
    reference = torch.randn(2, 3, model.width)
    position_ids = torch.tensor([[0, 2, 5], [1, 3, 7]])
    expected_rotary = modeling_gemma.GemmaRotaryEmbedding(model.hf_config)

    expected = expected_rotary(reference, position_ids)
    actual = model._position_embeddings(reference, position_ids)

    assert isinstance(model.rotary_emb, modeling_gemma.GemmaRotaryEmbedding)
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_prefix_view_supports_transformers_cache_structure():
    prefix_mask = torch.ones(2, 5, dtype=torch.bool)
    view = PrefixKVView.from_cache(_dynamic_cache(), layer_indices=(0, 3), prefix_mask=prefix_mask)

    assert len(view.layers) == 2
    assert view.valid_lengths.tolist() == [5, 5]
    assert all(tensor.requires_grad is False for layer in view.layers for tensor in layer)
    assert view.layers[0][0].shape == (2, 1, 5, 16)


def test_forward_shape_dtype_memory_conditioning_and_token_lengths():
    model = _tiny_model()
    prefix_mask = torch.tensor([[True, True, False, False, False], [True, True, True, False, False]])
    prefix_cache = PrefixKVView.from_cache(_legacy_cache(), layer_indices=model.layer_mapping, prefix_mask=prefix_mask)
    noisy_actions = torch.randn(2, 3, 2, dtype=torch.float64)
    memory_a = torch.randn(2, 1, model.width)
    memory_b = memory_a + 0.5
    timestep = torch.tensor([0.2, 0.8], dtype=torch.float32)

    out_a = model(prefix_cache, prefix_mask, memory_a, noisy_actions, timestep)
    out_b = model(prefix_cache, prefix_mask, memory_b, noisy_actions, timestep)

    assert out_a.shape == (2, 3, 2)
    assert out_a.dtype == noisy_actions.dtype
    assert not torch.allclose(out_a, out_b)
    assert model.last_attention_shapes == [
        (noisy_actions.shape[1], int(prefix_mask.sum(dim=-1).max().item()) + 1 + noisy_actions.shape[1]),
        (noisy_actions.shape[1], int(prefix_mask.sum(dim=-1).max().item()) + 1 + noisy_actions.shape[1]),
    ]


def test_prefix_padding_is_not_visible_to_action_queries():
    model = _tiny_model()
    prefix_mask = torch.tensor([[True, True, False, False, False], [True, True, True, False, False]])
    cache_a = _legacy_cache(prefix_len=5)
    cache_b = tuple((key.clone(), value.clone()) for key, value in cache_a)
    padding = ~prefix_mask[:, None, :, None]
    for key, value in cache_b:
        key.masked_fill_(padding, 10000.0)
        value.masked_fill_(padding, -10000.0)
    view_a = PrefixKVView.from_cache(cache_a, layer_indices=model.layer_mapping, prefix_mask=prefix_mask)
    view_b = PrefixKVView.from_cache(cache_b, layer_indices=model.layer_mapping, prefix_mask=prefix_mask)
    noisy_actions = torch.randn(2, 3, 2)
    memory_token = torch.randn(2, 1, model.width)
    timestep = torch.tensor([0.3, 0.7])

    torch.testing.assert_close(
        model(view_a, prefix_mask, memory_token, noisy_actions, timestep),
        model(view_b, prefix_mask, memory_token, noisy_actions, timestep),
        rtol=0,
        atol=1e-6,
    )


def test_gradients_reach_progress_memory_actions_but_not_prefix_or_action_expert():
    model = _tiny_model()
    prefix_source = _legacy_cache(requires_grad=True)
    prefix_mask = torch.ones(2, 5, dtype=torch.bool)
    prefix_cache = PrefixKVView.from_cache(prefix_source, layer_indices=model.layer_mapping, prefix_mask=prefix_mask)
    memory_token = torch.randn(2, 1, model.width, requires_grad=True)
    noisy_actions = torch.randn(2, 3, 2, requires_grad=True)
    timestep = torch.tensor([0.1, 0.9])
    source_before = [tensor.detach().clone() for tensor in iter_cache_tensors(prefix_source)]
    view_before = [tensor.detach().clone() for tensor in iter_cache_tensors(prefix_cache.layers)]

    loss = model(prefix_cache, prefix_mask, memory_token, noisy_actions, timestep).square().mean()
    loss.backward()

    assert memory_token.grad is not None and memory_token.grad.abs().sum() > 0
    assert noisy_actions.grad is not None and noisy_actions.grad.abs().sum() > 0
    for actual, expected in zip(iter_cache_tensors(prefix_source), source_before, strict=True):
        torch.testing.assert_close(actual, expected)
        assert actual.grad is None
    for actual, expected in zip(iter_cache_tensors(prefix_cache.layers), view_before, strict=True):
        torch.testing.assert_close(actual, expected)
        assert actual.grad is None
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0 for parameter in model.parameters())
    assert not any("action_expert" in name for name, _ in model.named_parameters())
    assert not hasattr(model, "action_expert")


def test_input_shape_errors_are_clear():
    model = _tiny_model()
    prefix_mask = torch.ones(2, 5, dtype=torch.bool)
    prefix_cache = PrefixKVView.from_cache(_legacy_cache(), layer_indices=model.layer_mapping, prefix_mask=prefix_mask)
    memory_token = torch.randn(2, 1, model.width)
    noisy_actions = torch.randn(2, 3, 2)
    timestep = torch.ones(2)

    with pytest.raises(ValueError, match="memory_token must have shape"):
        model(prefix_cache, prefix_mask, memory_token[:, 0], noisy_actions, timestep)
    with pytest.raises(ValueError, match="noisy_actions must have shape"):
        model(prefix_cache, prefix_mask, memory_token, noisy_actions.transpose(1, 2), timestep)
    with pytest.raises(ValueError, match="timestep must have shape"):
        model(prefix_cache, prefix_mask, memory_token, noisy_actions, timestep[:, None])
    with pytest.raises(ValueError, match="prefix contains no valid token"):
        model(prefix_cache, torch.zeros_like(prefix_mask), memory_token, noisy_actions, timestep)
