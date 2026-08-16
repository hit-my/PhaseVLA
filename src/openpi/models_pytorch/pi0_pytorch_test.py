import dataclasses
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.cache_utils import DynamicCache

from openpi.models_pytorch import gemma_pytorch
from openpi.models_pytorch.gemma_pytorch import (
    PaliGemmaWithExpertModel,
    clone_cache,
    clone_selected_prefix_cache,
    detach_cache,
    iter_cache_tensors,
)
from openpi.models_pytorch.pi0_pytorch import FrozenPrefix, PI0Pytorch
from openpi.models_pytorch import preprocessing_pytorch


@dataclasses.dataclass
class _Observation:
    images: dict[str, torch.Tensor]
    image_masks: dict[str, torch.Tensor]
    tokenized_prompt: torch.Tensor
    tokenized_prompt_mask: torch.Tensor
    state: torch.Tensor
    token_ar_mask: torch.Tensor | None = None
    token_loss_mask: torch.Tensor | None = None


@dataclasses.dataclass
class _Config:
    action_horizon: int = 3
    action_dim: int = 2



def test_preprocess_observation_outputs_channels_first_for_hwc_and_chw_images():
    batch_size = 2
    common = {
        "image_masks": {"base_0_rgb": torch.ones(batch_size, dtype=torch.bool)},
        "tokenized_prompt": torch.zeros(batch_size, 1, dtype=torch.long),
        "tokenized_prompt_mask": torch.ones(batch_size, 1, dtype=torch.bool),
        "state": torch.zeros(batch_size, 2),
    }
    hwc = _Observation(
        images={"base_0_rgb": torch.zeros(batch_size, 224, 224, 3)},
        **common,
    )
    chw = _Observation(
        images={"base_0_rgb": torch.zeros(batch_size, 3, 224, 224)},
        **common,
    )

    processed_hwc = preprocessing_pytorch.preprocess_observation_pytorch(
        hwc, train=False, image_keys=("base_0_rgb",)
    )
    processed_chw = preprocessing_pytorch.preprocess_observation_pytorch(
        chw, train=False, image_keys=("base_0_rgb",)
    )

    assert processed_hwc.images["base_0_rgb"].shape == (batch_size, 3, 224, 224)
    assert processed_chw.images["base_0_rgb"].shape == (batch_size, 3, 224, 224)

def test_paligemma_with_expert_drops_unused_action_language_head(monkeypatch):
    class FakePaliGemma(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

    class FakeActionDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(2, 2)

    class FakeGemmaForCausalLM(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.model = FakeActionDecoder()
            self.lm_head = nn.Linear(2, 2, bias=False)

    paligemma_config = SimpleNamespace(
        text_config=SimpleNamespace(),
        vision_config=SimpleNamespace(),
    )
    monkeypatch.setattr(
        gemma_pytorch,
        "CONFIG_MAPPING",
        {
            "paligemma": lambda: paligemma_config,
            "gemma": lambda **kwargs: SimpleNamespace(**kwargs),
        },
    )
    monkeypatch.setattr(gemma_pytorch, "PaliGemmaForConditionalGeneration", FakePaliGemma)
    monkeypatch.setattr(gemma_pytorch, "GemmaForCausalLM", FakeGemmaForCausalLM)
    vlm_config = SimpleNamespace(width=2, mlp_dim=4, num_heads=1, head_dim=2, depth=1, num_kv_heads=1)
    action_config = SimpleNamespace(width=2, mlp_dim=4, num_heads=1, head_dim=2, depth=1, num_kv_heads=1)

    model = PaliGemmaWithExpertModel(vlm_config, action_config, precision="float32")

    assert model.gemma_expert.model.embed_tokens is None
    assert model.gemma_expert.lm_head is None
    assert all("gemma_expert.lm_head" not in name for name, _ in model.named_parameters())

class _TinyPrefixModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prefix_forward_calls = 0
        self.paligemma = type(
            "PaliGemmaStub",
            (),
            {"language_model": type("LanguageModelStub", (), {"config": type("ConfigStub", (), {})()})()},
        )()
        self.gemma_expert = type(
            "GemmaExpertStub",
            (),
            {"model": type("ExpertModelStub", (), {"config": type("ConfigStub", (), {})()})()},
        )()

    def forward(self, *, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, adarms_cond=None):
        del attention_mask, position_ids, past_key_values, adarms_cond
        if inputs_embeds[1] is None:
            self.prefix_forward_calls += 1
            hidden = inputs_embeds[0] + 100.0
            cache = (
                (
                    torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4).requires_grad_(),
                    (torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4) + 50).requires_grad_(),
                ),
            )
            return [hidden, None], cache
        suffix = inputs_embeds[1]
        return [None, suffix + 0.25], None


class _RecordingSuffixModel(_TinyPrefixModel):
    def __init__(self):
        super().__init__()
        self.suffix_caches = []

    def forward(self, *, attention_mask, position_ids, past_key_values, inputs_embeds, use_cache, adarms_cond=None):
        if inputs_embeds[1] is None:
            return super().forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                adarms_cond=adarms_cond,
            )
        self.suffix_caches.append(past_key_values)
        for layer_idx in range(len(past_key_values.key_cache)):
            past_key_values.update(
                torch.full((1, 1, 1, 2), 100.0 + layer_idx),
                torch.full((1, 1, 1, 2), 200.0 + layer_idx),
                layer_idx,
            )
        return super().forward(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            adarms_cond=adarms_cond,
        )


def dynamic_cache_from_layers(*layer_lengths):
    legacy_cache = tuple(
        (
            torch.arange(length * 2, dtype=torch.float32).reshape(1, 1, length, 2) + layer_idx * 100,
            torch.arange(length * 2, dtype=torch.float32).reshape(1, 1, length, 2) + layer_idx * 100 + 50,
        )
        for layer_idx, length in enumerate(layer_lengths)
    )
    return DynamicCache.from_legacy_cache(legacy_cache)


def dynamic_cache_lengths(cache):
    return [cache.get_seq_length(layer_idx) for layer_idx in range(len(cache.key_cache))]

@pytest.fixture
def observation():
    batch = 2
    return _Observation(
        images={},
        image_masks={},
        tokenized_prompt=torch.zeros(batch, 1, dtype=torch.long),
        tokenized_prompt_mask=torch.ones(batch, 1, dtype=torch.bool),
        state=torch.tensor([[0.5, -0.5], [1.0, 0.25]], dtype=torch.float32),
    )


@pytest.fixture
def tiny_pi0(monkeypatch):
    model = PI0Pytorch.__new__(PI0Pytorch)
    nn.Module.__init__(model)
    model.config = _Config()
    model.pi05 = True
    model.paligemma_with_expert = _TinyPrefixModel()
    model.action_out_proj = nn.Identity()

    def preprocess(observation, *, train=True):
        del train
        return [], [], observation.tokenized_prompt, observation.tokenized_prompt_mask, observation.state

    def embed_prefix(images, img_masks, lang_tokens, lang_masks):
        del images, img_masks, lang_tokens
        hidden = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.0], [99.0, 99.0]],
                [[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
            ],
            requires_grad=True,
        )
        pad_mask = torch.tensor([[True, True, False], [True, True, True]]) & lang_masks.expand(-1, 3)
        att_mask = torch.zeros_like(pad_mask)
        return hidden, pad_mask, att_mask

    def embed_suffix(state, noisy_actions, timestep):
        del state, timestep
        pad_mask = torch.ones(noisy_actions.shape[:2], dtype=torch.bool, device=noisy_actions.device)
        att_mask = torch.zeros_like(pad_mask)
        return noisy_actions + 1.0, pad_mask, att_mask, None

    monkeypatch.setattr(model, "_preprocess_observation", preprocess)
    monkeypatch.setattr(model, "embed_prefix", embed_prefix)
    monkeypatch.setattr(model, "embed_suffix", embed_suffix)
    return model


def legacy_sample_actions(model, device, observation, noise=None, num_steps=10):
    bsize = observation.state.shape[0]
    if noise is None:
        noise = model.sample_noise((bsize, model.config.action_horizon, model.config.action_dim), device)

    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(observation, train=False)
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(images, img_masks, lang_tokens, lang_masks)
    prefix_att_2d_masks = model.make_att_2d_masks_for_test(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(prefix_att_2d_masks)
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )

    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    x_t = noise
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    while time >= -dt / 2:
        expanded_time = time.expand(bsize)
        v_t = model.denoise_step(state, prefix_pad_masks, past_key_values, x_t, expanded_time)
        x_t = x_t + dt * v_t
        time += dt
    return x_t


def test_last_valid_prefix_token_ignores_padding(tiny_pi0, observation):
    frozen = tiny_pi0.encode_frozen_prefix(observation, train=False)
    index = frozen.pad_mask.long().sum(dim=-1) - 1
    expected = frozen.hidden[torch.arange(index.numel()), index]
    torch.testing.assert_close(tiny_pi0.last_valid_prefix(frozen), expected)
def test_last_valid_prefix_ignores_noncontiguous_padding(tiny_pi0):
    frozen = FrozenPrefix(
        hidden=torch.tensor([[[10.0, 0.0], [20.0, 0.0], [30.0, 0.0]]]),
        pad_mask=torch.tensor([[True, False, True]]),
        kv_cache=((torch.zeros(1, 1, 3, 2), torch.zeros(1, 1, 3, 2)),),
    )
    torch.testing.assert_close(tiny_pi0.last_valid_prefix(frozen), torch.tensor([[30.0, 0.0]]))


def test_last_valid_prefix_rejects_empty_pad_mask(tiny_pi0):
    frozen = FrozenPrefix(
        hidden=torch.zeros(1, 2, 3),
        pad_mask=torch.zeros(1, 2, dtype=torch.bool),
        kv_cache=((torch.zeros(1, 1, 2, 3), torch.zeros(1, 1, 2, 3)),),
    )
    with pytest.raises(ValueError, match="^prefix contains no valid token$"):
        tiny_pi0.last_valid_prefix(frozen)


def test_frozen_prefix_is_detached_normal_tensor(tiny_pi0, observation):
    frozen = tiny_pi0.encode_frozen_prefix(observation, train=False)
    assert frozen.hidden.requires_grad is False
    assert frozen.hidden.is_inference() is False
    assert frozen.pad_mask.requires_grad is False
    assert all(t.requires_grad is False and not t.is_inference() for t in iter_cache_tensors(frozen.kv_cache))

def test_extract_prefix_context_is_stable_public_alias(tiny_pi0, observation):
    frozen = tiny_pi0.extract_prefix_context(observation, train=False)

    assert isinstance(frozen, FrozenPrefix)
    assert frozen.hidden.requires_grad is False
    assert frozen.hidden.is_inference() is False
    assert frozen.pad_mask.requires_grad is False
    assert all(t.requires_grad is False and not t.is_inference() for t in iter_cache_tensors(frozen.kv_cache))


def test_detach_and_clone_cache_do_not_mutate_or_share_storage():
    key0 = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2).requires_grad_()
    value0 = (torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2) + 10).requires_grad_()
    key1 = (torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2) + 20).requires_grad_()
    value1 = (torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2) + 30).requires_grad_()
    cache = ((key0, value0), (key1, value1))
    before = [tensor.detach().clone() for tensor in iter_cache_tensors(cache)]

    detached = detach_cache(cache)
    cloned = clone_selected_prefix_cache(detached, [1])

    for actual, expected in zip(iter_cache_tensors(cache), before, strict=True):
        torch.testing.assert_close(actual, expected)
        assert actual.requires_grad is True
    assert all(t.requires_grad is False and not t.is_inference() for t in iter_cache_tensors(detached))

    cloned_tensors = list(iter_cache_tensors(cloned))
    detached_tensors = list(iter_cache_tensors(detached))
    assert len(cloned_tensors) == 2
    for cloned_tensor, source_tensor in zip(cloned_tensors, detached_tensors[2:], strict=True):
        torch.testing.assert_close(cloned_tensor, source_tensor)
        assert cloned_tensor.data_ptr() != source_tensor.data_ptr()
    cloned_tensors[0].add_(1000.0)
    torch.testing.assert_close(detached_tensors[2], before[2])


def test_clone_cache_preserves_dynamic_cache_type_layers_and_storage_isolation():
    cache = dynamic_cache_from_layers(3, 4)

    cloned = clone_cache(cache)

    assert isinstance(cloned, DynamicCache)
    assert len(cloned.key_cache) == 2
    assert dynamic_cache_lengths(cloned) == [3, 4]
    for cloned_tensor, source_tensor in zip(iter_cache_tensors(cloned), iter_cache_tensors(cache), strict=True):
        torch.testing.assert_close(cloned_tensor, source_tensor)
        assert cloned_tensor.data_ptr() != source_tensor.data_ptr()

    selected = clone_selected_prefix_cache(cache, [1])
    assert isinstance(selected, DynamicCache)
    assert len(selected.key_cache) == 1
    assert dynamic_cache_lengths(selected) == [4]
    selected_source_tensors = iter_cache_tensors(cache.to_legacy_cache()[1])
    for selected_tensor, source_tensor in zip(iter_cache_tensors(selected), selected_source_tensors, strict=True):
        torch.testing.assert_close(selected_tensor, source_tensor)
        assert selected_tensor.data_ptr() != source_tensor.data_ptr()


def test_denoise_step_clones_dynamic_cache_before_suffix_forward(tiny_pi0, observation):
    tiny_pi0.paligemma_with_expert = _RecordingSuffixModel()
    frozen_cache = dynamic_cache_from_layers(4, 4)
    before_lengths = dynamic_cache_lengths(frozen_cache)
    before_tensors = [tensor.detach().clone() for tensor in iter_cache_tensors(frozen_cache)]

    prefix_pad_masks = torch.ones(observation.state.shape[0], 4, dtype=torch.bool)
    x_t = torch.zeros(observation.state.shape[0], tiny_pi0.config.action_horizon, tiny_pi0.config.action_dim)
    timestep = torch.ones(observation.state.shape[0])

    tiny_pi0.denoise_step(observation.state, prefix_pad_masks, frozen_cache, x_t, timestep)
    tiny_pi0.denoise_step(observation.state, prefix_pad_masks, frozen_cache, x_t, timestep)

    assert dynamic_cache_lengths(frozen_cache) == before_lengths
    for actual, expected in zip(iter_cache_tensors(frozen_cache), before_tensors, strict=True):
        torch.testing.assert_close(actual, expected)
    assert len(tiny_pi0.paligemma_with_expert.suffix_caches) == 2
    first_cache, second_cache = tiny_pi0.paligemma_with_expert.suffix_caches
    assert first_cache is not frozen_cache
    assert second_cache is not frozen_cache
    assert first_cache is not second_cache
    assert isinstance(first_cache, DynamicCache)
    assert isinstance(second_cache, DynamicCache)
    for recorded_cache in (first_cache, second_cache):
        assert dynamic_cache_lengths(recorded_cache) == [length + 1 for length in before_lengths]
        recorded_tensors = iter_cache_tensors(recorded_cache)
        frozen_tensors = iter_cache_tensors(frozen_cache)
        for recorded_tensor, frozen_tensor in zip(recorded_tensors, frozen_tensors, strict=False):
            assert recorded_tensor.data_ptr() != frozen_tensor.data_ptr()

def test_action_expert_velocity_preserves_frozen_cache(tiny_pi0, observation):
    frozen = tiny_pi0.extract_prefix_context(observation, train=False)
    before_tensors = [tensor.detach().clone() for tensor in iter_cache_tensors(frozen.kv_cache)]
    x_t = torch.zeros(observation.state.shape[0], tiny_pi0.config.action_horizon, tiny_pi0.config.action_dim)
    timestep = torch.ones(observation.state.shape[0])

    velocity = tiny_pi0.action_expert_velocity(observation.state, frozen.pad_mask, frozen.kv_cache, x_t, timestep)

    torch.testing.assert_close(velocity, tiny_pi0.denoise_step(observation.state, frozen.pad_mask, frozen.kv_cache, x_t, timestep))
    for actual, expected in zip(iter_cache_tensors(frozen.kv_cache), before_tensors, strict=True):
        torch.testing.assert_close(actual, expected)


def test_refactored_sampling_matches_original_fixed_noise_and_runs_prefix_once(tiny_pi0, observation):
    tiny_pi0.make_att_2d_masks_for_test = __import__(
        "openpi.models_pytorch.pi0_pytorch", fromlist=["make_att_2d_masks"]
    ).make_att_2d_masks
    noise = torch.linspace(-0.25, 0.25, steps=12, dtype=torch.float32).reshape(2, 3, 2)

    before = legacy_sample_actions(tiny_pi0, "cpu", observation, noise=noise.clone(), num_steps=10)
    calls_before_refactor = tiny_pi0.paligemma_with_expert.prefix_forward_calls
    after = tiny_pi0.sample_actions("cpu", observation, noise=noise.clone(), num_steps=10)

    torch.testing.assert_close(after, before, rtol=0, atol=1e-6)
    assert tiny_pi0.paligemma_with_expert.prefix_forward_calls - calls_before_refactor == 1
