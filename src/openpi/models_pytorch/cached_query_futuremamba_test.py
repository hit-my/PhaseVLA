from __future__ import annotations

import dataclasses

import pytest
import torch
from torch import nn

from openpi.models.model import Observation
from openpi.models_pytorch.futuremamba import FutureMambaPytorch
from openpi.models_pytorch.futuremamba_config import FutureMambaPytorchConfig, MambaMemoryConfig
from openpi.training.cached_query_data_loader import CachedQueryBatch
from openpi.training.episode_data_loader import TorchEpisodeBatch


def _case():
    torch.manual_seed(817)
    config = FutureMambaPytorchConfig(
        action_expert_variant="dummy",
        paligemma_variant="dummy",
        action_dim=2,
        action_horizon=3,
        execution_horizon=3,
        action_history_chunk_size=2,
        memory_backend="none",
        memory=MambaMemoryConfig(d_model=8, depth=1, d_state=4, expand=2, headdim=4),
        progress_depth=2,
        progress_layer_mapping=(0, 3),
        frozen_prefix_microbatch_size=2,
        dtype="float32",
        discrete_state_input=False,
    )
    model = FutureMambaPytorch(config, base=nn.Linear(2, 2))
    with torch.no_grad():
        model.futuremamba.empty_history.copy_(torch.linspace(-0.7, 0.9, 8))
    count, horizon, dim = 5, 3, 2
    actions = torch.randn(count, horizon, dim)
    action_mask = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    prefix_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    keys = torch.randn(count, 2, 1, 4, 16)
    values = torch.randn_like(keys)
    executed = torch.zeros(1, count, 1, dim)
    executed[0, 1:, 0] = actions[:-1, 0]
    executed_mask = torch.tensor([[[0], [1], [1], [1], [1]]], dtype=torch.bool)
    episode = TorchEpisodeBatch(
        observation=Observation(images={}, image_masks={}, state=torch.zeros(1, count, dim)),
        actions=actions[None],
        action_mask=action_mask[None],
        executed_actions=executed,
        executed_action_mask=executed_mask,
        query_mask=torch.ones(1, count, dtype=torch.bool),
        reset_mask=torch.tensor([[1, 0, 0, 0, 0]], dtype=torch.bool),
        episode_index=torch.tensor([0]),
        train_query_mask=torch.ones(1, count, dtype=torch.bool),
        conditioning_cache={
            "prefix_mask": prefix_mask[None],
            "action_expert_keys": keys[None],
            "action_expert_values": values[None],
        },
    )
    cached = CachedQueryBatch(
        actions=actions,
        action_mask=action_mask,
        has_history=torch.tensor([0, 1, 1, 1, 1], dtype=torch.bool),
        prefix_mask=prefix_mask,
        action_expert_keys=keys,
        action_expert_values=values,
    )
    noise = torch.randn_like(actions)
    time = torch.tensor([0.61, 0.73, 0.82, 0.94, 0.99])
    return model, episode, cached, noise, time


@pytest.mark.parametrize("microbatch_size", [2, 5])
def test_cached_queries_preserve_episode_loss_and_plugin_gradients(microbatch_size):
    model, episode, cached, noise, time = _case()
    reference = model.compute_episode_loss(episode, noise=noise[None], time=time[None])["loss"]
    reference.backward()
    gradients = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    assert gradients["futuremamba.empty_history"].abs().sum() > 0
    assert all(p.grad is None for p in model.base.parameters())
    model.zero_grad(set_to_none=True)
    total = torch.zeros(())
    for start in range(0, len(noise), microbatch_size):
        stop = min(start + microbatch_size, len(noise))
        result = model.compute_cached_query_loss(
            cached.slice(start, stop), noise=noise[start:stop], time=time[start:stop]
        )
        weighted = result["loss"] * ((stop - start) / len(noise))
        weighted.backward()
        total += weighted.detach()
    torch.testing.assert_close(total, reference.detach(), rtol=2e-5, atol=2e-6)
    actual = {name: p.grad for name, p in model.named_parameters() if p.grad is not None}
    assert actual.keys() == gradients.keys()
    for name, expected in gradients.items():
        torch.testing.assert_close(actual[name], expected, rtol=3e-4, atol=3e-6, msg=name)
    assert all(p.grad is None for p in model.base.parameters())


def test_cached_queries_ignore_padded_actions_and_noise():
    model, _, cached, noise, time = _case()
    expected = model.compute_cached_query_loss(cached, noise=noise, time=time)["loss"]
    actions = cached.actions.clone()
    changed_noise = noise.clone()
    actions[~cached.action_mask] = 10000
    changed_noise[~cached.action_mask] = -10000
    changed = dataclasses.replace(cached, actions=actions)
    actual = model.compute_cached_query_loss(changed, noise=changed_noise, time=time)["loss"]
    torch.testing.assert_close(actual, expected)


def test_cached_batch_slicing_and_transfer_preserve_bfloat16_prefix():
    _, _, cached, _, _ = _case()
    cached = dataclasses.replace(
        cached,
        action_expert_keys=cached.action_expert_keys.bfloat16(),
        action_expert_values=cached.action_expert_values.bfloat16(),
    )
    part = cached.slice(1, 4).to(torch.device("cpu"))
    assert part.action_expert_keys.dtype == torch.bfloat16
    assert part.action_expert_values.dtype == torch.bfloat16
    torch.testing.assert_close(part.actions, cached.actions[1:4])
    torch.testing.assert_close(part.has_history, cached.has_history[1:4])
    torch.testing.assert_close(part.prefix_mask, cached.prefix_mask[1:4])
