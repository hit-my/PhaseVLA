"""Regression tests for gradients through committed action-history blocks."""
import importlib.util

import pytest
import torch

from openpi.models_pytorch.futuremamba_config import MambaMemoryConfig
from openpi.models_pytorch.mamba_memory import Mamba2MemoryBackend

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("mamba_ssm") is None, reason="pinned Mamba runtime required"
)


def _backend(device):
    return Mamba2MemoryBackend(
        MambaMemoryConfig(d_model=128, depth=2, d_state=128, d_conv=4, expand=2, headdim=64)
    ).to(device)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_final_readout_reaches_all_prior_inputs_and_preserves_state(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(9)
    model = _backend(device)
    x = torch.randn(2, 5, 128, device=device, requires_grad=True)
    state = model.initial_state(2, device=x.device, dtype=x.dtype)
    for pos in range(x.shape[1]):
        before = tuple(t.detach().clone() for layer in state.layers for t in layer)
        y, next_state = model.step(x[:, pos], state)
        assert all(torch.equal(a, b) for a, b in zip(before, (t for layer in state.layers for t in layer), strict=True))
        state = next_state
    gradient = torch.autograd.grad((y * torch.randn_like(y)).sum(), x)[0]
    assert torch.isfinite(gradient).all()
    assert torch.all(gradient.norm(dim=-1) > 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_training_forward_matches_native_inference(dtype, monkeypatch):
    torch.manual_seed(9)
    model = _backend("cuda").to(dtype)
    x = torch.randn(2, 5, 128, device="cuda", dtype=dtype)
    def unroll():
        state = model.initial_state(2, device=x.device, dtype=x.dtype)
        values = []
        for pos in range(x.shape[1]):
            y, state = model.step(x[:, pos], state)
            values.append(y)
        return torch.stack(values, dim=1)
    differentiable = unroll()
    def forbidden(*args, **kwargs):
        raise AssertionError("no_grad inference must retain the native step")
    monkeypatch.setattr(model, "_step_mixer_differentiable", forbidden)
    with torch.no_grad():
        native = unroll()
    tolerance = 2e-2 if dtype == torch.bfloat16 else 2e-5
    torch.testing.assert_close(differentiable, native, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_executed_action_history_partial_readout_keeps_committed_gradients():
    from openpi.models_pytorch.futuremamba import FutureMambaPluginPytorch
    from openpi.training import config

    torch.manual_seed(123)
    cfg = config.get_config("futuremamba_action_history_handoff04_libero_mem_bowl_t6").model
    plugin = FutureMambaPluginPytorch(cfg, include_progress_expert=False).cuda()
    actions = torch.randn(1, 61, 32, device="cuda", requires_grad=True)
    history = plugin.initial_history_state(1, actions.device, actions.dtype)
    for start in [0, 20, 40, 60]:
        chunk = actions[:, start:start + 20]
        token, history, _ = plugin.advance_history(
            history, chunk, torch.ones(chunk.shape[:2], device="cuda", dtype=torch.bool)
        )
    assert history.committed_chunks.item() == 3
    assert history.pending_mask.sum().item() == 1
    gradient = torch.autograd.grad((token.float() * torch.randn_like(token.float())).sum(), actions)[0]
    assert torch.isfinite(gradient).all()
    for start, end in [(0, 20), (20, 40), (40, 60), (60, 61)]:
        assert gradient[:, start:end].norm() > 0


def test_unsupported_group_count_fails_explicitly():
    model = _backend("cpu")
    model.layers[0].mixer.ngroups = 2
    x = torch.randn(1, 128)
    state = model.initial_state(1, device=x.device, dtype=x.dtype)
    with pytest.raises(ValueError, match="ngroups=1"):
        model.step(x, state)
