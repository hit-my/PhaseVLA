from __future__ import annotations

import importlib.util

import pytest
import torch

from openpi.models_pytorch.futuremamba_config import MambaMemoryConfig
from openpi.models_pytorch.mamba_memory import (
    FrameStackMemoryBackend,
    GRUMemoryBackend,
    LSTMMemoryBackend,
    Mamba2MemoryBackend,
    MemorySnapshot,
    NoMemoryBackend,
)


def _small_config() -> MambaMemoryConfig:
    return MambaMemoryConfig(d_model=8, depth=2, d_state=4, expand=2, headdim=4, ngroups=1, d_conv=3, use_mem_eff_path=False)


def _mamba_available() -> bool:
    return importlib.util.find_spec("mamba_ssm") is not None


def _mamba_cuda_available() -> bool:
    return _mamba_available() and torch.cuda.is_available()


def _assert_state_close(actual: MemorySnapshot, expected: MemorySnapshot, *, rtol: float = 1e-5, atol: float = 1e-5) -> None:
    assert actual.backend_id == expected.backend_id
    assert actual.state_schema_version == expected.state_schema_version
    assert actual.batch_size == expected.batch_size
    assert len(actual.layers) == len(expected.layers)
    for actual_layer, expected_layer in zip(actual.layers, expected.layers, strict=True):
        assert len(actual_layer) == len(expected_layer)
        for actual_tensor, expected_tensor in zip(actual_layer, expected_layer, strict=True):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=rtol, atol=atol)


def _assert_state_row_close(
    actual: MemorySnapshot,
    actual_row: int,
    expected: MemorySnapshot,
    expected_row: int,
    *,
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> None:
    assert actual.backend_id == expected.backend_id
    assert actual.state_schema_version == expected.state_schema_version
    for actual_layer, expected_layer in zip(actual.layers, expected.layers, strict=True):
        for actual_tensor, expected_tensor in zip(actual_layer, expected_layer, strict=True):
            torch.testing.assert_close(
                actual_tensor[actual_row], expected_tensor[expected_row], rtol=rtol, atol=atol
            )


def _snapshot_bytes(snapshot: MemorySnapshot) -> int:
    return sum(tensor.numel() * tensor.element_size() for layer in snapshot.layers for tensor in layer)


def _backend_factories():
    config = _small_config()
    return [
        lambda: GRUMemoryBackend(config),
        lambda: LSTMMemoryBackend(config),
        lambda: FrameStackMemoryBackend(config, frame_stack_window=3),
        lambda: NoMemoryBackend(config),
    ]


def _stateful_backend_factories():
    config = _small_config()
    return [
        pytest.param(lambda: GRUMemoryBackend(config), id="gru"),
        pytest.param(lambda: LSTMMemoryBackend(config), id="lstm"),
        pytest.param(lambda: FrameStackMemoryBackend(config, frame_stack_window=3), id="frame_stack"),
    ]


def _sequence_bptt_backend_cases():
    config = _small_config()
    return [
        pytest.param(lambda: GRUMemoryBackend(config), 3, 0, id="gru"),
        pytest.param(lambda: LSTMMemoryBackend(config), 3, 0, id="lstm"),
        pytest.param(lambda: FrameStackMemoryBackend(config, frame_stack_window=3), 2, 0, id="frame_stack"),
    ]


def test_module_imports_without_official_mamba_dependency():
    assert MemorySnapshot.__name__ == "MemorySnapshot"
    if not _mamba_available():
        with pytest.raises(ImportError, match="mamba_ssm.*official Mamba-2"):
            Mamba2MemoryBackend(_small_config())


@pytest.mark.skipif(not _mamba_cuda_available(), reason="official Mamba-2 CUDA runtime is unavailable")
def test_mamba2_sequence_matches_steps():
    torch.manual_seed(0)
    model = Mamba2MemoryBackend(_small_config()).to(device=torch.device("cuda"))
    x = torch.randn(2, 7, _small_config().d_model, device=torch.device("cuda"))

    sequence_y, sequence_state = model.forward_sequence(x)
    state = model.initial_state(2, device=x.device, dtype=x.dtype)
    step_y = []
    for query in range(x.shape[1]):
        y, state = model.step(x[:, query], state)
        step_y.append(y)

    torch.testing.assert_close(sequence_y, torch.stack(step_y, dim=1), rtol=1e-4, atol=1e-4)
    _assert_state_close(sequence_state, state, rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(not _mamba_cuda_available(), reason="official Mamba-2 CUDA runtime is unavailable")
def test_mamba2_state_shape_and_bytes_do_not_depend_on_sequence_length():
    torch.manual_seed(0)
    model = Mamba2MemoryBackend(_small_config()).to(device=torch.device("cuda"))
    short_state = model.forward_sequence(
        torch.randn(2, 2, _small_config().d_model, device=torch.device("cuda"))
    )[1]
    long_state = model.forward_sequence(
        torch.randn(2, 16, _small_config().d_model, device=torch.device("cuda"))
    )[1]

    assert [[tuple(t.shape) for t in layer] for layer in short_state.layers] == [
        [tuple(t.shape) for t in layer] for layer in long_state.layers
    ]
    assert _snapshot_bytes(short_state) == _snapshot_bytes(long_state)


@pytest.mark.parametrize("factory, loss_query, source_query", _sequence_bptt_backend_cases())
def test_forward_sequence_preserves_recurrent_gradients(factory, loss_query, source_query):
    torch.manual_seed(3)
    model = factory()
    x = torch.randn(2, 4, _small_config().d_model, requires_grad=True)

    y, _ = model.forward_sequence(x)
    y[:, loss_query].square().sum().backward()

    assert x.grad is not None
    assert torch.any(x.grad[:, source_query].abs() > 0)


@pytest.mark.parametrize("factory", _stateful_backend_factories())
def test_step_accepts_float16_inputs_and_state_on_float32_modules(factory):
    torch.manual_seed(4)
    model = factory()
    state = model.initial_state(2, device=torch.device("cpu"), dtype=torch.float16)
    x = torch.randn(2, _small_config().d_model, dtype=torch.float16)

    y, next_state = model.step(x, state)

    assert y.dtype == torch.float16
    for layer in next_state.layers:
        for tensor in layer:
            assert tensor.dtype == torch.float16


@pytest.mark.parametrize("factory", _stateful_backend_factories())
def test_snapshot_restore_preserves_float16_state_dtype(factory):
    model = factory()
    state = model.initial_state(2, device=torch.device("cpu"), dtype=torch.float16)

    snapshot = model.snapshot(state)
    restored = model.restore(snapshot, batch_size=2, device=torch.device("cpu"))

    for snapshot_layer, restored_layer in zip(snapshot.layers, restored.layers, strict=True):
        for snapshot_tensor, restored_tensor in zip(snapshot_layer, restored_layer, strict=True):
            assert restored_tensor.dtype == torch.float16
            assert restored_tensor.data_ptr() != snapshot_tensor.data_ptr()

@pytest.mark.parametrize("factory", _backend_factories())
def test_forward_sequence_rejects_non_right_padding_and_zero_valid_queries(factory):
    model = factory()
    config = _small_config()
    x = torch.randn(2, 3, config.d_model)

    with pytest.raises(ValueError, match="right padding"):
        model.forward_sequence(x, query_mask=torch.tensor([[True, False, True], [True, True, False]]))
    with pytest.raises(ValueError, match="zero valid"):
        model.forward_sequence(x, query_mask=torch.tensor([[False, False, False], [True, True, False]]))


@pytest.mark.parametrize("factory", _backend_factories())
def test_padding_queries_are_zero_and_do_not_advance_state(factory):
    torch.manual_seed(1)
    model = factory()
    config = _small_config()
    x = torch.randn(2, 3, config.d_model)
    mask = torch.tensor([[True, True, False], [True, True, True]])

    y, state = model.forward_sequence(x, query_mask=mask)
    expected_row0 = model.forward_sequence(x[:1, :2])[1]
    expected_row1 = model.forward_sequence(x[1:2, :3])[1]

    assert torch.count_nonzero(y[0, 2]).item() == 0
    _assert_state_row_close(state, 0, expected_row0, 0)
    _assert_state_row_close(state, 1, expected_row1, 0)


@pytest.mark.parametrize("factory", _backend_factories())
def test_step_partial_reset_and_snapshot_restore_contract(factory):
    torch.manual_seed(2)
    model = factory()
    config = _small_config()
    x0 = torch.randn(2, config.d_model)
    x1 = torch.randn(2, config.d_model)

    state = model.initial_state(2, device=x0.device, dtype=x0.dtype)
    _, advanced = model.step(x0, state)
    reset = model.reset(advanced, torch.tensor([True, False]))
    zero = model.initial_state(2, device=x0.device, dtype=x0.dtype)
    _assert_state_row_close(reset, 0, zero, 0)
    _assert_state_row_close(reset, 1, advanced, 1)

    snapshot = model.snapshot(reset)
    assert snapshot == reset
    assert all(tensor.data_ptr() != cloned.data_ptr() for layer, cloned_layer in zip(reset.layers, snapshot.layers, strict=True) for tensor, cloned in zip(layer, cloned_layer, strict=True))

    _, changed = model.step(x1, reset)
    restored = model.restore(snapshot, batch_size=2, device=x0.device)
    _assert_state_close(restored, reset)
    first_after_restore, _ = model.step(x1, restored)
    if reset.layers:
        first_without_restore, _ = model.step(x1, changed)
        assert not torch.allclose(first_after_restore, first_without_restore)


@pytest.mark.parametrize("factory", _backend_factories())
def test_state_restore_rejects_backend_schema_batch_and_shape(factory):
    model = factory()
    config = _small_config()
    state = model.initial_state(2, device=torch.device("cpu"), dtype=torch.float32)

    wrong_backend = MemorySnapshot("other", state.state_schema_version, state.batch_size, state.layers)
    with pytest.raises(ValueError, match="backend"):
        model.restore(wrong_backend, batch_size=2, device=torch.device("cpu"))

    wrong_schema = MemorySnapshot(state.backend_id, state.state_schema_version + 1, state.batch_size, state.layers)
    with pytest.raises(ValueError, match="schema"):
        model.restore(wrong_schema, batch_size=2, device=torch.device("cpu"))

    wrong_batch = MemorySnapshot(state.backend_id, state.state_schema_version, 3, state.layers)
    with pytest.raises(ValueError, match="batch"):
        model.restore(wrong_batch, batch_size=2, device=torch.device("cpu"))

    if state.layers:
        first_layer = list(state.layers[0])
        first_layer[0] = first_layer[0][:1]
        wrong_shape_layers = (tuple(first_layer), *state.layers[1:])
        wrong_shape = MemorySnapshot(state.backend_id, state.state_schema_version, state.batch_size, wrong_shape_layers)
        with pytest.raises(ValueError, match="shape"):
            model.restore(wrong_shape, batch_size=2, device=torch.device("cpu"))



@pytest.mark.parametrize("factory", _backend_factories())
def test_initial_state_step_shape_dtype_and_metadata(factory):
    model = factory()
    config = _small_config()
    state = model.initial_state(3, device=torch.device("cpu"), dtype=torch.float32)
    x = torch.randn(3, config.d_model)

    assert state.backend_id == model.backend_id
    assert state.state_schema_version == model.state_schema_version
    assert state.batch_size == 3
    _assert_state_close(state, model.snapshot(state))
    y, next_state = model.step(x, state)

    assert y.shape == (3, config.d_model)
    assert y.dtype == x.dtype
    assert next_state.batch_size == 3
    assert isinstance(model.parameter_count, int)
    assert isinstance(model.mamba_parameter_count, int)
    assert isinstance(model.parameter_error, float)
    assert isinstance(model.parameter_matched, bool)
    if model.parameter_error > 0.05:
        assert model.parameter_matched is False


def test_no_memory_outputs_zero_and_has_empty_state():
    model = NoMemoryBackend(_small_config())
    x = torch.randn(2, 4, _small_config().d_model)

    y, state = model.forward_sequence(x)

    assert torch.count_nonzero(y).item() == 0
    assert state.layers == ()
    assert model.parameter_count == 0
    assert model.parameter_matched is False

@pytest.mark.skipif(not _mamba_available(), reason="official Mamba-2 runtime is unavailable")
def test_mamba2_step_runs_on_cpu_without_cuda_context():
    torch.manual_seed(5)
    config = _small_config()
    model = Mamba2MemoryBackend(config)
    state = model.initial_state(1, device=torch.device("cpu"), dtype=torch.float32)
    x = torch.randn(1, config.d_model)

    output, next_state = model.step(x, state)

    assert output.device.type == "cpu"
    assert output.shape == x.shape
    assert next_state.batch_size == 1
    assert all(tensor.device.type == "cpu" for layer in next_state.layers for tensor in layer)
