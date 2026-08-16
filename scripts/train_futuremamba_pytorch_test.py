from __future__ import annotations

import importlib.util
from pathlib import Path

import sys
import torch
from torch import nn


SCRIPT = Path(__file__).with_name("train_futuremamba_pytorch.py")
spec = importlib.util.spec_from_file_location("train_futuremamba_pytorch", SCRIPT)
assert spec is not None and spec.loader is not None
train = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = train
spec.loader.exec_module(train)


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2)
        self.futuremamba = nn.Linear(2, 2)
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def base_checksum(self):
        import hashlib

        digest = hashlib.sha256()
        for name, tensor in self.base.state_dict().items():
            digest.update(name.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    def compute_episode_loss(self, batch):
        prediction = self.futuremamba(batch)
        loss = prediction.square().mean()
        zero = loss.new_zeros(())
        return {
            "loss": loss,
            "flow_loss": loss,
            "handoff_loss": zero,
            "handoff_error": zero,
            "boundary_loss": zero,
            "boundary_error": zero,
        }


class _CountingChecksumModel(_TinyModel):
    def __init__(self):
        super().__init__()
        self.checksum_calls = 0

    def base_checksum(self):
        self.checksum_calls += 1
        return super().base_checksum()


def _metadata(model):
    return {
        "schema_version": 3,
        "base_checkpoint_uri": "fake://base",
        "base_checkpoint_checksum": model.base_checksum(),
        "base_assets_checksum": "assets",
        "robomme_policy_commit": "policy-commit",
        "robomme_benchmark_commit": "benchmark-commit",
        "robomme_dataset_checksum": "dataset-checksum",
        "robomme_task_suite": "counting",
        "train_seed": 7,
        "mamba_repo_commit": "77069de5cdb55cbe98b670889c80df211e031039",
        "memory_backend": "mamba2",
        "memory_state_schema_version": 1,
        "memory_config": {"d_model": 2},
        "progress_depth": 1,
        "progress_layer_mapping": [0],
        "handoff_ratio": 0.2,
        "num_denoise_steps": 10,
        "prediction_horizon": 1,
        "execution_horizon": 1,
        "action_expert_gradient_checkpointing": False,
        "terminal_loss_batch_fraction": 1.0,
        "terminal_loss_queries_per_episode": None,
        "frozen_prefix_microbatch_size": 2,
        "loss_weights": {"terminal": 1.0, "handoff": 0.0, "boundary": 0.0},
        "training_dtype": "float32",
        "state_dtypes": {},
        "kernel_mode": "fallback",
        "torch_version": torch.__version__,
        "triton_version": None,
        "cuda_version": torch.version.cuda,
        "gpu_name": None,
        "compute_capability": None,
    }


def _batches():
    return [
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[0.5, 1.0], [1.5, -0.5]]),
        torch.tensor([[2.0, 1.0], [-1.0, 0.5]]),
    ]


def _model_with_seed(seed: int):
    torch.manual_seed(seed)
    return _TinyModel()


def test_trainable_parameters_are_plugin_only():
    model = _model_with_seed(0)
    names, parameters = train.plugin_trainable_parameters(model)
    assert names
    assert all(name.startswith("futuremamba.") for name in names)
    assert list(parameters) == [parameter for parameter in model.futuremamba.parameters()]


def test_trainable_parameter_validation_rejects_base_parameter():
    model = _model_with_seed(0)
    model.base.weight.requires_grad_(True)
    try:
        train.plugin_trainable_parameters(model)
    except RuntimeError as error:
        assert "invalid trainable parameters" in str(error)
    else:
        raise AssertionError("base trainable parameter was accepted")


def test_training_hashes_frozen_base_only_at_checkpoint_boundaries(tmp_path: Path):
    model = _CountingChecksumModel()
    metadata = _metadata(model)
    model.checksum_calls = 0

    train.run_training(
        model,
        _batches(),
        checkpoint_root=tmp_path / "checksum",
        metadata=metadata,
        num_train_steps=3,
        save_interval=3,
    )

    assert model.checksum_calls == 1


def test_two_step_restore_to_four_matches_uninterrupted_training(tmp_path: Path):
    continuous = _model_with_seed(7)
    continuous_result = train.run_training(
        continuous,
        _batches(),
        checkpoint_root=tmp_path / "continuous",
        metadata=_metadata(continuous),
        num_train_steps=4,
        save_interval=4,
        learning_rate=0.01,
    )

    split = _model_with_seed(7)
    first = train.run_training(
        split,
        _batches(),
        checkpoint_root=tmp_path / "split",
        metadata=_metadata(split),
        num_train_steps=2,
        save_interval=2,
        learning_rate=0.01,
    )
    resumed = _model_with_seed(7)
    second = train.run_training(
        resumed,
        _batches(),
        checkpoint_root=tmp_path / "split",
        metadata=_metadata(resumed),
        num_train_steps=4,
        save_interval=2,
        learning_rate=0.01,
        resume=True,
    )

    assert first.start_step == 0 and first.end_step == 2
    assert second.start_step == 2 and second.end_step == 4
    assert second.data_iterator_step == 4
    assert first.losses + second.losses == continuous_result.losses
    assert resumed.base_checksum() == continuous.base_checksum()
    for name, tensor in resumed.futuremamba.state_dict().items():
        torch.testing.assert_close(tensor, continuous.futuremamba.state_dict()[name])


def test_training_rejects_nonfinite_loss(tmp_path: Path):
    model = _model_with_seed(2)

    def bad_loss(batch):
        del batch
        loss = model.futuremamba.weight.sum() * torch.tensor(float("nan"))
        return {"loss": loss}

    model.compute_episode_loss = bad_loss
    try:
        train.run_training(
            model,
            _batches(),
            checkpoint_root=tmp_path / "bad",
            metadata=_metadata(model),
            num_train_steps=1,
            save_interval=1,
        )
    except FloatingPointError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite loss was accepted")

def test_robomme_training_requires_explicit_episode_pickle_dir():
    config = train._config.get_config("futuremamba_robomme_mamba2")

    try:
        train.create_training_data(config, shuffle=True)
    except ValueError as error:
        assert "episode_data_dir" in str(error)
    else:
        raise AssertionError("RoboMME training silently accepted the generic dataset")

def test_robomme_training_builds_pytorch_window_loader(monkeypatch, tmp_path: Path):
    config = train._config.get_config("futuremamba_robomme_mamba2")
    config = train.dataclasses.replace(
        config,
        data=train.dataclasses.replace(config.data, episode_data_dir=str(tmp_path)),
        batch_size=1,
        num_workers=0,
    )
    dataset = ["window"]
    captured = {}

    def fake_dataset(data_config, model_config, episode_config):
        captured["data_config"] = data_config
        captured["model_config"] = model_config
        captured["episode_config"] = episode_config
        return dataset

    class FakeTorchLoader:
        def __init__(self, loaded_dataset, **kwargs):
            captured["dataset"] = loaded_dataset
            captured["kwargs"] = kwargs

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(train._robomme_episode_dataset, "create_robomme_episode_dataset", fake_dataset)
    monkeypatch.setattr(train._data_loader, "TorchDataLoader", FakeTorchLoader)

    loader = train.create_training_data(config, shuffle=True)

    assert isinstance(loader, train._data_loader.EpisodeDataLoaderImpl)
    assert captured["dataset"] is dataset
    assert captured["kwargs"]["framework"] == "pytorch"
    assert captured["kwargs"]["shuffle"] is True
    assert isinstance(captured["kwargs"]["collate_fn"], train._episode_data_loader.EpisodeCollator)

def test_episode_data_dir_cli_override_reaches_robomme_data_config(tmp_path: Path):
    args = train._parser().parse_args(
        ["futuremamba_robomme_mamba2", "--episode-data-dir", str(tmp_path / "pickles")]
    )

    config = train.apply_cli_overrides(train._config.get_config(args.config), args)

    assert config.data.episode_data_dir == str(tmp_path / "pickles")


def test_robomme_provenance_cli_overrides_checkpoint_identity(tmp_path: Path):
    args = train._parser().parse_args(
        [
            "futuremamba_robomme_mamba2",
            "--episode-data-dir",
            str(tmp_path / "pickles"),
            "--robomme-dataset-checksum",
            "sha256:dataset",
            "--robomme-task-suite",
            "PickXtimes",
            "--seed",
            "7",
        ]
    )

    config = train.apply_cli_overrides(train._config.get_config(args.config), args)

    assert config.model.robomme_dataset_checksum == "sha256:dataset"
    assert config.model.robomme_task_suite == "PickXtimes"
    assert config.seed == 7
    assert config.model.train_seed == 7


def test_seed_cli_override_and_runtime_seeding_are_reproducible():
    args = train._parser().parse_args(["futuremamba_robomme_mamba2", "--seed", "7"])
    config = train.apply_cli_overrides(train._config.get_config(args.config), args)

    assert config.seed == 7
    assert config.model.train_seed == 7
    train.seed_training_runtime(config.seed)
    first = (train.random.random(), train.np.random.random(), torch.rand(3))
    train.seed_training_runtime(config.seed)
    second = (train.random.random(), train.np.random.random(), torch.rand(3))

    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


def test_default_seed_populates_checkpoint_identity():
    args = train._parser().parse_args(["futuremamba_robomme_mamba2"])
    config = train.apply_cli_overrides(train._config.get_config(args.config), args)

    assert config.seed == 42
    assert config.model.train_seed == 42


def test_checkpoint_metadata_records_training_seed():
    model = _model_with_seed(5)
    model_config = train.dataclasses.replace(
        train._config.get_config("futuremamba_robomme_mamba2").model,
        train_seed=7,
    )

    metadata = train.build_checkpoint_metadata(model, model_config)

    assert metadata["train_seed"] == 7

def test_cli_overrides_save_and_log_intervals():
    args = train._parser().parse_args(
        [
            "futuremamba_robomme_mamba2",
            "--save-interval",
            "7",
            "--log-interval",
            "3",
        ]
    )

    config = train.apply_cli_overrides(train._config.get_config(args.config), args)

    assert config.save_interval == 7
    assert config.log_interval == 3


def test_training_logs_progress_at_configured_interval(tmp_path: Path, caplog):
    model = _model_with_seed(11)
    caplog.set_level(train.logging.INFO, logger=train.__name__)

    train.run_training(
        model,
        _batches(),
        checkpoint_root=tmp_path / "progress",
        metadata=_metadata(model),
        num_train_steps=3,
        save_interval=3,
        log_interval=2,
    )

    assert any("step=2" in record.message for record in caplog.records)

def test_training_emits_decomposed_metrics_each_step(tmp_path: Path):
    model = _model_with_seed(13)
    captured = []

    result = train.run_training(
        model,
        _batches(),
        checkpoint_root=tmp_path / "metrics",
        metadata=_metadata(model),
        num_train_steps=2,
        save_interval=2,
        metric_logger=lambda metrics, step: captured.append((step, dict(metrics))),
    )

    assert [step for step, _ in captured] == [1, 2]
    assert captured[-1][1]["train/loss"] == result.losses[-1]
    assert set(captured[-1][1]) >= {
        "train/loss",
        "train/flow_loss",
        "train/handoff_loss",
        "train/boundary_loss",
        "train/grad_norm",
        "train/learning_rate",
        "train/data_iterator_step",
    }

def test_scalar_training_metrics_preserves_sampled_time_statistics():
    metrics = train._scalar_training_metrics(
        {
            "loss": torch.tensor(0.5),
            "sample_time_mean": torch.tensor(0.92),
            "sample_time_min": torch.tensor(0.81),
            "sample_time_max": torch.tensor(0.99),
        }
    )

    assert metrics == {
        "train/loss": 0.5,
        "train/sample_time_mean": torch.tensor(0.92).item(),
        "train/sample_time_min": torch.tensor(0.81).item(),
        "train/sample_time_max": torch.tensor(0.99).item(),
    }
def test_configure_logging_enables_progress_logger():
    previous_level = train.LOGGER.level
    try:
        train.LOGGER.setLevel(train.logging.WARNING)
        train._configure_logging()
        assert train.LOGGER.isEnabledFor(train.logging.INFO)
    finally:
        train.LOGGER.setLevel(previous_level)

def test_wandb_logger_initializes_with_metadata_and_logs_steps():
    class FakeRun:
        def __init__(self):
            self.logged = []
            self.finished = False

        def log(self, metrics, *, step):
            self.logged.append((dict(metrics), step))

        def finish(self):
            self.finished = True

    class FakeWandb:
        def __init__(self):
            self.calls = []
            self.run = FakeRun()

        def init(self, **kwargs):
            self.calls.append(kwargs)
            return self.run

    fake_wandb = FakeWandb()
    run, logger = train.create_wandb_metric_logger(
        enabled=True,
        wandb_module=fake_wandb,
        project="futuremamba",
        name="pickxtimes-seed42",
        config={"train_seed": 42, "memory_backend": "mamba2"},
    )

    logger({"train/loss": 0.25}, 7)
    run.finish()

    assert run is fake_wandb.run
    assert fake_wandb.calls == [
        {
            "project": "futuremamba",
            "name": "pickxtimes-seed42",
            "config": {"train_seed": 42, "memory_backend": "mamba2"},
            "resume": "allow",
        }
    ]
    assert fake_wandb.run.logged == [({"train/loss": 0.25}, 7)]
    assert fake_wandb.run.finished is True
