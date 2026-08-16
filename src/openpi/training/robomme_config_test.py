from __future__ import annotations

import pathlib

from openpi import transforms as _transforms
from openpi.models_pytorch.futuremamba_config import FutureMambaPytorchConfig
from openpi.policies.robomme_policy import RoboMMEInputs, RoboMMEOutputs
from openpi.training import config as _config


def test_robomme_pytorch_base_config_is_registered():
    config = _config.get_config("pi05_robomme_pytorch")

    assert config.model.pi05 is True
    assert config.model.action_dim == 32
    assert config.model.action_horizon == 20
    assert config.model.discrete_state_input is False
    assert config.data.repo_id == "robomme"
    assert isinstance(config.data, _config.RoboMMEDataConfig)


def test_robomme_futuremamba_mamba2_config_is_registered():
    config = _config.get_config("futuremamba_robomme_mamba2")

    assert isinstance(config.model, FutureMambaPytorchConfig)
    assert config.model.pi05 is True
    assert config.model.discrete_state_input is False
    assert config.model.action_dim == 32
    assert config.model.action_horizon == 20
    assert config.model.execution_horizon == 16
    assert not hasattr(config.model, "executed_horizon")
    assert config.model.progress_depth == 6
    assert config.model.memory_backend == "mamba2"
    assert config.model.terminal_loss_queries_per_episode == 1
    assert config.model.frozen_prefix_microbatch_size == 2
    assert config.episode_data.query_stride == 16
    assert config.episode_data.executed_horizon == 16
    assert config.data.repo_id == "robomme"
    assert isinstance(config.data, _config.RoboMMEDataConfig)
    assert config.pytorch_weight_path == "./runs/ckpts/pi05_baseline_pytorch/79999"
    assert config.model.base_checkpoint_uri == "./runs/ckpts/pi05_baseline_pytorch/79999"
    assert config.model.base_assets_checksum == "3f15cc514b5a1941325bbeb673a776d80823cdbbff47629da8b9c7d84911f945"
    assert config.model.robomme_policy_commit == "ecf086c3be7c2223167d9bb2f6ef1f0a6e24353b"
    assert config.model.robomme_benchmark_commit == "856bc3a189d4172f3f47dbee4424d585f8d78db3"
    assert config.data.assets.assets_dir == "./runs/ckpts/pi05_baseline_pytorch/79999/assets"
    assert config.data.assets.asset_id == "robomme"
    assert config.num_workers == 0
    data = config.data.create(config.assets_dirs, config.model)
    assert data.norm_stats is not None
    assert set(data.norm_stats) == {"state", "actions"}


def test_robomme_futuremamba_structural_and_loss_ablations_are_registered():
    expected = {
        "futuremamba_robomme_mamba2_depth4": {"progress_depth": 4},
        "futuremamba_robomme_mamba2_depth9": {"progress_depth": 9},
        "futuremamba_robomme_mamba2_handoff_0p4": {"handoff_ratio": 0.4},
        "futuremamba_robomme_mamba2_handoff_0p6": {"handoff_ratio": 0.6},
        "futuremamba_robomme_mamba2_handoff_k0": {"handoff_ratio": 0.0},
        "futuremamba_robomme_mamba2_handoff_kn": {"handoff_ratio": 1.0},
        "futuremamba_robomme_mamba2_flow_only": {"terminal_loss_weight": 0.0},
    }

    for config_name, model_fields in expected.items():
        train_config = _config.get_config(config_name)
        assert isinstance(train_config.model, FutureMambaPytorchConfig)
        for field, value in model_fields.items():
            assert getattr(train_config.model, field) == value
        assert train_config.model.checkpoint_metadata() != _config.get_config(
            "futuremamba_robomme_mamba2"
        ).model.checkpoint_metadata()



def test_robomme_data_config_builds_quantile_and_official_transforms(tmp_path: pathlib.Path):
    model = FutureMambaPytorchConfig(
        action_horizon=20,
        discrete_state_input=False,
        execution_horizon=16,
        progress_depth=6,
    )
    data = _config.RoboMMEDataConfig(repo_id="robomme").create(tmp_path, model)
    repack = data.repack_transforms.inputs[0]
    assert repack.structure == {
        "observation/image": "image",
        "observation/wrist_image": "wrist_image",
        "observation/state": "state",
        "actions": "actions",
        "prompt": "prompt",
        "simple_subgoal": "simple_subgoal",
        "grounded_subgoal": "grounded_subgoal",
    }
    assert isinstance(data.data_transforms.inputs[1], _transforms.DeltaActions)
    assert isinstance(data.data_transforms.outputs[0], _transforms.AbsoluteActions)

    assert data.use_quantile_norm is True
    assert isinstance(data.data_transforms.inputs[0], RoboMMEInputs)
    assert data.data_transforms.inputs[0].model_type is model.model_type
    assert isinstance(data.data_transforms.outputs[1], RoboMMEOutputs)
    assert data.action_sequence_keys == ("actions",)

def test_robomme_data_config_carries_explicit_episode_pickle_dir(tmp_path: pathlib.Path):
    factory = _config.RoboMMEDataConfig(repo_id="robomme", episode_data_dir=str(tmp_path / "pickles"))
    data = factory.create(tmp_path, FutureMambaPytorchConfig(action_horizon=20, discrete_state_input=False, execution_horizon=16))

    assert data.episode_data_dir == str(tmp_path / "pickles")
