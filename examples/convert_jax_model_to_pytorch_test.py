import json
import pathlib

import numpy as np
import pytest
import torch

from examples import convert_jax_model_to_pytorch as converter


LM_HEAD_KEY = "paligemma_with_expert.paligemma.lm_head.weight"


def test_validate_convertible_parameter_tree_rejects_lora_paths():
    flat = {
        "PaliGemma/llm/layers/0/attn/lora_a/kernel": object(),
        "PaliGemma/llm/layers/0/attn/kernel": object(),
    }

    with pytest.raises(ValueError, match="LoRA"):
        converter.validate_convertible_parameter_tree(flat)


def test_validate_convertible_parameter_tree_accepts_base_parameter_tree():
    converter.validate_convertible_parameter_tree(
        {
            "PaliGemma/llm/layers/0/attn/kernel": object(),
            "action_in_proj/kernel": object(),
        }
    )


def test_validate_convertible_parameter_tree_rejects_tuple_lora_paths():
    flat = {("PaliGemma", "llm", "lora_b", "kernel"): object()}

    with pytest.raises(ValueError, match="LoRA"):
        converter.validate_convertible_parameter_tree(flat)


def test_validate_convertible_parameter_tree_rejects_nested_lora_paths():
    nested = {"action_in_proj": {"kernel": {"lora_a": {"value": object()}}}}

    with pytest.raises(ValueError, match="LoRA"):
        converter.validate_convertible_parameter_tree(nested)


@pytest.mark.parametrize(
    ("missing", "unexpected", "tied_weight_verified"),
    [
        ([LM_HEAD_KEY, "extra.weight"], [], True),
        ([LM_HEAD_KEY], ["unused.weight"], True),
        ([LM_HEAD_KEY], [], False),
        ([], ["unused.weight"], True),
    ],
)
def test_validate_load_result_rejects_unexplained_state_dict_keys(missing, unexpected, tied_weight_verified):
    with pytest.raises(ValueError):
        converter.validate_load_result(missing, unexpected, tied_weight_verified)


def test_validate_load_result_allows_only_verified_tied_lm_head_missing():
    converter.validate_load_result([LM_HEAD_KEY], [], tied_weight_verified=True)
    converter.validate_load_result([], [], tied_weight_verified=False)


def test_validate_load_result_accepts_single_tied_key_tuple():
    converter.validate_load_result((LM_HEAD_KEY,), (), tied_weight_verified=True)


def test_validate_load_result_rejects_duplicate_missing_keys():
    with pytest.raises(ValueError, match="missing"):
        converter.validate_load_result([LM_HEAD_KEY, LM_HEAD_KEY], [], tied_weight_verified=True)


def test_float32_conversion_config_prevents_bfloat16_load_truncation():
    source = converter.openpi.models.pi0_config.Pi0Config(
        dtype="bfloat16",
        action_horizon=20,
        pytorch_compile_mode="max-autotune",
    )

    conversion = converter._float32_conversion_config(source)

    assert conversion.dtype == "float32"
    assert conversion.pytorch_compile_mode is None
    assert conversion.action_horizon == 20


def test_copy_checkpoint_assets_copies_only_checkpoint_root_assets(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "output"
    root_asset = checkpoint / "assets" / "physical-intelligence" / "norm_stats.json"
    parent_asset = checkpoint.parent / "assets" / "wrong.txt"
    root_asset.parent.mkdir(parents=True)
    parent_asset.parent.mkdir(parents=True)
    root_asset.write_text('{"mean": [1]}', encoding="utf-8")
    parent_asset.write_text("wrong", encoding="utf-8")

    converter.copy_checkpoint_assets(checkpoint, output)

    copied = output / "assets" / "physical-intelligence" / "norm_stats.json"
    assert copied.read_text(encoding="utf-8") == '{"mean": [1]}'
    assert not (output / "assets" / "wrong.txt").exists()


def test_copy_checkpoint_assets_requires_checkpoint_root_assets(tmp_path):
    with pytest.raises(FileNotFoundError):
        converter.copy_checkpoint_assets(tmp_path / "missing_assets_checkpoint", tmp_path / "output")


def test_copy_checkpoint_assets_fails_when_destination_is_incomplete(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "output"
    (checkpoint / "assets" / "physical-intelligence").mkdir(parents=True)
    (checkpoint / "assets" / "physical-intelligence" / "norm_stats.json").write_text("{}", encoding="utf-8")

    original_copytree = converter.shutil.copytree
    def copytree_then_drop_file(src: pathlib.Path, dst: pathlib.Path, *args, **kwargs):
        result = original_copytree(src, dst, *args, **kwargs)
        if pathlib.Path(src) == checkpoint / "assets":
            (pathlib.Path(dst) / "physical-intelligence" / "norm_stats.json").unlink()
        return result
    monkeypatch.setattr(converter.shutil, "copytree", copytree_then_drop_file)

    with pytest.raises(RuntimeError, match="assets"):
        converter.copy_checkpoint_assets(checkpoint, output)



def test_directory_checksum_covers_relative_paths_and_contents(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "a.txt").write_text("same", encoding="utf-8")
    original = converter.directory_checksum(tree)

    (tree / "a.txt").rename(tree / "b.txt")
    renamed = converter.directory_checksum(tree)
    (tree / "b.txt").write_text("changed", encoding="utf-8")
    changed = converter.directory_checksum(tree)

    assert original != renamed
    assert renamed != changed


def test_parameter_metadata_is_sorted_and_normalizes_paths():
    metadata = converter.parameter_metadata(
        {
            ("z", "kernel"): np.zeros((2, 3), dtype=np.float32),
            "a/bias": np.zeros((3,), dtype=np.float16),
        }
    )

    assert list(metadata) == ["a/bias", "z/kernel"]
    assert metadata["a/bias"] == {"shape": [3], "dtype": "float16"}
    assert metadata["z/kernel"] == {"shape": [2, 3], "dtype": "float32"}


def test_write_conversion_manifest_is_deterministic_and_complete(tmp_path):
    manifest = converter.build_conversion_manifest(
        source_parameters={"source/kernel": np.zeros((2, 2), dtype=np.float32)},
        target_parameters={"target.weight": np.zeros((2, 2), dtype=np.float32)},
        source_checkpoint_checksum="source-sha256",
        assets_checksum="assets-sha256",
        model_checksum="model-sha256",
        config_name="pi05_libero",
        model_config={"action_dim": 32, "pi05": True},
        precision="float32",
    )
    destination = tmp_path / "conversion_manifest.json"

    converter.write_conversion_manifest(destination, manifest)
    first = destination.read_bytes()
    converter.write_conversion_manifest(destination, manifest)

    assert destination.read_bytes() == first
    payload = json.loads(first)
    assert payload["format_version"] == 1
    assert payload["converter"] == "openpi.convert_jax_model_to_pytorch"
    assert payload["config_name"] == "pi05_libero"
    assert payload["precision"] == "float32"
    assert payload["source_checkpoint_checksum"] == "source-sha256"
    assert payload["assets_checksum"] == "assets-sha256"
    assert payload["model_checksum"] == "model-sha256"
    assert payload["source_parameters"]["source/kernel"]["shape"] == [2, 2]
    assert payload["target_parameters"]["target.weight"]["dtype"] == "float32"


def test_manifest_normalizes_nested_json_values(tmp_path):
    manifest = converter.build_conversion_manifest(
        source_parameters={"source/kernel": np.zeros((1,), dtype=np.float32)},
        target_parameters={"target.weight": np.zeros((1,), dtype=np.float32)},
        source_checkpoint_checksum="source-sha256",
        assets_checksum="assets-sha256",
        model_checksum="model-sha256",
        config_name="pi05_libero",
        model_config={"nested": {"dtype": torch.float32, "path": pathlib.Path("assets")}},
        precision="float32",
    )
    destination = tmp_path / "manifest.json"

    converter.write_conversion_manifest(destination, manifest)
    payload = json.loads(destination.read_text(encoding="utf-8"))

    assert payload["model_config"]["nested"] == {"dtype": "torch.float32", "path": "assets"}