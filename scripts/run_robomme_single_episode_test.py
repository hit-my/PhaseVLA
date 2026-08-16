from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).with_name("run_robomme_single_episode.py")
spec = importlib.util.spec_from_file_location("run_robomme_single_episode", SCRIPT)
assert spec is not None and spec.loader is not None
launcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = launcher
spec.loader.exec_module(launcher)


class _FakeRunner:
    instances: list["_FakeRunner"] = []

    def __init__(self, task_name: str, video_dir: Path, max_steps: int, split: str = "test"):
        self.task_name = task_name
        self.video_dir = video_dir
        self.max_steps = max_steps
        self.split = split
        self.dataset = "val" if split == "validation" else split
        self.num_episodes = 3
        self.made_episode: int | None = None
        self.closed = False
        self.__class__.instances.append(self)

    def make_env(self, episode_id: int) -> None:
        self.made_episode = episode_id

    def close_env(self) -> None:
        self.closed = True


class _FakeEvaluator:
    outcome = "success"

    def __init__(self, args, save_dir: Path):
        self.args = args
        self.save_dir = save_dir

    def eval_each_episode(self, runner, predictor, video_dir: Path) -> str:
        assert runner.made_episode is not None
        assert predictor == "predictor"
        assert video_dir == self.save_dir / "videos"
        return self.outcome


def _runtime(tmp_path: Path):
    return SimpleNamespace(
        TASK_NAME_LIST=["PickXtimes", "BinFill"],
        EnvRunner=_FakeRunner,
        EpisodeEvaluator=_FakeEvaluator,
        build_subgoal_predictor=lambda args, save_dir: "predictor",
        setup_save_directory=lambda args: tmp_path / "run",
        check_args=lambda args: (_ for _ in ()).throw(AssertionError("obs_horizon must be 16"))
        if args.obs_horizon != 16
        else None,
    )


def _args(tmp_path: Path, *, obs_horizon: int = 16, split: str = "test", dataset: str | None = None):
    return SimpleNamespace(save_dir=str(tmp_path), max_steps=1300, obs_horizon=obs_horizon, split=split, dataset=dataset)


def _result_identity() -> dict[str, object]:
    return {
        "experiment_id": "futuremamba__PickXtimes__validation__ep000__train0__eval2026",
        "method_id": "futuremamba_mamba2",
        "train_seed": 0,
        "task_name": "PickXtimes",
        "episode_id": 0,
        "split": "validation",
        "checkpoint": {"path": "/ckpts/futuremamba/0", "checkpoint_id": 5000},
        "config": {"name": "futuremamba_robomme_mamba2"},
        "provenance": {"checkpoint_source": "futuremamba_mamba2", "train_seed": 0},
    }


def test_run_one_episode_uses_official_runner_evaluator_and_writes_results(tmp_path: Path):
    _FakeRunner.instances.clear()
    _FakeEvaluator.outcome = "success"

    result = launcher.run_one_episode(
        _args(tmp_path),
        task_name="PickXtimes",
        episode_id=0,
        runtime=_runtime(tmp_path),
    )

    runner = _FakeRunner.instances[-1]
    assert runner.task_name == "PickXtimes"
    assert runner.made_episode == 0
    assert runner.closed
    assert runner.split == "test"
    assert runner.dataset == "test"
    assert result == {
        "task_name": "PickXtimes",
        "episode_id": 0,
        "split": "test",
        "dataset": "test",
        "outcome": "success",
        "success": True,
    }
    run_dir = tmp_path / "run"
    assert json.loads((run_dir / "progress.json").read_text()) == {
        "PickXtimes": {"0": True}
    }
    assert json.loads((run_dir / "log.json").read_text()) == {
        "success_rate": {"PickXtimes": 1.0},
        "total_success_rate": 1.0,
    }


def test_run_one_episode_uses_validation_split_without_test_dataset(tmp_path: Path):
    _FakeRunner.instances.clear()
    _FakeEvaluator.outcome = "success"

    result = launcher.run_one_episode(
        _args(tmp_path, split="validation", dataset="val"),
        task_name="PickXtimes",
        episode_id=0,
        runtime=_runtime(tmp_path),
    )

    runner = _FakeRunner.instances[-1]
    assert runner.split == "validation"
    assert runner.dataset == "val"
    assert runner.dataset != "test"
    assert result["split"] == "validation"
    assert result["dataset"] == "val"



def test_run_one_episode_returns_manifest_identity_for_analyzer(tmp_path: Path):
    identity = _result_identity()

    result = launcher.run_one_episode(
        _args(tmp_path, split="validation", dataset="val"),
        task_name="PickXtimes",
        episode_id=0,
        runtime=_runtime(tmp_path),
        result_identity=identity,
    )

    assert {field: result[field] for field in identity} == identity
    assert result["dataset"] == "val"
    assert result["outcome"] == "success"
    assert result["success"] is True


def test_run_one_episode_rejects_result_identity_selection_mismatch(tmp_path: Path):
    identity = _result_identity()
    identity["episode_id"] = 1

    with pytest.raises(ValueError, match="result identity episode_id"):
        launcher.run_one_episode(
            _args(tmp_path, split="validation", dataset="val"),
            task_name="PickXtimes",
            episode_id=0,
            runtime=_runtime(tmp_path),
            result_identity=identity,
        )

def test_run_one_episode_rejects_split_dataset_mismatch(tmp_path: Path):
    with pytest.raises(ValueError, match="does not match split"):
        launcher.run_one_episode(
            _args(tmp_path, split="validation", dataset="test"),
            task_name="PickXtimes",
            episode_id=0,
            runtime=_runtime(tmp_path),
        )


def test_run_one_episode_merges_existing_progress_and_recomputes_rates(tmp_path: Path):
    _FakeRunner.instances.clear()
    _FakeEvaluator.outcome = "success"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "progress.json").write_text(
        json.dumps(
            {
                "PickXtimes": {"0": False},
                "BinFill": {"2": True},
            }
        ),
        encoding="utf-8",
    )

    launcher.run_one_episode(
        _args(tmp_path),
        task_name="PickXtimes",
        episode_id=1,
        runtime=_runtime(tmp_path),
    )

    assert json.loads((run_dir / "progress.json").read_text()) == {
        "PickXtimes": {"0": False, "1": True},
        "BinFill": {"2": True},
    }
    assert json.loads((run_dir / "log.json").read_text()) == {
        "success_rate": {"PickXtimes": 0.5, "BinFill": 1.0},
        "total_success_rate": 0.75,
    }


def test_main_forwards_official_subgoal_predictor_arguments(monkeypatch, tmp_path: Path):
    captured = {}

    class OfficialArgs:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.__dict__.update(kwargs)

    runtime = SimpleNamespace(Args=OfficialArgs)
    monkeypatch.setattr(launcher, "_official_runtime", lambda: runtime)
    monkeypatch.setattr(
        launcher,
        "run_one_episode",
        lambda args, **kwargs: {
            "task_name": kwargs["task_name"],
            "episode_id": kwargs["episode_id"],
            "split": args.split,
            "dataset": args.dataset,
            "outcome": "success",
            "success": True,
        }
    )
    args = launcher.LauncherArgs(
        save_dir=str(tmp_path),
        use_memer=True,
        subgoal_type="grounded_subgoal",
        gemini_model_name="gemini-test",
        qwenvl_simpleSG_adapter_path="simple-adapter",
        qwenvl_groundSG_adapter_path="grounded-adapter",
        memer_adapter_path="memer-adapter",
        subgoal_keep_period=3,
    )

    launcher.main(args)

    assert captured["use_oracle"] is False
    assert captured["use_qwenvl"] is False
    assert captured["use_memer"] is True
    assert captured["use_gemini"] is False
    assert captured["subgoal_type"] == "grounded_subgoal"
    assert captured["gemini_model_name"] == "gemini-test"
    assert captured["qwenvl_simpleSG_adapter_path"] == "simple-adapter"
    assert captured["qwenvl_groundSG_adapter_path"] == "grounded-adapter"
    assert captured["memer_adapter_path"] == "memer-adapter"
    assert captured["subgoal_keep_period"] == 3
    assert captured["split"] == "test"
    assert captured["dataset"] == "test"


def test_run_one_episode_closes_environment_when_evaluator_fails(tmp_path: Path):
    _FakeRunner.instances.clear()

    class FailingEvaluator(_FakeEvaluator):
        def eval_each_episode(self, runner, predictor, video_dir: Path) -> str:
            raise RuntimeError("policy server disconnected")

    runtime = _runtime(tmp_path)
    runtime.EpisodeEvaluator = FailingEvaluator

    with pytest.raises(RuntimeError, match="policy server disconnected"):
        launcher.run_one_episode(
            _args(tmp_path),
            task_name="PickXtimes",
            episode_id=1,
            runtime=runtime,
        )

    assert _FakeRunner.instances[-1].closed


@pytest.mark.parametrize(
    ("task_name", "episode_id", "message"),
    [
        ("Unknown", 0, "unknown RoboMME task"),
        ("PickXtimes", -1, "episode_id must be non-negative"),
        ("PickXtimes", 3, "outside available range"),
    ],
)
def test_run_one_episode_rejects_invalid_selection(
    tmp_path: Path, task_name: str, episode_id: int, message: str
):
    _FakeRunner.instances.clear()

    with pytest.raises(ValueError, match=message):
        launcher.run_one_episode(
            _args(tmp_path),
            task_name=task_name,
            episode_id=episode_id,
            runtime=_runtime(tmp_path),
        )

    if _FakeRunner.instances:
        assert _FakeRunner.instances[-1].closed


def test_run_one_episode_applies_official_argument_validation(tmp_path: Path):
    with pytest.raises(AssertionError, match="obs_horizon must be 16"):
        launcher.run_one_episode(
            _args(tmp_path, obs_horizon=20),
            task_name="PickXtimes",
            episode_id=0,
            runtime=_runtime(tmp_path),
        )
