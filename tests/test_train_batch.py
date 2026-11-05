"""Sequential experiment scheduling and native task configuration, without GPU training."""

import argparse
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_spec = importlib.util.spec_from_file_location(
    "m20_train_batch_test", Path(__file__).parents[1] / "scripts/train_batch.py"
)
assert _spec is not None and _spec.loader is not None
batch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(batch)


def _config(tmp_path, entries=None, defaults=None):
    path = tmp_path / "experiments.json"
    batch.write_json(
        path,
        {
            "defaults": {
                "task": "M20NewtonTraining-M20-M20-WalkOnSandProxy",
                "num_envs": 2,
                "max_iterations": 3,
                "reward_weights": {"feet_air_time": 5.0},
                **(defaults or {}),
            },
            "runs": entries or [{"name": "one"}, {"name": "two"}],
        },
    )
    return path


def _args(tmp_path, config, **changes):
    return argparse.Namespace(
        **{
            "config": config,
            "only": None,
            "dry_run": False,
            "validate_only": False,
            "continue_on_error": False,
            "output_dir": tmp_path / "batch",
            **changes,
        }
    )


def _fake_executor(tmp_path, calls, fail=None, no_checkpoint=None):
    def execute(command, spec_path, log_path):
        spec = batch.read_json(spec_path)
        name = spec["run"]["name"]
        validating = "--_validate" in command
        calls.append(("validate" if validating else "train", name, spec))
        if fail == ("validate" if validating else "train", name):
            return 1
        if validating:
            batch.write_json(
                spec_path.with_name("validation.json"),
                {
                    "training_log_dir": str(tmp_path / "native_logs" / name),
                    "active_reward_weights": spec["run"]["reward_weights"],
                },
            )
        elif name != no_checkpoint:
            log_dir = tmp_path / "native_logs" / name
            log_dir.mkdir(parents=True)
            # Natural iteration order must win over lexicographic order.
            for iteration in (9, 10):
                (log_dir / f"model_{iteration}.pt").touch()
        return 0

    return execute


def test_group_merge_is_independent_and_does_not_inherit_the_previous_group(tmp_path):
    path = _config(
        tmp_path,
        [
            {"name": "one", "reward_weights": {"feet_air_time": 20.0, "action_rate_l2": 0.0}},
            {"name": "two", "overrides": {"agent.algorithm.learning_rate": 0.0001}},
        ],
    )
    first, second = batch.load_runs(path)
    assert first["reward_weights"] == {"feet_air_time": 20.0, "action_rate_l2": 0.0}
    assert second["reward_weights"] == {"feet_air_time": 5.0}
    assert first["overrides"] == {}
    first["reward_weights"]["feet_air_time"] = 200.0
    assert second["reward_weights"]["feet_air_time"] == 5.0


@pytest.mark.parametrize(
    "changes",
    [
        {"num_envs": 0},
        {"max_iterations": True},
        {"seed": -2},
        {"reward_weights": {"feet_air_time": float("nan")}},
        {"reward_weights": {"feet_air_time": True}},
        {"overrides": {"env.rewards.feet_air_time.weight": 10.0}},
        {"overrides": {"agent.run_name": "collision"}},
        {"overrides": {"env.scene.num_envs": 100}},
        {"overrides": {"wrong.path": 1}},
        {"typo": 3},
    ],
)
def test_invalid_configuration_fails_before_training(tmp_path, changes):
    with pytest.raises(ValueError):
        batch.load_runs(_config(tmp_path, [{"name": "one", **changes}]))


def test_duplicate_names_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Duplicate"):
        batch.load_runs(_config(tmp_path, [{"name": "same"}, {"name": "same"}]))


def test_train_arguments_keep_paths_as_one_argument_and_use_native_hydra_literals(tmp_path):
    run = batch.load_runs(
        _config(
            tmp_path,
            [
                {
                    "name": "one",
                    "physics": "newton_mjwarp",
                    "overrides": {"env.commands.base_velocity.ranges.lin_vel_x": [0.15, 0.25]},
                }
            ],
        )
    )[0]
    args = batch.build_train_args(run, "batch_01", "/tmp/path with spaces/model_10.pt")
    assert args[args.index("--checkpoint") + 1] == "/tmp/path with spaces/model_10.pt"
    assert "--no-resume_optimizer_state" in args
    assert "physics=newton_mjwarp" in args
    assert "env.commands.base_velocity.ranges.lin_vel_x=[0.15,0.25]" in args


def test_queue_checks_every_group_then_trains_in_order_and_chains_exact_checkpoint(tmp_path, monkeypatch):
    config = _config(tmp_path, [{"name": "one"}, {"name": "two", "checkpoint": "previous"}])
    calls = []
    monkeypatch.setattr(batch, "execute", _fake_executor(tmp_path, calls))
    assert batch.run_batch(_args(tmp_path, config)) == 0
    assert [(kind, name) for kind, name, _ in calls] == [
        ("validate", "one"),
        ("validate", "two"),
        ("train", "one"),
        ("train", "two"),
    ]
    spec = calls[-1][2]
    expected = str(tmp_path / "native_logs" / "one" / "model_10.pt")
    assert spec["resolved_checkpoint"] == expected
    assert spec["train_args"][spec["train_args"].index("--checkpoint") + 1] == expected
    manifest = batch.read_json(tmp_path / "batch" / "manifest.json")
    assert [run["status"] for run in manifest["runs"]] == ["completed", "completed"]


def test_independent_runs_load_the_same_explicit_checkpoint(tmp_path, monkeypatch):
    checkpoint = tmp_path / "starting policy.pt"
    checkpoint.touch()
    config = _config(tmp_path, defaults={"checkpoint": str(checkpoint)})
    calls = []
    monkeypatch.setattr(batch, "execute", _fake_executor(tmp_path, calls))
    assert batch.run_batch(_args(tmp_path, config)) == 0
    trained = [spec for kind, _, spec in calls if kind == "train"]
    assert [spec["resolved_checkpoint"] for spec in trained] == [str(checkpoint), str(checkpoint)]


@pytest.mark.parametrize("failure", [("validate", "two"), ("train", "one")])
def test_failure_stops_the_queue(tmp_path, monkeypatch, failure):
    calls = []
    monkeypatch.setattr(batch, "execute", _fake_executor(tmp_path, calls, fail=failure))
    assert batch.run_batch(_args(tmp_path, _config(tmp_path))) == 1
    trained = [name for kind, name, _ in calls if kind == "train"]
    assert trained == ([] if failure[0] == "validate" else ["one"])


def test_continue_after_failure_never_uses_an_older_checkpoint_for_previous(tmp_path, monkeypatch):
    config = _config(
        tmp_path,
        [
            {"name": "one"},
            {"name": "two"},
            {"name": "three", "checkpoint": "previous"},
        ],
    )
    calls = []
    monkeypatch.setattr(batch, "execute", _fake_executor(tmp_path, calls, fail=("train", "two")))
    assert batch.run_batch(_args(tmp_path, config, continue_on_error=True)) == 1
    assert [name for kind, name, _ in calls if kind == "train"] == ["one", "two"]
    manifest = batch.read_json(tmp_path / "batch" / "manifest.json")
    assert [run["status"] for run in manifest["runs"]] == ["completed", "failed", "failed"]


def test_success_exit_without_a_checkpoint_is_not_treated_as_completed(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "execute", _fake_executor(tmp_path, calls, no_checkpoint="one"))
    assert batch.run_batch(_args(tmp_path, _config(tmp_path))) == 1
    manifest = batch.read_json(tmp_path / "batch" / "manifest.json")
    assert manifest["runs"][0]["status"] == "failed"
    assert "without a model checkpoint" in manifest["runs"][0]["error"]


def test_validation_only_does_not_train(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "execute", _fake_executor(tmp_path, calls))
    assert batch.run_batch(_args(tmp_path, _config(tmp_path), validate_only=True)) == 0
    assert all(kind == "validate" for kind, _, _ in calls)


def test_dry_run_creates_no_artifacts_or_processes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(batch, "execute", lambda *args: pytest.fail("dry run launched a process"))
    assert batch.run_batch(_args(tmp_path, _config(tmp_path), dry_run=True)) == 0
    assert not (tmp_path / "batch").exists()
    assert "reward_weights" in capsys.readouterr().out


def test_only_preserves_configuration_order(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "execute", _fake_executor(tmp_path, calls))
    assert batch.run_batch(_args(tmp_path, _config(tmp_path), only=["two", "one"])) == 0
    assert [name for kind, name, _ in calls if kind == "train"] == ["one", "two"]


def test_ctrl_c_marks_interrupted_and_starts_no_later_group(tmp_path, monkeypatch):
    calls = []
    fake = _fake_executor(tmp_path, calls)

    def interrupt(command, spec_path, log_path):
        if "--_validate" not in command:
            raise KeyboardInterrupt
        return fake(command, spec_path, log_path)

    monkeypatch.setattr(batch, "execute", interrupt)
    assert batch.run_batch(_args(tmp_path, _config(tmp_path))) == 130
    manifest = batch.read_json(tmp_path / "batch" / "manifest.json")
    assert [run["status"] for run in manifest["runs"]] == ["interrupted", "validated"]


def test_real_child_process_receives_isolated_spec_and_console_log(tmp_path, capsys):
    spec_path = tmp_path / "requested.json"
    batch.write_json(spec_path, {"name": "test"})
    code = (
        "import os,sys; "
        f"assert os.environ[{batch.SPEC_VARIABLE!r}] == sys.argv[1]; "
        f"assert os.getcwd() == {str(batch.PROJECT_ROOT)!r}; "
        "print('child output'); sys.exit(7)"
    )
    log = tmp_path / "train.log"
    assert batch.execute([sys.executable, "-c", code, str(spec_path)], spec_path, log) == 7
    assert log.read_text() == "child output\n"
    assert "child output" in capsys.readouterr().out


def test_reweight_then_prune_does_not_drop_terms_that_are_being_enabled():
    cfg = SimpleNamespace(
        rewards=SimpleNamespace(
            disabled=SimpleNamespace(weight=0.0),
            enabled=SimpleNamespace(weight=10.0),
            absent=None,
        )
    )
    batch.apply_reward_weights(cfg, {"disabled": 5.0, "enabled": 0.0})
    assert cfg.rewards.disabled.weight == 5.0
    assert cfg.rewards.enabled is None
    with pytest.raises(ValueError, match="does not exist"):
        batch.apply_reward_weights(cfg, {"absent": 2.0})


@pytest.mark.parametrize("task", ["Flat", "Walk", "Rough", "Stand", "Sand", "WalkOnSand", "WalkOnSandProxy"])
def test_native_factory_restores_pruning_and_preserves_original_task_class(tmp_path, monkeypatch, task):
    import gymnasium as gym

    from M20_newton_training.tasks.m20.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

    task_id = f"M20NewtonTraining-M20-M20-{task}"
    task_spec = gym.spec(task_id)
    original = task_spec.kwargs["env_cfg_entry_point"]
    prune = LocomotionVelocityRoughEnvCfg.disable_zero_weight_rewards
    spec_path = tmp_path / "requested.json"
    batch.write_json(
        spec_path,
        {
            "run": {
                "task": task_id,
                "reward_weights": {"joint_acc_l2": -0.001, "action_rate_l2": 0.0},
            }
        },
    )
    monkeypatch.setenv(batch.SPEC_VARIABLE, str(spec_path))
    try:
        batch.configure_training()
        cfg = task_spec.kwargs["env_cfg_entry_point"]()
        assert cfg.rewards.joint_acc_l2.weight == -0.001
        assert cfg.rewards.action_rate_l2 is None
        assert LocomotionVelocityRoughEnvCfg.disable_zero_weight_rewards is prune
        assert type(cfg).__name__ == original.split(":")[1]
        cfg.validate()
    finally:
        task_spec.kwargs["env_cfg_entry_point"] = original
