#!/usr/bin/env python3
"""Run JSON-defined RSL-RL experiments sequentially, using the native Isaac Lab CLI."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import math
import os
import re
import shlex
import signal
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_VARIABLE = "M20_TRAIN_BATCH_SPEC"
CALLBACK = "train_batch.configure_training"
RUN_KEYS = {
    "name",
    "task",
    "physics",
    "num_envs",
    "max_iterations",
    "seed",
    "checkpoint",
    "resume_optimizer_state",
    "device",
    "logger",
    "viz",
    "experiment_name",
    "agent",
    "reward_weights",
    "overrides",
}
CLI_KEYS = (
    "task",
    "num_envs",
    "max_iterations",
    "seed",
    "device",
    "logger",
    "viz",
    "experiment_name",
    "agent",
)


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_runs(config_path: Path) -> list[dict]:
    """Merge defaults without sharing mutable dictionaries between experiments."""
    config = read_json(config_path)
    if unknown := set(config) - {"defaults", "runs"}:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    defaults, runs = config.get("defaults", {}), config.get("runs")
    if not isinstance(defaults, dict) or not isinstance(runs, list) or not runs:
        raise ValueError("defaults must be an object; runs must be a nonempty list")
    if "name" in defaults or set(defaults) - RUN_KEYS:
        raise ValueError("defaults contains a name or an unknown option")
    merged_runs, names = [], set()
    for entry in runs:
        if not isinstance(entry, dict) or set(entry) - RUN_KEYS:
            raise ValueError(f"Invalid run options: {entry!r}")
        run = copy.deepcopy(defaults)
        run.update(copy.deepcopy(entry))
        for key in ("reward_weights", "overrides"):
            base, changes = defaults.get(key, {}), entry.get(key, {})
            if not isinstance(base, dict) or not isinstance(changes, dict):
                raise ValueError(f"{key} must be a JSON object")
            run[key] = {**copy.deepcopy(base), **copy.deepcopy(changes)}
        validate_run(run)
        if run["name"] in names:
            raise ValueError(f"Duplicate run name: {run['name']}")
        names.add(run["name"])
        run.setdefault("resume_optimizer_state", False)
        run.setdefault("logger", "tensorboard")
        run.setdefault("viz", "none")
        merged_runs.append(run)
    return merged_runs


def validate_run(run: dict) -> None:
    name = run.get("name")
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise ValueError("Each run needs a unique name of 1..80 letters, digits, underscores or hyphens")
    if not isinstance(run.get("task"), str) or not run["task"].startswith("M20NewtonTraining-"):
        raise ValueError(f"{name}: task must be a registered M20NewtonTraining task id")
    for key in ("num_envs", "max_iterations"):
        value = run.get(key)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name}: {key} must be a positive integer")
    if "seed" in run and (type(run["seed"]) is not int or run["seed"] < -1):
        raise ValueError(f"{name}: seed must be an integer >= -1")
    if "resume_optimizer_state" in run and type(run["resume_optimizer_state"]) is not bool:
        raise ValueError(f"{name}: resume_optimizer_state must be true or false")
    for key in ("physics", "device", "logger", "viz", "experiment_name", "agent"):
        if key in run and (not isinstance(run[key], str) or not run[key]):
            raise ValueError(f"{name}: {key} must be a nonempty string")
    checkpoint = run.get("checkpoint")
    if checkpoint is not None and (not isinstance(checkpoint, str) or not checkpoint):
        raise ValueError(f"{name}: checkpoint must be null, a local path, or 'previous'")
    for key, value in run["reward_weights"].items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", key):
            raise ValueError(f"{name}: invalid reward name {key!r}")
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{name}: reward weight {key} must be finite")
    for key, value in run["overrides"].items():
        if not re.fullmatch(r"(?:env|agent)(?:\.[A-Za-z_][A-Za-z0-9_]*)+", key):
            raise ValueError(f"{name}: override {key!r} must be an env.* or agent.* path")
        if key.startswith("env.rewards.") and (key.endswith(".weight") or key.count(".") == 2):
            raise ValueError(f"{name}: use reward_weights to enable/disable or change reward weights")
        if key in {"agent.run_name", "agent.experiment_name", "agent.max_iterations", "env.scene.num_envs"}:
            raise ValueError(f"{name}: use the top-level option instead of {key}")
        json.dumps(value, allow_nan=False)


def build_train_args(run: dict, timestamp: str, checkpoint: str | None = None) -> list[str]:
    args = ["--rl_library", "rsl_rl", "--external_callback", CALLBACK]
    for key in CLI_KEYS:
        if key in run:
            args.extend((f"--{key}", str(run[key])))
    args.extend(("--run_timestamp", timestamp, "--run_name", run["name"]))
    args.append("--resume_optimizer_state" if run["resume_optimizer_state"] else "--no-resume_optimizer_state")
    if checkpoint:
        args.extend(("--checkpoint", checkpoint))
    if "physics" in run:
        args.append(f"physics={run['physics']}")
    for key, value in run["overrides"].items():
        args.append(f"{key}={json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))}")
    return args


def apply_reward_weights(cfg, weights: dict) -> None:
    """Apply after task-specific configuration, before zero-weight pruning."""
    rewards = cfg.rewards
    for name, weight in weights.items():
        term = getattr(rewards, name, None)
        if term is None or not hasattr(term, "weight"):
            raise ValueError(f"Reward {name!r} does not exist or has no reusable definition in this task")
        term.weight = float(weight)
    for name, term in vars(rewards).items():
        if getattr(term, "weight", None) == 0:
            setattr(rewards, name, None)


def configure_training() -> None:
    """Native --external_callback hook; changes only this child process's task factory."""
    import gymnasium as gym

    importlib.import_module("M20_newton_training.tasks")
    from M20_newton_training.tasks.m20.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

    spec = read_json(Path(os.environ[SPEC_VARIABLE]))
    run = spec["run"]
    task_spec = gym.spec(run["task"])
    entry = task_spec.kwargs["env_cfg_entry_point"]
    if isinstance(entry, str):
        module, symbol = entry.split(":", 1)
        env_class = getattr(importlib.import_module(module), symbol)
    else:
        env_class = entry
    if not isinstance(env_class, type) or not issubclass(env_class, LocomotionVelocityRoughEnvCfg):
        raise ValueError("Batch reward configuration currently supports this project's M20 velocity tasks")

    def configured_env():
        # Keep the actual config class (including its identity checks and pickle support).
        # Deferral is temporary and restored even when construction fails.
        with patch.object(LocomotionVelocityRoughEnvCfg, "disable_zero_weight_rewards", lambda self: None):
            cfg = env_class()
        apply_reward_weights(cfg, run["reward_weights"])
        return cfg

    task_spec.kwargs["env_cfg_entry_point"] = configured_env


def validate_spec(spec_path: Path) -> None:
    """Resolve the native parser, physics presets and Hydra config without starting simulation."""
    from isaaclab_rl.entrypoints.backends import train_rsl_rl

    spec = read_json(spec_path)
    args = list(spec["train_args"])
    # This is the parser used by the installed train backend; it also invokes our callback.
    library_index = args.index("--rl_library")
    del args[library_index : library_index + 2]
    parsed = train_rsl_rl._parse_args(args)
    cfg, agent = train_rsl_rl.resolve_task_config(parsed.task, parsed.agent)
    agent = train_rsl_rl.cli_args.update_rsl_rl_cfg(agent, parsed)
    train_rsl_rl.apply_env_overrides(parsed, cfg)
    agent.max_iterations = parsed.max_iterations
    cfg.validate()
    # The native agent schema deliberately retains MISSING deprecated aliases.
    # A generic configclass.validate() would reject a valid current RSL-RL config.
    agent = train_rsl_rl.handle_deprecated_rsl_rl_cfg(agent, train_rsl_rl.check_rsl_rl_version())
    weights = {name: term.weight for name, term in vars(cfg.rewards).items() if hasattr(term, "weight")}
    for name, expected in spec["run"]["reward_weights"].items():
        if weights.get(name, 0.0) != expected:
            raise ValueError(f"Reward {name!r} was overridden after batch configuration")
    log_dir = PROJECT_ROOT / "logs" / "rsl_rl" / agent.experiment_name / f"{spec['timestamp']}_{agent.run_name}"
    write_json(
        spec_path.with_name("validation.json"),
        {
            "task": parsed.task,
            "num_envs": cfg.scene.num_envs,
            "max_iterations": agent.max_iterations,
            "active_reward_weights": weights,
            "training_log_dir": str(log_dir),
        },
    )
    print(f"[VALID] {parsed.task}: {len(weights)} active rewards, {cfg.scene.num_envs} environments", flush=True)


def child_environment(spec_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env[SPEC_VARIABLE] = str(spec_path)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (
                str(PROJECT_ROOT / "scripts"),
                str(PROJECT_ROOT),
                env.get("PYTHONPATH", ""),
            ),
        )
    )
    return env


def stop_child(process: subprocess.Popen) -> None:
    """Forward Ctrl+C to the isolated training group, with bounded shutdown waits."""
    for sig, timeout in ((signal.SIGINT, 15), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            continue


def execute(command: list[str], spec_path: Path, log_path: Path) -> int:
    """Tee one child to its console log; never run another child until it has exited."""
    with (
        log_path.open("w", encoding="utf-8") as log,
        subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=child_environment(spec_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
        ) as process,
    ):
        try:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return process.wait()
        except BaseException:
            stop_child(process)
            raise


def resolve_checkpoint(value: str | None, previous: str | None) -> str | None:
    if value is None:
        return None
    if value == "previous":
        if previous is None:
            raise ValueError("checkpoint='previous' requires the immediately preceding run to have succeeded")
        return previous
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise ValueError(f"Checkpoint does not exist: {path}")
    return str(path.resolve())


def latest_checkpoint(log_dir: Path) -> str:
    candidates = [(int(m[1]), p) for p in log_dir.glob("model_*.pt") if (m := re.fullmatch(r"model_(\d+)\.pt", p.name))]
    if not candidates:
        raise ValueError(f"Training exited without a model checkpoint in {log_dir}")
    return str(max(candidates)[1])


def run_batch(args: argparse.Namespace) -> int:
    config_path = args.config.expanduser().resolve()
    runs = load_runs(config_path)
    if args.only:
        if unknown := set(args.only) - {run["name"] for run in runs}:
            raise ValueError(f"Unknown --only names: {sorted(unknown)}")
        runs = [run for run in runs if run["name"] in args.only]
    if runs[0].get("checkpoint") == "previous":
        raise ValueError("The first selected run cannot use checkpoint='previous'")
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + uuid.uuid4().hex[:6]
    output = (args.output_dir or PROJECT_ROOT / "output" / "training_batches" / stamp).expanduser().resolve()
    if args.dry_run:
        for index, run in enumerate(runs, 1):
            checkpoint = run.get("checkpoint")
            checkpoint = "<previous-run-checkpoint>" if checkpoint == "previous" else checkpoint
            command = [
                sys.executable,
                "-m",
                "isaaclab",
                "train",
                *build_train_args(run, f"{stamp}_{index:02}", checkpoint),
            ]
            print(f"\n[{index}/{len(runs)}] {run['name']}\n{shlex.join(command)}")
            print("reward_weights =", json.dumps(run["reward_weights"], ensure_ascii=False))
        print("\nDry run: no simulation, training or output directory was created.")
        return 0
    # Verify all explicit checkpoints before creating a training queue.
    for run in runs:
        if run.get("checkpoint") != "previous":
            resolve_checkpoint(run.get("checkpoint"), None)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"config_path": str(config_path), "runs": []}
    manifest_path = output / "manifest.json"
    write_json(output / "config.json", read_json(config_path))
    print(f"Batch output: {output}", flush=True)
    current = None
    try:
        # Every run is checked first, so a typo in group 2 cannot waste a group 1 training run.
        for index, run in enumerate(runs, 1):
            folder = output / f"{index:02}_{run['name']}"
            folder.mkdir()
            timestamp = f"{stamp}_{index:02}"
            checkpoint = (
                None if run.get("checkpoint") == "previous" else resolve_checkpoint(run.get("checkpoint"), None)
            )
            spec = {"run": run, "timestamp": timestamp, "train_args": build_train_args(run, timestamp, checkpoint)}
            spec_path = folder / "requested.json"
            write_json(spec_path, spec)
            current = {"name": run["name"], "status": "validating", "directory": str(folder)}
            manifest["runs"].append(current)
            write_json(manifest_path, manifest)
            code = execute(
                [sys.executable, str(Path(__file__).resolve()), "--_validate", str(spec_path)],
                spec_path,
                folder / "validation.log",
            )
            if code != 0:
                current.update(status="validation_failed", returncode=code)
                write_json(manifest_path, manifest)
                return 1
            current.update(read_json(folder / "validation.json"), status="validated")
            write_json(manifest_path, manifest)
        if args.validate_only:
            print("All configurations validated. Training was not started.", flush=True)
            return 0
        previous = None
        failed = False
        for current, run in zip(manifest["runs"], runs, strict=True):
            folder = Path(current["directory"])
            spec_path = folder / "requested.json"
            spec = read_json(spec_path)
            try:
                checkpoint = resolve_checkpoint(run.get("checkpoint"), previous)
                previous = None
                spec["train_args"] = build_train_args(run, spec["timestamp"], checkpoint)
                spec["resolved_checkpoint"] = checkpoint
                write_json(spec_path, spec)
                command = [sys.executable, "-m", "isaaclab", "train", *spec["train_args"]]
                current.update(status="running", command=command, started_at=datetime.now().isoformat())
                write_json(manifest_path, manifest)
                print(f"\nStarting {run['name']}: {shlex.join(command)}", flush=True)
                code = execute(command, spec_path, folder / "train.log")
                current["returncode"] = code
                if code != 0:
                    raise ValueError(f"Trainer exited with code {code}; see {folder / 'train.log'}")
                previous = latest_checkpoint(Path(current["training_log_dir"]))
                current.update(status="completed", checkpoint=previous)
            except (ValueError, OSError) as error:
                previous = None
                failed = True
                current.update(status="failed", error=str(error))
                print(f"[FAILED] {run['name']}: {error}", file=sys.stderr, flush=True)
            current["finished_at"] = datetime.now().isoformat()
            write_json(manifest_path, manifest)
            if failed and not args.continue_on_error:
                return 1
        return int(failed)
    except KeyboardInterrupt:
        if current is not None:
            current["status"] = "interrupted"
        write_json(manifest_path, manifest)
        print("\nBatch interrupted; no further training groups will start.", file=sys.stderr)
        return 130


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("train_batches.json"))
    parser.add_argument("--only", nargs="+", help="Select run names; keep their order in the configuration")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and weights without importing Isaac Lab")
    parser.add_argument("--validate-only", action="store_true", help="Resolve all configs without launching simulation")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue after a training failure")
    parser.add_argument("--output-dir", type=Path, help="New directory for batch configuration, logs and manifest")
    parser.add_argument("--_validate", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        if args._validate:
            validate_spec(args._validate)
            return 0
        return run_batch(args)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
