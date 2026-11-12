#!/usr/bin/env python3
"""Run a short, real PPO training and plot 50 Hz foot depth/support-force traces.

The native external callback observes contact publication before episode reset.
It does not change task rewards, actions, observations or the PPO update loop.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEET = ("FL", "FR", "HL", "HR")
TRACE_OUTPUT = "M20_PROXY_TRAINING_TRACE_OUTPUT"
DEFAULT_CHECKPOINT = "logs/rsl_rl/deeprobotics_m20_walk_on_sand_proxy/height0.62deg15/model_11000.pt"


class TrainingTrace:
    def __init__(self, output):
        self.output = Path(output)
        self.samples = []

    def record(self, term):

        from M20_newton_training.tasks.m20.mdp.foot_sand_proxy import rotate_xyzw

        env = term._env
        if env.num_envs != 1:
            raise ValueError("This validation records exactly one training environment")
        data, ids = term._asset.data, term.body_ids
        quat = data.body_link_quat_w.torch[:, ids]
        center = data.body_link_pos_w.torch[:, ids] + rotate_xyzw(quat, term.local_center)
        axis = rotate_xyzw(quat, term.local_axis)
        bottom = center[..., 2] - term.radius - term.half_length * axis[..., 2].abs()

        def cpu(value):
            return value.detach().cpu().numpy().copy()

        state = term._contact_state
        proxy = cpu(term.force_w[0])
        combined = cpu(term.contact_force_w[0])
        dt = env.physics_dt * env.cfg.decimation
        self.samples.append(
            {
                "time_s": (len(self.samples) + 1) * dt,
                "depth_m": cpu((term.surface_height - bottom)[0]),
                "input_depth_m": cpu(term.model.depth[0]),
                "proxy_force_w_N": proxy,
                "rigid_force_w_N": combined - proxy,
                "combined_force_w_N": combined,
                "support": cpu(state.vertical_contact[0]),
                "touchdown": cpu(state.touchdown[0, 0]),
                "air_time_s": cpu(state.current_air_time[0]),
                "reset": bool(env.reset_buf[0].item()),
                "episode_step": int(env.episode_length_buf[0].item()),
                "command": cpu(env.command_manager.get_command("base_velocity")[0]),
            }
        )
        if len(self.samples) == 1:
            weights = {name: term.weight for name, term in vars(env.cfg.rewards).items() if hasattr(term, "weight")}
            details = {
                "sample_dt_s": dt,
                "surface_height_m": term.cfg.surface_height,
                "support_threshold_N": term.cfg.contact_force_threshold,
                "reward_weights": weights,
                "model_kind": term.cfg.model_kind,
                "model_path": term.cfg.model_path,
                "rigid_contact_sensor": term.cfg.rigid_contact_sensor,
                "force_frame": "world XYZ; positive Z is upward",
                "depth_m": "post-physics fixed surface minus capsule bottom; signed",
                "input_depth_m": "depth passed to the last proxy force prediction, before the last physics step",
                "collection_phase": "native PPO training, contact publication after physics and before reset",
            }
            (self.output / "signals.json").write_text(json.dumps(details, indent=2) + "\n")
        if len(self.samples) % 250 == 0:
            self.save()

    def save(self):
        if not self.samples:
            return
        arrays = {key: np.stack([sample[key] for sample in self.samples]) for key in self.samples[0]}
        arrays["foot_names"] = np.array(FEET)
        temporary = self.output / "trace.tmp.npz"
        np.savez_compressed(temporary, **arrays)
        temporary.replace(self.output / "trace.npz")


def configure_training():
    """Native --external_callback entry; the patch lives only in this child process."""
    import M20_newton_training.tasks  # noqa: F401
    from M20_newton_training.tasks.m20.mdp.foot_sand_proxy import FootSandProxyAction

    recorder = TrainingTrace(os.environ[TRACE_OUTPUT])
    original = FootSandProxyAction.update_contact_state

    def traced_update(term):
        pending = term._contact_pending
        original(term)
        if pending and not term._contact_pending:
            recorder.record(term)

    FootSandProxyAction.update_contact_state = traced_update
    atexit.register(recorder.save)


def plot_trace(output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    with np.load(output / "trace.npz") as raw:
        trace = {key: raw[key] for key in raw.files}
    details = json.loads((output / "signals.json").read_text())
    proxy, rigid, combined = [trace[key] for key in ("proxy_force_w_N", "rigid_force_w_N", "combined_force_w_N")]
    if not all(np.isfinite(value).all() for value in (trace["depth_m"], proxy, rigid, combined)):
        raise ValueError("Trace contains nonfinite depth or force values")
    if not np.allclose(proxy + rigid, combined, rtol=1e-5, atol=1e-3):
        raise ValueError("Combined force does not equal proxy plus rigid force")
    threshold = details["support_threshold_N"]
    if not np.array_equal(trace["support"], combined[..., 2] > threshold):
        raise ValueError("Recorded support predicate does not match combined Fz")
    time = trace["time_s"]
    fig, axes = plt.subplots(4, 2, figsize=(16, 12), sharex=True, layout="constrained")
    for foot, name in enumerate(FEET):
        left, right = axes[foot]
        left.plot(time, trace["depth_m"][:, foot] * 100, label="Post-physics depth", lw=1)
        left.plot(time, trace["input_depth_m"][:, foot] * 100, label="Proxy input depth", lw=0.7, alpha=0.6)
        left.axhline(0, color="gray", lw=0.7)
        left.axhline(details["surface_height_m"] * 100, color="brown", ls="--", lw=0.8, label="Rigid floor")
        left.set_ylabel(f"{name} depth [cm]")
        right.plot(time, proxy[:, foot, 2], color="tab:orange", label="Proxy Fz", lw=0.9)
        right.plot(time, rigid[:, foot, 2], color="tab:blue", label="Rigid floor Fz", lw=0.9)
        right.plot(time, combined[:, foot, 2], color="black", label="Combined Fz", lw=0.8, alpha=0.8)
        right.axhline(threshold, color="green", ls="--", lw=0.8, label=f"Support threshold ({threshold:g} N)")
        right.set_ylabel(f"{name} force [N]")
        for ax in (left, right):
            ax.grid(alpha=0.2)
            for reset_time in time[trace["reset"]]:
                ax.axvline(reset_time, color="red", alpha=0.25, lw=0.7)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    axes[0, 1].legend(fontsize=8, loc="upper right")
    for ax in axes[-1]:
        ax.set_xlabel("Training simulation time [s]; red lines mark episode resets")
    fig.suptitle("WalkOnSandProxy: one-environment PPO training, sampled at 50 Hz", fontsize=14)
    fig.savefig(output / "foot_depth_forces.png", dpi=170)
    fig.savefig(output / "foot_depth_forces.pdf")
    for row in axes:
        for ax in row:
            ax.set_xlim(0, min(8.0, float(time[-1])))
    fig.suptitle("WalkOnSandProxy: first 8 s of PPO training, sampled at 50 Hz", fontsize=14)
    fig.savefig(output / "foot_depth_forces_first_8s.png", dpi=170)
    plt.close(fig)
    stats = {
        "samples": len(time),
        "duration_s": float(time[-1]),
        "resets": int(trace["reset"].sum()),
        "feet": {
            name: {
                "depth_range_m": [float(trace["depth_m"][:, i].min()), float(trace["depth_m"][:, i].max())],
                "proxy_fz_range_N": [float(proxy[:, i, 2].min()), float(proxy[:, i, 2].max())],
                "rigid_fz_max_N": float(rigid[:, i, 2].max()),
                "combined_fz_max_N": float(combined[:, i, 2].max()),
                "rigid_supported_samples": int((rigid[:, i, 2] > threshold).sum()),
                "floor_corrected_support_samples": int(((proxy[:, i, 2] <= threshold) & trace["support"][:, i]).sum()),
            }
            for i, name in enumerate(FEET)
        },
    }
    (output / "summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--from-scratch", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--plot-only", type=Path, metavar="RUN_DIRECTORY")
    args = parser.parse_args(argv)
    if args.plot_only is not None:
        plot_trace(args.plot_only)
        return 0
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    output = args.output_dir or ROOT / "output/proxy_training_trace" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-m",
        "isaaclab",
        "train",
        "--rl_library",
        "rsl_rl",
        "--task",
        "M20NewtonTraining-M20-M20-WalkOnSandProxy",
        "--num_envs",
        "1",
        "--max_iterations",
        str(args.iterations),
        "--no-resume_optimizer_state",
        "--logger",
        "tensorboard",
        "--viz",
        "none",
        "--seed",
        "42",
        "--run_name",
        "single_env_force_validation",
        "--external_callback",
        "proxy_training_trace.configure_training",
        "physics=newton_mjwarp",
    ]
    if not args.from_scratch:
        checkpoint = Path(args.checkpoint).expanduser()
        checkpoint = checkpoint if checkpoint.is_absolute() else ROOT / checkpoint
        if not checkpoint.is_file():
            parser.error(f"Checkpoint does not exist: {checkpoint}")
        command.extend(["--checkpoint", str(checkpoint)])
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    env = os.environ.copy()
    env[TRACE_OUTPUT] = str(output)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "scripts"), str(ROOT), env.get("PYTHONPATH")]))
    print(f"Training trace: {output}", flush=True)
    with (output / "training.log").open("w") as log, subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as child:
        try:
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
        except KeyboardInterrupt:
            child.terminate()
        status = child.wait()
    if (output / "trace.npz").exists():
        plot_trace(output)
    else:
        print(f"No training samples were recorded. See {output / 'training.log'}", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
