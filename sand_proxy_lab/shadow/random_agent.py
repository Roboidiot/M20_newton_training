"""Longer MPM-only shadow evaluation using IsaacLab's uniform RandomAgent and explicit resets."""

from __future__ import annotations

import argparse
import hashlib
import json
import signal
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

from sand_proxy_lab.paths import add_terrain_argument, terrain_directory

from .__main__ import experiment_config, settled_surface


@contextmanager
def pause_signals():
    """Request a save at the next control boundary; avoid interrupts inside Warp destructors."""
    state = {"requested": False, "signal": None}

    def request_pause(signum, _frame):
        state.update(requested=True, signal=signum)

    previous = {sig: signal.signal(sig, request_pause) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield state
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def reset_reason(pose, surface, episode_time, max_episode):
    """World pose is xyz + xyzw; detect falls without unsupported contact sensors."""
    x, y, z, w = pose[3:]
    up_z = 1 - 2 * (x * x + y * y)
    if up_z < np.cos(np.pi / 3):
        return "fall_tilt_over_60deg"
    if pose[2] - surface < 0.20:
        return "fall_base_clearance_under_020m"
    if np.max(np.abs(pose[:2])) > 0.85:
        return "tray_boundary"
    if episode_time >= max_episode - 1e-8:
        return "episode_time_limit"
    return None


def run(args):
    import gymnasium as gym
    import newton
    import torch
    import warp as wp

    from isaaclab.app import launch_simulation
    from isaaclab.utils import to_dict
    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics.mpm_manager import NewtonMPMManager

    from isaaclab_rl.entrypoints.simple_agents import create_random_action_policy

    import M20_newton_training.tasks  # noqa: F401
    from M20_newton_training.sand_proxy.foot_model import DEFAULT_MODEL_PATH

    from .random_report import analyze
    from .record import ShadowRecorder

    model = (args.model or DEFAULT_MODEL_PATH).resolve()
    terrain = terrain_directory(args.terrain, model)
    output = args.output_dir or terrain / "shadow" / ("random_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = experiment_config(args)
    # All resets are explicit here: capture terminal-window forces before resetting the scene.
    for name in vars(cfg.terminations):
        setattr(cfg.terminations, name, None)
    for name in vars(cfg.curriculum):
        setattr(cfg.curriculum, name, None)
    (output / "environment.json").write_text(json.dumps(to_dict(cfg), indent=2, default=str) + "\n")
    (output / "model_snapshot.json").write_bytes(model.read_bytes())
    metadata = {
        "schema_version": 2,
        "terrain": terrain.name,
        "agent": "isaaclab.random_agent",
        "action_distribution": "Independent U(-1,1) for all 16 actions, resampled every 20 ms",
        "checkpoint": None,
        "model_path": str(model),
        "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "num_envs": 1,
        "seed": args.seed,
        "command": [0, 0, 0],
        "start_xy_m": args.start_xy,
        "duration_requested_s": args.duration,
        "duration_definition": "Total recorded episode physics, including per-episode warmup; settling excluded",
        "warmup_s": args.warmup,
        "settle_s_per_episode": args.settle,
        "max_episode_s": args.max_episode,
        "physics_dt_s": cfg.sim.dt,
        "policy_dt_s": cfg.sim.dt * cfg.decimation,
        "mpm_iterations": args.mpm_iterations,
        "newton_version": newton.__version__,
        "warp_version": wp.__version__,
        "proxy_force_applied": False,
        "coordinate_frame": "world XYZ; force ON collider, Z up",
        "sampling": "MPM input colliders after staggered sync/rewind; impulse over following solver interval",
        "reset": "Reset robot, particle positions/velocities, MPM solver internals, all proxy histories; settle again",
        "status": "initializing",
        "episodes": [],
    }
    recorder = None
    failure = None
    started = time.monotonic()

    def save():
        metadata["wall_seconds"] = time.monotonic() - started
        if recorder is not None:
            metadata["physics_samples"] = len(recorder.rows)
            metadata["recorded_duration_s"] = recorder.elapsed
            recorder.save(output / "traces.npz", metadata)
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    save()
    print(f"RandomAgent shadow output: {output}", flush=True)
    with pause_signals() as stop, launch_simulation(cfg, {"device": args.device, "visualizer": [args.viz]}):
        raw = gym.make("M20NewtonTraining-M20-M20-WalkOnSand", cfg=cfg).unwrapped
        try:
            torch.manual_seed(args.seed)
            policy = create_random_action_policy(raw)
            solver = NewtonManager._solver.solver("sand")
            robot = raw.scene["robot"]
            mass = float(robot.data.default_mass.torch.sum())
            recorder = ShadowRecorder(solver, output / "model_snapshot.json", mass / 4, args.device, 0.30)
            metadata.update(robot_mass_kg=mass, robot_weight_N=mass * 9.81, mapping=recorder.mapping, status="running")
            zeros = torch.zeros((1, raw.action_manager.total_action_dim), device=args.device)
            assert zeros.shape == (1, 16)
            command_term = raw.command_manager.get_term("base_velocity")
            spacing = cfg.scene.sand.spawn.voxel_size / cfg.scene.sand.spawn.particles_per_cell
            episode = 0
            step = 0
            with recorder, torch.inference_mode():
                while recorder.elapsed < args.duration - raw.step_dt / 2 and not stop["requested"]:
                    recorder.enabled = False
                    raw.reset()
                    # MPM history is not implicitly reset by normal robot autoresets.
                    raw.scene["sand"].reset()
                    raw.scene.write_data_to_sim()
                    raw.sim.forward()
                    NewtonMPMManager.reset_solver_state()
                    for _ in range(round(args.settle / raw.step_dt)):
                        raw.step(zeros)
                        if stop["requested"]:
                            break
                    if stop["requested"]:
                        break
                    points = NewtonManager.get_state_0().particle_q.numpy()
                    surface = settled_surface(points, spacing)
                    recorder.reset_episode(episode, surface)
                    robot.reset()
                    root = robot.data.default_root_state.torch.clone()
                    root[:, :3] = raw.scene.env_origins
                    root[:, :2] += torch.tensor(args.start_xy, device=args.device)
                    root[:, 2] += surface + 0.58
                    root[:, 7:] = 0
                    robot.write_root_link_pose_to_sim_index(root_pose=root[:, :7])
                    robot.write_root_link_velocity_to_sim_index(root_velocity=root[:, 7:])
                    robot.write_joint_state_to_sim_index(
                        position=robot.data.default_joint_pos.torch,
                        velocity=torch.zeros_like(robot.data.default_joint_vel.torch),
                    )
                    raw.action_manager.reset()
                    raw.observation_manager.reset()
                    raw.episode_length_buf.zero_()
                    raw.scene.write_data_to_sim()
                    raw.sim.forward()
                    raw.scene.update(dt=0.0)
                    entry = {
                        "episode_id": episode,
                        "recorded_start_s": recorder.elapsed,
                        "surface_height_m": surface,
                        "status": "running",
                        "start_xy_m": args.start_xy,
                    }
                    metadata["episodes"].append(entry)
                    metadata["particle_count"] = len(points)
                    recorder.enabled = True
                    while recorder.elapsed < args.duration - raw.step_dt / 2 and not stop["requested"]:
                        command_term.vel_command_b.zero_()
                        command_term.is_standing_env.zero_()
                        command_term.is_heading_env.zero_()
                        obs = raw.observation_manager.compute()
                        action = zeros if recorder.episode_elapsed < args.warmup - 1e-8 else policy()
                        recorder.action = action[0].cpu().numpy()
                        recorder.observation = obs["policy"][0].cpu().numpy()
                        recorder.control_step = step
                        _, _, terminated, truncated, _ = raw.step(action)
                        if terminated.any() or truncated.any():
                            raise RuntimeError(
                                "Unexpected automatic reset; experiment uses explicit episode boundaries"
                            )
                        if not all(torch.isfinite(value).all() for value in raw.obs_buf.values()):
                            raise FloatingPointError("Environment returned nonfinite observations")
                        step += 1
                        pose = robot.data.root_link_pose_w.torch[0].cpu().numpy()
                        reason = reset_reason(pose, surface, recorder.episode_elapsed, args.max_episode)
                        if reason is not None:
                            entry.update(
                                status="reset",
                                reset_reason=reason,
                                duration_s=recorder.episode_elapsed,
                                recorded_end_s=recorder.elapsed,
                                terminal_pose_w=pose.tolist(),
                            )
                            print(
                                f"  Episode {episode}: {reason}; length={recorder.episode_elapsed:.2f}s; "
                                f"total={recorder.elapsed:.2f}s",
                                flush=True,
                            )
                            save()
                            break
                        if step % 50 == 0:
                            print(
                                f"  {recorder.elapsed:.2f}/{args.duration:.2f} s; "
                                f"episode {episode}; {len(recorder.rows)} MPM samples",
                                flush=True,
                            )
                            save()
                    if entry["status"] == "running":
                        entry.update(
                            status="finished",
                            reset_reason="user_pause" if stop["requested"] else "experiment_duration",
                            duration_s=recorder.episode_elapsed,
                            recorded_end_s=recorder.elapsed,
                        )
                    episode += 1
                metadata.update(
                    status="paused" if stop["requested"] else "complete",
                    stop_reason="user requested stop; saved at control boundary"
                    if stop["requested"]
                    else "requested recorded simulation duration reached",
                )
        except BaseException as exc:
            metadata.update(status="failed", stop_reason=f"{type(exc).__name__}: {exc}")
            failure = exc
        finally:
            save()
            raw.close()
    if recorder is not None and recorder.rows:
        analyze(output / "traces.npz")
    if failure is not None:
        raise failure
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_terrain_argument(parser)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--warmup", type=float, default=0.5)
    parser.add_argument("--settle", type=float, default=0.3)
    parser.add_argument("--max-episode", type=float, default=6.0)
    parser.add_argument("--physics-dt", type=float, default=0.0025)
    parser.add_argument("--mpm-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--start-xy", type=float, nargs=2, default=[0.065, 0.0])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--viz", choices=("none", "newton"), default="none")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--analyze", type=Path)
    args = parser.parse_args()
    if args.analyze:
        from .random_report import analyze

        analyze(args.analyze)
        return
    if not np.isfinite(
        [args.duration, args.warmup, args.settle, args.physics_dt, args.max_episode, *args.start_xy]
    ).all():
        parser.error("All numeric arguments must be finite")
    if args.duration <= 0 or not 0 <= args.warmup < args.max_episode or not 0 <= args.settle <= 0.4:
        parser.error("Need duration>0, 0<=warmup<max-episode, 0<=settle<=0.4")
    if args.physics_dt <= 0 or not np.isclose(round(0.02 / args.physics_dt) * args.physics_dt, 0.02):
        parser.error("physics-dt must divide the 20 ms action period")
    if args.mpm_iterations < 1:
        parser.error("mpm-iterations must be positive")
    run(args)


if __name__ == "__main__":
    main()
