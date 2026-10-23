"""Run a Walk checkpoint on one MPM bed and record a passive foot-force proxy."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from sand_proxy_lab.paths import add_terrain_argument, terrain_directory

ROOT = Path(__file__).resolve().parents[2]


def checkpoint_path(argument):
    if argument:
        result = Path(argument).expanduser().resolve()
    else:
        runs = [
            p
            for p in (ROOT / "logs/rsl_rl/deeprobotics_m20_walk").glob("*/model_*.pt")
            if p.stem.split("_")[-1].isdigit()
        ]
        if not runs:
            raise FileNotFoundError("No Walk checkpoint found; specify --checkpoint")
        result = max(runs, key=lambda p: (p.parent.name, int(p.stem.split("_")[-1])))
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def experiment_config(args):
    from isaaclab.utils import replace

    from M20_newton_training.tasks.m20.config.M20.walk_on_sand_env_cfg import DeeproboticsM20WalkOnSandEnvCfg

    cfg = DeeproboticsM20WalkOnSandEnvCfg()
    cfg.sim.physics = cfg.sim.physics.newton_sand
    cfg.sim.physics.use_cuda_graph = False
    mpm = next(entry for entry in cfg.sim.physics.solver_cfg.entries if entry.name == "sand")
    mpm.solver_cfg.max_iterations = args.mpm_iterations
    # Keep the MPM reference cadence used for collected traces, with a 50 Hz policy.
    cfg.sim.dt = args.physics_dt
    cfg.decimation = round(0.02 / args.physics_dt)
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.num_envs = 1
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.episode_length_s = args.duration + args.warmup + args.settle + 10
    cfg.observations.policy.enable_corruption = False
    # Let the sand settle before the robot reaches it; later place only the robot.
    robot = cfg.scene.robot
    start_xy = getattr(args, "start_xy", (0.0, 0.0))
    cfg.scene.robot = replace(robot, init_state=replace(robot.init_state, pos=(*start_xy, 1.8)))
    for name in vars(cfg.events):
        if name not in ("randomize_reset_base", "randomize_reset_joints"):
            setattr(cfg.events, name, None)
    cfg.events.randomize_reset_base.params["pose_range"] = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.events.randomize_reset_base.params["velocity_range"] = {
        key: (0.0, 0.0) for key in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.events.randomize_reset_joints.params.update(position_range=(1.0, 1.0), velocity_range=(0.0, 0.0))
    command = cfg.commands.base_velocity
    command.heading_command = False
    command.ranges.heading = None
    command.rel_standing_envs = 0.0
    for name in ("rel_zero_vel_envs", "rel_only_lin_x_envs", "rel_only_lin_y_envs", "rel_only_ang_z_envs"):
        setattr(command, name, 0.0)
    command.resampling_time_range = (1e9, 1e9)
    for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        setattr(command.ranges, name, (0.0, 0.0))
    return cfg


def settled_surface(points, spacing):
    """Median column-top height, in an undisturbed central patch before robot release."""
    points = np.asarray(points)
    points = points[(np.abs(points[:, 0]) < 0.3) & (np.abs(points[:, 1]) < 0.3)]
    if not len(points) or not np.isfinite(points).all():
        raise ValueError("No finite particles to measure the settled reference surface")
    xy = np.floor(points[:, :2] / (2 * spacing)).astype(int)
    _, inverse = np.unique(xy, axis=0, return_inverse=True)
    tops = np.full(inverse.max() + 1, -np.inf)
    np.maximum.at(tops, inverse, points[:, 2])
    return float(np.median(tops) + spacing / 2)


def run(args):
    import gymnasium as gym
    import newton
    import torch
    import warp as wp

    from isaaclab.app import launch_simulation
    from isaaclab.utils import to_dict
    from isaaclab_newton.physics import NewtonManager

    from isaaclab_rl.rsl_rl import (
        RslRlVecEnvWrapper,
        check_rsl_rl_version,
        create_rsl_rl_runner,
        handle_deprecated_rsl_rl_cfg,
    )

    import M20_newton_training.tasks  # noqa: F401
    from M20_newton_training.sand_proxy.foot_model import DEFAULT_MODEL_PATH
    from M20_newton_training.tasks.m20.config.M20.agents.rsl_rl_ppo_cfg import DeeproboticsM20WalkPPORunnerCfg

    from .record import ShadowRecorder
    from .report import analyze

    checkpoint = checkpoint_path(args.checkpoint)
    model_path = (args.model or DEFAULT_MODEL_PATH).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    terrain = terrain_directory(args.terrain, model_path)
    output = args.output_dir or terrain / "shadow" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = experiment_config(args)
    (output / "environment.json").write_text(json.dumps(to_dict(cfg), indent=2, default=str) + "\n")
    metadata = {
        "schema_version": 1,
        "terrain": terrain.name,
        "status": "initializing",
        "stop_reason": "initializing",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "model_path": str(model_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "command": args.command,
        "start_xy_m": args.start_xy,
        "seed": args.seed,
        "num_envs": 1,
        "duration_requested_s": args.duration,
        "warmup_s": args.warmup,
        "settle_s": args.settle,
        "physics_dt_s": cfg.sim.dt,
        "policy_dt_s": cfg.sim.dt * cfg.decimation,
        "mpm_iterations": args.mpm_iterations,
        "newton_version": newton.__version__,
        "warp_version": wp.__version__,
        "proxy_force_applied": False,
        "coordinate_frame": "world XYZ; force ON collider, Z up",
        "sampling": "MPM input colliders after staggered sync/rewind; impulse over the following solver interval",
        "evaluation": "No PPO update, no refit, no domain randomization, no observation corruption",
        "timestep_note": "MPM reference cadence; proxy training may use smaller dt for explicit-force stability",
    }
    (output / "model_snapshot.json").write_bytes(model_path.read_bytes())
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Checkpoint: {checkpoint}\nOutput: {output}", flush=True)
    recorder = None
    started = time.monotonic()
    failure = None
    with launch_simulation(cfg, {"device": args.device, "visualizer": [args.viz]}):
        raw = gym.make("M20NewtonTraining-M20-M20-WalkOnSand", cfg=cfg).unwrapped
        try:
            agent_cfg = DeeproboticsM20WalkPPORunnerCfg()
            agent_cfg.device = args.device
            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, check_rsl_rl_version())
            env = RslRlVecEnvWrapper(raw, clip_actions=agent_cfg.clip_actions)
            runner = create_rsl_rl_runner(env, agent_cfg)
            runner.load(str(checkpoint))
            policy = runner.get_inference_policy(device=args.device)
            root_solver = NewtonManager._solver  # Coupled entry access is backend-specific in IsaacLab 3.0.
            solver = root_solver.solver("sand")
            recorder = ShadowRecorder(solver, output / "model_snapshot.json", 1.0, args.device, 0.30)
            with recorder, torch.inference_mode():
                zeros = torch.zeros((1, 16), device=args.device)
                for _ in range(round(args.settle / raw.step_dt)):
                    _, _, done, _ = env.step(zeros)
                    if done.any():
                        raise RuntimeError("Environment reset while sand was settling")
                points = NewtonManager.get_state_0().particle_q.numpy()
                spacing = cfg.scene.sand.spawn.voxel_size / cfg.scene.sand.spawn.particles_per_cell
                surface = settled_surface(points, spacing)
                robot = raw.scene["robot"]
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
                raw.episode_length_buf.zero_()
                raw.scene.write_data_to_sim()
                raw.sim.forward()
                raw.scene.update(dt=0.0)
                mass = float(robot.data.default_mass.torch.sum())
                recorder.mass.fill_(mass / 4)
                recorder.surface_height = surface
                metadata.update(
                    surface_height_m=surface,
                    robot_weight_N=mass * 9.81,
                    robot_mass_kg=mass,
                    mapping=recorder.mapping,
                    particle_count=len(points),
                    status="running",
                    stop_reason="running",
                )
                metadata["observation_shapes"] = {
                    k: list(v.shape) for k, v in raw.observation_manager.compute().items()
                }
                if metadata["observation_shapes"]["policy"] != [1, 57] or raw.action_manager.total_action_dim != 16:
                    raise ValueError("Walk checkpoint expects 57 actor inputs and 16 actions")
                print(f"Settled surface: {surface:.4f} m; mass: {mass:.3f} kg; begin shadow recording", flush=True)
                recorder.enabled = True
                command_term = raw.command_manager.get_term("base_velocity")
                for step in range(round((args.duration + args.warmup) / raw.step_dt)):
                    t = step * raw.step_dt
                    command = (0.0, 0.0, 0.0) if t < args.warmup else args.command
                    command_term.vel_command_b[:] = torch.tensor(command, device=args.device)
                    command_term.is_standing_env.zero_()
                    command_term.is_heading_env.zero_()
                    # Recompute after setting command, so the actor sees the command of this step.
                    raw.obs_buf = raw.observation_manager.compute()
                    obs = env.get_observations()
                    action = policy(obs)
                    if not torch.isfinite(action).all():
                        raise FloatingPointError("Policy returned nonfinite actions")
                    recorder.command = np.asarray(command)
                    recorder.action = action[0].clamp(-agent_cfg.clip_actions, agent_cfg.clip_actions).cpu().numpy()
                    recorder.observation = obs["policy"][0].cpu().numpy()
                    recorder.control_step = step
                    _, _, done, _ = env.step(action)
                    if step % 50 == 0:
                        print(f"  {recorder.elapsed:.2f} s; {len(recorder.rows)} MPM samples", flush=True)
                        recorder.save(output / "traces.npz", metadata)
                        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                    if done.any():
                        metadata.update(
                            status="ended_early",
                            stop_reason="environment termination",
                            termination_terms={
                                name: bool(raw.termination_manager.get_term(name)[0])
                                for name in raw.termination_manager.active_terms
                            },
                        )
                        break
                    base = recorder.rows[-1]["base_pose_w"]
                    if np.max(np.abs(base[:2])) > 0.85:
                        metadata.update(status="ended_early", stop_reason="robot approaching tray boundary")
                        break
                else:
                    metadata.update(status="complete", stop_reason="requested duration reached")
        except BaseException as exc:
            metadata.update(status="failed", stop_reason=f"{type(exc).__name__}: {exc}")
            failure = exc
        finally:
            metadata["wall_seconds"] = time.monotonic() - started
            if recorder is not None:
                metadata["physics_samples"] = len(recorder.rows)
                recorder.save(output / "traces.npz", metadata)
            (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
            raw.close()
    if recorder is not None and recorder.rows:
        summary = analyze(output / "traces.npz")
        print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2), flush=True)
    if failure is not None:
        raise failure
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Walk .pt checkpoint; default latest run's largest model number")
    parser.add_argument("--model", type=Path, help="Default: deployed m20_foot_sand.json")
    add_terrain_argument(parser)
    parser.add_argument("--output-dir", type=Path, help="Override terrain/shadow/<timestamp>; must be a new directory")
    parser.add_argument("--duration", type=float, default=4.0, help="Recorded walking duration, in addition to warmup")
    parser.add_argument("--warmup", type=float, default=0.5, help="Recorded zero-command warmup, excluded from metrics")
    parser.add_argument("--settle", type=float, default=0.30, help="Sand settling with robot clear of the surface")
    parser.add_argument("--command", type=float, nargs=3, default=[0.15, 0.0, 0.0], metavar=("VX", "VY", "WZ"))
    parser.add_argument("--physics-dt", type=float, default=0.0025)
    parser.add_argument("--mpm-iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-xy", type=float, nargs=2, default=[0.0, 0.0], metavar=("X", "Y"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--viz", choices=("none", "newton"), default="none")
    parser.add_argument("--analyze", type=Path, help="Rebuild report from traces.npz without CUDA or simulation")
    args = parser.parse_args()
    if args.analyze:
        from .report import analyze

        analyze(args.analyze)
        return
    if not np.isfinite([args.duration, args.warmup, args.settle, args.physics_dt, *args.command]).all():
        parser.error("All durations, timestep and command components must be finite")
    if args.duration <= 0 or args.warmup < 0 or not 0 <= args.settle <= 0.4:
        parser.error("duration >0, warmup >=0, and settle in [0,0.4] seconds required")
    if args.physics_dt <= 0 or not np.isclose(round(0.02 / args.physics_dt) * args.physics_dt, 0.02):
        parser.error("physics-dt must divide the 20 ms policy period")
    if args.mpm_iterations < 1:
        parser.error("mpm-iterations must be positive")
    run(args)


if __name__ == "__main__":
    main()
