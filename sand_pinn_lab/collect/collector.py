"""Explicit fixed-duration episodes with complete sand and coupling-history resets."""

import hashlib
import json
import signal
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .config import DATA_ROOT, environment_config
from .geometry import read_wheel_geometry
from .policy import CheckpointPolicy, SlowRandomPolicy, set_command
from .storage import ChunkWriter, save_json, save_npz
from .surface import reference_surface


@contextmanager
def stop_signals():
    requested = {"signal": None}

    def stop(signum, _frame):
        requested["signal"] = signum

    previous = {s: signal.signal(s, stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield requested
    finally:
        for s, handler in previous.items():
            signal.signal(s, handler)


def reset_episode(raw, args, stop):
    import torch
    from newton import StateFlags

    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics.mpm_manager import NewtonMPMManager

    raw.reset()
    raw.scene["sand"].reset()
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    # Reset both manager buffers, MPM constitutive/warmstart state and proxy feedback.
    NewtonMPMManager.reset_solver_state()
    zeros = torch.zeros((1, 16), device=args.device)
    for _ in range(round(args.settle / args.physics.control_dt_s)):
        raw.step(zeros)
        if stop["signal"] is not None:
            return None
    spacing = raw.cfg.scene.sand.spawn.voxel_size / raw.cfg.scene.sand.spawn.particles_per_cell
    surface = reference_surface(NewtonManager.get_state_0().particle_q.numpy(), spacing)
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
        position=robot.data.default_joint_pos.torch, velocity=torch.zeros_like(robot.data.default_joint_vel.torch)
    )
    raw.action_manager.reset()
    raw.observation_manager.reset()
    raw.episode_length_buf.zero_()
    raw.scene.write_data_to_sim()
    raw.sim.forward()
    # Teleport only the robot: preserve the settled sand's constitutive state.
    NewtonMPMManager.reset_solver_state(flags=StateFlags.BODY_Q | StateFlags.BODY_QD)
    raw.scene.update(dt=0.0)
    return surface


def run(args):
    import gymnasium as gym
    import newton
    import torch
    import warp as wp

    from isaaclab.app import launch_simulation
    from isaaclab.utils import to_dict
    from isaaclab_newton.physics import NewtonManager

    import M20_newton_training.tasks  # noqa: F401

    from .record import MPMRecorder

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run this command on the M20 Newton machine with its GPU accessible.")
    cfg = environment_config(args)
    checkpoint_policy = (
        CheckpointPolicy(args.checkpoint, args.agent_config, args.device) if args.checkpoint is not None else None
    )
    physics_dt, control_dt = args.physics.physics_dt_s, args.physics.control_dt_s
    args.leg_joint_names = list(cfg.leg_joint_names)
    args.wheel_joint_names = list(cfg.wheel_joint_names)
    geometry = read_wheel_geometry(cfg.scene.robot.spawn.usd_path)
    stamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d_%H-%M-%S_%f")
    prefix = "random" if args.checkpoint is None else f"policy_{args.checkpoint.stem}"
    output = (args.output_dir or DATA_ROOT / f"{prefix}_{stamp}").expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "environment.json", json.loads(json.dumps(to_dict(cfg), default=str)))
    save_json(output / "physics.json", args.physics.to_dict())
    save_json(output / "geometry.json", geometry)
    sources = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    metadata = {
        "schema_version": 2,
        "particle_state_saved": False,
        "status": "initializing",
        "created_at": stamp,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "physics"},
        "physics": args.physics.to_dict(),
        "versions": {"newton": newton.__version__, "warp": wp.__version__, "torch": torch.__version__},
        "source_sha256": sources,
        "physics_dt_s": physics_dt,
        "control_dt_s": control_dt,
        "mpm_max_iterations": args.physics.mpm["max_iterations"],
        "mpm_tolerance": args.physics.mpm["tolerance"],
        "num_envs": 1,
        "requested_episodes": args.episodes,
        "episode_seconds": args.episode_seconds,
        "coordinate_convention": "world XYZ, Z up; pose=[x,y,z,qx,qy,qz,qw]; twist=[vx,vy,vz,wx,wy,wz] at COM",
        "force_convention": "MPM impulse/dt, force ON robot; per-body torque about that proxy body's input COM",
        "sampling": {
            "robot": "Fresh rigid-entry end state, immediately before MPM; time=episode_time_s",
            "proxy": "Actual MPM input collider state after coupling sync/rewind, on the final coupling pass",
            "proxy_pose_time": "episode_time_start_s"
            if args.physics.coupling["mode"] == "lagged"
            else "episode_time_s",
            "proxy_twist": "Actual input body_qd; backward collider velocity mode derives motion from pose history",
            "surface": "Particle state BEFORE MPM; time=episode_time_start_s",
            "force": "Average over [episode_time_start_s, episode_time_s]; not instantaneous end force",
            "time_s": "Concatenated recorded episodes, excludes settling",
            "simulation_time_s": "Includes all per-episode settling",
        },
        "reset": "Fixed-duration episodes; no early fall/edge reset; reset robot, particles, MPM and proxy history",
        "randomization": "Slow independent normalized targets; same initial bed, separately seeded episode actions",
        "force_scope": "Sand forces only; gravity, actuator torques and rigid-floor contacts are not included",
        "episodes": [],
    }
    if checkpoint_policy is not None:
        metadata["policy"] = {
            "kind": "rsl_rl_checkpoint",
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            "agent_config": str(args.agent_config),
            "agent_config_sha256": hashlib.sha256(args.agent_config.read_bytes()).hexdigest(),
            "command_body": args.command,
            "command_during_warmup": [0.0, 0.0, 0.0],
            "inference": "deterministic actor mean; saved observation normalizer and action clipping; no PPO updates",
            "random_target": "Executed checkpoint action; zeros during warmup (legacy field name)",
            "random_segment": "-1 (not a random segment)",
        }
        metadata["randomization"] = (
            "Fixed trained actor and body command; same initial bed; episode seed=seed+episode_id"
        )
    started = time.monotonic()
    writer = None
    entry = None
    recorder = None

    def save(flush=True):
        if flush and writer is not None:
            writer.flush()
        metadata["wall_seconds"] = time.monotonic() - started
        metadata["saved_frames"] = sum(e.get("frames", 0) for e in metadata["episodes"])
        save_json(output / "metadata.json", metadata)

    save()
    print(f"Output: {output}\n{args.episodes} episodes x {args.episode_seconds}s, dt={physics_dt}s", flush=True)
    try:
        with stop_signals() as stop, launch_simulation(cfg, {"device": args.device, "visualizer": [args.viz]}):
            raw = gym.make("M20NewtonTraining-M20-M20-WalkOnSand", cfg=cfg).unwrapped
            try:
                if raw.action_manager.total_action_dim != 16:
                    raise RuntimeError("Expected 12 position + 4 wheel velocity actions")
                robot = raw.scene["robot"]
                root_solver = NewtonManager._solver
                recorder = MPMRecorder(root_solver, robot, raw.action_manager, geometry, args)
                metadata.update(
                    mapping=recorder.mapping,
                    particle_count=recorder.solver.model.particle_count,
                    robot_mass_kg=float(robot.data.default_mass.torch.sum()),
                    status="running",
                )
                model = recorder.rigid.view
                save_npz(
                    output / "body_properties.npz",
                    mass_kg=model.body_mass.numpy()[recorder.rigid_ids],
                    inertia_body_kg_m2=model.body_inertia.numpy()[recorder.rigid_ids],
                    com_body_m=model.body_com.numpy()[recorder.rigid_ids],
                )
                save()
                with recorder, torch.inference_mode():
                    for episode in range(args.episodes):
                        if stop["signal"] is not None:
                            break
                        recorder.enabled = False
                        torch.manual_seed(args.seed + episode)
                        height = reset_episode(raw, args, stop)
                        if height is None:
                            break
                        writer = ChunkWriter(output / f"episode_{episode:03d}", round(args.chunk_seconds / physics_dt))
                        entry = {
                            "episode_id": episode,
                            "directory": writer.directory.name,
                            "status": "running",
                            "seed": args.seed + episode,
                            "reference_surface_height_w_m": height,
                            "frames": 0,
                        }
                        metadata["episodes"].append(entry)
                        recorder.begin_episode(
                            episode, writer, height, episode * (args.episode_seconds + args.settle) + args.settle
                        )
                        policy = checkpoint_policy or SlowRandomPolicy(
                            args.seed + episode,
                            limit=args.action_limit,
                            hold=tuple(args.hold_seconds),
                            transition=args.transition_seconds,
                        )
                        recorder.enabled = True
                        print(f"Episode {episode + 1}/{args.episodes}: surface={height:.4f} m", flush=True)
                        save()
                        for step in range(round(args.episode_seconds / control_dt)):
                            if stop["signal"] is not None:
                                break
                            t = step * control_dt
                            warming_up = t < args.warmup - 1e-9
                            if checkpoint_policy is not None:
                                set_command(raw, (0.0, 0.0, 0.0) if warming_up else args.command)
                                if warming_up:
                                    action = np.zeros(16, dtype=np.float32)
                                    policy.target = action.copy()
                                else:
                                    # Refresh after teleport/command assignment; never feed stale obs_buf.
                                    raw.obs_buf = raw.observation_manager.compute()
                                    action = policy(raw.obs_buf)
                            else:
                                action = np.zeros(16, dtype=np.float32) if warming_up else policy(t - args.warmup)
                            recorder.action, recorder.target = action, policy.target.copy()
                            recorder.segment, recorder.control_step = policy.segment, step
                            previous_frame = recorder.frame
                            _, _, terminated, truncated, _ = raw.step(torch.as_tensor(action[None], device=args.device))
                            if terminated.any() or truncated.any():
                                raise RuntimeError("Unexpected reset: collection requires explicit episode boundaries")
                            if recorder.frame - previous_frame != round(control_dt / physics_dt):
                                raise RuntimeError("Missing or duplicated MPM samples in control step")
                            entry["frames"] = recorder.frame
                            if (step + 1) % max(1, round(1 / control_dt)) == 0:
                                print(
                                    f"  {t + control_dt:.2f}/{args.episode_seconds:.2f}s; {recorder.frame} frames",
                                    flush=True,
                                )
                                save(flush=False)
                        recorder.enabled = False
                        entry.update(
                            status="interrupted" if stop["signal"] is not None else "complete",
                            frames=recorder.frame,
                            duration_s=recorder.frame * physics_dt,
                            stop_reason="signal" if stop["signal"] is not None else "time_limit",
                        )
                        if entry["status"] == "complete" and recorder.frame != round(args.episode_seconds / physics_dt):
                            raise RuntimeError("Episode frame count does not match requested duration")
                        save_json(writer.directory / "episode.json", entry)
                        save()
                metadata.update(
                    status="interrupted" if stop["signal"] is not None else "complete", stop_signal=stop["signal"]
                )
            finally:
                if recorder is not None:
                    recorder.enabled = False
                raw.close()
    except BaseException as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if entry is not None and entry["status"] == "running":
            entry.update(status="failed", frames=0 if writer is None else writer.frames, stop_reason=metadata["error"])
            save_json(writer.directory / "episode.json", entry)
        raise
    finally:
        save()
    print(f"Collection {metadata['status']}: {metadata['saved_frames']} frames saved to {output}", flush=True)
    return output
