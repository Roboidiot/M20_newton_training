"""Collection settings; importing CLI help does not initialize a simulator."""

import argparse
import math
from pathlib import Path

from .physics import apply_physics_config, load_physics_config

DATA_ROOT = Path(__file__).resolve().parents[1] / "data"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Independent M20 random-action MPM data collection.")
    p.add_argument(
        "--physics-config", type=Path, help="Partial/full physics JSON; overlays collect/configs/default.json"
    )
    p.add_argument(
        "--set",
        dest="physics_overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help="Repeatable physics override, e.g. --set material.friction=0.8 --set mpm.collider_basis=S2",
    )
    p.add_argument(
        "--print-config", action="store_true", help="Print resolved physics JSON and exit without a simulator"
    )
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--episode-seconds", type=float, default=10.0)
    p.add_argument("--settle", type=float, default=0.3, help="Unrecorded settling before EACH episode, at most 0.4 s")
    p.add_argument("--warmup", type=float, default=0.5, help="Zero-action time INCLUDED in each episode")
    p.add_argument("--action-limit", type=float, default=1.5, help="Uniform normalized action target bounds +/- this")
    p.add_argument("--hold-seconds", type=float, nargs=2, default=[0.5, 1.0], metavar=("MIN", "MAX"))
    p.add_argument("--transition-seconds", type=float, default=0.2, help="Smoothstep transition to each new target")
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--start-xy", type=float, nargs=2, default=[0.0, 0.0])
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--viz", choices=["none", "newton"], default="none")
    p.add_argument("--output-dir", type=Path, help="New run directory; default sand_pinn_lab/data/<timestamp>")
    p.add_argument("--chunk-seconds", type=float, default=1.0, help="Flush one compressed chunk per simulated second")
    args = p.parse_args(argv)
    try:
        args.physics = load_physics_config(args.physics_config, args.physics_overrides)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    control_dt = args.physics.control_dt_s
    numbers = [
        args.episode_seconds,
        args.settle,
        args.warmup,
        args.action_limit,
        *args.hold_seconds,
        args.transition_seconds,
        args.chunk_seconds,
        *args.start_xy,
    ]
    if not all(math.isfinite(x) for x in numbers):
        p.error("Numeric parameters must be finite")
    if args.episodes < 1 or args.seed < 0:
        p.error("Need episodes>=1, seed>=0")
    if args.episode_seconds <= 0 or not 0 <= args.warmup < args.episode_seconds or not 0 <= args.settle <= 0.4:
        p.error("Need episode-seconds>0, 0<=warmup<episode-seconds, 0<=settle<=0.4")
    if args.action_limit <= 0 or not 0 < args.hold_seconds[0] <= args.hold_seconds[1]:
        p.error("Action limit and hold bounds must be positive and ordered")
    if not 0 <= args.transition_seconds <= args.hold_seconds[0] or args.chunk_seconds <= 0:
        p.error("Need 0<=transition-seconds<=minimum hold, chunk-seconds>0")
    for name, value in (
        ("episode-seconds", args.episode_seconds),
        ("settle", args.settle),
        ("warmup", args.warmup),
        ("chunk-seconds", args.chunk_seconds),
    ):
        if not math.isclose(value / control_dt, round(value / control_dt), abs_tol=1e-7):
            p.error(f"{name} must be a multiple of {control_dt} s")
    return args


def environment_config(args):
    """Reuse the physical M20 scene, independently configure collection (no shadow imports)."""
    from isaaclab.utils import replace

    from M20_newton_training.tasks.m20.config.M20.walk_on_sand_env_cfg import DeeproboticsM20WalkOnSandEnvCfg

    cfg = DeeproboticsM20WalkOnSandEnvCfg()
    cfg.sim.physics = cfg.sim.physics.newton_sand
    apply_physics_config(cfg, args.physics)
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.seed = args.seed
    cfg.episode_length_s = args.episode_seconds + args.settle + 10
    cfg.observations.policy.enable_corruption = False
    cfg.observations.critic.enable_corruption = False
    robot = cfg.scene.robot
    cfg.scene.robot = replace(robot, init_state=replace(robot.init_state, pos=(*args.start_xy, 1.8)))
    # Collection owns timing, force sampling and resets; no reward-side solver hooks.
    cfg.actions.mpm_contact = None
    for group in (cfg.terminations, cfg.curriculum, cfg.rewards):
        for name in vars(group):
            if not name.startswith("_"):
                setattr(group, name, None)
    for name in vars(cfg.events):
        if not name.startswith("_") and name not in ("randomize_reset_base", "randomize_reset_joints"):
            setattr(cfg.events, name, None)
    for key in ("pose_range", "velocity_range"):
        cfg.events.randomize_reset_base.params[key] = {
            name: (0.0, 0.0) for name in ("x", "y", "z", "roll", "pitch", "yaw")
        }
    cfg.events.randomize_reset_joints.params.update(position_range=(1.0, 1.0), velocity_range=(0.0, 0.0))
    command = cfg.commands.base_velocity
    command.heading_command = False
    command.ranges.heading = None
    command.rel_standing_envs = 0.0
    command.resampling_time_range = (1e9, 1e9)
    for name in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        setattr(command.ranges, name, (0.0, 0.0))
    return cfg
