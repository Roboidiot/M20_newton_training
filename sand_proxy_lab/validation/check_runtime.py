"""Headless nominal hold test: confirm finite forces and support above the rigid floor."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import torch

from isaaclab.app import launch_simulation

import M20_newton_training.tasks  # noqa: F401
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_proxy_env_cfg import (
    DeeproboticsM20WalkOnSandProxyEnvCfg,
    M20FootProxyPhysicsCfg,
)
from sand_proxy_lab.paths import add_terrain_argument, terrain_directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_terrain_argument(parser)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--model", type=Path, help="Optional candidate parameter JSON")
    parser.add_argument("--physics-dt", type=float, help="Optional physical step; policy remains 50 Hz")
    parser.add_argument("--output", type=Path, help="Override terrain/validation/runtime_support.json")
    args = parser.parse_args()
    args.output = args.output or terrain_directory(args.terrain, args.model) / "validation/runtime_support.json"
    cfg = DeeproboticsM20WalkOnSandProxyEnvCfg()
    if args.model:
        cfg.actions.sand_proxy.model_path = str(args.model.resolve())
    if args.physics_dt:
        cfg.sim.dt = args.physics_dt
        cfg.decimation = round(0.02 / args.physics_dt)
        cfg.sim.render_interval = cfg.decimation
    # Direct scripts must resolve presets before constructing SimulationContext.
    physics = cfg.sim.physics
    assert isinstance(physics, M20FootProxyPhysicsCfg)
    cfg.sim.physics = physics.newton_mjwarp
    cfg.scene.num_envs = args.num_envs
    for name in (
        "randomize_rigid_body_mass",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_inertia",
        "randomize_com_positions",
        "randomize_apply_external_force_torque",
    ):
        setattr(cfg.events, name, None)
    cfg.events.randomize_reset_base.params["pose_range"] = {
        k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.events.randomize_reset_base.params["velocity_range"] = {
        k: (0.0, 0.0) for k in ("x", "y", "z", "roll", "pitch", "yaw")
    }
    cfg.events.randomize_reset_joints.params.update(position_range=(1.0, 1.0), velocity_range=(0.0, 0.0))
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
    cfg.commands.base_velocity.heading_command = False
    cfg.seed = 42
    samples = []
    resets = 0
    with launch_simulation(cfg, {"device": "cuda:0", "visualizer": ["none"]}):
        env = gym.make("M20NewtonTraining-M20-M20-WalkOnSandProxy", cfg=cfg).unwrapped
        try:
            env.reset()
            term = env.action_manager.get_term("sand_proxy")
            force_updates = 0
            original_step = term.model.step

            def counted_step(*args, **kwargs):
                nonlocal force_updates
                force_updates += 1
                return original_step(*args, **kwargs)

            term.model.step = counted_step
            assert env.action_manager.total_action_dim == 16
            for i in range(args.steps):
                with torch.inference_mode():
                    obs, reward, terminated, truncated, _ = env.step(
                        torch.zeros((args.num_envs, 16), device=env.device)
                    )
                assert all(torch.isfinite(x).all() for x in obs.values()) and torch.isfinite(reward).all()
                assert torch.isfinite(term.force_w).all()
                resets += int((terminated | truncated).sum())
                if i >= args.steps // 2:
                    height = env.scene["robot"].data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
                    foot_bottom = term.application_points_w[..., 2] - env.scene.env_origins[:, 2:3]
                    samples.append(
                        [
                            float(term.force_w[..., 2].sum(-1).mean()),
                            float(height.mean()),
                            float(foot_bottom.min()),
                            float(term.model.depth.max()),
                        ]
                    )
            data = torch.tensor(samples)
            weight = float(env.scene["robot"].data.default_mass.torch.sum(-1).mean() * 9.81)
            result = {
                "steps": args.steps,
                "num_envs": args.num_envs,
                "nominal_reset": True,
                "episode_resets": resets,
                "mean_proxy_support_N": float(data[:, 0].mean()),
                "robot_weight_N": weight,
                "support_weight_ratio": float(data[:, 0].mean() / weight),
                "mean_base_z_m": float(data[:, 1].mean()),
                "minimum_foot_bottom_above_floor_m": float(data[:, 2].min()),
                "max_foot_penetration_m": float(data[:, 3].max()),
                "observation_shapes": {k: list(v.shape) for k, v in obs.items()},
                "action_dimension": 16,
                "physics_dt_s": env.physics_dt,
                "force_update_count": force_updates,
                "expected_force_update_count": args.steps * env.cfg.decimation,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
            assert force_updates == args.steps * env.cfg.decimation, "Sand force was not updated every physics step"
            assert resets == 0, "Nominal zero-action pose failed to remain supported"
            assert 0.7 < result["support_weight_ratio"] < 1.3
            assert result["minimum_foot_bottom_above_floor_m"] > 0.02, "Rigid floor may be supporting the robot"
        finally:
            env.close()


if __name__ == "__main__":
    main()
