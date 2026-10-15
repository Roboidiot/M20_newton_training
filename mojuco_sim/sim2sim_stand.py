"""Replay a 57-input/16-output Stand TorchScript policy in CPU MuJoCo."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).with_name("stand_config.json")


def latest_policy() -> Path:
    policies = list((ROOT / "logs/rsl_rl/deeprobotics_m20_stand").glob("*/exported/policy.pt"))
    if not policies:
        raise FileNotFoundError("No exported Stand policy.pt found. Export it with isaaclab play, or pass --policy.")
    return max(policies, key=lambda path: path.stat().st_mtime)


def build_observation(angular_velocity, gravity, command, joint_positions, joint_velocities, last_action, config):
    """Match the training term order, clipping-before-scaling, and position mask."""
    clip = float(config["observation_clip"])
    positions = np.asarray(joint_positions) - np.asarray(config["default_joint_positions"])
    positions = positions.copy()
    positions[config["joint_position_zero_indices"]] = 0.0
    return np.concatenate(
        (
            np.clip(angular_velocity, -clip, clip) * config["angular_velocity_scale"],
            np.clip(gravity, -clip, clip),
            np.clip(command, -clip, clip),
            np.clip(positions, -clip, clip),
            np.clip(joint_velocities, -clip, clip) * config["joint_velocity_scale"],
            np.clip(last_action, -clip, clip),
        )
    ).astype(np.float32)


def action_targets(action, config):
    """Convert raw policy actions into leg positions and wheel velocities."""
    action = np.clip(np.asarray(action), -config["action_clip"], config["action_clip"])
    positions = (
        np.asarray(config["default_joint_positions"][:12]) + np.asarray(config["leg_action_scales"]) * action[:12]
    )
    velocities = config["wheel_action_scale"] * action[12:]
    clip = config["processed_action_clip"]
    return action, np.clip(positions, -clip, clip), np.clip(velocities, -clip, clip)


class CommandInput:
    """Optional viewer key presses increment commands; Backspace stops and R resets."""

    def __init__(self, command, limits):
        self._command = np.asarray(command, dtype=float).copy()
        self._limits = np.asarray(limits, dtype=float)
        self._lock = threading.Lock()
        self._reset = False
        self._paused = False

    def key_callback(self, keycode: int):
        changes = {"I": (0, 0.1), "K": (0, -0.1), "J": (1, 0.1), "L": (1, -0.1), "U": (2, 0.1), "O": (2, -0.1)}
        with self._lock:
            if keycode == 259:  # GLFW_KEY_BACKSPACE
                self._command[:] = 0.0
            elif keycode == ord("R"):
                self._reset = True
                self._command[:] = 0.0
            elif keycode == ord(" "):
                self._paused = not self._paused
            elif 0 <= keycode < 128 and chr(keycode).upper() in changes:
                axis, delta = changes[chr(keycode).upper()]
                self._command[axis] += delta
                self._command[:] = np.clip(self._command, -self._limits, self._limits)
            else:
                return
            print(f"[Input] command={self._command.round(3).tolist()}, paused={self._paused}")

    def read(self):
        with self._lock:
            reset = self._reset
            self._reset = False
            return self._command.copy(), reset, self._paused


class StandSimulation:
    """Named joint/actuator mapping with nominal PD and the original MJCF flat floor."""

    def __init__(self, config_path: Path, policy_path: Path, *, pd_only: bool = False):
        self.config = json.loads(config_path.read_text())
        cfg = self.config
        model_path = (config_path.parent / cfg["model"]).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = cfg["physics_dt"]
        self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
        self.data = mujoco.MjData(self.model)
        self.decimation = round(cfg["control_dt"] / cfg["physics_dt"])
        if self.decimation < 1 or not np.isclose(self.decimation * cfg["physics_dt"], cfg["control_dt"]):
            raise ValueError("control_dt must be a positive integer multiple of physics_dt.")
        self.base_id = self._id(mujoco.mjtObj.mjOBJ_BODY, "base_link")
        root_joint = self._id(mujoco.mjtObj.mjOBJ_JOINT, "floating_base")
        self.root_qpos = int(self.model.jnt_qposadr[root_joint])
        names = cfg["policy_joint_names"]
        if len(names) != 16 or len(cfg["default_joint_positions"]) != 16 or len(cfg["leg_action_scales"]) != 12:
            raise ValueError("Stand requires 12 leg joints followed by 4 wheel joints.")
        joints = [self._id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
        self.qpos_ids = self.model.jnt_qposadr[joints]
        self.qvel_ids = self.model.jnt_dofadr[joints]
        self.actuator_ids = np.asarray([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names])
        if not np.array_equal(self.model.actuator_trnid[self.actuator_ids, 0], joints):
            raise ValueError("The named actuators must drive their corresponding joints.")
        if not np.allclose(self.model.actuator_gear[self.actuator_ids, 0], 1.0):
            raise ValueError("This torque controller expects actuator gear=1.")
        self.model.dof_damping[self.qvel_ids] = 0.0
        self.model.dof_frictionloss[self.qvel_ids] = 0.0
        self.model.dof_armature[self.qvel_ids[:12]] = 0.0
        self.model.dof_armature[self.qvel_ids[12:]] = cfg["wheel_armature"]
        # Training disables robot self-collisions. Bitmasks retain robot-ground contacts.
        for geom in range(self.model.ngeom):
            if self.model.geom_contype[geom] or self.model.geom_conaffinity[geom]:
                ground = self.model.geom_bodyid[geom] == 0
                self.model.geom_contype[geom] = 1 if ground else 2
                self.model.geom_conaffinity[geom] = 2 if ground else 1
        self.policy = None if pd_only else torch.jit.load(str(policy_path), map_location="cpu").eval()
        if self.policy is not None:
            with torch.inference_mode():
                output = self.policy(torch.zeros(1, 57))
            if tuple(output.shape) != (1, 16):
                raise ValueError(f"Expected policy output (1,16), got {tuple(output.shape)}.")
        self.reset()

    def _id(self, kind, name):
        result = mujoco.mj_name2id(self.model, kind, name)
        if result < 0:
            raise ValueError(f"MJCF is missing {kind.name}: {name}")
        return result

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.root_qpos : self.root_qpos + 3] = (0.0, 0.0, self.config["base_height"])
        self.data.qpos[self.root_qpos + 3 : self.root_qpos + 7] = self.config["base_quaternion_wxyz"]
        self.data.qpos[self.qpos_ids] = self.config["default_joint_positions"]
        self.last_action = np.zeros(16)
        self.position_targets = np.asarray(self.config["default_joint_positions"][:12])
        self.velocity_targets = np.zeros(4)
        self.control_steps = 0
        self.fall_steps = 0
        self.error_sum = np.zeros(2)
        self.max_torque = np.zeros(2)
        self.observation = np.zeros(57, dtype=np.float32)
        mujoco.mj_forward(self.model, self.data)

    def body_state(self):
        rotation = self.data.xmat[self.base_id].reshape(3, 3)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.base_id, velocity, 1)
        gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
        return velocity[:3], velocity[3:], gravity

    def step(self, command):
        cfg = self.config
        angular, _, gravity = self.body_state()
        self.observation = build_observation(
            angular,
            gravity,
            command,
            self.data.qpos[self.qpos_ids],
            self.data.qvel[self.qvel_ids],
            self.last_action,
            cfg,
        )
        with torch.inference_mode():
            action = (
                np.zeros(16)
                if self.policy is None
                else self.policy(torch.from_numpy(self.observation).unsqueeze(0))[0].numpy()
            )
        if not np.isfinite(action).all():
            raise FloatingPointError("The policy produced non-finite actions.")
        self.last_action, self.position_targets, self.velocity_targets = action_targets(action, cfg)
        for _ in range(self.decimation):
            q, dq = self.data.qpos[self.qpos_ids], self.data.qvel[self.qvel_ids]
            leg_torque = cfg["leg_kp"] * (self.position_targets - q[:12]) - cfg["leg_kd"] * dq[:12]
            wheel_torque = cfg["wheel_kd"] * (self.velocity_targets - dq[12:])
            torque = np.concatenate(
                (
                    np.clip(leg_torque, -cfg["leg_torque_limit"], cfg["leg_torque_limit"]),
                    np.clip(wheel_torque, -cfg["wheel_torque_limit"], cfg["wheel_torque_limit"]),
                )
            )
            self.data.ctrl[self.actuator_ids] = torque
            self.max_torque = np.maximum(self.max_torque, [np.max(np.abs(torque[:12])), np.max(np.abs(torque[12:]))])
            mujoco.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
                raise FloatingPointError("MuJoCo state became non-finite.")
        mujoco.mj_forward(self.model, self.data)
        angular, linear, gravity = self.body_state()
        self.control_steps += 1
        fallen = gravity[0] > 0.0 or np.any(np.abs(gravity[1:]) > 0.7)
        self.fall_steps += int(fallen)
        self.error_sum += [np.linalg.norm(np.asarray(command[:2]) - linear[1:3]), abs(command[2] - angular[0])]
        return bool(fallen)

    def summary(self):
        return {
            "sim_seconds": float(self.data.time),
            "control_steps": self.control_steps,
            "final_base_height": float(self.data.xpos[self.base_id, 2]),
            "projected_gravity": self.body_state()[2].tolist(),
            "mean_linear_error_m_s": float(self.error_sum[0] / max(self.control_steps, 1)),
            "mean_angular_error_rad_s": float(self.error_sum[1] / max(self.control_steps, 1)),
            "steps_outside_stand_orientation": self.fall_steps,
            "max_leg_torque_Nm": float(self.max_torque[0]),
            "max_wheel_torque_Nm": float(self.max_torque[1]),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy", type=Path, help="Exported TorchScript policy.pt; defaults to the latest Stand export."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--command", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument(
        "--duration", type=float, default=20.0, help="Simulation seconds; 0 runs until the window closes."
    )
    parser.add_argument("--headless", action="store_true", help="CPU simulation without a viewer.")
    parser.add_argument(
        "--keyboard", action="store_true", help="Key presses increment commands by 0.1; Backspace stops."
    )
    parser.add_argument("--stop-on-fall", action="store_true")
    parser.add_argument(
        "--pd-only", action="store_true", help="Hold the default pose with zero actions to diagnose the model."
    )
    parser.add_argument("--summary", type=Path, help="Optional output JSON path for validation metrics.")
    args = parser.parse_args()
    if args.duration < 0 or (args.headless and args.duration == 0):
        parser.error("Use a non-negative duration, and a positive duration for headless runs.")
    if args.headless and args.keyboard:
        parser.error("Keyboard input requires a viewer; remove --headless.")
    torch.set_num_threads(1)
    policy = args.policy or (Path() if args.pd_only else latest_policy())
    sim = StandSimulation(args.config.resolve(), policy, pd_only=args.pd_only)
    if np.any(np.abs(args.command) > np.asarray(sim.config["command_limits"])):
        parser.error("--command exceeds the configured Stand training range.")
    inputs = CommandInput(args.command, sim.config["command_limits"])
    print(f"[Sim2Sim] policy={policy}, dt={sim.model.opt.timestep}, control_dt={sim.config['control_dt']}")
    print(f"[Sim2Sim] flat terrain; joint position zero indices={sim.config['joint_position_zero_indices']}")
    if args.keyboard:
        print("[Keyboard] I/K: vx +/-0.1; J/L: vy +/-0.1; U/O: wz +/-0.1; Backspace: zero; R: reset; Space: pause.")
    viewer = None
    try:
        if not args.headless:
            import mujoco.viewer

            viewer = mujoco.viewer.launch_passive(
                sim.model, sim.data, key_callback=inputs.key_callback if args.keyboard else None
            )
            with viewer.lock():
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                viewer.cam.trackbodyid = sim.base_id
                viewer.cam.distance = 3.0
                viewer.cam.azimuth = 135.0
                viewer.cam.elevation = -15.0
        elapsed_steps = 0
        while (args.duration == 0 or elapsed_steps * sim.config["control_dt"] < args.duration) and (
            viewer is None or viewer.is_running()
        ):
            start = time.perf_counter()
            command, reset, paused = inputs.read()
            if reset:
                sim.reset()
            if not paused:
                fallen = sim.step(command)
                elapsed_steps += 1
                if fallen and args.stop_on_fall:
                    print("[Sim2Sim] Stand orientation termination reached.")
                    break
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, sim.config["control_dt"] - (time.perf_counter() - start)))
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
    summary = sim.summary()
    print(json.dumps(summary, indent=2))
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
