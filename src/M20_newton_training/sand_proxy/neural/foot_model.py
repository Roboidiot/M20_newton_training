"""Adapt the trained 19-input wheel model to a fixed, undeformed sand plane."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .wheel_model_runtime import WheelModelRuntime, WheelMotionHistory

DEFAULT_MODEL_PATH = Path(__file__).parent / "best.pt"


class NeuralFootSandModel:
    """World-frame force/COM torque; one 5 ms history per environment, no footprint memory."""

    def __init__(self, path=DEFAULT_MODEL_PATH, *, num_envs, num_feet=4, device="cpu"):
        if num_feet != 4:
            raise ValueError("The neural wheel model requires four ordered FL/FR/HL/HR feet")
        self.parameters = json.loads((Path(__file__).parent / "deployment.json").read_text())
        self.num_envs, self.num_feet, self.device = num_envs, num_feet, torch.device(device)
        self.runtime = WheelModelRuntime(path, device=device)
        self.history = WheelMotionHistory(num_envs, device=device)
        self.depth = torch.zeros((num_envs, 4), device=device)
        self.force = torch.zeros((num_envs, 4, 3), device=device)
        self.torque = torch.zeros_like(self.force)
        self.features = torch.zeros((num_envs, 4, 19), device=device)
        self.reference_com_b = torch.tensor(self.parameters["reference_com_body_m"], device=device)

    @classmethod
    def load(cls, path=DEFAULT_MODEL_PATH, **kwargs):
        return cls(path, **kwargs)

    @torch.no_grad()
    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.history.reset(env_ids)
        for tensor in (self.depth, self.force, self.torque, self.features):
            tensor[ids] = 0

    @torch.no_grad()
    def step(self, depth, velocity, positions_xy, dt, *, angular_velocity_w, axis_w, joint_velocity):
        """velocity is capsule-CENTER velocity in WORLD axes, not bottom velocity or yaw-frame velocity.

        Both depth inputs equal the original action's fixed-plane bottom depth;
        local surface is valid everywhere. positions_xy does not alter the plane.
        """
        self.runtime.check_physics_dt(dt)
        self.depth.copy_(depth)
        current = torch.cat(
            (
                depth[..., None],
                depth[..., None],
                torch.ones_like(depth)[..., None],
                velocity,
                angular_velocity_w,
                axis_w,
                joint_velocity[..., None],
            ),
            dim=-1,
        )
        inputs, ready = self.history.append(current)
        self.features.copy_(inputs)
        prediction = self.runtime(inputs)
        active = ready[:, None] & (depth > 0)
        self.force.copy_(torch.where(active[..., None], prediction[..., :3], 0.0))
        self.torque.copy_(torch.where(active[..., None], prediction[..., 3:], 0.0))
        return self.force
