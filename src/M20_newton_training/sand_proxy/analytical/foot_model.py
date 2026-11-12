"""Batched Torch foot contact with elastoplastic bearing and footprint memory.

Inspired by newton-terrain's analytic bearing/drag/pull/footprint construction.
Coefficients are calibrated on M20's USD wheel, not copied from Go2. No shin,
thigh or belly forces are computed. All buffers stay on the caller's device.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from .shadow_response import sliding_response

DEFAULT_MODEL_PATH = Path(__file__).parent / "calibrated" / "m20_foot_sand.json"


def power_response(depth: torch.Tensor, law: dict, speed: torch.Tensor | float = 0.0):
    return law["a"] * depth.clamp_min(1e-8).pow(law["n"]) * (1 + law.get("b", 0.0) * speed)


class FootSandModel:
    """State per environment/foot; return forces in a yaw-aligned frame with world-up Z.

    depth is the collision primitive's lowest-point depth; velocity is the velocity
    at the application point. positions_xy uses world coordinates for shared holes.
    Passing stability_mass_kg enables timestep-dependent explicit-force regularization.
    """

    def __init__(self, parameters: dict, num_envs: int, num_feet: int = 4, device="cpu", keep=0.7):
        if parameters.get("model_type") != "m20_foot_elastoplastic_sand" or parameters.get("schema_version") not in (
            1,
            2,
        ):
            raise ValueError("Expected calibrated M20 foot contact parameters")
        if parameters["schema_version"] == 2 and "shadow_contact" not in parameters:
            raise ValueError("Version 2 requires whole-robot shadow contact parameters")
        self.parameters = parameters
        self.num_envs, self.num_feet, self.device = num_envs, num_feet, torch.device(device)
        shape = (num_envs, num_feet)
        self.memory = torch.full(shape, -1.0, device=device)
        self.force = torch.zeros((*shape, 3), device=device)
        self.depth = torch.zeros(shape, device=device)
        self.contact = torch.zeros(shape, dtype=torch.bool, device=device)
        self.air_time = torch.zeros(shape, device=device)
        self.slip_speed_sq = torch.zeros(shape, device=device)
        self.out_of_domain = torch.zeros(shape, dtype=torch.bool, device=device)
        self.hole_count = int(parameters["history"]["slots"])
        self.hole_xy = torch.full((num_envs, self.hole_count, 2), 1e6, device=device)
        self.hole_depth = torch.zeros((num_envs, self.hole_count), device=device)
        self.hole_next = torch.zeros(num_envs, dtype=torch.long, device=device)
        self._env_ids = torch.arange(num_envs, device=device)
        self.keep = keep

    @classmethod
    def load(cls, path=DEFAULT_MODEL_PATH, **kwargs):
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run the independent foot-contact calibration described in README")
        return cls(json.loads(path.read_text()), **kwargs)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.memory[ids] = -1
        self.force[ids] = 0
        self.depth[ids] = 0
        self.contact[ids] = False
        self.air_time[ids] = 0
        self.slip_speed_sq[ids] = 0
        self.out_of_domain[ids] = False
        self.hole_xy[ids] = 1e6
        self.hole_depth[ids] = 0
        self.hole_next[ids] = 0

    def step(self, depth, velocity, positions_xy, dt, *, bearing_scale=1.0, drag_scale=1.0, stability_mass_kg=None):
        p = self.parameters
        base_domain = p.get("intruder_domain", p["domain"])
        max_depth, max_speed = base_domain["max_depth_m"], base_domain["max_speed_m_s"]
        active = depth > 0
        d = depth.clamp(0, max_depth)
        horizontal = velocity[..., :2]
        speed = torch.linalg.vector_norm(horizontal, dim=-1)
        domain = p["domain"]
        self.out_of_domain.copy_(
            (depth > domain["max_depth_m"])
            | (speed > domain["max_speed_m_s"])
            | (velocity[..., 2].abs() > domain.get("max_vertical_speed_m_s", domain["max_speed_m_s"]))
            | (velocity[..., 1].abs() > domain.get("max_lateral_speed_m_s", domain["max_speed_m_s"]))
        )

        # Store footprints at liftoff. Four serial foot passes avoid ring-slot races.
        for foot in range(self.num_feet):
            store = (depth[:, foot] < -0.01) & (self.memory[:, foot] > 0.005)
            slot = self.hole_next.remainder(self.hole_count)
            self.hole_xy[self._env_ids, slot] = torch.where(
                store[:, None], positions_xy[:, foot], self.hole_xy[self._env_ids, slot]
            )
            self.hole_depth[self._env_ids, slot] = torch.where(
                store, self.memory[:, foot], self.hole_depth[self._env_ids, slot]
            )
            self.hole_next += store.long()
        memory = torch.where(depth < -0.01, -1.0, self.memory)
        distance = torch.linalg.vector_norm(positions_xy[:, :, None, :] - self.hole_xy[:, None, :, :], dim=-1)
        old_floor = (
            self.keep * self.hole_depth[:, None, :] * (1 - distance / p["history"]["diameter_m"]).clamp_min(0)
        ).amax(-1)
        memory = torch.where(active & (memory < 0), old_floor, memory)

        stiffness = power_response(d, p["unloading_stiffness"])
        capacity = bearing_scale * power_response(d, p["bearing"])
        elastic = (d - memory.clamp_min(0)).clamp_min(0)
        memory = torch.where(active, torch.maximum(memory, d - capacity / stiffness.clamp_min(1)), memory)
        elastic = torch.minimum(elastic, capacity / stiffness.clamp_min(1))
        # Shear loosens the supporting floor; its decay length is measured from drag transients.
        direction_x = horizontal[..., 0].abs() / speed.clamp_min(1e-8)
        direction_y = horizontal[..., 1].abs() / speed.clamp_min(1e-8)
        inverse_length = direction_x / p["shear_length_x_m"] + direction_y / p["shear_length_y_m"]
        elastic = elastic * torch.exp(-speed * dt * inverse_length)
        memory = torch.where(active, torch.maximum(memory, d - elastic), memory)
        down_speed = (-velocity[..., 2]).clamp(0, max_speed)
        up_speed = velocity[..., 2].clamp_min(0)
        down_damping = capacity * p["bearing"].get("b", 0.0)
        up_damping = power_response(d, p["unloading_damping"])
        if stability_mass_kg is not None:
            # Limit the damping slope of the explicit external-force update.
            limit = 0.5 * stability_mass_kg / dt
            up_damping = torch.minimum(up_damping, limit)
            down_damping = torch.minimum(down_damping, limit)
        normal = (stiffness * elastic + down_damping * down_speed * (elastic > 0) - up_damping * up_speed).clamp_min(0)
        normal -= p["pull_p_N_per_m"] * d * (up_speed / 0.05).clamp_max(1)

        rate = speed.clamp_max(max_speed)
        dx = drag_scale * power_response(d, p["drag_x"], rate)
        dy = drag_scale * power_response(d, p["drag_y"], rate)
        epsilon = torch.full_like(d, p["low_speed_epsilon_m_s"])
        if stability_mass_kg is not None:
            epsilon = torch.maximum(epsilon, torch.maximum(dx, dy) * dt / stability_mass_kg.clamp_min(0.1))
        denominator = torch.sqrt(speed.square() + epsilon.square())
        fx, fy = -dx * horizontal[..., 0] / denominator, -dy * horizontal[..., 1] / denominator
        rx = power_response(d, p["lift_ratio_x"]).clamp(0, 1)
        ry = power_response(d, p["lift_ratio_y"]).clamp(0, 1)
        lift = torch.sqrt((rx * fx).square() + (ry * fy).square())
        self.force.copy_(torch.stack([fx, fy, normal + lift], dim=-1) * active[..., None])
        if "shadow_contact" in p:
            correction = p["shadow_contact"]
            # A short speed envelope avoids selecting a different law at every velocity zero crossing.
            alpha = math.exp(-dt / correction["speed_filter_tau_s"])
            self.slip_speed_sq.copy_(
                torch.where(active, alpha * self.slip_speed_sq + (1 - alpha) * speed.square(), 0.0)
            )
            gate = -torch.expm1(-self.slip_speed_sq / correction["coefficients"]["blend"] ** 2)
            dynamic = sliding_response(
                depth.clamp(0, domain["max_depth_m"]),
                velocity.clamp(-domain["max_speed_m_s"], domain["max_speed_m_s"]),
                correction["coefficients"],
                stability_mass_kg=stability_mass_kg,
                dt=dt,
                bearing_scale=bearing_scale,
                drag_scale=drag_scale,
            )
            self.force.copy_(((1 - gate[..., None]) * self.force + gate[..., None] * dynamic) * active[..., None])
        self.memory.copy_(memory)
        self.depth.copy_(depth)
        support = torch.linalg.vector_norm(self.force, dim=-1) if "shadow_contact" in p else self.force[..., 2]
        self.contact.copy_(active & (support > 1))
        self.air_time.copy_(torch.where(self.contact, 0.0, self.air_time + dt))
        return self.force
