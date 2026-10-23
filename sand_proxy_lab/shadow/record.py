"""Read MPM collider impulses and evaluate the training contact law without applying forces.

The local hook runs at the MPM solver boundary, after staggered coupling has
prepared its collider poses/velocities. CUDA graph replay must be disabled.
No task, environment, or installed solver source is modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import warp as wp

from M20_newton_training.sand_proxy.foot_model import FootSandModel
from M20_newton_training.tasks.m20.mdp.foot_sand_proxy import rotate_xyzw

FEET = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")


def body_slots(labels):
    """Resolve entry-local labels, independent of global/IsaacLab body ordering."""
    ids = []
    for name in (*FEET, "base_link"):
        matches = [i for i, label in enumerate(labels) if label.rstrip("/").split("/")[-1] == name]
        if len(matches) != 1:
            raise ValueError(f"Expected one {name}, found {matches} in MPM entry labels {labels}")
        ids.append(matches[0])
    slots = np.full(len(labels), -1, dtype=np.int32)
    for i, label in enumerate(labels):
        if "/Robot/" in label:
            slots[i] = 4  # Unmodeled robot contacts, excluding tray/static colliders.
    slots[ids[:4]] = np.arange(4)
    return ids[:4], ids[4], slots


@wp.kernel
def sum_wrenches(
    impulses: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    colliders: wp.array[int],
    collider_body: wp.array[int],
    body_slot: wp.array[int],
    centers_com: wp.array[wp.vec3],
    inverse_dt: float,
    forces: wp.array[wp.vec3],
    torques: wp.array[wp.vec3],
):
    i = wp.tid()
    collider = colliders[i]
    if collider >= 0:
        body = collider_body[collider]
        if body >= 0:
            slot = body_slot[body]
            if slot >= 0:
                force = impulses[i] * inverse_dt
                wp.atomic_add(forces, slot, force)
                if slot < 4:
                    wp.atomic_add(torques, slot, wp.cross(positions[i] - centers_com[slot], force))


class ShadowRecorder:
    def __init__(self, solver, model_path, effective_mass, device, surface_height):
        self.solver, self.device = solver, device
        self.parameters = json.loads(Path(model_path).read_text())
        self.runtime = FootSandModel(self.parameters, 1, device=device)
        self.unlimited = FootSandModel(self.parameters, 1, device=device)
        self.mass = torch.tensor([[effective_mass]], device=device)
        self.surface_height = surface_height
        labels = list(solver.model.body_label)
        self.feet, self.base, slots = body_slots(labels)
        self.mapping = {"body_labels": labels, "foot_local_ids": self.feet, "base_local_id": self.base}
        self.slots = wp.array(slots, dtype=int, device=device)
        self.forces = wp.zeros(5, dtype=wp.vec3, device=device)
        self.torques = wp.zeros(4, dtype=wp.vec3, device=device)
        self.centers_com = wp.zeros(4, dtype=wp.vec3, device=device)
        shape = self.parameters["geometry"]
        if shape["type"] != "capsule":
            raise ValueError("The current training model requires capsule wheel feet")
        self.radius, self.half_length = shape["radius"], shape["length"] / 2
        self.local_center = torch.tensor(shape["xyz"], device=device).expand(1, 4, 3)
        self.axis = rotate_xyzw(
            torch.tensor(shape["quat_xyzw"], device=device), torch.tensor([0.0, 0.0, 1.0], device=device)
        ).expand(1, 4, 3)
        self.body_com = wp.to_torch(solver.model.body_com)[self.feet][None].clone()
        self.rows = []
        self.enabled = False
        self.elapsed = 0.0
        self.episode_id = 0
        self.episode_elapsed = 0.0
        self.control_step = 0
        self.command = np.zeros(3)
        self.action = np.zeros(16)
        self.observation = np.zeros(57)

    def reset_episode(self, episode_id, surface_height):
        """Reset proxy history at a physical scene reset, retaining recorded data/time."""
        self.runtime.reset()
        self.unlimited.reset()
        self.episode_id = episode_id
        self.episode_elapsed = 0.0
        self.surface_height = surface_height

    def __enter__(self):
        self._original = self.solver.step
        self._had_override = "step" in self.solver.__dict__
        self.solver.step = self._step
        return self

    def __exit__(self, *args):
        if self._had_override:
            self.solver.step = self._original
        else:
            del self.solver.step

    @torch.no_grad()
    def _sample(self, state, dt):
        wp.synchronize_device(self.device)
        pose = wp.to_torch(state.body_q)
        q, position = pose[self.feet, 3:][None], pose[self.feet, :3][None]
        center = position + rotate_xyzw(q, self.local_center)
        axis = rotate_xyzw(q, self.axis)
        point = center.clone()
        point[..., 2] -= self.radius + self.half_length * axis[..., 2].abs()
        com = position + rotate_xyzw(q, self.body_com)
        twist = wp.to_torch(state.body_qd)[self.feet][None]
        velocity = twist[..., :3] + torch.linalg.cross(twist[..., 3:], point - com)
        x, y, z, w = pose[self.base, 3:].unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = yaw.cos(), yaw.sin()
        local_v = torch.stack(
            [
                c * velocity[..., 0] + s * velocity[..., 1],
                -s * velocity[..., 0] + c * velocity[..., 1],
                velocity[..., 2],
            ],
            dim=-1,
        )
        depth = self.surface_height - point[..., 2]

        def world(force):
            return torch.stack(
                [c * force[..., 0] - s * force[..., 1], s * force[..., 0] + c * force[..., 1], force[..., 2]], dim=-1
            )

        force = world(self.runtime.step(depth, local_v, center[..., :2], dt, stability_mass_kg=self.mass))
        unlimited = world(self.unlimited.step(depth, local_v, center[..., :2], dt))
        self.centers_com.assign(com[0].cpu().numpy())
        result = {
            "depth_m": depth[0],
            "foot_point_w_m": point[0],
            "foot_velocity_w_m_s": velocity[0],
            "foot_velocity_yaw_m_s": local_v[0],
            "proxy_force_w_N": force[0],
            "proxy_unlimited_force_w_N": unlimited[0],
            "proxy_torque_com_w_Nm": torch.linalg.cross(point - com, force)[0],
            "out_of_domain": self.runtime.out_of_domain[0],
            "plastic_depth_m": self.runtime.memory[0],
            "base_pose_w": pose[self.base].clone(),
            "base_twist_w": wp.to_torch(state.body_qd)[self.base].clone(),
        }
        return {key: value.cpu().numpy().copy() for key, value in result.items()}

    def _step(self, state_in, state_out, control, contacts, dt):
        if not self.enabled:
            return self._original(state_in, state_out, control, contacts, dt)
        row = self._sample(state_in, dt)
        self._original(state_in, state_out, control, contacts, dt)
        impulses, positions, colliders = self.solver.collect_collider_impulses(state_out)
        self.forces.zero_()
        self.torques.zero_()
        wp.launch(
            sum_wrenches,
            dim=len(impulses),
            device=self.device,
            inputs=[
                impulses,
                positions,
                colliders,
                self.solver.collider_body_index,
                self.slots,
                self.centers_com,
                1 / dt,
                self.forces,
                self.torques,
            ],
        )
        forces = self.forces.numpy().copy()
        row.update(
            time_s=self.elapsed + dt,
            dt_s=dt,
            control_step=self.control_step,
            episode_id=self.episode_id,
            episode_time_s=self.episode_elapsed + dt,
            command=self.command.copy(),
            action=self.action.copy(),
            policy_observation=self.observation.copy(),
            mpm_force_w_N=forces[:4],
            mpm_other_robot_force_w_N=forces[4],
            mpm_torque_com_w_Nm=self.torques.numpy().copy(),
        )
        if not all(np.isfinite(v).all() for v in row.values()):
            raise FloatingPointError(f"Nonfinite shadow data at {self.elapsed:.4f} s")
        self.rows.append(row)
        self.elapsed += dt
        self.episode_elapsed += dt

    def save(self, path, metadata):
        if not self.rows:
            return
        data = {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]}
        path = Path(path)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **data, metadata=json.dumps(metadata, ensure_ascii=False))
        temporary.replace(path)
