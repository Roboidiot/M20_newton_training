"""Passive per-MPM-step recorder, independent of shadow and fitted contact models."""

import numpy as np
import torch
import warp as wp

from .geometry import FEET, SHINS, body_ids, capsule_kinematics, joint_indices, rotate_xyzw
from .surface import LocalSurface


@wp.kernel
def reduce_wrenches(
    impulses: wp.array[wp.vec3],
    positions: wp.array[wp.vec3],
    collider_ids: wp.array[int],
    collider_body: wp.array[int],
    body_slots: wp.array[int],
    com: wp.array[wp.vec3],
    inverse_dt: float,
    force: wp.array[wp.vec3],
    torque: wp.array[wp.vec3],
):
    i = wp.tid()
    collider = collider_ids[i]
    if collider >= 0:
        body = collider_body[collider]
        if body >= 0:
            slot = body_slots[body]
            if slot >= 0:
                f = impulses[i] * inverse_dt
                wp.atomic_add(force, slot, f)
                wp.atomic_add(torque, slot, wp.cross(positions[i] - com[slot], f))


class MPMRecorder:
    def __init__(self, root_solver, robot, action_manager, geometry, args):
        # The private entry state is necessary: IsaacLab data caches update only AFTER the solver returns.
        self.solver = root_solver.solver("sand")
        self.rigid = root_solver._entries["robot"]
        self.args = args
        self.physics_dt = args.physics.physics_dt_s
        self.coupling_iterations = args.physics.coupling["iterations"]
        self.device = self.solver.model.device
        self.actions = action_manager
        self.names = list(robot.body_names)
        self.joint_names = list(robot.joint_names)
        self.rigid_ids = body_ids(self.rigid.view.body_label, self.names)
        self.mpm_ids = body_ids(self.solver.model.body_label, self.names)
        self.q_ids, self.qd_ids = joint_indices(self.rigid.view, self.joint_names)
        self.feet = [self.names.index(n) for n in FEET]
        self.shins = [self.names.index(n) for n in SHINS]
        self.other = [i for i in range(len(self.names)) if i not in self.feet + self.shins]
        self.base = self.names.index("base_link")
        slots = np.full(len(self.solver.model.body_label), -1, dtype=np.int32)
        slots[self.mpm_ids] = np.arange(len(self.names))
        self.slots = wp.array(slots, dtype=int, device=self.device)
        self.force = wp.zeros(len(self.names), dtype=wp.vec3, device=self.device)
        self.torque = wp.zeros_like(self.force)
        self.com_buffer = wp.zeros_like(self.force)
        self.local_com = wp.to_torch(self.rigid.view.body_com)[self.rigid_ids].clone()
        self.mpm_local_com = wp.to_torch(self.solver.model.body_com)[self.mpm_ids].clone()
        wheels = geometry["wheels"]
        self.center = torch.tensor([w["local_center"] for w in wheels], device=args.device, dtype=torch.float32)
        self.axis = torch.tensor([w["local_axis"] for w in wheels], device=args.device, dtype=torch.float32)
        self.radius = torch.tensor([w["radius"] for w in wheels], device=args.device, dtype=torch.float32)
        self.half_length = torch.tensor([w["half_length"] for w in wheels], device=args.device, dtype=torch.float32)
        self.surface = LocalSurface(args.device)
        self.particle_radius = wp.to_torch(self.solver.model.particle_radius)
        self.mapping = {
            "body_names": self.names,
            "joint_names": self.joint_names,
            "rigid_body_ids": self.rigid_ids,
            "mpm_body_ids": self.mpm_ids,
            "joint_coordinate_ids": self.q_ids,
            "joint_velocity_ids": self.qd_ids,
            "wheel_body_slots": self.feet,
            "shin_body_slots": self.shins,
            "other_body_slots": self.other,
            "base_body_slot": self.base,
            "position_target_joint_names": list(args.leg_joint_names),
            "velocity_target_joint_names": list(args.wheel_joint_names),
            "surface_cell_m": self.surface.cell,
            "surface_patch_size": self.surface.patch_size,
            "surface_annulus_inner_outer_m": [0.15, 0.24],
        }
        self.enabled = False
        self.writer = None
        self.frame = 0
        self.mpm_calls = 0
        self.episode = 0
        self.control_step = 0
        self.segment = -1
        self.action = np.zeros(16, dtype=np.float32)
        self.target = self.action.copy()

    def begin_episode(self, episode, writer, reference_height, simulation_start):
        self.episode, self.writer = episode, writer
        self.reference_height = reference_height
        self.simulation_start = simulation_start
        self.frame = 0
        self.mpm_calls = 0

    def __enter__(self):
        self.original = self.solver.step
        self.had_override = "step" in self.solver.__dict__
        self.solver.step = self._step
        return self

    def __exit__(self, *exc):
        if self.had_override:
            self.solver.step = self.original
        else:
            del self.solver.step

    @torch.no_grad()
    def inputs(self, state):
        rigid = self.rigid.state_1
        pose = wp.to_torch(rigid.body_q)[self.rigid_ids]
        twist = wp.to_torch(rigid.body_qd)[self.rigid_ids]
        com = pose[:, :3] + rotate_xyzw(pose[:, 3:], self.local_com)
        proxy_pose = wp.to_torch(state.body_q)[self.mpm_ids]
        proxy_twist = wp.to_torch(state.body_qd)[self.mpm_ids]
        proxy_com = proxy_pose[:, :3] + rotate_xyzw(proxy_pose[:, 3:], self.mpm_local_com)
        wp.copy(self.com_buffer, wp.from_torch(proxy_com.contiguous(), dtype=wp.vec3))
        center, center_v, bottom, bottom_v, axis = capsule_kinematics(
            pose[self.feet],
            twist[self.feet],
            com[self.feet],
            self.center,
            self.axis,
            self.radius,
            self.half_length,
        )
        surface = self.surface.sample(wp.to_torch(state.particle_q), self.particle_radius, center[:, :2])
        row = {
            "body_pose_w": pose,
            "body_com_position_w_m": com,
            "body_com_twist_w": twist,
            "body_link_velocity_w_m_s": twist[:, :3] + torch.linalg.cross(twist[:, 3:], pose[:, :3] - com),
            "mpm_proxy_body_pose_w": proxy_pose,
            "mpm_proxy_body_com_twist_w": proxy_twist,
            "mpm_proxy_body_com_position_w_m": proxy_com,
            "base_pose_w": pose[self.base],
            "base_com_twist_w": twist[self.base],
            "joint_position_rad": wp.to_torch(rigid.joint_q)[self.q_ids],
            "joint_velocity_rad_s": wp.to_torch(rigid.joint_qd)[self.qd_ids],
            "joint_effort_input_Nm": wp.to_torch(self.rigid.control.joint_f)[self.qd_ids],
            "joint_position_target_rad": self.actions.get_term("joint_pos").processed_actions[0],
            "wheel_velocity_target_rad_s": self.actions.get_term("joint_vel").processed_actions[0],
            "wheel_pose_w": pose[self.feet],
            "wheel_center_w_m": center,
            "wheel_center_velocity_w_m_s": center_v,
            "wheel_angular_velocity_w_rad_s": twist[self.feet, 3:],
            "wheel_axis_w": axis,
            "wheel_bottom_w_m": bottom,
            "wheel_bottom_velocity_w_m_s": bottom_v,
            "wheel_sinkage_reference_m": self.reference_height - bottom[:, 2],
            "wheel_sinkage_local_m": torch.where(
                surface["wheel_surface_valid"], surface["wheel_surface_height_w_m"] - bottom[:, 2], 0
            ),
            **surface,
        }
        # Copy BEFORE in-place MPM modifies its input; do not retain GPU array views.
        return {key: value.detach().cpu().numpy().copy() for key, value in row.items()}

    def _step(self, state_in, state_out, control, contacts, dt):
        if not self.enabled:
            return self.original(state_in, state_out, control, contacts, dt)
        if not np.isclose(dt, self.physics_dt, rtol=0, atol=1e-9):
            raise RuntimeError(f"Unexpected MPM step {dt}; expected {self.physics_dt}")
        # Outer coupling iterations re-solve the SAME interval. Only the final pass is a sample.
        self.mpm_calls += 1
        if self.mpm_calls % self.coupling_iterations:
            return self.original(state_in, state_out, control, contacts, dt)
        row = self.inputs(state_in)
        result = self.original(state_in, state_out, control, contacts, dt)
        impulse, position, collider = self.solver.collect_collider_impulses(state_out)
        self.force.zero_()
        self.torque.zero_()
        wp.launch(
            reduce_wrenches,
            dim=len(impulse),
            inputs=[
                impulse,
                position,
                collider,
                self.solver.collider_body_index,
                self.slots,
                self.com_buffer,
                1 / dt,
                self.force,
                self.torque,
            ],
            device=self.device,
        )
        force, torque = self.force.numpy().copy(), self.torque.numpy().copy()
        start, end = self.frame * self.physics_dt, (self.frame + 1) * self.physics_dt
        reference = row["mpm_proxy_body_com_position_w_m"][self.base]
        torque_at_base = torque + np.cross(row["mpm_proxy_body_com_position_w_m"] - reference, force)
        base_q = row["base_pose_w"][3:]
        row.update(
            episode_id=self.episode,
            frame_index=self.frame,
            control_step=self.control_step,
            episode_time_start_s=start,
            episode_time_s=end,
            dt_s=dt,
            time_s=self.episode * self.args.episode_seconds + end,
            simulation_time_s=self.simulation_start + end,
            action=self.action.copy(),
            random_target=self.target.copy(),
            random_segment=self.segment,
            warmup=end <= self.args.warmup + 1e-8,
            reference_surface_height_w_m=self.reference_height,
            body_mpm_force_w_N=force,
            body_mpm_torque_com_w_Nm=torque,
            wheel_mpm_force_w_N=force[self.feet],
            wheel_mpm_torque_com_w_Nm=torque[self.feet],
            shin_mpm_force_w_N=force[self.shins],
            shin_mpm_torque_com_w_Nm=torque[self.shins],
            other_mpm_force_w_N=force[self.other].sum(axis=0),
            other_mpm_torque_base_com_w_Nm=torque_at_base[self.other].sum(axis=0),
            total_mpm_force_w_N=force.sum(axis=0),
            total_mpm_torque_base_com_w_Nm=torque_at_base.sum(axis=0),
            wheel_contact=np.linalg.norm(force[self.feet], axis=1) > 5.0,
            base_up_z=1 - 2 * (base_q[0] ** 2 + base_q[1] ** 2),
            base_clearance_m=row["base_pose_w"][2] - self.reference_height,
            base_outside_tray=np.max(np.abs(row["base_pose_w"][:2])) > 1.2,
        )
        self.writer.append(row)
        self.frame += 1
        return result
