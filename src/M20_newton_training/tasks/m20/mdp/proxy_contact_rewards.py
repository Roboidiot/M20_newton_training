"""Contact-driven Walk rewards backed by applied foot proxy forces, without a ContactSensor."""

from __future__ import annotations

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply_inverse


class ProxyFootContactState:
    """Physics-rate contact timers and policy-window touchdown events.

    The first landing after reset establishes support and earns no air-time credit.
    Touchdown events remain visible until the next policy action, even if a foot
    lifts again before rewards are evaluated. Multiple touchdowns within a policy
    step accumulate signed credit through the recorded per-physics-step durations.
    """

    def __init__(self, num_envs, device, decimation):
        shape = (num_envs, 4)
        self.contact = torch.zeros(shape, dtype=torch.bool, device=device)
        self.has_contact = torch.zeros_like(self.contact)
        self.current_air_time = torch.zeros(shape, device=device)
        self.current_contact_time = torch.zeros_like(self.current_air_time)
        self.touchdown_air_time = torch.zeros((num_envs, decimation, 4), device=device)
        self.touchdown = torch.zeros_like(self.touchdown_air_time, dtype=torch.bool)
        self._tick = 0

    def begin_step(self):
        self.touchdown.zero_()
        self.touchdown_air_time.zero_()
        self._tick = 0

    def update(self, force_w, dt, threshold):
        contact = torch.linalg.vector_norm(force_w, dim=-1) > threshold
        landed = contact & ~self.contact & self.has_contact
        # The previous sample was airborne, so include the final integration interval.
        duration = self.current_air_time + dt
        slot = self._tick % self.touchdown.shape[1]
        self.touchdown[:, slot].copy_(landed)
        self.touchdown_air_time[:, slot].copy_(torch.where(landed, duration, 0.0))
        self.current_air_time.copy_(torch.where(contact, 0.0, duration))
        self.current_contact_time.copy_(torch.where(contact, self.current_contact_time + dt, 0.0))
        self.has_contact |= contact
        self.contact.copy_(contact)
        self._tick += 1

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for buffer in (
            self.contact,
            self.has_contact,
            self.current_air_time,
            self.current_contact_time,
            self.touchdown_air_time,
            self.touchdown,
        ):
            buffer[ids] = 0


def proxy_feet_air_time_command_scaled(
    env,
    command_name,
    threshold=0.2,
    max_air_time=0.5,
    command_threshold=0.1,
    side_multiplier=2.0,
    turn_multiplier=3.0,
    action_term_name="sand_proxy",
):
    """Walk's signed touchdown air-time score, including all touchdowns in the control interval."""
    state = env.action_manager.get_term(action_term_name).contact_state
    credit = ((state.touchdown_air_time.clamp(max=max_air_time) - threshold) * state.touchdown).sum(dim=(1, 2))
    command = env.command_manager.get_command(command_name)
    side = command[:, 1].abs() > command_threshold
    turn = command[:, 2].abs() > command_threshold
    moving = (torch.linalg.vector_norm(command[:, :2], dim=1) > command_threshold) | turn
    scale = 1.0 + (side_multiplier - 1.0) * side + (turn_multiplier - 1.0) * turn
    return credit * scale * moving


def proxy_feet_slide_ang_z_cmd(
    env,
    command_name,
    asset_cfg,
    cmd_lin_threshold=0.1,
    cmd_ang_threshold=0.1,
    action_term_name="sand_proxy",
):
    """Preserve Walk's body-relative foot sliding penalty during near-pure yaw commands."""
    term = env.action_manager.get_term(action_term_name)
    data = env.scene[asset_cfg.name].data
    velocity_w = data.body_lin_vel_w.torch[:, term.body_ids] - data.root_lin_vel_w.torch[:, None, :]
    rotation = data.root_quat_w.torch[:, None, :].expand(-1, 4, -1)
    velocity_b = quat_apply_inverse(rotation, velocity_w)
    contact = term.force_history_w.norm(dim=-1).amax(dim=1) > term.cfg.contact_force_threshold
    penalty = (velocity_b[..., :2].norm(dim=-1) * contact).sum(dim=1)
    command = env.command_manager.get_command(command_name)
    gate = (command[:, :2].norm(dim=1) < cmd_lin_threshold) & (command[:, 2].abs() > cmd_ang_threshold)
    return penalty * gate


def _group_ids(term, names):
    return [term.cfg.foot_names.index(name) for name in names]


def proxy_rotation_gait_status(
    env,
    command_name,
    asset_cfg,
    group_a_body_names,
    group_b_body_names,
    target_height=0.05,
    lin_vel_threshold=0.5,
    ang_vel_threshold=0.05,
    action_term_name="sand_proxy",
):
    """Walk's diagonal support/height pattern, using current proxy-force contacts."""
    term = env.action_manager.get_term(action_term_name)
    a = _group_ids(term, group_a_body_names)
    b = _group_ids(term, group_b_body_names)
    height = env.scene[asset_cfg.name].data.body_pos_w.torch[:, term.body_ids, 2]
    delta = height[:, b].mean(dim=1) - height[:, a].mean(dim=1)
    contact = term.contact_state.contact
    pattern = (contact[:, a].all(dim=1) & (delta > target_height)) | (
        contact[:, b].all(dim=1) & (-delta > target_height)
    )
    command = env.command_manager.get_command(command_name)
    gate = (command[:, :2].norm(dim=1) < lin_vel_threshold) & (command[:, 2].abs() > ang_vel_threshold)
    return (pattern & gate).float()


class ProxyRotationGaitSymmetry(ManagerTermBase):
    """Walk's five-second diagonal duty-cycle score, with per-environment reset warm-up."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.buffer_size = max(1, int(5.0 / env.step_dt))
        self.contact_buffer = torch.zeros((env.num_envs, 4, self.buffer_size), device=env.device)
        self.samples = torch.zeros(env.num_envs, device=env.device)
        self.buffer_idx = 0

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.contact_buffer[ids] = 0
        self.samples[ids] = 0

    def __call__(
        self,
        env,
        command_name,
        group_a_body_names,
        group_b_body_names,
        target_duty=0.5,
        std=0.2,
        lin_vel_threshold=0.5,
        ang_vel_threshold=0.05,
        action_term_name="sand_proxy",
    ):
        term = env.action_manager.get_term(action_term_name)
        self.contact_buffer[:, :, self.buffer_idx] = term.contact_state.contact
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        self.samples.add_(1).clamp_(max=self.buffer_size)
        a = _group_ids(term, group_a_body_names)
        b = _group_ids(term, group_b_body_names)
        duty_a = self.contact_buffer[:, a].mean(dim=(1, 2))
        duty_b = self.contact_buffer[:, b].mean(dim=(1, 2))
        reward = torch.exp(-((duty_a - target_duty).square() + (duty_b - target_duty).square()) / std**2)
        command = env.command_manager.get_command(command_name)
        gate = (command[:, :2].norm(dim=1) < lin_vel_threshold) & (command[:, 2].abs() > ang_vel_threshold)
        return reward * (self.samples / self.buffer_size) * gate
