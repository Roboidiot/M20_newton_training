"""Contact-driven Walk rewards backed by applied foot proxy forces, without a ContactSensor."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase
from isaaclab.utils.math import quat_apply_inverse


class ProxyFootContactState:
    """Control-rate contact timers and one sampled touchdown per foot and step.

    Air-time events and rotation gait rewards use upward world-Z force only.
    The full-force ``contact`` state remains available separately for diagnostics.
    The first landing after reset establishes support and earns no air-time credit.
    The last applied force of each control interval is sampled once, like
    newton-terrain. Intermediate physics-rate contact switches are not counted.
    Touchdown events remain visible until the next policy action.
    """

    def __init__(self, num_envs, device):
        shape = (num_envs, 4)
        self.contact = torch.zeros(shape, dtype=torch.bool, device=device)
        self.vertical_contact = torch.zeros_like(self.contact)
        self.has_contact = torch.zeros_like(self.contact)
        self.current_air_time = torch.zeros(shape, device=device)
        self.current_contact_time = torch.zeros_like(self.current_air_time)
        self.touchdown_air_time = torch.zeros((num_envs, 1, 4), device=device)
        self.touchdown = torch.zeros_like(self.touchdown_air_time, dtype=torch.bool)

    def begin_step(self):
        self.touchdown.zero_()
        self.touchdown_air_time.zero_()

    def update(self, force_w, dt, threshold):
        """Update once per control step with dt=env.step_dt (0.02 s in this task)."""
        # Upward support establishes contact; horizontal drag and downward pull-out
        # forces must not prevent air-time credit after a foot has unloaded.
        vertical_contact = force_w[..., 2] > threshold
        landed = vertical_contact & ~self.vertical_contact & self.has_contact
        # The previous sample was airborne, so include the final integration interval.
        duration = self.current_air_time + dt
        self.touchdown[:, 0].copy_(landed)
        self.touchdown_air_time[:, 0].copy_(torch.where(landed, duration, 0.0))
        self.current_air_time.copy_(torch.where(vertical_contact, 0.0, duration))
        self.current_contact_time.copy_(torch.where(vertical_contact, self.current_contact_time + dt, 0.0))
        self.has_contact |= vertical_contact
        self.vertical_contact.copy_(vertical_contact)
        self.contact.copy_(torch.linalg.vector_norm(force_w, dim=-1) > threshold)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for buffer in (
            self.contact,
            self.vertical_contact,
            self.has_contact,
            self.current_air_time,
            self.current_contact_time,
            self.touchdown_air_time,
            self.touchdown,
        ):
            buffer[ids] = 0


def _gait_command_gate(env, command_name, min_command_norm):
    """Use the full [vx, vy, wz] command so pure yaw can still activate stepping."""
    return env.command_manager.get_command(command_name).norm(dim=1) >= min_command_norm


def proxy_feet_air_time_command_scaled(
    env,
    command_name,
    threshold=0.0,
    max_air_time=0.5,
    command_threshold=0.1,
    side_multiplier=2.0,
    turn_multiplier=3.0,
    action_term_name="sand_proxy",
    min_command_norm=0.1,
):
    """Score touchdowns only when the full velocity-command norm is large enough.

    ``command_threshold`` selects side/turn multipliers; ``min_command_norm``
    independently gates both positive credit and the short-air penalty.
    """
    state = env.action_manager.get_term(action_term_name).contact_state
    credit = ((state.touchdown_air_time.clamp(max=max_air_time) - threshold) * state.touchdown).sum(dim=(1, 2))
    command = env.command_manager.get_command(command_name)
    side = command[:, 1].abs() > command_threshold
    turn = command[:, 2].abs() > command_threshold
    scale = 1.0 + (side_multiplier - 1.0) * side + (turn_multiplier - 1.0) * turn
    return credit * scale * _gait_command_gate(env, command_name, min_command_norm)


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


def _diagonal_patterns(contact, a, b):
    """A supported/B unloaded, or B supported/A unloaded; never mixed or all-foot support."""
    return torch.stack(
        (contact[:, a].all(dim=1) & ~contact[:, b].any(dim=1), contact[:, b].all(dim=1) & ~contact[:, a].any(dim=1)),
        dim=1,
    )


def proxy_hipx_symmetry_l2(
    env,
    asset_cfg,
    command_name,
    lateral_cmd_threshold=0.05,
    yaw_cmd_threshold=0.1,
    maneuver_scale=0.25,
):
    """Mirror left/right HipX angles; ordered slots are FL, FR, HL, HR.

    M20's left/right HipX axes point in the same direction, so geometric
    reflection requires opposite angles. Sideways/turning commands relax the
    constraint without changing the straight/zero-command penalty.
    """
    angles = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]
    pair_sums = torch.stack((angles[:, 0] + angles[:, 1], angles[:, 2] + angles[:, 3]), dim=1)
    command = env.command_manager.get_command(command_name)
    maneuver = (command[:, 1].abs() > lateral_cmd_threshold) | (command[:, 2].abs() > yaw_cmd_threshold)
    scale = torch.where(maneuver, maneuver_scale, 1.0)
    return pair_sums.square().mean(dim=1) * scale


class ProxyDiagonalLiftConsistency(ManagerTermBase):
    """Compare completed diagonal swings, never simultaneous stance/swing heights.

    Record each foot's maximum lift from its support position at swing start.
    A sample starts only with both diagonal feet unloaded and the opposite pair
    supporting. Initial drops, flights and swings shorter than min_air_time are
    ignored. Once both groups have a fresh completed sample, score within-pair
    and between-pair differences once, then consume both samples. Command changes
    preserve history while active; low commands and environment resets clear it per world.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.params.get("height_scale", 0.25) <= 0:
            raise ValueError("height_scale must be positive")
        if cfg.params.get("min_air_time", 0.1) < 0 or cfg.params.get("max_pair_age_s", 2.0) <= 0:
            raise ValueError("min_air_time must be nonnegative and max_pair_age_s must be positive")
        shape = (env.num_envs, 4)
        self.support_height = torch.zeros(shape, device=env.device)
        self.support_seen = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.swing_origin = torch.zeros(shape, device=env.device)
        self.peak_height = torch.zeros(shape, device=env.device)
        self.completed_height = torch.zeros(shape, device=env.device)
        pairs = (env.num_envs, 2)
        self.swing_active = torch.zeros(pairs, dtype=torch.bool, device=env.device)
        self.swing_valid = torch.zeros_like(self.swing_active)
        self.swing_steps = torch.zeros(pairs, dtype=torch.long, device=env.device)
        self.pending = torch.zeros_like(self.swing_active)
        self.pending_steps = torch.zeros_like(self.swing_steps)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in (
            self.support_height,
            self.support_seen,
            self.swing_origin,
            self.peak_height,
            self.completed_height,
            self.swing_active,
            self.swing_valid,
            self.swing_steps,
            self.pending,
            self.pending_steps,
        ):
            value[ids] = 0

    def __call__(
        self,
        env,
        group_a_body_names,
        group_b_body_names,
        height_scale=0.25,
        min_air_time=0.1,
        max_pair_age_s=2.0,
        action_term_name="sand_proxy",
        command_name="base_velocity",
        min_command_norm=0.1,
    ):
        gate = _gait_command_gate(env, command_name, min_command_norm)
        self.reset(~gate)
        term = env.action_manager.get_term(action_term_name)
        contact = term.contact_state.vertical_contact
        height = term.application_points_w[..., 2]
        a, b = _group_ids(term, group_a_body_names), _group_ids(term, group_b_body_names)
        patterns = _diagonal_patterns(contact, a, b)
        minimum_steps = max(1, math.ceil(min_air_time / env.step_dt - 1e-6))
        maximum_age_steps = max(1, math.floor(max_pair_age_s / env.step_dt + 1e-6))
        self.pending_steps.add_(self.pending)
        self.pending &= self.pending_steps <= maximum_age_steps
        supported = contact & gate[:, None]
        self.support_height.copy_(torch.where(supported, height, self.support_height))
        self.support_seen |= supported
        flight = ~contact.any(dim=1)
        for i, swing in enumerate((a, b)):
            supported = contact[:, swing].all(dim=1)
            start = patterns[:, 1 - i] & ~self.swing_active[:, i] & self.support_seen[:, swing].all(dim=1) & gate
            self.swing_origin[:, swing] = torch.where(
                start[:, None], self.support_height[:, swing], self.swing_origin[:, swing]
            )
            self.peak_height[:, swing] = torch.where(start[:, None], 0.0, self.peak_height[:, swing])
            self.swing_steps[:, i] = torch.where(start, 0, self.swing_steps[:, i])
            self.swing_valid[:, i] |= start
            self.swing_active[:, i] |= start
            active = self.swing_active[:, i]
            self.swing_steps[:, i].add_(active)
            self.swing_valid[:, i] &= ~(active & flight)
            lift = (height[:, swing] - self.swing_origin[:, swing]).clamp_min(0)
            self.peak_height[:, swing] = torch.where(
                active[:, None] & ~contact[:, swing],
                torch.maximum(self.peak_height[:, swing], lift),
                self.peak_height[:, swing],
            )
            finished = active & supported
            completed = finished & self.swing_valid[:, i] & (self.swing_steps[:, i] >= minimum_steps)
            self.completed_height[:, swing] = torch.where(
                completed[:, None], self.peak_height[:, swing], self.completed_height[:, swing]
            )
            self.pending[:, i] |= completed
            self.pending_steps[:, i] = torch.where(completed, 0, self.pending_steps[:, i])
            self.swing_active[:, i] &= ~finished
            self.swing_valid[:, i] &= ~finished
        ready = self.pending.all(dim=1) & gate
        ha, hb = self.completed_height[:, a], self.completed_height[:, b]
        within = 0.5 * ((ha[:, 0] - ha[:, 1]).square() + (hb[:, 0] - hb[:, 1]).square())
        between = (ha.mean(dim=1) - hb.mean(dim=1)).square()
        penalty = ((within + between) / height_scale**2).clamp(max=4.0)
        self.pending[ready] = False
        self.pending_steps[ready] = 0
        return torch.where(ready, penalty, 0.0)


class ProxyRotationGaitStatus(ManagerTermBase):
    """Reward new paired lift progress above each foot's last support position.

    Each swing pair can earn at most one normalized unit per swing. Both feet
    must unload while the opposite diagonal supports the robot. Holding a pose
    or lowering and raising within the already credited height earns nothing.
    Low commands and environment resets clear support history.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.params.get("target_height", 0.25) <= 0:
            raise ValueError("target_height must be positive")
        self.support_height = torch.zeros((env.num_envs, 4), device=env.device)
        self.support_seen = torch.zeros_like(self.support_height, dtype=torch.bool)
        self.peak_progress = torch.zeros((env.num_envs, 2), device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in (self.support_height, self.support_seen, self.peak_progress):
            value[ids] = 0

    def __call__(
        self,
        env,
        group_a_body_names,
        group_b_body_names,
        target_height=0.25,
        action_term_name="sand_proxy",
        command_name="base_velocity",
        min_command_norm=0.1,
    ):
        gate = _gait_command_gate(env, command_name, min_command_norm)
        self.reset(~gate)
        term = env.action_manager.get_term(action_term_name)
        contact = term.contact_state.vertical_contact
        height = term.application_points_w[..., 2]  # calibrated capsule bottoms, in ordered proxy slots
        supported = contact & gate[:, None]
        self.support_height.copy_(torch.where(supported, height, self.support_height))
        self.support_seen |= supported
        a = _group_ids(term, group_a_body_names)
        b = _group_ids(term, group_b_body_names)
        patterns = _diagonal_patterns(contact, a, b)
        progress = ((height - self.support_height) / target_height).clamp(0.0, 1.0)
        reward = torch.zeros(env.num_envs, device=env.device)
        # Pattern A means A supports and B swings; progress slots follow the swing pairs.
        for i, swing in enumerate((b, a)):
            self.peak_progress[:, i] = torch.where(contact[:, swing].all(dim=1), 0.0, self.peak_progress[:, i])
            valid = patterns[:, i] & self.support_seen[:, swing].all(dim=1) & gate
            paired = progress[:, swing].amin(dim=1)
            increment = (paired - self.peak_progress[:, i]).clamp(min=0.0)
            reward += torch.where(valid, increment, 0.0)
            self.peak_progress[:, i] = torch.where(
                valid, torch.maximum(self.peak_progress[:, i], paired), self.peak_progress[:, i]
            )
        return reward


class ProxyRotationGaitSymmetry(ManagerTermBase):
    """Balance diagonal support while commanded to move; low commands clear the window."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        window_s = cfg.params.get("window_s", 1.0)
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        self.buffer_size = max(1, round(window_s / env.step_dt))
        self.pattern_buffer = torch.zeros((env.num_envs, 2, self.buffer_size), device=env.device)
        self.samples = torch.zeros(env.num_envs, device=env.device)
        self.buffer_idx = 0

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.pattern_buffer[ids] = 0
        self.samples[ids] = 0

    def __call__(
        self,
        env,
        group_a_body_names,
        group_b_body_names,
        window_s=1.0,
        action_term_name="sand_proxy",
        command_name="base_velocity",
        min_command_norm=0.1,
    ):
        gate = _gait_command_gate(env, command_name, min_command_norm)
        self.reset(~gate)
        term = env.action_manager.get_term(action_term_name)
        a = _group_ids(term, group_a_body_names)
        b = _group_ids(term, group_b_body_names)
        patterns = _diagonal_patterns(term.contact_state.vertical_contact, a, b) & gate[:, None]
        self.pattern_buffer[:, :, self.buffer_idx] = patterns
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        self.samples.add_(gate).clamp_(max=self.buffer_size)
        # Unfilled slots are zero: the score warms up without counting pre-reset history.
        occupancy = self.pattern_buffer.mean(dim=2)
        return 2.0 * torch.sqrt(occupancy[:, 0] * occupancy[:, 1])
