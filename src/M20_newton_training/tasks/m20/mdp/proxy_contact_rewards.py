"""Contact-driven Walk rewards backed by applied foot proxy forces, without a ContactSensor."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

HIPX_BODY_NAMES = ("fl_hipx", "fr_hipx", "hl_hipx", "hr_hipx")


def _ordered_hipx_body_ids(env):
    """Resolve HipX joint origins in the same FL/FR/HL/HR order as the foot providers.

    The deployed M20 USD has localPos1=(0,0,0) for each HipX joint, so the child
    link origin is exactly the joint pivot. Use body_link_pos_w, never link CoM.
    Reference-axis conventions remain world Z / chassis-yaw-aligned XY; these
    are position references, not frames rotating with the HipX motor angle.
    """
    ids, names = env.scene["robot"].find_bodies(list(HIPX_BODY_NAMES), preserve_order=True)
    if list(names) != list(HIPX_BODY_NAMES) or len(ids) != 4 or len(set(ids)) != 4:
        raise ValueError(f"Expected four ordered HipX bodies {HIPX_BODY_NAMES}; got {names} / {ids}")
    return ids


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
        # World-Z force diagnostics sampled with the contact state, once per control step.
        self.fz_integral = torch.zeros(shape, device=device)
        self.force_duration = torch.zeros(num_envs, device=device)

    def begin_step(self):
        self.touchdown.zero_()
        self.touchdown_air_time.zero_()

    def update(self, force_w, dt, threshold):
        """Update once per control step with dt=env.step_dt (0.02 s in this task)."""
        fz = force_w[..., 2]
        self.fz_integral.add_(fz * dt)
        self.force_duration.add_(dt)
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

    def log_force_metrics(self, env, env_ids=None):
        """Log signed episode mean Fz before clearing resetting worlds.

        All samples count, including airborne zeros and negative pull-out forces.
        Scalars average the resetting worlds' own episode statistics, rather than
        summing parallel robots or taking the norm of the complete force vector.
        """
        extras = getattr(env, "extras", None)
        if extras is None:
            return
        ids = slice(None) if env_ids is None else env_ids
        duration = self.force_duration[ids]
        if duration.numel() == 0:
            return
        valid = (duration > 0).to(self.fz_integral.dtype)
        count = valid.sum().clamp_min(1.)
        divisor = duration.clamp_min(1e-12)[:, None]
        values = self.fz_integral[ids] / divisor
        means = (values * valid[:, None]).sum(dim=0) / count
        log = extras.setdefault("log", {})
        for foot, value in zip(("fl", "fr", "hl", "hr"), means, strict=True):
            log[f"Metrics/foot_forces/{foot}_fz_mean"] = value

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
            self.fz_integral,
            self.force_duration,
        ):
            buffer[ids] = 0


def _gait_command_gate(env, command_name, min_command_norm):
    """Use the full [vx, vy, wz] command so pure yaw can still activate stepping."""
    return env.command_manager.get_command(command_name).norm(dim=1) >= min_command_norm


def sand_base_height_l2(env, target_height, surface_height=0.30):
    """Squared root height error relative to the nominal sand surface, in meters.

    Environment origin Z is removed explicitly. Ray casters see the rigid floor
    rather than the analytic/MPM sand surface and must not supply this reference.
    """
    height = env.scene["robot"].data.root_pos_w.torch[:, 2] - env.scene.env_origins[:, 2] - surface_height
    return (height - target_height).square()


def front_wheels_height_reward(env, asset_cfg: SceneEntityCfg):
    """Reward positive front-minus-rear mean wheel height in world Z, in meters.

    asset_cfg must resolve wheel link origins in FL/FR/HL/HR order. Equal wheel
    radii make their center-height difference equal to the bottom-height
    difference. This is instantaneous geometry: no contact or command gate,
    target-height normalization, or upper clamp is applied. Lowering the rear
    wheels also increases this relative-height reward.
    """
    heights = env.scene[asset_cfg.name].data.body_link_pos_w.torch[:, asset_cfg.body_ids, 2]
    if heights.shape[1] != 4:
        raise ValueError("Expected four ordered FL/FR/HL/HR wheel bodies")
    front_mean = heights[:, :2].mean(dim=1)
    rear_mean = heights[:, 2:].mean(dim=1)
    return (front_mean - rear_mean).clamp_min(0.0)


def forward_pitch_orientation_l2(env, command_name, forward_pitch_deg=-5.0, forward_threshold=0.1):
    """Keep zero roll and a small nose-up pitch while commanded forward.

    M20 moves forward along body +X with +Z up. Negative right-handed pitch
    about +Y raises its nose. For pitch theta, projected gravity in the body is
    [sin(theta), 0, -cos(theta)]. Retain the original XY gravity-error penalty,
    shifting its X target only for vx commands above forward_threshold.
    Other commands retain the level-body target; yaw does not affect the error.
    """
    gravity = env.scene["robot"].data.projected_gravity_b.torch
    forward = env.command_manager.get_command(command_name)[:, 0] > forward_threshold
    target_x = torch.where(forward, math.sin(math.radians(forward_pitch_deg)), 0.0)
    return (gravity[:, 0] - target_x).square() + gravity[:, 1].square()


def no_turn_ang_vel_z_l2(env, command_name, zero_command_threshold=1e-6):
    """Penalize body yaw speed squared only when no yaw motion is commanded."""
    command = env.command_manager.get_command(command_name)[:, 2]
    yaw_speed = env.scene["robot"].data.root_ang_vel_b.torch[:, 2]
    return yaw_speed.square() * (command.abs() <= zero_command_threshold)


def com_forward_support_l2(
    env,
    command_name="base_velocity",
    action_term_name="sand_proxy",
    allowed_forward_offset=0.03,
    distance_scale=0.10,
    min_command_norm=0.1,
    min_support_feet=2,
):
    """Penalize whole-robot CoM bias toward the planar command, relative to supporting feet.

    The reference is the unweighted mean XY position of feet with upward support
    (the provider's existing vertical_contact predicate), not a force-weighted
    center of pressure. Swing feet are excluded. Current per-link masses and CoM
    positions include mass/CoM randomization; the root link is not the robot CoM.
    Both the reference and the robot move freely in world space. Only their
    relative displacement projected onto the yaw-aligned command is penalized.

    Returns a nonnegative, normalized squared excess; bind a negative weight.
    Idle/pure-yaw commands and fewer than min_support_feet supports return zero.
    This is a directional posture cost, not a support-polygon/ZMP stability test.
    """
    if not math.isfinite(allowed_forward_offset) or allowed_forward_offset < 0:
        raise ValueError("allowed_forward_offset must be finite and nonnegative")
    if not math.isfinite(distance_scale) or distance_scale <= 0:
        raise ValueError("distance_scale must be finite and positive")
    if not math.isfinite(min_command_norm) or min_command_norm <= 0:
        raise ValueError("min_command_norm must be finite and positive")
    if type(min_support_feet) is not int or not 1 <= min_support_feet <= 4:
        raise ValueError("min_support_feet must be an integer in [1, 4]")

    command = env.command_manager.get_command(command_name)[:, :2]
    norm = command.norm(dim=1)
    direction_b = torch.cat((command / norm.clamp_min(1e-8)[:, None], torch.zeros_like(norm[:, None])), dim=1)
    data = env.scene["robot"].data
    direction_w = quat_apply(yaw_quat(data.root_link_quat_w.torch), direction_b)[:, :2]

    term = env.action_manager.get_term(action_term_name)
    contact = term.contact_state.vertical_contact  # Publishes the MPM window before reading foot positions.
    count = contact.sum(dim=1)
    support_center = (term.application_points_w[..., :2] * contact[..., None]).sum(dim=1)
    support_center = support_center / count.clamp_min(1)[:, None]
    mass = data.body_mass.torch
    # Subtract the local support reference before summing to retain precision in
    # environments far from the world origin. This equals CoM_world - support_center.
    relative_com = (
        (data.body_com_pos_w.torch[..., :2] - support_center[:, None, :]) * mass[..., None]
    ).sum(dim=1) / mass.sum(dim=1).clamp_min(1e-8)[:, None]
    offset = (relative_com * direction_w).sum(dim=1)
    penalty = ((offset - allowed_forward_offset).clamp_min(0.0) / distance_scale).square()
    return torch.where((norm >= min_command_norm) & (count >= min_support_feet), penalty, 0.0)


def _foot_participation_scale(progress, repeat_decay, max_progress_lead=None):
    """Discount a foot's lead over the least-participating foot, without fixing order."""
    lead = (progress - progress.amin(dim=1, keepdim=True)).clamp_min(0.0)
    if max_progress_lead is not None:
        scale = (1.0 - lead / max_progress_lead).clamp(0.0, 1.0)
        return torch.where(lead + 1e-6 >= max_progress_lead, 0.0, scale)
    return 1.0 / (1.0 + repeat_decay * lead)


def _advance_participation_balance(balance, increment, max_progress_lead):
    """Cap the internal lead; unpaid repeats must not build an ever-growing catch-up debt."""
    ceiling = balance.amin(dim=1, keepdim=True) + max_progress_lead
    balance.copy_(torch.minimum(balance + increment, ceiling))


def _log_foot_participation(env, name, counts, progress, env_ids):
    """Log per-foot completion counts and full-step-equivalent progress at resets."""
    extras = getattr(env, "extras", None)
    if extras is None:
        return
    ids = slice(None) if env_ids is None else env_ids
    selected = counts[ids]
    if selected.shape[0] == 0:
        return
    log = extras.setdefault("log", {})
    mean_counts, mean_progress = selected.float().mean(dim=0), progress[ids].mean(dim=0)
    for i, foot in enumerate(("fl", "fr", "hl", "hr")):
        log[f"Metrics/{name}/{foot}_count"] = mean_counts[i]
        log[f"Metrics/{name}/{foot}_progress"] = mean_progress[i]


class _SandFootSwingReward(ManagerTermBase):
    """Track independent, supported swings at the reward/control rate.

    A swing must start from observed support, retain at least two other supports,
    and keep the chassis upright and above its minimum sand-relative height.
    Initial drops, flights, long swings and reversed commands cannot earn credit.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        for name, default in (
            ("target_clearance", 0.04), ("target_stride", 0.08), ("min_stride", 0.03),
            ("min_air_time", 0.1), ("max_swing_time", 1.0),
            ("min_base_height", 0.35), ("max_tilt", 0.5), ("min_command_norm", 0.1),
        ):
            value = cfg.params.get(name, default)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if cfg.params.get("min_stride", 0.03) > cfg.params.get("target_stride", 0.08):
            raise ValueError("min_stride must not exceed target_stride")
        if cfg.params.get("min_air_time", 0.1) > cfg.params.get("max_swing_time", 1.0):
            raise ValueError("min_air_time must not exceed max_swing_time")
        if cfg.params.get("max_tilt", 0.5) >= math.pi / 2:
            raise ValueError("max_tilt must be less than pi/2")
        shape = (env.num_envs, 4)
        self.previous_contact = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.swing_active = torch.zeros_like(self.previous_contact)
        self.swing_valid = torch.zeros_like(self.previous_contact)
        self.cleared = torch.zeros_like(self.previous_contact)
        self.swing_steps = torch.zeros(shape, dtype=torch.long, device=env.device)
        self.peak_clearance = torch.zeros(shape, device=env.device)
        self.peak_forward = torch.zeros_like(self.peak_clearance)
        self.forward_origin_w = torch.zeros((*shape, 3), device=env.device)
        self.forward_origin_b = torch.zeros_like(self.forward_origin_w)
        self.direction_b = torch.zeros((*shape, 2), device=env.device)
        self.direction_w = torch.zeros_like(self.direction_b)
        self.previous_direction = torch.zeros((env.num_envs, 2), device=env.device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for name in (
            "previous_contact", "swing_active", "swing_valid", "cleared", "swing_steps",
            "peak_clearance", "peak_forward", "forward_origin_w", "forward_origin_b",
            "direction_b", "direction_w", "previous_direction",
        ):
            getattr(self, name)[ids] = 0

    def __call__(
        self, env, target_clearance=0.04, target_stride=0.08, min_stride=0.03,
        min_air_time=0.1, max_swing_time=1.0, surface_height=0.30,
        min_base_height=0.35, max_tilt=0.5, action_term_name="sand_proxy",
        command_name="base_velocity", min_command_norm=0.1,
    ):
        command = env.command_manager.get_command(command_name)[:, :2]
        norm = command.norm(dim=1)
        direction = command / norm.clamp_min(1e-6)[:, None]
        gate = norm >= min_command_norm
        changed = (direction * self.previous_direction).sum(dim=1) < 0.95
        self.reset(~gate | changed)
        self.previous_direction.copy_(direction)

        term = env.action_manager.get_term(action_term_name)
        contact = term.contact_state.vertical_contact  # publishes the MPM window once if needed
        point = term.application_points_w
        data = env.scene["robot"].data
        root = data.root_pos_w.torch
        surface = env.scene.env_origins[:, 2] + surface_height
        upright = -data.projected_gravity_b.torch[:, 2] >= math.cos(max_tilt)
        posture = upright & (root[:, 2] - surface >= min_base_height)
        supported = contact.sum(dim=1) >= 2
        valid_env = gate & posture & supported
        rotation = yaw_quat(data.root_quat_w.torch)
        relative = quat_apply_inverse(rotation[:, None, :].expand(-1, 4, -1), point - root[:, None, :])

        start = self.previous_contact & ~contact & valid_env[:, None]
        for buffer in (self.swing_steps, self.peak_clearance, self.peak_forward, self.cleared):
            buffer.masked_fill_(start, 0)
        self.swing_active |= start
        self.swing_valid |= start
        self.direction_b.copy_(torch.where(start[..., None], direction[:, None, :], self.direction_b))
        direction3 = torch.cat((direction, torch.zeros_like(norm[:, None])), dim=1)
        world_direction = quat_apply(rotation, direction3)[:, :2]
        self.direction_w.copy_(torch.where(start[..., None], world_direction[:, None, :], self.direction_w))
        self.swing_steps.add_(self.swing_active)
        minimum_steps = max(1, math.ceil(min_air_time / env.step_dt - 1e-6))
        maximum_steps = max(1, math.floor(max_swing_time / env.step_dt + 1e-6))
        self.swing_valid &= valid_env[:, None] & (self.swing_steps <= maximum_steps)
        eligible = self.swing_active & self.swing_valid & ~contact
        clearance = point[..., 2] - surface[:, None]
        first_clear = eligible & ~self.cleared & (clearance >= target_clearance)
        # Measure the forward stroke from extraction, not from buried liftoff.
        self.forward_origin_w.copy_(torch.where(first_clear[..., None], point, self.forward_origin_w))
        self.forward_origin_b.copy_(torch.where(first_clear[..., None], relative, self.forward_origin_b))
        self.cleared |= first_clear
        advance_b = ((relative - self.forward_origin_b)[..., :2] * self.direction_b).sum(dim=-1)
        advance_w = ((point - self.forward_origin_w)[..., :2] * self.direction_w).sum(dim=-1)
        # Translation/rotation of a rigidly held leg and backward chassis drift
        # alone must not masquerade as a forward foot stroke.
        advance = torch.minimum(advance_b, advance_w).clamp_min(0.0)
        forward = (advance / target_stride).clamp(max=1.0)
        clear_forward = eligible & self.cleared & (clearance >= target_clearance)
        self.peak_forward.copy_(
            torch.where(clear_forward, torch.maximum(self.peak_forward, forward), self.peak_forward)
        )
        mature = self.swing_steps >= minimum_steps
        completed = self.swing_active & self.swing_valid & contact & mature & self.cleared
        score = self._score(
            (clearance / target_clearance).clamp(0.0, 1.0),
            forward, advance >= min_stride, eligible & mature, completed,
        )
        self.swing_active &= ~contact
        self.swing_valid &= ~contact
        self.previous_contact.copy_(contact & gate[:, None])
        return score.sum(dim=1)


class ProxyFootClearanceReward(_SandFootSwingReward):
    """Credit new foot-bottom clearance above sand, capped at one unit per swing.

    No diagonal pattern is required. Buried unloading, initial falls, hovering and
    revisiting a previously credited clearance earn zero. Units are incremental;
    RewardManager still applies weight * control dt, as for the legacy lift term.
    """

    def _score(self, clearance, forward, enough_stride, eligible, completed):
        increment = (clearance - self.peak_clearance).clamp_min(0.0)
        score = torch.where(eligible, increment, 0.0)
        self.peak_clearance.copy_(
            torch.where(eligible, torch.maximum(self.peak_clearance, clearance), self.peak_clearance)
        )
        return score


class ProxyExtractedFootForwardStepReward(_SandFootSwingReward):
    """Reward extracted forward strokes only upon a valid subsequent touchdown.

    Forward distance must increase both in the chassis yaw frame and in the world
    command direction. It starts at the first observed target clearance. The
    touchdown keeps at most the advance attained above sand: lifting in place,
    buried dragging, forward-then-retracted feet and holding in air earn zero.
    One completed foot earns at most one unit, without a fixed gait phase.
    """

    def _score(self, clearance, forward, enough_stride, eligible, completed):
        return torch.where(completed & enough_stride, torch.minimum(self.peak_forward, forward), 0.0)


class ProxyFootForwardStepReward(ManagerTermBase):
    """Score a foot's world-XY advance between successive sampled touchdowns.

    Store the calibrated foot-bottom application point in world coordinates.
    Project its displacement onto the current planar command rotated into world
    XY by the chassis yaw. No chassis or HipX position is subtracted. Chassis
    motion alone with a fixed world-space foot earns nothing; a foot carried
    forward with the chassis can earn credit after qualifying unloading.
    Only positive progress earns credit, capped at one raw unit per foot at
    target_stride. A touchdown only
    scores after min_air_time of consecutive sampled unloading. No sand clearance,
    additional minimum stride, posture or support-count gate is used.
    The first support after reset establishes an anchor without earning credit.
    Continuous support cannot earn repeated progress; commands changing direction
    or becoming inactive discard anchors. All state updates run at control rate.
    Optional repeat_decay discounts feet ahead of the least-participating foot.
    Participation counts pre-discount progress: one full stride is one effective
    step; tiny contact jitter contributes proportionally little to catching up.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        for name, default in (("target_stride", 0.08), ("min_command_norm", 0.1)):
            if not math.isfinite(cfg.params.get(name, default)) or cfg.params.get(name, default) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(cfg.params.get("repeat_decay", 0.0)) or cfg.params.get("repeat_decay", 0.0) < 0:
            raise ValueError("repeat_decay must be finite and nonnegative")
        if not math.isfinite(cfg.params.get("min_air_time", 0.1)) or cfg.params.get("min_air_time", 0.1) < 0:
            raise ValueError("min_air_time must be finite and nonnegative")
        self.previous_touchdown_w = torch.zeros((env.num_envs, 4, 3), device=env.device)
        self.touchdown_seen = torch.zeros((env.num_envs, 4), dtype=torch.bool, device=env.device)
        self.previous_direction = torch.zeros((env.num_envs, 2), device=env.device)
        self.completed_steps = torch.zeros((env.num_envs, 4), dtype=torch.long, device=env.device)
        self.participation_progress = torch.zeros((env.num_envs, 4), device=env.device)

    def reset(self, env_ids=None):
        _log_foot_participation(
            self._env, "forward_steps", self.completed_steps, self.participation_progress, env_ids
        )
        self._reset_state(env_ids)

    def _reset_state(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for buffer in (
            self.previous_touchdown_w, self.touchdown_seen, self.previous_direction,
            self.completed_steps, self.participation_progress,
        ):
            buffer[ids] = 0

    def __call__(
        self, env, target_stride=0.08, action_term_name="sand_proxy",
        command_name="base_velocity", min_command_norm=0.1, repeat_decay=0.0, min_air_time=0.1,
    ):
        command = env.command_manager.get_command(command_name)[:, :2]
        norm = command.norm(dim=1)
        direction = command / norm.clamp_min(1e-6)[:, None]
        gate = norm >= min_command_norm
        changed = (direction * self.previous_direction).sum(dim=1) < 0.95
        self._reset_state(~gate | changed)
        self.previous_direction.copy_(direction)
        term = env.action_manager.get_term(action_term_name)
        state = term.contact_state  # Also publishes the MPM control-window force.
        contact = state.vertical_contact
        landed = state.touchdown[:, 0] & contact
        point = term.application_points_w
        rotation = yaw_quat(env.scene["robot"].data.root_quat_w.torch)
        direction_w = quat_apply(rotation, torch.cat((direction, torch.zeros_like(direction[:, :1])), dim=1))[:, :2]
        displacement = point[..., :2] - self.previous_touchdown_w[..., :2]
        advance = (displacement * direction_w[:, None, :]).sum(dim=-1)
        score = (advance / target_stride).clamp(0.0, 1.0)
        # The shared touchdown timer includes the landing interval for legacy
        # feet_air_time credit. Exclude that interval here: 0.1 s requires five
        # unloaded 50 Hz samples, not four unloaded samples plus one landing.
        unloaded_time = (state.touchdown_air_time[:, 0] - env.step_dt).clamp_min(0.0)
        long_enough = unloaded_time + 1e-6 >= min_air_time
        score = torch.where(landed & self.touchdown_seen & gate[:, None] & long_enough, score, 0.0)
        scaled = score * _foot_participation_scale(self.participation_progress, repeat_decay)
        self.completed_steps.add_((score > 0).long())
        self.participation_progress.add_(score)
        # Save every landing, including backward/too-short ones, so rejected
        # contact jitter cannot bank displacement toward a later valid stride.
        update = contact & (~self.touchdown_seen | landed) & gate[:, None]
        self.previous_touchdown_w.copy_(
            torch.where(update[..., None], point, self.previous_touchdown_w)
        )
        self.touchdown_seen |= update
        return scaled.sum(dim=1)


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


class ProxyFootLiftConsistency(ManagerTermBase):
    """Compare four independently completed foot swings, regardless of gait pairing.

    Each foot records its own liftoff support height, peak relative lift and
    touchdown. Once all four have fresh valid samples, score their squared
    deviation from their mean once and consume all samples. No diagonal support
    pattern or simultaneous liftoff/touchdown is required. Flights, initial drops
    and short swings are ignored; low commands and partial resets clear history.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.params.get("height_scale", 0.25) <= 0:
            raise ValueError("height_scale must be positive")
        if cfg.params.get("min_air_time", 0.1) < 0 or cfg.params.get("max_sample_age_s", 4.0) <= 0:
            raise ValueError("min_air_time must be nonnegative and max_sample_age_s must be positive")
        shape = (env.num_envs, 4)
        self.support_height = torch.zeros(shape, device=env.device)
        self.support_seen = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.previous_contact = torch.zeros_like(self.support_seen)
        self.swing_origin = torch.zeros(shape, device=env.device)
        self.peak_height = torch.zeros(shape, device=env.device)
        self.completed_height = torch.zeros(shape, device=env.device)
        self.swing_active = torch.zeros_like(self.support_seen)
        self.swing_valid = torch.zeros_like(self.support_seen)
        self.swing_steps = torch.zeros(shape, dtype=torch.long, device=env.device)
        self.pending = torch.zeros_like(self.support_seen)
        self.pending_steps = torch.zeros_like(self.swing_steps)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in (
            self.support_height,
            self.support_seen,
            self.previous_contact,
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
        height_scale=0.25,
        min_air_time=0.1,
        max_sample_age_s=4.0,
        action_term_name="sand_proxy",
        command_name="base_velocity",
        min_command_norm=0.1,
    ):
        gate = _gait_command_gate(env, command_name, min_command_norm)
        self.reset(~gate)
        term = env.action_manager.get_term(action_term_name)
        contact = term.contact_state.vertical_contact
        height = term.application_points_w[..., 2]
        minimum_steps = max(1, math.ceil(min_air_time / env.step_dt - 1e-6))
        maximum_age_steps = max(1, math.floor(max_sample_age_s / env.step_dt + 1e-6))
        self.pending_steps.add_(self.pending)
        self.pending &= self.pending_steps <= maximum_age_steps
        supported = contact & gate[:, None]
        self.support_height.copy_(torch.where(supported, height, self.support_height))
        self.support_seen |= supported
        has_support = contact.any(dim=1, keepdim=True)
        start = self.previous_contact & ~contact & ~self.swing_active & gate[:, None] & has_support
        self.swing_origin.copy_(torch.where(start, self.support_height, self.swing_origin))
        self.peak_height.copy_(torch.where(start, 0.0, self.peak_height))
        self.swing_steps.copy_(torch.where(start, 0, self.swing_steps))
        self.swing_active |= start
        self.swing_valid |= start
        self.swing_steps.add_(self.swing_active)
        self.swing_valid &= ~(self.swing_active & ~has_support)
        lift = (height - self.swing_origin).clamp_min(0)
        self.peak_height.copy_(
            torch.where(self.swing_active & ~contact, torch.maximum(self.peak_height, lift), self.peak_height)
        )
        finished = self.swing_active & contact
        completed = finished & self.swing_valid & (self.swing_steps >= minimum_steps)
        self.completed_height.copy_(torch.where(completed, self.peak_height, self.completed_height))
        self.pending |= completed
        self.pending_steps.copy_(torch.where(completed, 0, self.pending_steps))
        self.swing_active &= ~finished
        self.swing_valid &= ~finished
        self.previous_contact.copy_(supported)
        ready = self.pending.all(dim=1) & gate
        deviations = self.completed_height - self.completed_height.mean(dim=1, keepdim=True)
        penalty = (deviations.square().sum(dim=1) / height_scale**2).clamp(max=4.0)
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


def _single_swing_patterns(contact):
    """One slot per ordered proxy foot: exactly that foot unloaded and the other three supported."""
    return ~contact & (contact.sum(dim=1, keepdim=True) == 3)


class ProxyCrawlGaitStatus(ManagerTermBase):
    """Reward new single-foot lift progress while all three other feet support.

    Heights use the calibrated capsule bottom's world Z. Each foot's last
    supported world Z is the reference: lift = foot_z - foot_z_last_support.
    Neither chassis nor HipX height is subtracted. Whole-robot upward translation
    therefore contributes if this foot qualifies as the single unloaded foot;
    chassis/HipX motion with a fixed world-space foot does not. Support may be below the sand surface.
    There is no surface-clearance condition or phase clock. Each foot earns at
    most one unit per swing; support
    resets its credited progress. Initial drops and low commands earn no credit.
    min_air_time consecutive sampled unloading is required before credit starts;
    earlier peaks are not banked. Short swings cannot add participation progress.
    Optional repeat_decay uses each foot's completed, pre-discount lift progress
    to reduce the gain from repeating the same foot. One full target lift equals
    one effective step, independently of the forward-touchdown reward's counts.
    max_progress_lead selects a finite lead limit instead of the legacy inverse
    discount. Its internal balance ledger is capped so unpaid repeats do not
    create an unbounded catch-up debt; diagnostic progress remains undiscounted.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        target = cfg.params.get("target_height", 0.30)
        if not math.isfinite(target) or target <= 0:
            raise ValueError("target_height must be finite and positive")
        if not math.isfinite(cfg.params.get("repeat_decay", 0.0)) or cfg.params.get("repeat_decay", 0.0) < 0:
            raise ValueError("repeat_decay must be finite and nonnegative")
        if not math.isfinite(cfg.params.get("min_air_time", 0.1)) or cfg.params.get("min_air_time", 0.1) < 0:
            raise ValueError("min_air_time must be finite and nonnegative")
        lead = cfg.params.get("max_progress_lead")
        if lead is not None and (not math.isfinite(lead) or lead <= 0):
            raise ValueError("max_progress_lead must be finite and positive")
        self.support_height = torch.zeros((env.num_envs, 4), device=env.device)
        self.support_seen = torch.zeros_like(self.support_height, dtype=torch.bool)
        self.peak_progress = torch.zeros_like(self.support_height)
        self.completed_steps = torch.zeros_like(self.support_height, dtype=torch.long)
        self.participation_progress = torch.zeros_like(self.support_height)
        self.balance_progress = torch.zeros_like(self.support_height)

    def reset(self, env_ids=None):
        _log_foot_participation(
            self._env, "lift_steps", self.completed_steps, self.participation_progress, env_ids
        )
        self._reset_state(env_ids)

    def _reset_state(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for value in (
            self.support_height, self.support_seen, self.peak_progress,
            self.completed_steps, self.participation_progress,
            self.balance_progress,
        ):
            value[ids] = 0

    def __call__(
        self,
        env,
        target_height=0.30,
        action_term_name="sand_proxy",
        command_name="base_velocity",
        min_command_norm=0.1,
        repeat_decay=0.0,
        min_air_time=0.1,
        max_progress_lead=None,
    ):
        gate = _gait_command_gate(env, command_name, min_command_norm)
        self._reset_state(~gate)
        term = env.action_manager.get_term(action_term_name)
        state = term.contact_state
        contact = state.vertical_contact
        height = term.application_points_w[..., 2]
        supported = contact & gate[:, None]
        self.support_height.copy_(torch.where(supported, height, self.support_height))
        self.support_seen |= supported
        completed = contact & (self.peak_progress > 0)
        self.completed_steps.add_(completed.long())
        self.participation_progress.add_(torch.where(completed, self.peak_progress, 0.0))
        if max_progress_lead is not None:
            _advance_participation_balance(
                self.balance_progress, torch.where(completed, self.peak_progress, 0.0), max_progress_lead
            )
        self.peak_progress.masked_fill_(contact, 0.0)
        long_enough = state.current_air_time + 1e-6 >= min_air_time
        valid = _single_swing_patterns(contact) & self.support_seen & gate[:, None] & long_enough
        progress = ((height - self.support_height) / target_height).clamp(0.0, 1.0)
        increment = (progress - self.peak_progress).clamp_min(0.0)
        self.peak_progress.copy_(torch.where(valid, torch.maximum(self.peak_progress, progress), self.peak_progress))
        balance = self.participation_progress if max_progress_lead is None else self.balance_progress
        scale = _foot_participation_scale(balance, repeat_decay, max_progress_lead)
        rewarded = increment * valid * scale
        return rewarded.sum(dim=1)


class ProxyCrawlGaitSymmetry(ManagerTermBase):
    """Balance the four three-support/one-swing states within a sliding window.

    Score is 4*(p_FL*p_FR*p_HL*p_HR)**0.25: all four states must occur to earn
    credit; balanced full-window occupancy scores one. Other contact patterns
    occupy the window but earn no state credit. No fixed cycle or order is imposed.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        window = cfg.params.get("window_s", 2.0)
        if not math.isfinite(window) or window <= 0:
            raise ValueError("window_s must be finite and positive")
        self.buffer_size = max(1, round(window / env.step_dt))
        self.pattern_buffer = torch.zeros((env.num_envs, 4, self.buffer_size), device=env.device)
        self.samples = torch.zeros(env.num_envs, device=env.device)
        self.buffer_idx = 0

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.pattern_buffer[ids] = 0
        self.samples[ids] = 0

    def __call__(
        self,
        env,
        window_s=2.0,
        action_term_name="sand_proxy",
        command_name="base_velocity",
        min_command_norm=0.1,
    ):
        gate = _gait_command_gate(env, command_name, min_command_norm)
        self.reset(~gate)
        term = env.action_manager.get_term(action_term_name)
        patterns = _single_swing_patterns(term.contact_state.vertical_contact) & gate[:, None]
        self.pattern_buffer[:, :, self.buffer_idx] = patterns
        self.buffer_idx = (self.buffer_idx + 1) % self.buffer_size
        self.samples.add_(gate).clamp_(max=self.buffer_size)
        occupancy = self.pattern_buffer.mean(dim=2)
        return 4.0 * occupancy.prod(dim=1).pow(0.25)
