"""Passive MPM foot impulses exposed through the existing force-based gait interface.

The zero-dimensional action owns a control-window accumulator, not a force model.
It never applies a wrench. Every MPM substep contributes its world-space collider
impulses; the first reward read publishes their mean force once per policy step.
"""

from __future__ import annotations

import math
import re
import weakref

import torch
import warp as wp

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import NewtonManager

from .foot_sand_proxy import rotate_xyzw
from .proxy_contact_rewards import ProxyFootContactState


def foot_body_slots(labels, num_envs, foot_names):
    """Map entry-local bodies to env-major FL/FR/HL/HR slots using full labels.

    Entry-local body order need not match the rigid solver or IsaacLab's asset.
    Other robot links, tray bodies and static colliders are deliberately excluded.
    """
    slots = [-1] * len(labels)
    seen = set()
    for body, label in enumerate(labels):
        match = re.search(r"/env_(\d+)/Robot/([^/]+)$", label.rstrip("/"))
        if match is None or match[2] not in foot_names:
            continue
        env_id = int(match[1])
        if not 0 <= env_id < num_envs:
            raise ValueError(f"MPM foot body is outside configured environments: {label}")
        slot = env_id * len(foot_names) + foot_names.index(match[2])
        if slot in seen:
            raise ValueError(f"Duplicate MPM foot body: {label}")
        seen.add(slot)
        slots[body] = slot
    if len(seen) != num_envs * len(foot_names):
        raise ValueError(f"Missing MPM foot bodies: found {len(seen)}, expected {num_envs * len(foot_names)}")
    return slots


@wp.kernel
def accumulate_foot_impulses(
    impulses: wp.array[wp.vec3],
    collider_ids: wp.array[int],
    collider_body: wp.array[int],
    body_slots: wp.array[int],
    total_impulse: wp.array[wp.vec3],
):
    i = wp.tid()
    collider = collider_ids[i]
    if collider >= 0:
        body = collider_body[collider]
        if body >= 0:
            slot = body_slots[body]
            if slot >= 0:
                wp.atomic_add(total_impulse, slot, impulses[i])


class MPMFootContactAction(ActionTerm):
    """Observe actual sand reactions without adding policy dimensions or forces.

    The passive hook runs after each MPM step (also for coupled-entry substeps).
    Contact timers, force history and capsule-bottom heights publish together on
    the first reward access, after physics and before reset. Later reward reads
    reuse that same 50 Hz sample, independent of the number/order of rewards.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._export_IO_descriptor = False
        physics = env.cfg.sim.physics
        root = NewtonManager._solver  # IsaacLab 3.0 has no public coupled-solver accessor.
        if not callable(getattr(root, "solver", None)) or cfg.entry_name not in root.entry_names():
            raise ValueError(
                "MPM foot contacts require the coupled sand backend: use physics=newton_sand "
                "instead of physics=newton_mjwarp. The rigid-only backend does not simulate sand particles."
            )
        if getattr(physics, "use_cuda_graph", False):
            raise ValueError(
                "MPM foot impulse sampling requires sim.physics.use_cuda_graph=False. "
                "Use physics=newton_sand env.sim.physics.use_cuda_graph=False; "
                "Python impulse sampling cannot run during CUDA Graph replay."
            )
        self.solver = root.solver(cfg.entry_name)
        if not callable(getattr(self.solver, "collect_collider_impulses", None)):
            raise ValueError(f"Entry {cfg.entry_name!r} does not expose MPM collider impulses")
        self.body_ids, names = self._asset.find_bodies(cfg.foot_names, preserve_order=True)
        if names != list(cfg.foot_names):
            raise ValueError(f"Expected four ordered wheel bodies, got {names}")
        slots = foot_body_slots(self.solver.model.body_label, env.num_envs, cfg.foot_names)
        self._body_slots = wp.array(slots, dtype=int, device=env.device)
        self._impulses = wp.zeros(env.num_envs * 4, dtype=wp.vec3, device=env.device)
        self.impulse_w = wp.to_torch(self._impulses).view(env.num_envs, 4, 3)
        self._force_w = torch.zeros_like(self.impulse_w)
        self._force_history_w = torch.zeros((env.num_envs, 3, 4, 3), device=env.device)
        self._application_points_w = torch.zeros_like(self._force_w)
        self._contact_state = ProxyFootContactState(env.num_envs, env.device)
        self._empty = torch.empty((env.num_envs, 0), device=env.device)
        self._local_axis = torch.tensor(cfg.capsule_axis, device=env.device).expand(env.num_envs, 4, 3)
        self._pending = False
        self._collecting = False
        self.sample_duration = 0.0
        self._install_hook()

    def _install_hook(self):
        original = self.solver.step
        self._original_step = original
        self._had_step_override = "step" in self.solver.__dict__
        owner = weakref.ref(self)

        def step(state_in, state_out, control, contacts, dt):
            result = original(state_in, state_out, control, contacts, dt)
            term = owner()
            if term is not None and term._collecting:
                term._accumulate(state_out, dt)
            return result

        self._step_hook = step
        self.solver.step = step

    def detach(self):
        """Restore the solver method without replacing a later experiment hook."""
        if getattr(self.solver, "step", None) is self._step_hook:
            if self._had_step_override:
                self.solver.step = self._original_step
            else:
                del self.solver.step

    def __del__(self):
        if hasattr(self, "_step_hook"):
            self.detach()
        super().__del__()

    @property
    def action_dim(self):
        return 0

    @property
    def raw_actions(self):
        return self._empty

    @property
    def processed_actions(self):
        return self._empty

    def process_actions(self, actions):
        self._contact_state.begin_step()
        self._impulses.zero_()
        self.sample_duration = 0.0
        self._pending = True
        self._collecting = True

    def apply_actions(self):
        """No external forces: the coupled MPM solver supplies the real reaction."""

    def _accumulate(self, state, dt):
        impulses, _, collider_ids = self.solver.collect_collider_impulses(state)
        wp.launch(
            accumulate_foot_impulses,
            dim=len(impulses),
            inputs=[impulses, collider_ids, self.solver.collider_body_index, self._body_slots, self._impulses],
            device=self._env.device,
        )
        self.sample_duration += dt

    @torch.no_grad()
    def update_contact_state(self):
        if not self._pending:
            return
        dt = self._env.step_dt
        if not math.isclose(self.sample_duration, dt, rel_tol=1e-5, abs_tol=1e-8):
            raise RuntimeError(f"MPM impulse window is {self.sample_duration:g} s; expected {dt:g} s")
        self._force_w.copy_(self.impulse_w / dt)
        self._force_history_w.copy_(self._force_history_w.roll(1, dims=1))
        self._force_history_w[:, 0].copy_(self._force_w)
        data = self._asset.data
        # The same capsule bottom used by Proxy gait heights, without loading a fitted force model.
        axis = rotate_xyzw(data.body_link_quat_w.torch[:, self.body_ids], self._local_axis)
        self._application_points_w.copy_(data.body_link_pos_w.torch[:, self.body_ids])
        bottom_offset = self.cfg.capsule_radius + self.cfg.capsule_half_length * axis[..., 2].abs()
        self._application_points_w[..., 2] -= bottom_offset
        self._contact_state.update(self._force_w, dt, self.cfg.contact_force_threshold)
        self._pending = False
        self._collecting = False

    @property
    def contact_state(self):
        self.update_contact_state()
        return self._contact_state

    @property
    def force_w(self):
        self.update_contact_state()
        return self._force_w

    @property
    def force_history_w(self):
        self.update_contact_state()
        return self._force_history_w

    @property
    def application_points_w(self):
        self.update_contact_state()
        return self._application_points_w

    def reset(self, env_ids=None):
        self._contact_state.log_force_metrics(self._env, env_ids)
        ids = slice(None) if env_ids is None else env_ids
        for value in (self.impulse_w, self._force_w, self._force_history_w, self._application_points_w):
            value[ids] = 0
        self._contact_state.reset(env_ids)
        if env_ids is None:
            self._pending = False
            self._collecting = False
            self.sample_duration = 0.0


@configclass
class MPMFootContactActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = MPMFootContactAction
    asset_name: str = "robot"
    entry_name: str = "sand"
    foot_names: tuple[str, ...] = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")
    contact_force_threshold: float = 5.0
    # M20 USD collision capsules: radius 9 cm, straight cylinder length 5.4 cm, local axis -Y.
    capsule_radius: float = 0.09
    capsule_half_length: float = 0.027
    capsule_axis: tuple[float, float, float] = (0.0, -1.0, 0.0)
