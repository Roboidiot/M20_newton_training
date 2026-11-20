"""Privileged sand observations sampled from current geometry and existing contact state."""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply


def _tensor(value):
    return value if isinstance(value, torch.Tensor) else value.torch


def sand_layer_depth(env, surface_height: float) -> torch.Tensor:
    """Layer thickness above the sand-walking rigid floor at environment-local Z=0."""
    return torch.full((env.num_envs, 1), surface_height, device=env.device)


def sand_feet_depth(env, surface_height: float, action_term_name: str) -> torch.Tensor:
    """Signed surface-minus-capsule-bottom depth, ordered FL/FR/HL/HR.

    Read post-physics link poses instead of the neural model's previous force-input
    depth. Positive values are submerged; negative values encode clearance. No
    proxy inference, force application or MPM stepping is performed here.
    """
    term = env.action_manager.get_term(action_term_name)
    data = env.scene[term.cfg.asset_name].data
    position = _tensor(data.body_link_pos_w)[:, term.body_ids]
    rotation = _tensor(data.body_link_quat_w)[:, term.body_ids]
    if hasattr(term, "local_center"):
        # Neural/analytical proxy geometry comes from its deployment calibration.
        center = position + quat_apply(rotation, term.local_center)
        axis = quat_apply(rotation, term.local_axis)
        offset = term.radius + term.half_length * axis[..., 2].abs()
    else:
        # The MPM observer exposes the same wheel capsule geometry in its config.
        center = position
        axis = quat_apply(rotation, term._local_axis)
        offset = term.cfg.capsule_radius + term.cfg.capsule_half_length * axis[..., 2].abs()
    bottom = center[..., 2] - offset
    surface = env.scene.env_origins[:, 2:3] + surface_height
    return surface - bottom


def sand_feet_contact(env, action_term_name: str) -> torch.Tensor:
    """Existing upward-force support predicate, ordered FL/FR/HL/HR, encoded as 0/1.

    The provider publishes at most once per control step. Proxy contacts already
    include rigid-floor reactions; MPM contacts use the accumulated MPM impulse.
    """
    term = env.action_manager.get_term(action_term_name)
    return term.contact_state.vertical_contact.to(dtype=torch.float32)


def sand_base_clearance(env, surface_height: float, asset_name: str = "robot") -> torch.Tensor:
    """Root world Z minus environment origin Z and nominal sand surface height."""
    root_z = _tensor(env.scene[asset_name].data.root_pos_w)[:, 2:3]
    return root_z - env.scene.env_origins[:, 2:3] - surface_height
