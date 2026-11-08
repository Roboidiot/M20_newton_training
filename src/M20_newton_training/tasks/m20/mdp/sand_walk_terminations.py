"""Sand-walking failure conditions and rewards tied to their termination flags."""

from __future__ import annotations

import math

import torch

from isaaclab.managers import SceneEntityCfg


def base_height_low(env, minimum_height: float = 0.60, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    """Terminate when the base-link origin reaches a low world-up height above the environment origin.

    This is not clearance above the nominal sand surface: a 0.30 m surface and
    minimum_height=0.60 correspond to 0.30 m nominal base clearance.
    """
    if not math.isfinite(minimum_height) or minimum_height <= 0:
        raise ValueError("minimum_height must be finite and positive")
    asset_name = "robot" if asset_cfg is None else asset_cfg.name
    height = env.scene[asset_name].data.root_link_pos_w.torch[:, 2] - env.scene.env_origins[:, 2]
    return height <= minimum_height + 1e-6


def front_wheels_low(env, height_difference: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Terminate when rear wheel centers are at least height_difference above front centers.

    asset_cfg must resolve the ordered FL/FR/HL/HR wheel link origins. Heights
    are world Z, without contact, command, sand-surface or body-pitch projection gates.
    """
    if not math.isfinite(height_difference) or height_difference <= 0:
        raise ValueError("height_difference must be finite and positive")
    heights = env.scene[asset_cfg.name].data.body_link_pos_w.torch[:, asset_cfg.body_ids, 2]
    if heights.shape[1] != 4:
        raise ValueError("Expected four ordered FL/FR/HL/HR wheel bodies")
    front_mean = heights[:, :2].mean(dim=1)
    rear_mean = heights[:, 2:].mean(dim=1)
    return rear_mean - front_mean >= height_difference - 1e-6


def sand_reset_penalty(env, termination_name: str) -> torch.Tensor:
    """Return this failure's flag; RewardManager applies its weight and control dt.

    Reading the computed flag avoids re-evaluating geometry or penalizing unrelated
    timeout resets. Two failures on the same step receive two separate costs.
    """
    return env.termination_manager.get_term(termination_name).float()
