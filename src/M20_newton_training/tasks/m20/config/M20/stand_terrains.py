# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

"""Random rough roads with one fixed height-noise range."""

from __future__ import annotations

from collections.abc import Callable

from isaaclab.terrains import HfRandomUniformTerrainCfg
from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain
from isaaclab.utils import clone, configclass


def stand_random_rough_terrain(difficulty: float, cfg: StandRandomRoughTerrainCfg):
    """Use the same noise range for every tile, independent of difficulty."""
    # The upstream generator temporarily changes its config's size. Work on a
    # copy so generating one difficulty cannot affect another terrain tile.
    return random_uniform_terrain(difficulty, clone(cfg))


@configclass
class StandRandomRoughTerrainCfg(HfRandomUniformTerrainCfg):
    """Only uneven roads, with a fixed 1--16 cm height-noise sampling range."""

    function: Callable = stand_random_rough_terrain
    proportion = 1.0
    noise_range = (0.01, 0.16)
    noise_step = 0.005
    downsampled_scale = 0.2
    border_width = 0.25
