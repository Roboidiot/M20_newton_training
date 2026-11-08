# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Shared sand-walking policy and rewards on the coupled MJWarp/MPM bed."""

from typing import cast

from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass

from M20_newton_training.tasks.m20.mdp.mpm_foot_contact import MPMFootContactActionCfg

from .rough_env_cfg import DeeproboticsM20ActionsCfg
from .sand_env_cfg import SAND_ENV_SPACING, SAND_NUM_ENVS, DeeproboticsM20SandSceneCfg, M20SandPhysicsCfg
from .walk_env_cfg import DeeproboticsM20WalkEnvCfg
from .walk_sand_common_cfg import (
    DeeproboticsM20WalkOnSandRewardsCfg,  # noqa: F401 - preserve the old import path
    SandWalkRewardsCfg,
    SandWalkTerminationsCfg,
    configure_sand_walk_task,
)


@configclass
class MPMFootContactActionsCfg(DeeproboticsM20ActionsCfg):
    mpm_contact = MPMFootContactActionCfg(contact_force_threshold=5.0)


@configclass
class DeeproboticsM20WalkOnSandEnvCfg(DeeproboticsM20WalkEnvCfg):
    """Walk's 57/60 observations and 16 actions, with MPM impulse-based foot contacts."""

    scene: DeeproboticsM20SandSceneCfg = DeeproboticsM20SandSceneCfg(
        num_envs=SAND_NUM_ENVS, env_spacing=SAND_ENV_SPACING
    )
    actions: MPMFootContactActionsCfg = MPMFootContactActionsCfg()
    rewards: SandWalkRewardsCfg = SandWalkRewardsCfg()
    terminations: SandWalkTerminationsCfg = SandWalkTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        physics = M20SandPhysicsCfg()
        # The passive solver hook reads every actual MPM substep, including sparse-grid rebuilds.
        # Keep this task eager, like the validated shadow recorder, rather than skipping Python
        # sampling during graph replay. The original Sand and Proxy presets are unchanged.
        physics.newton_sand.use_cuda_graph = False
        physics.default = physics.newton_sand
        self.sim.physics = cast(PhysicsCfg, physics)
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation
        configure_sand_walk_task(self, contact_term_name="mpm_contact")
        self.disable_zero_weight_rewards()
