# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Walk's policy interfaces on Sand's coupled MPM bed, without contact-sensor rewards."""

from collections.abc import Callable
from typing import cast

from torch import Tensor

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.utils import configclass, replace
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg

from M20_newton_training.tasks.m20.mdp.walk_on_sand_rewards import (
    WalkOnSandDiagonalStepReward,
    WalkOnSandStepCompletionReward,
)

from .rough_env_cfg import DeeproboticsM20RewardsCfg
from .sand_env_cfg import (
    SAND_BASE_CLEARANCE,
    SAND_ENV_SPACING,
    SAND_NUM_ENVS,
    DeeproboticsM20SandSceneCfg,
    M20SandPhysicsCfg,
)
from .walk_env_cfg import DeeproboticsM20WalkEnvCfg



@configclass
class DeeproboticsM20WalkOnSandRewardsCfg(DeeproboticsM20RewardsCfg):
    sand_diagonal_steps = RewTerm(
        func=cast(Callable[..., Tensor], WalkOnSandDiagonalStepReward),
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
            "command_name": "base_velocity",
            "target_lift": 0.07,
            "height_std": 0.035,
            "sync_std": 0.025,
            "command_threshold": 0.1,
            "side_multiplier": 2.0,
            "turn_multiplier": 3.0,
        },
    )
    sand_step_completion = RewTerm(
        func=cast(Callable[..., Tensor], WalkOnSandStepCompletionReward),
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
            "command_name": "base_velocity",
            "lift_threshold": 0.035,
            "lower_threshold": 0.01,
            "target_lift": 0.07,
            "min_stride": 0.06,
            "min_swing_time": 0.08,
            "max_swing_time": 0.8,
            "command_threshold": 0.1,
            "side_multiplier": 2.0,
            "turn_multiplier": 3.0,
        },
    )


@configclass
class DeeproboticsM20WalkOnSandEnvCfg(DeeproboticsM20WalkEnvCfg):
    """Keep Walk's 57/60 observations, 16 actions, and non-contact reward definitions."""

    scene: DeeproboticsM20SandSceneCfg = DeeproboticsM20SandSceneCfg(
        num_envs=SAND_NUM_ENVS, env_spacing=SAND_ENV_SPACING
    )
    rewards: DeeproboticsM20WalkOnSandRewardsCfg = DeeproboticsM20WalkOnSandRewardsCfg()

    def __post_init__(self):
        # Walk -> Flat removes the height scan from both observation groups.
        super().__post_init__()
        self.sim.physics = cast(PhysicsCfg, M20SandPhysicsCfg())

        grid = self.scene.sand.spawn
        if not isinstance(grid, MPMGridCfg):
            raise TypeError("WalkOnSand initialization requires an MPMGridCfg particle bed.")
        robot = self.scene.robot
        # Match the current Sand task's base clearance, including its 10 cm lowering offset.
        self.scene.robot = replace(
            robot,
            init_state=replace(
                robot.init_state,
                pos=(
                    robot.init_state.pos[0],
                    robot.init_state.pos[1],
                    float(grid.upper[2]) + SAND_BASE_CLEARANCE - 0.1,
                ),
            ),
        )
        pose_range = cast(dict[str, tuple[float, float]], self.events.randomize_reset_base.params["pose_range"])
        pose_range.update(x=(0.0, 0.0), y=(0.0, 0.0), roll=(-0.1, 0.1), pitch=(-0.1, 0.1))

        # MJWarp+MPM cannot expose contact forces/timing to ContactSensor. Keep the old functions unchanged
        # and disable every active reader until a calibrated contact proxy is introduced.
        self.scene.contact_forces = None
        for name in (
            "undesired_contacts",
            "contact_forces",
            "feet_air_time",
            "feet_slide_ang_z_cmd",
            "rotation_gait_status",
            "rotation_gait_symmetry",
        ):
            getattr(self.rewards, name).weight = 0.0

        if self.__class__.__name__ == "DeeproboticsM20WalkOnSandEnvCfg":
            self.disable_zero_weight_rewards()
