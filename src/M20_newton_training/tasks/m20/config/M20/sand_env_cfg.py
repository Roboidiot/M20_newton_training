# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Sand-locomotion environment: the M20 walks on a granular bed simulated by Newton's implicit MPM solver.

The rigid robot is advanced by MJWarp while the sand is advanced by implicit MPM. Both run inside one
:class:`CouplerProxyCfg`, which is required because MPM is not an articulation solver: it owns particles and
treats rigid geometry as colliders. The robot reaches the sand through a directed proxy mapping.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import numpy as np
from isaaclab_contrib.coupling import CouplerEntryCfg, CouplerProxyCfg, CouplerProxyMappingCfg
from torch import Tensor

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.sim.schemas import UsdPhysicsRigidBodyCfg
from isaaclab.sim.spawners.materials import RigidBodyMaterialBaseCfg
from isaaclab.utils import configclass, replace
from isaaclab_newton.assets import MPMObjectCfg
from isaaclab_newton.physics import (
    MJWarpSolverCfg,
    MPMSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_newton.sim.schemas import NewtonCollisionPropertiesCfg
from isaaclab_newton.sim.spawners.mpm import MPMGridCfg, MPMParticleMaterialCfg

from M20_newton_training.tasks.m20.mdp.keyboard_velocity import KeyboardVelocityCommandCfg
from M20_newton_training.tasks.m20.mdp.sand_rewards import (
    SandActionSmoothness,
    SandDiagonalStepReward,
    SandStepCompletionReward,
    SandSupportMotionPenalty,
    sand_stand_still,
    sand_wheel_velocity_l2,
)
from M20_newton_training.tasks.m20.velocity_env_cfg import M20PhysicsCfg, MySceneCfg

from .rough_env_cfg import DeeproboticsM20RewardsCfg, DeeproboticsM20RoughEnvCfg

##
# Sand bed geometry
##

SAND_HALF_EXTENT_XY = 1.2
"""Half side length of the square sand patch [m], measured to the inner face of the tray walls."""

SAND_DEPTH = 0.30
"""Undisturbed sand layer thickness [m]."""

SAND_VOXEL_SIZE = 0.02
"""MPM grid voxel size [m]. Keep it comparable to the contact patch of a wheel, not to the whole robot."""

SAND_PARTICLES_PER_CELL = 1.0
"""Particles per voxel axis. Raising this multiplies particle count cubically."""

SAND_JITTER_FRACTION = 0.3
"""Full width of the per-axis position jitter interval, relative to the particle spacing."""

SAND_COLLIDER_MARGIN = 0.01
"""Contact margin of the tray colliders [m]. Implicit MPM consumes the shape margin."""

SAND_TRAY_THICKNESS = 0.06
"""Thickness of the tray floor the sand rests on [m]."""

SAND_WALL_THICKNESS = 0.05
"""Thickness of the tray side walls [m]."""

SAND_WALL_HEIGHT = SAND_DEPTH + 0.2
"""Wall height [m]. Extends above the sand surface so a kicked-up ridge cannot escape the patch."""

SAND_NUM_ENVS = 64
"""Default parallel environments. MPM cost scales with ``num_envs * particles_per_env``."""

SAND_ENV_SPACING = 2.0 * (SAND_HALF_EXTENT_XY + SAND_WALL_THICKNESS) + 1.0
"""Environment-center spacing [m], leaving a 1 m gap between the tray exteriors."""

SAND_BASE_CLEARANCE = 0.58
"""Initial base-link height above the undeformed sand generation box [m]."""

RIGID_ENTRY = "robot"
MPM_ENTRY = "sand"

SAND_PRIM_PATH = "{ENV_REGEX_NS}/Sand"
SAND_TRAY_REGEX = r"/World/envs/env_.*/SandTray.*"
ROBOT_REGEX = r"/World/envs/env_.*/Robot"

# Only the values that depart from the dry-sand defaults are written out. In particular the inherited
# yield_pressure keeps the bed quasi-static instead of collapsing like a fluid, and friction sets the
# internal angle of repose that decides how steep a pushed-up ridge can hold.
SAND_MATERIAL = MPMParticleMaterialCfg(
    density=1700.0,
    friction=0.7,
    damping=0.005,
)


def _grid_bound(value: float, lower: float) -> float:
    """Nudge a float32 grid bound inward so an intended integral lattice resolution cannot round above it.

    Newton derives the lattice resolution from float32 bounds, so an intended two-layer stack such as
    ``0.1 / 0.05 == 2`` can resolve to three layers and emit a stray particle sheet against a tray wall.
    """
    return float(np.nextafter(np.float32(lower + value), np.float32(lower)))


def _sand_grid() -> MPMGridCfg:
    """Build the cell-centred particle lattice that fills the sand patch."""
    spacing = SAND_VOXEL_SIZE / SAND_PARTICLES_PER_CELL
    inset = SAND_HALF_EXTENT_XY - SAND_COLLIDER_MARGIN
    lower = (-inset, -inset, 0.0)
    layers = max(1, math.floor(SAND_DEPTH / spacing + 1.0e-9))
    upper = (
        _grid_bound(2.0 * inset, -inset),
        _grid_bound(2.0 * inset, -inset),
        _grid_bound(layers * spacing, 0.0),
    )
    return MPMGridCfg(
        lower=lower,
        upper=upper,
        voxel_size=SAND_VOXEL_SIZE,
        particles_per_cell=SAND_PARTICLES_PER_CELL,
        particle_placement="cell_center",
        jitter=SAND_JITTER_FRACTION * spacing,
        material=SAND_MATERIAL,
        visual_color=(0.82, 0.7, 0.45),
    )


def _tray_collider(
    prim_path: str,
    *,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
) -> RigidObjectCfg:
    """Build one kinematic tray collider owned exclusively by the MPM entry.

    The static terrain stays owned by the rigid entry, so the sand gets its own co-located mirrors instead of
    sharing a shape between the two sub-solvers.
    """
    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=UsdPhysicsRigidBodyCfg(rigid_body_enabled=True, kinematic_enabled=True),
            collision_props=NewtonCollisionPropertiesCfg(
                collision_enabled=True,
                contact_margin=SAND_COLLIDER_MARGIN,
                # Gap is a rigid-contact parameter; implicit MPM only reads the margin.
                contact_gap=0.0,
            ),
            physics_material=RigidBodyMaterialBaseCfg(static_friction=0.8, dynamic_friction=0.7),
            visible=False,
        ),
    )


@configclass
class DeeproboticsM20SandSceneCfg(MySceneCfg):
    """Flat rigid terrain carrying one sand patch and the tray that confines it, per environment."""

    sand = MPMObjectCfg(prim_path=SAND_PRIM_PATH, spawn=_sand_grid())

    sand_tray_floor = _tray_collider(
        "{ENV_REGEX_NS}/SandTrayFloor",
        size=(
            2.0 * SAND_HALF_EXTENT_XY + 2.0 * SAND_WALL_THICKNESS,
            2.0 * SAND_HALF_EXTENT_XY + 2.0 * SAND_WALL_THICKNESS,
            SAND_TRAY_THICKNESS,
        ),
        position=(0.0, 0.0, -0.5 * SAND_TRAY_THICKNESS),
    )
    sand_tray_north = _tray_collider(
        "{ENV_REGEX_NS}/SandTrayNorth",
        size=(SAND_WALL_THICKNESS, 2.0 * SAND_HALF_EXTENT_XY + 2.0 * SAND_WALL_THICKNESS, SAND_WALL_HEIGHT),
        position=(SAND_HALF_EXTENT_XY + 0.5 * SAND_WALL_THICKNESS, 0.0, 0.5 * SAND_WALL_HEIGHT - 0.1),
    )
    sand_tray_south = _tray_collider(
        "{ENV_REGEX_NS}/SandTraySouth",
        size=(SAND_WALL_THICKNESS, 2.0 * SAND_HALF_EXTENT_XY + 2.0 * SAND_WALL_THICKNESS, SAND_WALL_HEIGHT),
        position=(-SAND_HALF_EXTENT_XY - 0.5 * SAND_WALL_THICKNESS, 0.0, 0.5 * SAND_WALL_HEIGHT - 0.1),
    )
    sand_tray_east = _tray_collider(
        "{ENV_REGEX_NS}/SandTrayEast",
        size=(2.0 * SAND_HALF_EXTENT_XY, SAND_WALL_THICKNESS, SAND_WALL_HEIGHT),
        position=(0.0, SAND_HALF_EXTENT_XY + 0.5 * SAND_WALL_THICKNESS, 0.5 * SAND_WALL_HEIGHT - 0.1),
    )
    sand_tray_west = _tray_collider(
        "{ENV_REGEX_NS}/SandTrayWest",
        size=(2.0 * SAND_HALF_EXTENT_XY, SAND_WALL_THICKNESS, SAND_WALL_HEIGHT),
        position=(0.0, -SAND_HALF_EXTENT_XY - 0.5 * SAND_WALL_THICKNESS, 0.5 * SAND_WALL_HEIGHT - 0.1),
    )


def _sand_coupled_physics() -> NewtonCfg:
    """Build the MJWarp-plus-MPM coupled physics config used by the sand task."""
    return NewtonCfg(
        solver_cfg=CouplerProxyCfg(
            entries=[
                CouplerEntryCfg(
                    name=RIGID_ENTRY,
                    solver_cfg=MJWarpSolverCfg(
                        njmax=1000,
                        # Sand-wheel contact is far denser than contact on a rigid mesh.
                        nconmax=800,
                        cone="pyramidal",
                        impratio=1.0,
                        integrator="implicitfast",
                        use_mujoco_contacts=False,
                    ),
                    bodies=[ROBOT_REGEX],
                    # The static terrain is a body-less shape, so only this entry can own it.
                    include_static_shapes=True,
                    # Restores the 400 Hz rigid integration the non-coupled M20 preset runs at.
                    substeps=2,
                ),
                CouplerEntryCfg(
                    name=MPM_ENTRY,
                    solver_cfg=MPMSolverCfg(
                        voxel_size=SAND_VOXEL_SIZE,
                        grid_type="sparse",
                        grid_padding=1,
                        strain_basis="P0",
                        transfer_scheme="apic",
                        max_iterations=24,
                        tolerance=1.0e-4,
                        warmstart_mode="auto",
                        velocity_basis="Q1",
                        collider_basis="pic27",
                        collider_velocity_mode="forward",
                        solver="auto",
                        # One grid per environment keeps a displaced patch in one env from bleeding into a
                        # neighbour's grid, which the shared-world dense grid would otherwise allow.
                        separate_worlds=True,
                    ),
                    all_particles=True,
                    bodies=[SAND_TRAY_REGEX],
                    include_static_shapes=False,
                    include_child_joints=False,
                    substeps=1,
                    in_place=True,
                ),
            ],
            proxies=[
                CouplerProxyMappingCfg(
                    source=RIGID_ENTRY,
                    destination=MPM_ENTRY,
                    bodies=[ROBOT_REGEX],
                    # The robot is tens of times heavier than the sand it displaces, so no mass rescaling is
                    # needed to stop particle reaction from kicking the base around.
                    mass_scale=1.0,
                    mode="staggered",
                    collision_pipeline=None,
                )
            ],
            iterations=1,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(soft_contact_max=0),
        num_substeps=1,
        default_shape_cfg=NewtonShapeCfg(
            margin=SAND_COLLIDER_MARGIN,
            ke=160000.0,
            kd=1100.0,
        ),
        use_cuda_graph=True,
    )


@configclass
class M20SandPhysicsCfg(M20PhysicsCfg):
    """Physics presets for the sand task, defaulting to the coupled MJWarp-plus-MPM solve.

    The inherited rigid-body presets remain selectable, but none of them advances particles, so selecting one
    leaves the sand bed frozen and the robot drops through it onto the rigid plane.
    """

    newton_sand = _sand_coupled_physics()
    default = newton_sand


@configclass
class DeeproboticsM20SandRewardsCfg(DeeproboticsM20RewardsCfg):
    """Kinematic stepping rewards; no contact sensor, policy phase, or gait_level multiplier."""

    sand_diagonal_steps = RewTerm(
        func=cast(Callable[..., Tensor], SandDiagonalStepReward),
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
            "command_name": "base_velocity",
            "target_lift": 0.07,
            "height_std": 0.035,
            "sync_std": 0.025,
        },
    )
    sand_step_completion = RewTerm(
        func=cast(Callable[..., Tensor], SandStepCompletionReward),
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
        },
    )
    sand_support_motion = RewTerm(
        func=cast(Callable[..., Tensor], SandSupportMotionPenalty),
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel"], preserve_order=True
            ),
            "support_threshold": 0.01,
            "velocity_reference": 0.5,
        },
    )
    sand_wheel_velocity = RewTerm(
        func=sand_wheel_velocity_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_wheel_joint"), "velocity_reference": 5.0},
    )


@configclass
class DeeproboticsM20SandEnvCfg(DeeproboticsM20RoughEnvCfg):
    """Velocity-tracking locomotion on a deformable sand bed for the M20 wheeled quadruped."""

    scene: DeeproboticsM20SandSceneCfg = DeeproboticsM20SandSceneCfg(
        num_envs=SAND_NUM_ENVS, env_spacing=SAND_ENV_SPACING
    )
    rewards: DeeproboticsM20SandRewardsCfg = DeeproboticsM20SandRewardsCfg()

    def __post_init__(self):
        # The parent populates self.scene (robot, scanner prim paths, rewards) on the instance above.
        super().__post_init__()

        # Use the generated layer height, which can be rounded down from SAND_DEPTH to whole particle layers.
        sand_grid = self.scene.sand.spawn
        if not isinstance(sand_grid, MPMGridCfg):
            raise TypeError("Sand initialization requires an MPMGridCfg particle bed.")
        robot_cfg = self.scene.robot
        self.scene.robot = replace(
            robot_cfg,
            init_state=replace(
                robot_cfg.init_state,
                pos=(
                    robot_cfg.init_state.pos[0],
                    robot_cfg.init_state.pos[1],
                    float(sand_grid.upper[2]) + SAND_BASE_CLEARANCE - 0.1,
                ),
            ),
        )

        reset_pose_range = cast(
            dict[str, tuple[float, float]],
            self.events.randomize_reset_base.params["pose_range"],
        )
        reset_pose_range["x"] = (0.0, 0.0)
        reset_pose_range["y"] = (0.0, 0.0)
        reset_pose_range["roll"] = (-0.1, 0.1)
        reset_pose_range["pitch"] = (-0.1, 0.1)

        # The sand patch replaces procedural roughness: a flat plane under a granular layer.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        self.curriculum.gait_level = None

        self.sim.physics = cast(PhysicsCfg, M20SandPhysicsCfg())

        # Keep effort/smoothness regularization active independently of Rough's global curriculum scalar.
        self.rewards.joint_torques_l2.func = base_mdp.joint_torques_l2
        self.rewards.action_rate_l2.func = base_mdp.action_rate_l2
        self.rewards.action_smooth_l2.func = cast(Callable[..., Tensor], SandActionSmoothness)
        # Foot-space diagonal coordination replaces the inherited raw joint-angle mirror penalty.
        self.rewards.joint_mirror.weight = 0.0

        # Allow stepping away from the default pose; retain a mild hold penalty at zero command.
        self.rewards.hipx_joint_pos_penalty.weight = -0.3
        self.rewards.hipy_joint_pos_penalty.weight = -0.15
        self.rewards.knee_joint_pos_penalty.weight = -0.075
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.bad_orientation_penalty.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.stand_still.func = sand_stand_still
        self.rewards.stand_still.weight = -0.1
        self.rewards.stand_still.params = {
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=self.leg_joint_names, preserve_order=True),
        }

        # The bed surface is itself the quantity that moves, so no terrain-relative height target.
        self.rewards.base_height_l2.params["sensor_cfg"] = None

        # The coupled MJWarp-plus-MPM solver cannot report contact forces to sensors, so the contact sensor
        # and every reward that reads it are disabled. This drops the gait-shaping terms (rotation symmetry,
        # foot air time) and the slide/contact penalties; re-introducing them needs a contact-free sensing
        # scheme, e.g. inferring foot contact from link kinematics instead of a ContactSensor.
        self.scene.contact_forces = None
        for term in (
            "undesired_contacts",
            "contact_forces",
            "feet_air_time_ang_z_M20",
            "feet_slide_ang_z_cmd",
            "rotation_gait_status",
            "rotation_gait_symmetry",
        ):
            getattr(self.rewards, term).weight = 0.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "DeeproboticsM20SandEnvCfg":
            self.disable_zero_weight_rewards()

    def play_mode(self):
        """Use body-frame keyboard velocities for playback; training retains random commands."""
        super().play_mode()
        self.scene.num_envs = 1
        command = self.commands.base_velocity
        self.commands.base_velocity = KeyboardVelocityCommandCfg(
            asset_name=command.asset_name,
            ranges=replace(command.ranges),
            debug_vis=command.debug_vis,
            keyboard_speeds=(0.5, 0.3, 0.5),
        )
