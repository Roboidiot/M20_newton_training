"""Walk-compatible training with frozen neural wheel wrenches on a fixed sand plane."""

from typing import cast

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.physics import PhysicsCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg

from isaaclab_tasks.utils import PresetCfg

from M20_newton_training.tasks.m20.mdp.foot_sand_proxy import FootSandProxyActionCfg
from M20_newton_training.tasks.m20.velocity_env_cfg import MySceneCfg

from .rough_env_cfg import DeeproboticsM20ActionsCfg
from .walk_env_cfg import DeeproboticsM20WalkEnvCfg
from .walk_sand_common_cfg import (
    SAND_WALK_SURFACE_HEIGHT,
    SandWalkRewardsCfg,
    SandWalkTerminationsCfg,
    configure_sand_walk_task,
)


@configclass
class M20FootProxyPhysicsCfg(PresetCfg):
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(njmax=100, nconmax=64, use_mujoco_contacts=True, integrator="implicitfast"),
        num_substeps=2,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )
    default = newton_mjwarp


@configclass
class FootProxySceneCfg(MySceneCfg):
    foot_ground_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/[fh][lr]_wheel",
        filter_shape_prim_expr=["/World/ground(?:/.*)?"],
        update_period=0.005,
        history_length=0,
        track_air_time=False,
    )
    sand_surface_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SandSurfaceVisual",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, SAND_WALK_SURFACE_HEIGHT - 0.002)),
        spawn=sim_utils.CuboidCfg(
            size=(2.4, 2.4, 0.003),
            collision_props=None,
            rigid_props=None,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.70, 0.53, 0.31), opacity=0.85),
        ),
    )


@configclass
class FootProxyActionsCfg(DeeproboticsM20ActionsCfg):
    sand_proxy = FootSandProxyActionCfg(contact_force_threshold=5.0, rigid_contact_sensor="foot_ground_contacts")


@configclass
class FootProxyRewardsCfg(SandWalkRewardsCfg):
    """Compatibility name for the shared sand-walking reward definitions."""


@configclass
class DeeproboticsM20WalkOnSandProxyEnvCfg(DeeproboticsM20WalkEnvCfg):
    scene: FootProxySceneCfg = FootProxySceneCfg(num_envs=2048, env_spacing=3.5)
    actions: FootProxyActionsCfg = FootProxyActionsCfg()
    rewards: FootProxyRewardsCfg = FootProxyRewardsCfg()
    terminations: SandWalkTerminationsCfg = SandWalkTerminationsCfg()



    def __post_init__(self):
        super().__post_init__()
        self.sim.physics = cast(PhysicsCfg, M20FootProxyPhysicsCfg())
        # Neural history was trained at 200 Hz; actor, critic and rewards stay at 50 Hz.
        self.sim.dt = 0.005
        self.decimation = 4
        self.sim.render_interval = self.decimation
        configure_sand_walk_task(self, contact_term_name="sand_proxy")
        self.disable_zero_weight_rewards()
