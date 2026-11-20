"""Sand geometry, cached support state and native policy/critic observation assembly."""

from math import sqrt
from types import SimpleNamespace

import pytest
import torch

from isaaclab.managers import ObservationManager
from isaaclab.utils.string import resolve_matching_names

from M20_newton_training.tasks.m20.config.M20.walk_on_sand_env_cfg import DeeproboticsM20WalkOnSandEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_proxy_env_cfg import DeeproboticsM20WalkOnSandProxyEnvCfg
from M20_newton_training.tasks.m20.mdp.proxy_contact_rewards import ProxyFootContactState
from M20_newton_training.tasks.m20.mdp.sand_observations import (
    sand_base_clearance,
    sand_feet_contact,
    sand_feet_depth,
    sand_layer_depth,
)
from M20_newton_training.tasks.m20.sand_walk_settings import SAND_WALK_SURFACE_HEIGHT


class Scene(dict):
    pass


def make_env(mpm=False):
    cfg = (DeeproboticsM20WalkOnSandEnvCfg if mpm else DeeproboticsM20WalkOnSandProxyEnvCfg)()
    origins = torch.tensor([[0., 0., 0.], [3., -4., 1.]])
    ids = [2, 0, 3, 1]  # Asset storage differs from FL/FR/HL/HR observation order.
    expected_depth = torch.tensor([[-.10, 0., .05, .15], [.20, .10, -.05, -.20]])
    quat = torch.zeros(2, 4, 4)
    quat[..., 3] = 1.
    quat[:, ids[1::2]] = torch.tensor([sqrt(.5), 0., 0., sqrt(.5)])
    pos = origins[:, None, :].repeat(1, 4, 1)
    center_offset = 0. if mpm else .01
    # Identity wheels have a horizontal capsule axis; rolled wheels have a vertical one.
    offset = torch.tensor([.09, .117, .09, .117])
    pos[:, ids, 2] = origins[:, 2:3] + SAND_WALK_SURFACE_HEIGHT + offset - expected_depth - center_offset
    # The local +Z center offset rotates to -Y for the rolled wheels.
    pos[:, ids[1::2], 2] += center_offset

    wrap = lambda value: SimpleNamespace(torch=value)  # noqa: E731
    data = SimpleNamespace(
        body_link_pos_w=wrap(pos), body_link_quat_w=wrap(quat),
        root_pos_w=wrap(origins + torch.tensor(cfg.scene.robot.init_state.pos)),
        root_lin_vel_b=wrap(torch.tensor([[.3, -.2, .1], [-.1, .2, -.3]])),
        root_ang_vel_b=wrap(torch.zeros(2, 3)),
        projected_gravity_b=wrap(torch.tensor([[0., 0., -1.]]).repeat(2, 1)),
        joint_pos=wrap(torch.zeros(2, 16)), default_joint_pos=wrap(torch.zeros(2, 16)),
        joint_vel=wrap(torch.zeros(2, 16)), default_joint_vel=wrap(torch.zeros(2, 16)),
    )
    joint_names = cfg.observations.policy.joint_pos.params["asset_cfg"].joint_names
    robot = SimpleNamespace(data=data, joint_names=joint_names, num_joints=16)
    robot.find_joints = lambda names, preserve_order=False: resolve_matching_names(
        names, joint_names, preserve_order=preserve_order
    )
    scene = Scene(robot=robot)
    scene.env_origins = origins
    state = ProxyFootContactState(2, "cpu")
    force = torch.zeros(2, 4, 3)
    force[..., 0] = 100.  # Horizontal drag does not establish support.
    force[..., 2] = torch.tensor([[0., 5., 5.01, -20.], [20., 0., 10., 6.]])
    state.update(force, .02, 5.)
    term = SimpleNamespace(
        cfg=SimpleNamespace(asset_name="robot", capsule_radius=.09, capsule_half_length=.027),
        body_ids=ids, contact_state=state,
        model=SimpleNamespace(depth=torch.full((2, 4), 999.)),
    )
    axis = torch.tensor([0., -1., 0.]).expand(2, 4, 3)
    if mpm:
        term._local_axis = axis
    else:
        term.local_axis = axis
        term.local_center = torch.tensor([0., 0., center_offset]).expand(2, 4, 3)
        term.radius, term.half_length = torch.full((4,), .09), torch.full((4,), .027)
    name = "mpm_contact" if mpm else "sand_proxy"
    env = SimpleNamespace(
        cfg=cfg, num_envs=2, device="cpu", scene=scene,
        sim=SimpleNamespace(is_playing=lambda: True),
        action_manager=SimpleNamespace(get_term=lambda requested: {name: term}[requested], action=torch.zeros(2, 16)),
        command_manager=SimpleNamespace(get_command=lambda _: torch.tensor([[.2, 0., 0.]]).repeat(2, 1)),
    )
    return env, term, name, expected_depth


@pytest.mark.parametrize("mpm", [False, True])
def test_current_capsule_geometry_and_support_are_origin_invariant_and_read_only(mpm):
    env, term, name, expected_depth = make_env(mpm)
    air_time = term.contact_state.current_air_time.clone()
    surface = SAND_WALK_SURFACE_HEIGHT
    for _ in range(2):
        torch.testing.assert_close(sand_feet_depth(env, surface, name), expected_depth, atol=1e-6, rtol=0)
        torch.testing.assert_close(sand_layer_depth(env, surface), torch.full((2, 1), .15))
        torch.testing.assert_close(sand_base_clearance(env, surface), torch.full((2, 1), .53))
        torch.testing.assert_close(
            sand_feet_contact(env, name), torch.tensor([[0., 0., 1., 0.], [1., 0., 1., 1.]])
        )
    torch.testing.assert_close(term.contact_state.current_air_time, air_time)
    assert (term.model.depth == 999.).all()  # No stale model input or force update is used.
    env.scene["robot"].data.body_link_pos_w.torch[0, term.body_ids[0], 2] += .03
    expected_depth[0, 0] -= .03
    torch.testing.assert_close(sand_feet_depth(env, surface, name), expected_depth, atol=1e-6, rtol=0)
    term.contact_state.reset(torch.tensor([0]))
    assert not sand_feet_contact(env, name)[0].any()
    assert sand_feet_contact(env, name)[1].sum() == 3


@pytest.mark.parametrize("mpm", [False, True])
def test_native_manager_concatenates_60_policy_and_70_critic_inputs(mpm):
    env, term, name, depths = make_env(mpm)
    manager = ObservationManager(env.cfg.observations, env)
    assert manager.group_obs_dim == {"policy": (60,), "critic": (70,)}
    assert manager.active_terms["critic"][-4:] == [
        "sand_layer_depth", "feet_sand_depth", "feet_sand_contact", "base_sand_clearance"
    ]
    samples = manager.compute()
    torch.testing.assert_close(samples["critic"][:, :3], env.scene["robot"].data.root_lin_vel_b.torch)
    # Policy uses the same body-frame velocity with its configured observation noise.
    assert (samples["policy"][:, :3] - samples["critic"][:, :3]).abs().max() <= .100001
    torch.testing.assert_close(samples["critic"][:, 60:61], torch.full((2, 1), .15))
    torch.testing.assert_close(samples["critic"][:, 61:65], depths, atol=1e-6, rtol=0)
    torch.testing.assert_close(samples["critic"][:, 65:69], sand_feet_contact(env, name))
    torch.testing.assert_close(samples["critic"][:, 69:70], torch.full((2, 1), .53))
    before = term.contact_state.current_air_time.clone()
    manager.compute()
    torch.testing.assert_close(term.contact_state.current_air_time, before)
