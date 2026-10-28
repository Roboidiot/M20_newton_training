"""Exclusive direction commands, nonzero bounds, resampling and inherited reset metrics."""

from types import SimpleNamespace

import pytest
import torch

from M20_newton_training.tasks.m20.config.M20.walk_on_sand_proxy_env_cfg import DeeproboticsM20WalkOnSandProxyEnvCfg
from M20_newton_training.tasks.m20.mdp.single_axis_velocity import SingleAxisVelocityCommand


def _command(n=1024, cfg=None):
    cfg = cfg or DeeproboticsM20WalkOnSandProxyEnvCfg().commands.base_velocity
    cfg.debug_vis = False

    def wrap(value):
        return SimpleNamespace(torch=value)

    data = SimpleNamespace(
        root_lin_vel_b=wrap(torch.zeros((n, 3))),
        root_ang_vel_b=wrap(torch.zeros((n, 3))),
        root_pos_w=wrap(torch.tensor([[0.0, 0.0, 0.8]]).repeat(n, 1)),
        joint_pos=wrap(torch.zeros((n, 1))),
        default_joint_pos=wrap(torch.zeros((n, 1))),
    )
    env = SimpleNamespace(
        num_envs=n,
        device="cpu",
        extras={"log": {}},
        scene={"robot": SimpleNamespace(data=data, find_joints=lambda pattern: ([0], ["fl_knee_joint"]))},
        sim=SimpleNamespace(vis_marker_registry=SimpleNamespace(clear_debug_vis_callback=lambda term: None)),
    )
    return SingleAxisVelocityCommand(cfg, env)


def _check(values):
    assert (torch.count_nonzero(values, dim=1) == 1).all()
    assert (values.norm(dim=1) > 0.15).all()
    for axis, lo, hi in ((0, 0.16, 0.4), (1, 0.16, 0.3), (2, 0.16, 0.4)):
        active = values[:, axis][values[:, axis] != 0]
        assert (active.abs() >= lo - 1e-7).all() and (active.abs() <= hi + 1e-7).all()


def test_exclusive_sampler_preserves_bounds_after_update_and_uses_all_directions():
    command = _command()
    values = []
    for _ in range(8):
        command._resample_command(torch.arange(command.num_envs))
        command._update_command()
        _check(command.command)
        assert not command.is_standing_env.any() and not command.is_heading_env.any()
        values.append(command.command.clone())
    values = torch.cat(values)
    counts = torch.count_nonzero(values, dim=0)
    assert ((counts.float() / len(values) - 1 / 3).abs() < 0.05).all()
    for axis in (0, 1, 2):
        assert (values[:, axis] > 0).any() and (values[:, axis] < 0).any()
    # A lateral magnitude below 0.2 is valid and must survive the old sampler's dead zone.
    assert ((values[:, 1].abs() >= 0.16) & (values[:, 1].abs() < 0.2)).any()


@pytest.mark.parametrize("env_ids", [slice(None), slice(1, 8, 2), [1, 3], torch.tensor([1, 3]), []])
def test_partial_resampling_does_not_change_other_environments(env_ids):
    command = _command(8)
    command.vel_command_b[:] = torch.tensor([0.7, 0.6, 0.5])
    before = command.command.clone()
    selected = torch.zeros(8, dtype=torch.bool)
    selected[env_ids] = True
    command._resample_command(env_ids)
    _check(command.command[selected])
    torch.testing.assert_close(command.command[~selected], before[~selected])


def test_reset_preserves_inherited_episode_metrics_and_resamples_nonzero_commands():
    command = _command(8)
    command._update_metrics()
    command._update_metrics()
    extras = command.reset(torch.tensor([1, 3]))
    assert extras["base_z"] == pytest.approx(0.8)
    assert not command._metric_step_counter[[1, 3]].any()
    assert (command._metric_step_counter[[0, 2]] == 2).all()
    command._update_command()
    _check(command.command[[1, 3]])


def test_sampler_rejects_ranges_without_allowed_nonzero_values():
    cfg = DeeproboticsM20WalkOnSandProxyEnvCfg().commands.base_velocity
    cfg.ranges.lin_vel_y = (-0.15, 0.15)
    with pytest.raises(ValueError, match="contains no command"):
        _command(cfg=cfg)


