"""Check the current Stand leg regularization and sibling symmetry settings."""

from types import SimpleNamespace

import pytest
import torch

from isaaclab.envs.mdp import joint_deviation_l1

from M20_newton_training.tasks.m20 import mdp
from M20_newton_training.tasks.m20.config.M20.flat_env_cfg import DeeproboticsM20FlatEnvCfg
from M20_newton_training.tasks.m20.config.M20.rough_env_cfg import DeeproboticsM20RoughEnvCfg
from M20_newton_training.tasks.m20.config.M20.stand_env_cfg import DeeproboticsM20StandEnvCfg


def test_stand_uses_leg_deviation_instead_of_the_removed_front_mirror_term():
    cfg = DeeproboticsM20StandEnvCfg()
    assert cfg.rewards.joint_mirror is None
    term = cfg.rewards.joint_deviation_l1
    assert term.func is joint_deviation_l1
    assert term.weight == -0.01
    assert term.params["asset_cfg"].joint_names == cfg.leg_joint_names


@pytest.mark.parametrize("config_type", [DeeproboticsM20RoughEnvCfg, DeeproboticsM20FlatEnvCfg])
def test_locomotion_tasks_keep_diagonal_symmetry(config_type):
    term = config_type().rewards.joint_mirror
    assert term.func is mdp.joint_mirror
    assert term.weight == -0.03
    assert term.params["mirror_joints"] == [
        ["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
        ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"],
    ]


def test_leg_deviation_ignores_wheel_angles_and_does_not_modify_joint_state():
    cfg = DeeproboticsM20StandEnvCfg()
    term = cfg.rewards.joint_deviation_l1
    asset_cfg = term.params["asset_cfg"]
    asset_cfg.joint_ids = list(range(12))
    positions = torch.zeros(2, 16)
    positions[0, 0] = 0.3
    positions[1, 8] = -0.4
    positions[:, 12:] = 50.0
    before = positions.clone()
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    joint_pos=SimpleNamespace(torch=positions),
                    default_joint_pos=SimpleNamespace(torch=torch.zeros_like(positions)),
                )
            )
        }
    )
    torch.testing.assert_close(term.func(env, asset_cfg), torch.tensor([0.3, 0.4]))
    torch.testing.assert_close(positions, before)
