"""Front-leg mirror signs and native joint ordering for upright Stand."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.warp import ProxyArray

from M20_newton_training.tasks.m20 import mdp
from M20_newton_training.tasks.m20.config.M20.flat_env_cfg import DeeproboticsM20FlatEnvCfg
from M20_newton_training.tasks.m20.config.M20.rough_env_cfg import DeeproboticsM20RoughEnvCfg
from M20_newton_training.tasks.m20.config.M20.stand_env_cfg import (
    DeeproboticsM20StandEnvCfg,
    stand_front_joint_mirror_l2,
)

NATIVE_NAMES = [
    f"{limb}_{joint}_joint" for limb in ("fl", "fr", "hl", "hr") for joint in ("hipx", "hipy", "knee", "wheel")
]


def _term_and_env(names):
    term = DeeproboticsM20StandEnvCfg().rewards.joint_mirror
    positions = torch.zeros(2, 16)
    for side, values in (
        ("fl", [[0.2, -1.3, 0.8], [0.2, -1.1, 0.6]]),
        ("fr", [[-0.2, -1.3, 0.8], [0.1, -1.5, 0.9]]),
    ):
        ids = [names.index(f"{side}_{joint}_joint") for joint in ("hipx", "hipy", "knee")]
        positions[:, ids] = torch.tensor(values)
    asset = SimpleNamespace(
        num_joints=16,
        joint_names=names,
        find_joints=lambda patterns, preserve_order=False: resolve_matching_names(
            patterns, names, preserve_order=preserve_order
        ),
        data=SimpleNamespace(joint_pos=ProxyArray(wp.from_torch(positions))),
    )
    env = SimpleNamespace(scene={"robot": asset})
    for name in ("asset_cfg", "mirror_asset_cfg"):
        term.params[name].resolve(env.scene)
    return term, env, positions


@pytest.mark.parametrize("names", [NATIVE_NAMES, NATIVE_NAMES[::-1]], ids=["native", "reversed"])
def test_front_mirror_uses_physical_signs_and_resolved_joint_order(names):
    term, env, positions = _term_and_env(names)
    before = positions.clone()
    # First environment is mirrored. In the second, errors are 0.3, 0.4, -0.3 rad.
    torch.testing.assert_close(term.func(env, **term.params), torch.tensor([0.0, 0.34]))
    torch.testing.assert_close(positions, before)


def test_rear_and_wheel_angles_do_not_affect_front_mirror_reward():
    term, env, positions = _term_and_env(NATIVE_NAMES)
    before = term.func(env, **term.params)
    other_ids = [i for i, name in enumerate(NATIVE_NAMES) if name.startswith(("hl_", "hr_")) or "wheel" in name]
    positions[:, other_ids] = torch.arange(len(other_ids), dtype=torch.float32) + 10.0
    torch.testing.assert_close(term.func(env, **term.params), before)


def test_stand_enables_front_symmetry_without_modifying_locomotion_tasks():
    term = DeeproboticsM20StandEnvCfg().rewards.joint_mirror
    assert term.func is stand_front_joint_mirror_l2
    assert term.weight == -1.0
    assert term.params["joint_signs"] == (-1.0, 1.0, 1.0)
    assert term.params["asset_cfg"].preserve_order
    assert term.params["mirror_asset_cfg"].preserve_order
    assert term.params["asset_cfg"].joint_names == ["fl_hipx_joint", "fl_hipy_joint", "fl_knee_joint"]
    assert term.params["mirror_asset_cfg"].joint_names == ["fr_hipx_joint", "fr_hipy_joint", "fr_knee_joint"]
    for config_type in (DeeproboticsM20RoughEnvCfg, DeeproboticsM20FlatEnvCfg):
        sibling = config_type().rewards.joint_mirror
        assert sibling.func is mdp.joint_mirror
        assert sibling.weight == -0.03
        assert sibling.params["mirror_joints"] == [
            ["fl_(hipx|hipy|knee).*", "hr_(hipx|hipy|knee).*"],
            ["fr_(hipx|hipy|knee).*", "hl_(hipx|hipy|knee).*"],
        ]
