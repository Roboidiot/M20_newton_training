"""Wheel masking must follow joint identity across observation permutations."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.string import resolve_matching_names
from isaaclab.utils.warp import ProxyArray

from M20_newton_training.tasks.m20.config.M20.flat_env_cfg import DeeproboticsM20FlatEnvCfg
from M20_newton_training.tasks.m20.config.M20.rough_env_cfg import DeeproboticsM20RoughEnvCfg
from M20_newton_training.tasks.m20.config.M20.stand_env_cfg import DeeproboticsM20StandEnvCfg
from M20_newton_training.tasks.m20.mdp.observations import joint_pos_rel_without_wheel

LIMBS = ("fl", "fr", "hl", "hr")
NATIVE_NAMES = [f"{limb}_{joint}_joint" for limb in LIMBS for joint in ("hipx", "hipy", "knee", "wheel")]
POLICY_NAMES = [f"{limb}_{joint}_joint" for limb in LIMBS for joint in ("hipx", "hipy", "knee")] + [
    f"{limb}_wheel_joint" for limb in LIMBS
]


def _env(names):
    # Real Isaac Lab ProxyArray views, with distinct offsets in two environments.
    defaults = torch.arange(32, dtype=torch.float32).reshape(2, 16) * 0.125
    offsets = torch.arange(1, 33, dtype=torch.float32).reshape(2, 16)
    positions = defaults + offsets
    asset = SimpleNamespace(
        joint_names=names,
        num_joints=len(names),
        find_joints=lambda patterns, preserve_order=False: resolve_matching_names(
            patterns, names, preserve_order=preserve_order
        ),
        data=SimpleNamespace(
            joint_pos=ProxyArray(wp.from_torch(positions)),
            default_joint_pos=ProxyArray(wp.from_torch(defaults)),
        ),
    )
    return SimpleNamespace(scene={"robot": asset}), positions, defaults, offsets


def _expected(offsets, native_names, selected_names):
    return torch.stack(
        [
            torch.zeros(offsets.shape[0]) if name.endswith("_wheel_joint") else offsets[:, native_names.index(name)]
            for name in selected_names
        ],
        dim=1,
    )


@pytest.mark.parametrize(
    "config_type", [DeeproboticsM20StandEnvCfg, DeeproboticsM20RoughEnvCfg, DeeproboticsM20FlatEnvCfg]
)
@pytest.mark.parametrize("group", ["policy", "critic"])
@pytest.mark.parametrize("native_names", [NATIVE_NAMES, POLICY_NAMES], ids=["interleaved", "legs_then_wheels"])
def test_task_observations_zero_only_wheels_and_preserve_leg_angles(config_type, group, native_names):
    env, positions, defaults, offsets = _env(native_names)
    before_positions, before_defaults = positions.clone(), defaults.clone()
    term = getattr(config_type().observations, group).joint_pos
    for entity_cfg in term.params.values():
        entity_cfg.resolve(env.scene)
    actual = term.func(env, **term.params)
    selected_names = POLICY_NAMES if group == "policy" else native_names
    assert actual.shape == (2, 16)
    torch.testing.assert_close(actual, _expected(offsets, native_names, selected_names))
    # Masking observations must not modify the robot state or its default pose.
    torch.testing.assert_close(positions, before_positions)
    torch.testing.assert_close(defaults, before_defaults)


@pytest.mark.parametrize("selection", [[14, 0, 3, 7], [0, 1, 2], slice(2, 10, 2)])
def test_wheel_mask_handles_subsets_without_using_native_ids_as_output_slots(selection):
    env, _, _, offsets = _env(NATIVE_NAMES)
    actual = joint_pos_rel_without_wheel(
        env,
        asset_cfg=SceneEntityCfg("robot", joint_ids=selection),
        wheel_asset_cfg=SceneEntityCfg("robot", joint_ids=[3, 7, 11, 15]),
    )
    selected_ids = list(range(16))[selection] if isinstance(selection, slice) else selection
    torch.testing.assert_close(actual, _expected(offsets, NATIVE_NAMES, [NATIVE_NAMES[i] for i in selected_ids]))
