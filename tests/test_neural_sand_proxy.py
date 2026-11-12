"""Fixed-plane NN features, historical sampling, COM torques and legacy preservation on CPU."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from isaaclab.utils import to_dict

from M20_newton_training.sand_proxy.analytical.foot_model import FootSandModel
from M20_newton_training.sand_proxy.neural.foot_model import DEFAULT_MODEL_PATH, NeuralFootSandModel
from M20_newton_training.sand_proxy.neural.wheel_model_runtime import WheelModelRuntime
from M20_newton_training.tasks.m20.config.M20.walk_env_cfg import DeeproboticsM20WalkEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_env_cfg import DeeproboticsM20WalkOnSandEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_proxy_env_cfg import DeeproboticsM20WalkOnSandProxyEnvCfg
from M20_newton_training.tasks.m20.mdp.foot_sand_proxy import FootSandProxyAction, rotate_xyzw

ROOT = Path(__file__).resolve().parents[1]
FEET = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")


def neural_step(model, depth=0.1, vx=0.2):
    d = torch.full((model.num_envs, 4), depth)
    v = torch.zeros(model.num_envs, 4, 3)
    v[..., 0] = vx
    return model.step(
        d,
        v,
        torch.zeros(model.num_envs, 4, 2),
        0.005,
        angular_velocity_w=v * 0,
        axis_w=torch.tensor([0.0, -1.0, 0.0]).expand_as(v),
        joint_velocity=torch.full_like(d, 2.0),
    )


def test_neural_payload_is_unmodified_and_analytical_imports_remain_compatible():
    from M20_newton_training.sand_proxy.foot_model import FootSandModel as CompatibilityModel

    assert CompatibilityModel is FootSandModel
    assert FootSandModel.load(num_envs=1).parameters["schema_version"] == 2
    for name in ("best.pt", "wheel_model_runtime.py"):
        assert (DEFAULT_MODEL_PATH.parent / name).read_bytes() == (ROOT / "sand_pinn_lab/model" / name).read_bytes()
    metadata = json.loads((DEFAULT_MODEL_PATH.parent / "deployment.json").read_text())
    assert metadata["checkpoint_sha256"] == hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest()
    assert metadata["fixed_surface_height_m"] == 0.20


def test_features_history_and_frozen_prediction_match_the_packaged_network():
    model = NeuralFootSandModel.load(num_envs=3)
    assert not neural_step(model).any() and not model.torque.any()  # No previous 5 ms velocity yet.
    previous = model.features[..., 3:9].clone()
    force = neural_step(model, vx=0.4)
    torch.testing.assert_close(model.features[..., 0], torch.full((3, 4), 0.1))
    torch.testing.assert_close(model.features[..., 1], model.features[..., 0])
    assert (model.features[..., 2] == 1).all()
    torch.testing.assert_close(model.features[..., 13:19], previous)
    expected = WheelModelRuntime(DEFAULT_MODEL_PATH)(model.features)
    torch.testing.assert_close(force, expected[..., :3])
    torch.testing.assert_close(model.torque, expected[..., 3:])
    assert torch.isfinite(force).all() and torch.isfinite(model.torque).all()
    assert not model.runtime.training and all(not p.requires_grad for p in model.runtime.parameters())
    neural_step(model, depth=-0.02)
    assert not model.force.any() and not model.torque.any()
    assert (model.depth < 0).all()  # Raw, signed depth is retained.


def test_partial_reset_discards_only_reset_worlds_history_and_first_step_wrench():
    model = NeuralFootSandModel.load(num_envs=2)
    neural_step(model)
    neural_step(model)
    previous = model.history.previous[1].clone()
    model.reset([0])
    assert model.history.ready.tolist() == [False, True]
    torch.testing.assert_close(model.history.previous[1], previous)
    neural_step(model)
    assert not model.force[0].any() and not model.torque[0].any()
    assert model.force[1].any()
    model.reset()
    assert not model.history.ready.any() and not model.features.any()


@pytest.mark.parametrize("dt", [0.0005, 0.0025, 0.02])
def test_history_dt_mismatch_fails_instead_of_silently_retiming(dt):
    model = NeuralFootSandModel.load(num_envs=1)
    with pytest.raises(ValueError, match="history dt"):
        model.runtime.check_physics_dt(dt)


def make_action():
    cfg = DeeproboticsM20WalkOnSandProxyEnvCfg()
    n = 2
    origins = torch.tensor([[0.0, 0.0, 0.0], [3.0, -4.0, 1.0]])
    pos = origins[:, None].expand(n, 4, 3).clone()
    pos[..., 2] += cfg.actions.sand_proxy.surface_height + 0.05  # 4 cm penetration after subtracting the 9 cm radius.
    quat = torch.tensor([0.0, 0.0, 0.0, 1.0]).expand(n, 4, 4).clone()
    # Different current simulator COM: predicted torque must be shifted, not applied at the bottom again.
    com = pos + torch.tensor([0.01, -0.02, 0.03])
    linear = torch.tensor([0.2, -0.1, 0.05]).expand(n, 4, 3).clone()
    angular = torch.tensor([0.1, 2.0, -0.3]).expand(n, 4, 3).clone()
    joint_ids = [3, 0, 2, 1]  # Unrelated to body order and action-array positions.
    joint = torch.tensor([[2.0, -4.0, 3.0, -1.0]]).expand(n, 4).clone()
    data = SimpleNamespace(
        body_link_quat_w=SimpleNamespace(torch=quat),
        body_link_pos_w=SimpleNamespace(torch=pos),
        body_com_pos_w=SimpleNamespace(torch=com),
        body_com_lin_vel_w=SimpleNamespace(torch=linear),
        body_com_ang_vel_w=SimpleNamespace(torch=angular),
        joint_vel=SimpleNamespace(torch=joint),
        default_mass=SimpleNamespace(torch=torch.full((n, 4), 2.0)),
    )
    calls = []

    def capture(**kwargs):
        calls.append({key: val.clone() if isinstance(val, torch.Tensor) else val for key, val in kwargs.items()})

    robot = SimpleNamespace(
        data=data,
        find_bodies=lambda names, preserve_order: ([0, 1, 2, 3], list(names)),
        find_joints=lambda names, preserve_order: (joint_ids, list(names)),
        instantaneous_wrench_composer=SimpleNamespace(add_forces_and_torques_index=capture),
    )

    class Scene(dict):
        pass

    scene = Scene(robot=robot)
    scene.env_origins = origins
    scene.sensors = {
        "foot_ground_contacts": SimpleNamespace(
            find_bodies=lambda names, preserve_order: ([2, 0, 3, 1], list(names)),
            num_filter_objects=2,
            data=SimpleNamespace(force_matrix_w=SimpleNamespace(torch=torch.zeros(n, 4, 2, 3))),
        )
    }
    env = SimpleNamespace(
        num_envs=n,
        device="cpu",
        physics_dt=cfg.sim.dt,
        extras={},
        scene=scene,
        cfg=SimpleNamespace(sim=SimpleNamespace(physics=cfg.sim.physics.newton_mjwarp), decimation=cfg.decimation),
    )
    return env, FootSandProxyAction(cfg.actions.sand_proxy, env), calls, data, joint_ids


def test_action_reads_fixed_plane_depth_center_velocity_actual_joint_speed_and_applies_com_wrench():
    env, action, calls, data, joint_ids = make_action()
    assert action.action_dim == 0 and action.raw_actions.shape == (2, 0)
    action.process_actions(torch.empty(2, 0))
    action.apply_actions()
    assert not calls[-1]["forces"].any() and not calls[-1]["torques"].any()
    action.apply_actions()
    torch.testing.assert_close(action.model.depth, torch.full((2, 4), 0.04), atol=1e-6, rtol=0)
    torch.testing.assert_close(action.model.features[..., 1], action.model.depth)
    assert (action.model.features[..., 2] == 1).all()
    expected_v = data.body_com_lin_vel_w.torch + torch.linalg.cross(
        data.body_com_ang_vel_w.torch, data.body_link_pos_w.torch - data.body_com_pos_w.torch, dim=-1
    )
    torch.testing.assert_close(action.model.features[..., 3:6], expected_v)
    torch.testing.assert_close(action.model.features[..., 12], data.joint_vel.torch[:, joint_ids])
    source_com = data.body_link_pos_w.torch + rotate_xyzw(
        data.body_link_quat_w.torch, action.model.reference_com_b.expand(2, 4, 3)
    )
    expected_tau = action.model.torque + torch.linalg.cross(
        source_com - data.body_com_pos_w.torch, action.force_w, dim=-1
    )
    torch.testing.assert_close(calls[-1]["forces"], action.model.force)
    torch.testing.assert_close(calls[-1]["torques"], expected_tau)
    torch.testing.assert_close(calls[-1]["positions"], data.body_com_pos_w.torch)
    assert calls[-1]["is_global"] is True
    assert not action.contact_state.force_duration.any()
    action.apply_actions()
    action.apply_actions()
    torch.testing.assert_close(action.contact_state.force_duration, torch.full((2,), 0.02))
    action.reset([0])
    assert not action.model.history.ready[0] and action.model.history.ready[1]
    assert not action.torque_w[0].any() and not action.force_w[0].any()


def test_default_neural_task_keeps_policy_io_and_other_tasks_unchanged():
    before = to_dict(DeeproboticsM20WalkOnSandEnvCfg())
    walk = DeeproboticsM20WalkEnvCfg()
    cfg = DeeproboticsM20WalkOnSandProxyEnvCfg()
    assert cfg.actions.sand_proxy.model_kind == "neural"
    assert Path(cfg.actions.sand_proxy.model_path) == DEFAULT_MODEL_PATH
    assert cfg.actions.sand_proxy.surface_height == 0.20
    assert cfg.actions.sand_proxy.rigid_contact_sensor == "foot_ground_contacts"
    assert cfg.scene.foot_ground_contacts.filter_shape_prim_expr == ["/World/ground(?:/.*)?"]
    assert cfg.sim.dt == 0.005 and cfg.decimation == 4
    assert to_dict(cfg.observations) == to_dict(walk.observations)
    assert to_dict(cfg.actions.joint_pos) == to_dict(walk.actions.joint_pos)
    assert to_dict(cfg.actions.joint_vel) == to_dict(walk.actions.joint_vel)
    assert to_dict(DeeproboticsM20WalkOnSandEnvCfg()) == before


def test_rigid_floor_support_combines_with_proxy_after_physics_without_extra_wrenches():
    env, action, calls, _, _ = make_action()
    floor = env.scene.sensors["foot_ground_contacts"].data.force_matrix_w.torch
    # Isolate contact semantics from NN predictions: the solver receives only this sand force.
    proxy = torch.zeros_like(action.force_w)
    proxy[..., 2] = -2.0
    action.model.step = lambda *args, **kwargs: proxy

    def physics_window():
        action.process_actions(torch.empty(2, 0))
        for _ in range(env.cfg.decimation):
            action.apply_actions()

    physics_window()
    assert not action._contact_state.force_duration.any()  # No pre-physics force sampling.
    # Sensor order is FR/HR/FL/HL and force_matrix has two ground shapes.
    floor[:, 2, :, 2] = 4.0  # FL: 8 N rigid - 2 N sand = 6 N, supported.
    floor[:, 0, :, 2] = 3.5  # FR: 7 N rigid - 2 N sand = 5 N, unloaded.
    floor[:, 3, :, 0] = 100.0  # HL: horizontal force cannot establish support.
    state = action.contact_state
    assert state.vertical_contact.tolist() == [[True, False, False, False]] * 2
    assert not state.touchdown.any()  # Initial support is not a completed swing.
    torch.testing.assert_close(state.force_duration, torch.full((2,), 0.02))
    # Repeated rewards cannot advance timers or read changing sensor values twice.
    floor.zero_()
    assert action.contact_state.vertical_contact[:, 0].all()
    torch.testing.assert_close(action.contact_state.force_duration, torch.full((2,), 0.02))
    physics_window()
    assert not action.contact_state.vertical_contact.any()
    torch.testing.assert_close(action.contact_state.current_air_time[:, 0], torch.full((2,), 0.02))
    physics_window()
    floor[:, 2, :, 2] = 10.0
    assert action.contact_state.touchdown[:, 0, 0].all()
    torch.testing.assert_close(action.contact_state.touchdown_air_time[:, 0, 0], torch.full((2,), 0.04))
    assert len(calls) == 3 * env.cfg.decimation
    for call in calls:
        torch.testing.assert_close(call["forces"], proxy)
    # Preserve the existing proxy-only signed Fz metrics, and only reset requested worlds.
    action.reset([0])
    torch.testing.assert_close(env.extras["log"]["Metrics/foot_forces/fl_fz_mean"], torch.tensor(-2.0))
    assert not action.contact_force_w[0].any() and action.contact_state.has_contact[1, 0]


def test_ground_sensor_filter_must_match_a_collision_shape():
    env, action, _, _, _ = make_action()
    env.scene.sensors["foot_ground_contacts"].num_filter_objects = 0
    with pytest.raises(ValueError, match="match the ground collision shapes"):
        FootSandProxyAction(action.cfg, env)
