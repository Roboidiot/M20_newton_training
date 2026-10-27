"""Physical invariants, footprint state, checkpoint compatibility and force injection."""

import json
from types import SimpleNamespace

import torch

from isaaclab.utils import to_dict

from M20_newton_training.sand_proxy.foot_model import DEFAULT_MODEL_PATH, FootSandModel
from M20_newton_training.tasks.m20.config.M20.walk_env_cfg import DeeproboticsM20WalkEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_env_cfg import DeeproboticsM20WalkOnSandEnvCfg
from M20_newton_training.tasks.m20.config.M20.walk_on_sand_proxy_env_cfg import DeeproboticsM20WalkOnSandProxyEnvCfg
from M20_newton_training.tasks.m20.mdp.foot_sand_proxy import (
    FootSandProxyAction,
    proxy_contact_forces,
    proxy_depth_limit,
    rotate_xyzw,
)
from M20_newton_training.tasks.m20.mdp.proxy_contact_rewards import (
    ProxyDiagonalLiftConsistency,
    ProxyRotationGaitStatus,
    ProxyRotationGaitSymmetry,
    proxy_feet_air_time_command_scaled,
    proxy_hipx_symmetry_l2,
)


def model(num_envs=2):
    return FootSandModel.load(num_envs=num_envs)


def tick(proxy, depth=0.04, vx=0.0, vz=0.0, stability=None):
    d = torch.full((proxy.num_envs, 4), depth)
    v = torch.zeros((proxy.num_envs, 4, 3))
    v[..., 0] = vx
    v[..., 2] = vz
    xy = torch.zeros((proxy.num_envs, 4, 2))
    xy[:, :, 0] = torch.arange(4) * 0.5
    return proxy.step(d, v, xy, 0.0025, stability_mass_kg=stability)


def test_foot_law_is_finite_passive_and_contact_free_outside_sand():
    proxy = model()
    assert tick(proxy, depth=-0.02).count_nonzero() == 0
    force = tick(proxy, depth=0.03, vx=0.1)
    assert force.shape == (2, 4, 3) and torch.isfinite(force).all()
    assert (force[..., 0] < 0).all() and (force[..., 2] > 0).all()
    assert proxy.contact.all()
    force = tick(proxy, depth=0.04, vx=-0.1)
    assert (force[..., 0] > 0).all()


def test_elastoplastic_unloading_has_history_and_old_footprints_reduce_support():
    proxy = model()
    loaded = tick(proxy, depth=0.12).clone()
    assert (loaded[..., 2] > 0).all()
    unloaded = tick(proxy, depth=0.02).clone()
    fresh = tick(model(), depth=0.02).clone()
    assert (unloaded[..., 2] < fresh[..., 2]).all()
    tick(proxy, depth=-0.02)
    assert (proxy.hole_depth > 0).any()
    returned = tick(proxy, depth=0.02)
    assert (returned[..., 2] < fresh[..., 2]).all()


def test_partial_reset_preserves_other_worlds_and_clears_footprint_memory():
    proxy = model()
    tick(proxy, depth=0.1)
    tick(proxy, depth=-0.02)
    previous = proxy.hole_depth[1].clone()
    proxy.reset([0])
    assert (proxy.memory[0] == -1).all() and not proxy.hole_depth[0].any()
    torch.testing.assert_close(proxy.hole_depth[1], previous)


def test_explicit_drag_limiter_does_not_reverse_slip_in_one_step():
    proxy = model()
    mass = torch.full((2, 1), 2.0)
    force = tick(proxy, depth=0.18, vx=0.05, stability=mass)
    assert (torch.linalg.vector_norm(force[..., :2], dim=-1) <= mass * 0.05 / 0.0025 + 1e-4).all()
    assert torch.isfinite(tick(proxy, depth=0.3, vx=3.0, stability=mass)).all()
    assert proxy.out_of_domain.all()


def test_proxy_task_uses_no_mpm_preserves_actor_critic_interfaces_and_siblings():
    before_walk = to_dict(DeeproboticsM20WalkEnvCfg())
    before_mpm = to_dict(DeeproboticsM20WalkOnSandEnvCfg())
    cfg = DeeproboticsM20WalkOnSandProxyEnvCfg()
    assert to_dict(cfg.observations) == before_walk["observations"]
    assert to_dict(cfg.actions.joint_pos) == before_walk["actions"]["joint_pos"]
    assert to_dict(cfg.actions.joint_vel) == before_walk["actions"]["joint_vel"]
    assert to_dict(DeeproboticsM20WalkEnvCfg()) == before_walk
    assert to_dict(DeeproboticsM20WalkOnSandEnvCfg()) == before_mpm
    assert not hasattr(cfg.scene, "sand")
    assert cfg.scene.sand_surface_visual.spawn.collision_props is None
    assert cfg.sim.dt * cfg.decimation == 0.02
    assert cfg.scene.contact_forces is None
    for name in (
        "contact_forces",
        "hipy_joint_pos_penalty",
        "knee_joint_pos_penalty",
        "joint_acc_wheel_l2",
        "action_smooth_l2",
        "feet_slide_ang_z_cmd",
        "joint_mirror",
        "stand_still",
        "sand_diagonal_steps",
        "sand_step_completion",
        "undesired_contacts",
    ):
        assert getattr(cfg.rewards, name) is None
    for name, func, weight in (
        ("feet_air_time", proxy_feet_air_time_command_scaled, 5.0),
        ("rotation_gait_status", ProxyRotationGaitStatus, 10.0),
        ("rotation_gait_symmetry", ProxyRotationGaitSymmetry, 3.0),
        ("diagonal_lift_consistency", ProxyDiagonalLiftConsistency, -3.0),
    ):
        term = getattr(cfg.rewards, name)
        assert term.func is func and term.weight == weight
        assert term.params["action_term_name"] == "sand_proxy"
        assert "sensor_cfg" not in term.params
        assert term.params["min_command_norm"] == 0.1
    assert cfg.rewards.hipx_joint_pos_penalty.weight == -1.5
    assert cfg.rewards.hipx_symmetry.func is proxy_hipx_symmetry_l2
    assert cfg.rewards.hipx_symmetry.weight == -0.5
    assert cfg.rewards.hipx_symmetry.params["asset_cfg"].preserve_order
    assert cfg.rewards.rotation_gait_status.params["target_height"] == 0.25
    assert cfg.rewards.track_lin_vel_xy_exp.weight == 3.0
    assert cfg.rewards.track_ang_vel_z_exp.weight == 1.5
    assert cfg.actions.sand_proxy.contact_force_threshold == 5.0
    for term in vars(cfg.rewards).values():
        if term is not None:
            sensor = term.params.get("sensor_cfg")
            assert sensor is None or sensor.name != "contact_forces"
    cfg.play_mode()
    walk_play = DeeproboticsM20WalkEnvCfg()
    walk_play.play_mode()
    assert cfg.commands.base_velocity.keyboard_speeds == walk_play.commands.base_velocity.keyboard_speeds
    assert cfg.rewards.contact_forces is None


def test_proxy_force_penalty_uses_per_foot_peak_full_vector_without_sensor_or_model_step():
    # Four feet, two environments. Opposing force signs must not cancel.
    history = torch.zeros((2, 3, 4, 3))
    history[0, 0, 0] = torch.tensor([60.0, 80.0, 0.0])  # exactly 100 N: no excess
    history[0, 1, 1] = torch.tensor([-120.0, 0.0, 160.0])  # 200 N in older sample
    history[0, 2, 2] = torch.tensor([0.0, 0.0, -150.0])
    history[1] = 0.5  # all forces below threshold
    term = SimpleNamespace(force_history_w=history)
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: term))
    before = history.clone()
    for _ in range(2):
        torch.testing.assert_close(proxy_contact_forces(env, threshold=100.0), torch.tensor([150.0, 0.0]))
    torch.testing.assert_close(history, before)


def test_proxy_depth_termination_is_disabled_but_explicit_override_remains_available():
    assert getattr(DeeproboticsM20WalkOnSandProxyEnvCfg().terminations, "proxy_depth_limit", None) is None
    parameters = {"training_depth_limit_m": 0.20, "domain": {"max_depth_m": 0.28}}
    depth = torch.tensor([[0.0, 0.29, 0.30, -0.01], [0.0, 0.0, 0.0, 0.3001]])
    term = SimpleNamespace(model=SimpleNamespace(parameters=parameters, depth=depth))
    env = SimpleNamespace(action_manager=SimpleNamespace(get_term=lambda name: term))
    torch.testing.assert_close(proxy_depth_limit(env, limit_m=0.30), torch.tensor([False, True]))
    torch.testing.assert_close(proxy_depth_limit(env), torch.tensor([True, True]))
    assert parameters["training_depth_limit_m"] == 0.20


def test_force_action_has_zero_policy_dimensions_and_applies_at_wheel_bottom_in_world_frame():
    cfg = DeeproboticsM20WalkOnSandProxyEnvCfg()
    n = 2
    pos = torch.zeros((n, 4, 3))
    pos[..., 2] = cfg.actions.sand_proxy.surface_height + 0.05
    quat = torch.zeros((n, 4, 4))
    quat[..., 3] = 1
    root_quat = torch.zeros((n, 4))
    root_quat[..., 3] = 1

    def wrap(value):
        return SimpleNamespace(torch=value)

    data = SimpleNamespace(
        body_link_quat_w=wrap(quat),
        body_link_pos_w=wrap(pos),
        body_com_lin_vel_w=wrap(torch.zeros_like(pos)),
        body_com_ang_vel_w=wrap(torch.zeros_like(pos)),
        body_com_pos_w=wrap(pos),
        root_link_quat_w=wrap(root_quat),
        default_mass=wrap(torch.full((n, 4), 2.0)),
    )
    calls = []
    robot = SimpleNamespace(
        data=data,
        find_bodies=lambda names, preserve_order: ([0, 1, 2, 3], list(names)),
        instantaneous_wrench_composer=SimpleNamespace(
            add_forces_and_torques_index=lambda **kwargs: calls.append(kwargs)
        ),
    )
    env = SimpleNamespace(
        num_envs=n,
        device="cpu",
        physics_dt=cfg.sim.dt,
        cfg=SimpleNamespace(sim=SimpleNamespace(physics=cfg.sim.physics.newton_mjwarp), decimation=cfg.decimation),
        scene=SimpleNamespace(),
    )

    class Scene(dict):
        env_origins = torch.zeros((n, 3))

    env.scene = Scene(robot=robot)
    action = FootSandProxyAction(cfg.actions.sand_proxy, env)
    assert action.action_dim == 0 and action.raw_actions.shape == (n, 0)
    action.process_actions(torch.empty((n, 0)))
    action.apply_actions()
    assert len(calls) == 1 and calls[0]["body_ids"] == [0, 1, 2, 3] and calls[0]["is_global"]
    torch.testing.assert_close(
        calls[0]["positions"][..., 2],
        torch.full((n, 4), cfg.actions.sand_proxy.surface_height - 0.04),
        atol=1e-6,
        rtol=0,
    )
    assert (calls[0]["forces"][..., 2] > 0).all()
    assert not action.contact_state.vertical_contact.any()  # timers wait for the control boundary
    applied = action.force_w.clone()
    torch.testing.assert_close(action.force_history_w[:, 0], applied)
    assert not action.force_history_w[:, 1:].any()
    env.action_manager = SimpleNamespace(get_term=lambda name: action)
    torch.testing.assert_close(
        proxy_contact_forces(env, threshold=100.0), (applied.norm(dim=-1) - 100).clamp(min=0).sum(dim=1)
    )
    for _ in range(cfg.decimation - 1):
        action.apply_actions()
    assert action.contact_state.vertical_contact.all()
    assert not action.contact_state.touchdown.any()  # initial support is not a completed swing
    torch.testing.assert_close(action.contact_state.current_contact_time, torch.full((n, 4), 0.02))

    # Twenty force threshold crossings inside a control period must not become twenty landings.
    original_step = action.model.step

    def alternating_force(*args, **kwargs):
        force = torch.zeros_like(action.force_w)
        force[..., 2] = action.cfg.contact_force_threshold + 10.0 if action._physics_tick % 2 else 0.0
        return force

    action.model.step = alternating_force
    action.process_actions(torch.empty((n, 0)))
    for _ in range(cfg.decimation):
        action.apply_actions()
    assert len(calls) == 2 * cfg.decimation  # physical forces still update at every tick
    assert action.contact_state.vertical_contact.all()
    assert not action.contact_state.touchdown.any()
    assert not action.contact_state.current_air_time.any()
    torch.testing.assert_close(action.contact_state.current_contact_time, torch.full((n, 4), 0.04))
    action.model.step = original_step
    applied = action.force_w.clone()

    # Lift out of the sand: old applied forces persist for exactly three physics samples.
    pos[..., 2] = 0.6
    action.process_actions(torch.empty((n, 0)))
    action.apply_actions()
    assert not action.force_w.any()
    assert action.contact_state.vertical_contact.all()  # no new sample until the control boundary
    assert not action.contact_state.current_air_time.any()
    torch.testing.assert_close(action.force_history_w[:, 1], applied)
    action.apply_actions()
    torch.testing.assert_close(action.force_history_w[:, 2], applied)
    history_other_env = action.force_history_w[1].clone()
    action.reset([0])
    assert not action.model.force[0].any()
    assert not action.force_w[0].any()
    assert not action.force_history_w[0].any()
    assert not action.contact_state.current_air_time[0].any()
    assert action.contact_state.has_contact[1].all()
    torch.testing.assert_close(action.force_history_w[1], history_other_env)
    assert proxy_contact_forces(env, threshold=0.0)[0] == 0
    for _ in range(cfg.decimation - 2):
        action.apply_actions()
    assert not action.contact_state.vertical_contact.any()
    torch.testing.assert_close(action.contact_state.current_air_time, torch.full((n, 4), 0.02))
    assert not action.force_history_w.any()
    action.reset()
    assert not action.force_history_w.any()


def test_xyzw_geometry_and_only_foot_experiments_in_deployed_model():
    q = torch.tensor([2**-0.5, 0.0, 0.0, 2**-0.5])
    torch.testing.assert_close(
        rotate_xyzw(q, torch.tensor([0.0, 0.0, 1.0])), torch.tensor([0.0, -1.0, 0.0]), atol=1e-6, rtol=0
    )
    parameters = json.loads(DEFAULT_MODEL_PATH.read_text())
    assert all(name.startswith("foot_") for name in parameters["fit_cases"])
    assert len(parameters["fit_cases"]) == 114
    assert parameters["source_data_origins"] == ["newton_mpm"]
