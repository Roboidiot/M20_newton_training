"""CPU checks of the numerical labels, hook timing, persistence and configuration."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import warp as wp

from sand_pinn_lab.collect.config import environment_config, parse_args
from sand_pinn_lab.collect.geometry import FEET, SHINS, body_ids, capsule_kinematics, read_wheel_geometry
from sand_pinn_lab.collect.policy import SlowRandomPolicy
from sand_pinn_lab.collect.record import MPMRecorder, reduce_wrenches
from sand_pinn_lab.collect.storage import ChunkWriter, save_json
from sand_pinn_lab.collect.surface import LocalSurface, reference_surface
from sand_pinn_lab.collect.validate import validate_run


@pytest.fixture(scope="module", autouse=True)
def warp_cpu_cache(tmp_path_factory):
    wp.config.kernel_cache_dir = str(tmp_path_factory.mktemp("warp_cache"))
    wp.init()


def test_config_independent_and_fixed_physics():
    from isaaclab.utils import to_dict

    from M20_newton_training.tasks.m20.config.M20.walk_on_sand_env_cfg import DeeproboticsM20WalkOnSandEnvCfg

    before = to_dict(DeeproboticsM20WalkOnSandEnvCfg())
    args = parse_args([])
    cfg = environment_config(args)
    assert args.episodes == 20 and args.episode_seconds == 10
    assert args.action_limit == 1.5 and args.hold_seconds == [0.5, 1.0]
    assert not hasattr(args, "particle_samples") and not hasattr(args, "particle_every")
    assert cfg.scene.num_envs == 1 and cfg.sim.dt == 0.005 and cfg.decimation == 4
    assert not cfg.sim.physics.use_cuda_graph
    mpm = next(e.solver_cfg for e in cfg.sim.physics.solver_cfg.entries if e.name == "sand")
    assert mpm.max_iterations == 250 and mpm.tolerance == 1e-5
    assert (mpm.collider_basis, mpm.voxel_size) == ("pic27", 0.02)
    assert cfg.sim.physics.solver_cfg.proxies[0].mode == "staggered"
    assert cfg.actions.mpm_contact is None
    assert all(v is None for k, v in vars(cfg.terminations).items() if not k.startswith("_"))
    assert all(v is None for k, v in vars(cfg.rewards).items() if not k.startswith("_"))
    assert to_dict(DeeproboticsM20WalkOnSandEnvCfg()) == before
    geometry = read_wheel_geometry(cfg.scene.robot.spawn.usd_path)
    assert [g["body_name"] for g in geometry["wheels"]] == list(FEET)
    assert all(np.isclose(g["radius"], 0.09) for g in geometry["wheels"])


@pytest.mark.parametrize(
    "argv",
    [
        ["--episodes", "0"],
        ["--episode-seconds", "0.03", "--warmup", "0"],
        ["--transition-seconds", "2"],
        ["--action-limit", "nan"],
        ["--chunk-seconds", "0"],
        ["--seed", "-1"],
    ],
)
def test_cli_rejects_invalid_timing_and_ranges(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)


def test_random_policy_is_slow_bounded_smooth_and_repeatable():
    a, b = SlowRandomPolicy(42), SlowRandomPolicy(42)
    samples = []
    changes = []
    old_segment = -1
    for t in np.arange(0, 10, 0.02):
        action = a(t)
        np.testing.assert_array_equal(action, b(t))
        if a.segment != old_segment:
            changes.append(t)
            if samples:
                np.testing.assert_allclose(action, samples[-1])
            old_segment = a.segment
        samples.append(action)
    assert np.max(np.abs(samples)) <= 1.5 and np.max(np.abs(samples)) > 1
    assert np.all(np.diff(changes) >= 0.5 - 1e-8)
    assert np.all(np.diff(changes) <= 1.0 + 1e-8)
    assert len(changes) < 22


def test_tilted_capsule_bottom_and_point_velocity():
    pose = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]])
    twist = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    axis = torch.tensor([[0.6, 0.0, 0.8]])
    center, center_v, point, velocity, _ = capsule_kinematics(
        pose, twist, pose[:, :3], torch.zeros(1, 3), axis, torch.tensor([0.09]), torch.tensor([0.027])
    )
    np.testing.assert_allclose(point, [[-0.0162, 0, 0.8884]], atol=1e-6)
    np.testing.assert_allclose(velocity, [[0.8884, 0, 0.0162]], atol=1e-6)
    np.testing.assert_allclose(center_v, [[1, 0, 0]])
    np.testing.assert_allclose(center, [[0, 0, 1]])


def sand_points():
    xy = np.arange(-0.38, 0.4, 0.04)
    xx, yy = np.meshgrid(xy, xy, indexing="ij")
    return np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, 0.29)], axis=-1).astype(np.float32)


def test_surface_masks_and_reference():
    p = torch.tensor(sand_points())
    result = LocalSurface("cpu").sample(p, torch.full((len(p),), 0.01), torch.tensor([[0.0, 0.0], [3.0, 3.0]]))
    assert result["wheel_surface_valid"].tolist() == [True, False]
    assert result["wheel_surface_height_w_m"][0] == pytest.approx(0.3)
    assert not result["wheel_surface_patch_valid"][1].any()
    assert result["wheel_surface_height_w_m"][1] == 0
    assert reference_surface(sand_points(), 0.02) == pytest.approx(0.3)


def test_wrench_reduction_excludes_tray_and_sums_all_body_torques():
    impulse = wp.array([[0, 0, 0.005], [0, 0, 0.01], [0, 0, 999], [0, 0, 999]], dtype=wp.vec3, device="cpu")
    points = wp.array([[0.1, 0, 0]] * 4, dtype=wp.vec3, device="cpu")
    collider = wp.array([0, 1, 2, -1], dtype=int, device="cpu")
    bodies = wp.array([1, 2, 0], dtype=int, device="cpu")
    slots = wp.array([-1, 0, 1], dtype=int, device="cpu")
    com = wp.array([[0, 0, 0], [0, 0, 0]], dtype=wp.vec3, device="cpu")
    force, torque = wp.zeros(2, dtype=wp.vec3, device="cpu"), wp.zeros(2, dtype=wp.vec3, device="cpu")
    wp.launch(
        reduce_wrenches,
        dim=4,
        inputs=[impulse, points, collider, bodies, slots, com, 200.0, force, torque],
        device="cpu",
    )
    np.testing.assert_allclose(force.numpy(), [[0, 0, 1], [0, 0, 2]])
    np.testing.assert_allclose(torque.numpy(), [[0, -0.1, 0], [0, -0.2, 0]], atol=1e-7)


@pytest.mark.parametrize("physics_dt,coupling_iterations", [(0.005, 1), (0.0025, 1), (0.005, 3)])
def test_passive_hook_fresh_joint_state_time_alignment_and_chunk_roundtrip(tmp_path, physics_dt, coupling_iterations):
    args = parse_args(
        [
            "--device",
            "cpu",
            "--episodes",
            "1",
            "--episode-seconds",
            str(8 * physics_dt),
            "--warmup",
            "0",
            "--set",
            f"time.physics_dt_s={physics_dt}",
            "--set",
            f"coupling.iterations={coupling_iterations}",
        ]
    )
    args.leg_joint_names = [f"joint_{j}" for j in range(12)]
    args.wheel_joint_names = [f"joint_{j}" for j in range(12, 16)]
    names = ["base_link", *FEET, *SHINS]
    labels = [f"/World/envs/env_0/Robot/{n}" for n in names]
    mpm_labels = ["/World/envs/env_0/SandTrayFloor", *reversed(labels)]
    rid = body_ids(mpm_labels, names)
    pose = np.zeros((len(names), 7), dtype=np.float32)
    pose[:, 2], pose[:, 6] = 0.35, 1
    proxy_pose = np.zeros((len(mpm_labels), 7), dtype=np.float32)
    proxy_pose[:, 6] = 1
    proxy_pose[rid] = pose
    particles = sand_points()
    model = SimpleNamespace(
        body_label=mpm_labels,
        device=wp.get_device("cpu"),
        particle_count=len(particles),
        body_com=wp.zeros(len(mpm_labels), dtype=wp.vec3, device="cpu"),
        particle_radius=wp.full(len(particles), 0.01, dtype=float, device="cpu"),
    )
    rigid_state = SimpleNamespace(
        body_q=wp.array(pose, dtype=wp.transform, device="cpu"),
        body_qd=wp.zeros(len(names), dtype=wp.spatial_vector, device="cpu"),
        joint_q=wp.array(np.arange(16, dtype=np.float32), device="cpu"),
        joint_qd=wp.zeros(16, device="cpu"),
    )
    state = SimpleNamespace(
        body_q=wp.array(proxy_pose, dtype=wp.transform, device="cpu"),
        body_qd=wp.zeros(len(mpm_labels), dtype=wp.spatial_vector, device="cpu"),
        particle_q=wp.array(particles, dtype=wp.vec3, device="cpu"),
        body_f=wp.zeros(len(mpm_labels), dtype=wp.spatial_vector, device="cpu"),
    )

    class Solver:
        def step(self, state_in, state_out, control, contacts, dt):
            pass

        def collect_collider_impulses(self, state):
            return (
                wp.array([[0, 0, 0.005]], dtype=wp.vec3, device="cpu"),
                wp.array([[0.1, 0, 0.35]], dtype=wp.vec3, device="cpu"),
                wp.array([0], dtype=int, device="cpu"),
            )

    solver = Solver()
    solver.model = model
    solver.collider_body_index = wp.array([rid[1]], dtype=int, device="cpu")
    view = SimpleNamespace(
        body_label=labels,
        body_com=wp.zeros(len(names), dtype=wp.vec3, device="cpu"),
        joint_label=args.leg_joint_names + args.wheel_joint_names,
        joint_q_start=wp.array(np.arange(17), dtype=int, device="cpu"),
        joint_qd_start=wp.array(np.arange(17), dtype=int, device="cpu"),
    )
    rigid = SimpleNamespace(view=view, state_1=rigid_state, control=SimpleNamespace(joint_f=wp.zeros(16, device="cpu")))
    root = SimpleNamespace(solver=lambda name: solver, _entries={"robot": rigid})
    robot = SimpleNamespace(body_names=names, joint_names=view.joint_label)
    actions = SimpleNamespace(
        get_term=lambda name: SimpleNamespace(processed_actions=torch.zeros(1, 12 if name == "joint_pos" else 4))
    )
    geometry = {
        "wheels": [{"local_center": [0, 0, 0], "local_axis": [0, 1, 0], "radius": 0.09, "half_length": 0.027}] * 4
    }
    writer = ChunkWriter(tmp_path / "episode_000", 3)
    recorder = MPMRecorder(root, robot, actions, geometry, args)
    recorder.begin_episode(0, writer, 0.3, 0.3)
    with recorder:
        recorder.enabled = True
        for i in range(8):
            rigid_state.joint_q.fill_(float(i))
            for _ in range(coupling_iterations):
                solver.step(state, state, None, None, physics_dt)
        writer.flush()
    assert "step" not in solver.__dict__
    assert not state.body_f.numpy().any()
    assert writer.frames == 8 and len(writer.chunks) == 3
    with np.load(writer.directory / "chunk_00000.npz") as data:
        np.testing.assert_allclose(data["joint_position_rad"][:, 0], [0, 1, 2])
        np.testing.assert_allclose(data["episode_time_s"], np.arange(1, 4) * physics_dt)
        np.testing.assert_allclose(data["wheel_mpm_force_w_N"][:, 0], [[0, 0, 0.005 / physics_dt]] * 3)
        np.testing.assert_allclose(
            data["wheel_mpm_torque_com_w_Nm"][:, 0], [[0, -0.0005 / physics_dt, 0]] * 3, atol=1e-7
        )
    assert sorted(path.name for path in writer.directory.glob("*.npz")) == [
        "chunk_00000.npz",
        "chunk_00001.npz",
        "chunk_00002.npz",
    ]
    assert all("particles_file" not in chunk for chunk in writer.chunks)
    save_json(
        tmp_path / "metadata.json",
        {
            "schema_version": 2,
            "particle_state_saved": False,
            "status": "complete",
            "physics_dt_s": physics_dt,
            "episode_seconds": 8 * physics_dt,
            "requested_episodes": 1,
            "saved_frames": 8,
            "episodes": [{"episode_id": 0, "directory": "episode_000", "status": "complete", "frames": 8}],
        },
    )
    assert validate_run(tmp_path)["frames"] == 8


def test_partial_chunk_flush_and_nonfinite_rejection(tmp_path):
    writer = ChunkWriter(tmp_path / "partial", 200)
    row = {"episode_time_start_s": 0.0, "episode_time_s": 0.005, "force": np.ones(3)}
    writer.append(row)
    with pytest.raises(FloatingPointError):
        writer.append({**row, "force": np.array([np.nan, 0, 0])})
    writer.flush()
    assert json.loads((writer.directory / "chunks.json").read_text())["frames"] == 1
    assert not list(writer.directory.glob("*.tmp.*"))


def test_reset_clears_sand_history_before_settling_and_preserves_it_after_teleport(monkeypatch):
    from newton import StateFlags

    from isaaclab_newton.physics import NewtonManager
    from isaaclab_newton.physics.mpm_manager import NewtonMPMManager

    from sand_pinn_lab.collect.collector import reset_episode

    events = []
    monkeypatch.setattr(NewtonMPMManager, "reset_solver_state", lambda **kw: events.append(("solver_reset", kw)))
    monkeypatch.setattr(
        NewtonManager, "get_state_0", lambda: SimpleNamespace(particle_q=SimpleNamespace(numpy=lambda: sand_points()))
    )
    root = torch.zeros(1, 13)
    root[:, 6] = 1
    robot = SimpleNamespace(
        reset=lambda: events.append(("robot_reset", {})),
        data=SimpleNamespace(
            default_root_state=SimpleNamespace(torch=root),
            default_joint_pos=SimpleNamespace(torch=torch.zeros(1, 16)),
            default_joint_vel=SimpleNamespace(torch=torch.zeros(1, 16)),
        ),
        write_root_link_pose_to_sim_index=lambda **kw: events.append(("place", kw)),
        write_root_link_velocity_to_sim_index=lambda **kw: None,
        write_joint_state_to_sim_index=lambda **kw: None,
    )

    class Scene(dict):
        env_origins = torch.zeros(1, 3)

        def write_data_to_sim(self):
            pass

        def update(self, dt):
            pass

    scene = Scene(robot=robot, sand=SimpleNamespace(reset=lambda: events.append(("sand_reset", {}))))
    manager = SimpleNamespace(reset=lambda: None)
    raw = SimpleNamespace(
        reset=lambda: events.append(("env_reset", {})),
        scene=scene,
        sim=SimpleNamespace(forward=lambda: None),
        step=lambda action: events.append(("step", {})),
        cfg=SimpleNamespace(
            scene=SimpleNamespace(sand=SimpleNamespace(spawn=SimpleNamespace(voxel_size=0.02, particles_per_cell=1)))
        ),
        action_manager=manager,
        observation_manager=manager,
        episode_length_buf=torch.zeros(1),
    )
    surface = reset_episode(raw, parse_args(["--device", "cpu"]), {"signal": None})
    assert surface == pytest.approx(0.3)
    assert sum(name == "step" for name, _ in events) == 15
    resets = [(i, kw) for i, (name, kw) in enumerate(events) if name == "solver_reset"]
    assert len(resets) == 2 and resets[0][1] == {}
    step_ids = [i for i, (name, _) in enumerate(events) if name == "step"]
    assert resets[0][0] < min(step_ids) < max(step_ids) < resets[1][0]
    assert resets[1][1]["flags"] == StateFlags.BODY_Q | StateFlags.BODY_QD
    placed = next(kw["root_pose"] for name, kw in events if name == "place")
    assert placed[0, 2] == pytest.approx(0.88)
