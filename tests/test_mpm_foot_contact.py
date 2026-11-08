"""MPM impulse units, entry-local mapping and control-rate gait/contact timing."""

from types import SimpleNamespace

import pytest
import torch
import warp as wp

from isaaclab_newton.physics import NewtonManager

from M20_newton_training.tasks.m20.config.M20.walk_on_sand_env_cfg import DeeproboticsM20WalkOnSandEnvCfg
from M20_newton_training.tasks.m20.mdp.mpm_foot_contact import (
    MPMFootContactAction,
    MPMFootContactActionCfg,
    foot_body_slots,
)

FEET = ("fl_wheel", "fr_wheel", "hl_wheel", "hr_wheel")


def _labels(num_envs=2):
    # Deliberately scramble environment and link order.
    return [
        f"/World/envs/env_{env}/Robot/{name}"
        for env in reversed(range(num_envs))
        for name in ("fr_wheel", "base_link", "hr_wheel", "fl_wheel", "hl_wheel")
    ] + ["/World/envs/env_0/SandTrayFloor"]


def test_entry_local_mapping_separates_environments_and_excludes_nonfeet():
    slots = foot_body_slots(_labels(), 2, FEET)
    assert slots == [5, -1, 7, 4, 6, 1, -1, 3, 0, 2, -1]
    with pytest.raises(ValueError, match="Missing MPM foot"):
        foot_body_slots(_labels()[1:], 2, FEET)
    with pytest.raises(ValueError, match="Duplicate MPM foot"):
        foot_body_slots(_labels() + [_labels()[0]], 2, FEET)


@pytest.fixture
def contact_env(monkeypatch):
    wp.init()
    labels = _labels()
    mapping = foot_body_slots(labels, 2, FEET)
    # One collider per entry-local body, plus a body-less collider.
    collider_body = list(range(len(labels))) + [-1]
    colliders = list(range(len(collider_body))) + [8, -1]

    class Solver:
        model = SimpleNamespace(body_label=labels)
        collider_body_index = wp.array(collider_body, dtype=int, device="cpu")
        calls = 0

        def __init__(self):
            self.forces = torch.zeros(2, 4, 3)
            self.impulses = wp.zeros(len(colliders), dtype=wp.vec3, device="cpu")
            self.ids = wp.array(colliders, dtype=int, device="cpu")

        def step(self, state_in, state_out, control, contacts, dt):
            self.calls += 1
            impulses = []
            for collider in colliders:
                body = collider_body[collider] if collider >= 0 else -1
                slot = mapping[body] if body >= 0 else -1
                if slot < 0:
                    impulses.append([100.0, 100.0, 100.0])
                else:
                    # FL env_0 has two impulse samples; both must be accumulated.
                    share = 0.5 if slot == 0 else 1.0
                    impulses.append((self.forces.reshape(-1, 3)[slot] * dt * share).tolist())
            self.impulses.assign(impulses)

        def collect_collider_impulses(self, state):
            return self.impulses, None, self.ids

    solver = Solver()
    root = SimpleNamespace(solver=lambda name: solver, entry_names=lambda: ("robot", "sand"))
    monkeypatch.setattr(NewtonManager, "_solver", root)
    quaternion = torch.zeros(2, 4, 4)
    quaternion[..., 3] = 1
    data = SimpleNamespace(
        body_link_quat_w=SimpleNamespace(torch=quaternion),
        body_link_pos_w=SimpleNamespace(torch=torch.full((2, 4, 3), 0.3)),
    )
    asset = SimpleNamespace(data=data, find_bodies=lambda names, preserve_order: ([0, 1, 2, 3], list(FEET)))
    cfg = DeeproboticsM20WalkOnSandEnvCfg()
    env = SimpleNamespace(
        cfg=SimpleNamespace(sim=SimpleNamespace(physics=SimpleNamespace(use_cuda_graph=False))),
        device="cpu",
        num_envs=2,
        step_dt=0.02,
        scene={"robot": asset},
        command_manager=SimpleNamespace(get_command=lambda name: torch.tensor([[0.3, 0.0, 0.0]]).repeat(2, 1)),
        extras={},
    )
    term = MPMFootContactAction(MPMFootContactActionCfg(), env)
    env.action_manager = SimpleNamespace(get_term=lambda name: term)
    yield env, term, solver, cfg
    term.detach()
    assert "step" not in solver.__dict__


def _window(term, solver, forces, durations=(0.005,) * 4):
    term.process_actions(torch.empty(2, 0))
    for force, dt in zip(forces, durations, strict=True):
        solver.forces.copy_(force)
        term.apply_actions()
        solver.step(None, None, None, None, dt)


def test_force_is_whole_window_impulse_over_duration_not_last_substep(contact_env):
    _, term, solver, _ = contact_env
    assert term.action_dim == 0 and term.raw_actions.shape == term.processed_actions.shape == (2, 0)
    force = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    force[..., 2] += 20.0
    _window(term, solver, [force * 4, force * 0, force * 0, force * 0])
    torch.testing.assert_close(term.impulse_w, force * 0.02)
    # Physics substeps do not advance timers. Publication happens once, during reward access.
    assert not term._contact_state.current_contact_time.any()
    torch.testing.assert_close(term.force_w, force)
    torch.testing.assert_close(term.contact_state.current_contact_time, torch.full((2, 4), 0.02))
    for _ in range(3):
        torch.testing.assert_close(term.force_w, force)
        torch.testing.assert_close(term.contact_state.current_contact_time, torch.full((2, 4), 0.02))
        torch.testing.assert_close(term._contact_state.force_duration, torch.full((2,), .02))
    torch.testing.assert_close(term.force_history_w[:, 0], force)
    assert not term.force_history_w[:, 1:].any()
    torch.testing.assert_close(term.application_points_w[..., 2], torch.full((2, 4), 0.21))


def test_nonuniform_substeps_weight_impulses_by_actual_duration(contact_env):
    _, term, solver, _ = contact_env
    a, b = torch.zeros(2, 4, 3), torch.zeros(2, 4, 3)
    a[..., 2], b[..., 2] = 20.0, 100.0
    _window(term, solver, [a, b], durations=(0.015, 0.005))
    torch.testing.assert_close(term.force_w[..., 2], torch.full((2, 4), 40.0))


def test_airtime_touchdown_and_partial_reset_use_50hz_vertical_contact(contact_env):
    env, term, solver, cfg = contact_env
    reward = cfg.rewards.feet_air_time
    reward_func = reward.func
    support = torch.zeros(2, 4, 3)
    support[..., 2] = 50.0
    _window(term, solver, [support] * 4)
    assert not reward_func(env, **reward.params).any()  # Initial landing is not rewarded.
    unloaded = support.clone()
    unloaded[:, 0] = torch.tensor([100.0, 0.0, -50.0])
    unloaded[:, 1] = torch.tensor([100.0, 0.0, 5.0])
    for _ in range(5):
        _window(term, solver, [unloaded] * 4)
        assert not reward_func(env, **reward.params).any()
    torch.testing.assert_close(term.contact_state.current_air_time[:, :2], torch.full((2, 2), 0.10))
    assert term.contact_state.contact.all() and not term.contact_state.vertical_contact[:, :2].any()
    _window(term, solver, [support] * 4)
    torch.testing.assert_close(reward_func(env, **reward.params), torch.full((2,), 0.04))
    assert term.contact_state.touchdown.sum() == 4
    torch.testing.assert_close(reward_func(env, **reward.params), torch.full((2,), 0.04))
    term.reset([0])
    assert not term.contact_state.has_contact[0].any() and not term.force_history_w[0].any()
    assert term.contact_state.has_contact[1].all() and term.force_history_w[1].any()


def test_incomplete_window_fails_instead_of_silently_using_wrong_force(contact_env):
    _, term, solver, _ = contact_env
    _window(term, solver, [torch.zeros(2, 4, 3)], durations=(0.005,))
    with pytest.raises(RuntimeError, match="expected 0.02"):
        _ = term.contact_state


def test_mpm_force_logs_use_window_averaged_world_fz_and_publish_before_partial_reset(contact_env):
    env, term, solver, _ = contact_env
    a, b = torch.zeros(2, 4, 3), torch.zeros(2, 4, 3)
    a[..., 2], b[..., 2] = -20., 60.
    _window(term, solver, [a, b], durations=(.015, .005))
    torch.testing.assert_close(term.force_w[..., 2], torch.zeros(2, 4), atol=1e-5, rtol=0)
    b[..., 2] = -40.
    _window(term, solver, [b] * 4)
    torch.testing.assert_close(term.force_w[..., 2], torch.full((2, 4), -40.))
    term.reset([0])
    torch.testing.assert_close(env.extras["log"]["Metrics/foot_forces/fl_fz_mean"], torch.tensor(-20.))
    assert len(env.extras["log"]) == 4
    assert not term._contact_state.force_duration[0]
    torch.testing.assert_close(term._contact_state.force_duration[1], torch.tensor(.04))
