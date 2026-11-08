"""World-Z force diagnostics retain signs, sample once per control update, and reset independently."""

from types import SimpleNamespace

import torch

from M20_newton_training.tasks.m20.mdp.proxy_contact_rewards import ProxyFootContactState


def _sample(state, values, dt):
    force = torch.full((2, 4, 3), 10000.0)  # Horizontal force must not enter Fz statistics.
    force[0, :, 2] = torch.tensor(values)
    force[1, :, 2] = 2 * force[0, :, 2]
    state.begin_step()
    state.update(force, dt, 5.0)


def test_only_signed_mean_force_is_logged_including_pullout_and_airborne_samples():
    state = ProxyFootContactState(2, "cpu")
    env = SimpleNamespace(extras={"log": {"existing": 1.0}})
    _sample(state, [10.0, 20.0, -30.0, 0.0], 0.01)
    _sample(state, [-30.0, 40.0, 10.0, 0.0], 0.03)
    state.log_force_metrics(env)
    for foot, value in zip(("fl", "fr", "hl", "hr"), [-30.0, 52.5, 0.0, 0.0], strict=True):
        torch.testing.assert_close(env.extras["log"][f"Metrics/foot_forces/{foot}_fz_mean"], torch.tensor(value))
    assert env.extras["log"]["existing"] == 1.0
    assert len(env.extras["log"]) == 5
    # Logging is read-only; a second read must neither add samples nor mutate the logged result.
    before = state.fz_integral.clone()
    state.log_force_metrics(env)
    torch.testing.assert_close(state.fz_integral, before)


def test_partial_reset_preserves_other_worlds_and_published_scalars():
    state = ProxyFootContactState(2, "cpu")
    env = SimpleNamespace(extras={})
    _sample(state, [10.0, -20.0, 30.0, 0.0], 0.02)
    state.log_force_metrics(env, [0])
    integral_other = state.fz_integral[1].clone()
    state.reset([0])
    assert state.force_duration[0] == 0
    assert not state.fz_integral[0].any()
    torch.testing.assert_close(state.fz_integral[1], integral_other)
    torch.testing.assert_close(env.extras["log"]["Metrics/foot_forces/fr_fz_mean"], torch.tensor(-20.0))
    # A world with no samples must not dilute the statistics of a world that has sampled forces.
    state.log_force_metrics(env)
    torch.testing.assert_close(env.extras["log"]["Metrics/foot_forces/fr_fz_mean"], torch.tensor(-40.0))
    state.reset()
    assert not state.force_duration.any()
    assert not state.fz_integral.any()


def test_initial_and_empty_resets_do_not_emit_nan_or_modify_unrelated_logs():
    state = ProxyFootContactState(2, "cpu")
    env = SimpleNamespace(extras={"log": {"existing": 1.0}})
    state.log_force_metrics(env, [])
    assert env.extras["log"] == {"existing": 1.0}
    state.log_force_metrics(SimpleNamespace())  # Standalone calibration/tests need no environment logger.
    state.log_force_metrics(env)
    assert all(torch.isfinite(value) and value == 0 for key, value in env.extras["log"].items() if key != "existing")
