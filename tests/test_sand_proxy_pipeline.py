"""Active pipeline independence, frozen case planning, geometry and episode export."""

import hashlib
import json
import signal
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sand_proxy_lab.calibration.common import particle_grid
from sand_proxy_lab.calibration.config import cases, duration, sample_times, validate_config
from sand_proxy_lab.shadow.episode import extract_episode
from sand_proxy_lab.shadow.random_agent import pause_signals

ROOT = Path(__file__).resolve().parents[1]


def test_frozen_foot_only_suite_and_terminal_times():
    config = json.loads((ROOT / "sand_proxy_lab/configs/m20_foot_intruder.json").read_text())
    validate_config(config)
    suite = cases(config)
    assert len(suite) == 179 and {c.geometry for c in suite} == {"foot"}
    assert Counter(c.split for c in suite) == {"train": 114, "validation": 65}
    for c in suite:
        times = sample_times(config, c)
        assert np.all(np.diff(times) > 0)
        assert times[-1] == pytest.approx(duration(config, c))
    points, mass, _ = particle_grid(config, 0)
    assert points.shape == (168000, 3) and mass > 0


def test_active_source_has_no_import_or_default_path_into_retired_pipeline():
    for base in (ROOT / "sand_proxy_lab", ROOT / "src/M20_newton_training/sand_proxy"):
        for p in base.rglob("*.py"):
            text = p.read_text()
            for retired in (
                "from sand_calibration",
                "import sand_calibration",
                "experiments.sand_shadow",
                "M20_newton_training.sand.",
                "outputs/m20_sand_calibration",
                "output/sand_shadow",
            ):
                assert retired not in text, (p, retired)
    from M20_newton_training.sand_proxy.foot_model import DEFAULT_MODEL_PATH, FootSandModel

    assert DEFAULT_MODEL_PATH.is_file()
    assert FootSandModel.load(num_envs=1).parameters["schema_version"] == 2


def test_pause_signal_sets_save_request_without_raising_inside_solver():
    previous = signal.getsignal(signal.SIGINT)
    with pause_signals() as state:
        signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        assert state == {"requested": True, "signal": signal.SIGINT}
    assert signal.getsignal(signal.SIGINT) == previous


def test_episode_export_keeps_forces_and_source_bytes_and_rebases_time(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    force = np.arange(36).reshape(3, 4, 3)
    meta = {"episodes": [{"episode_id": 0}, {"episode_id": 1, "surface_height_m": 0.3}], "warmup_s": 0.5}
    path = source / "traces.npz"
    np.savez_compressed(
        path,
        episode_id=[0, 1, 1],
        time_s=[1, 6.02, 6.04],
        dt_s=[0.02] * 3,
        episode_time_s=[1, 0.02, 0.04],
        control_step=[49, 300, 301],
        mpm_force_w_N=force,
        proxy_force_w_N=force + 1,
        metadata=json.dumps(meta),
    )
    (source / "model_snapshot.json").write_text("{}\n")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    result = extract_episode(path, 1, tmp_path / "export")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    with np.load(result, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["mpm_force_w_N"], force[1:])
        np.testing.assert_array_equal(saved["time_s"], [0.02, 0.04])
        np.testing.assert_array_equal(saved["control_step"], [0, 1])
        np.testing.assert_array_equal(saved["episode_id"], [0, 0])
        assert json.loads(saved["metadata"].item())["source_episode_id"] == 1
