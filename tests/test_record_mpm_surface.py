"""Video timing must follow physics ticks, including Newton's internal decimation."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import imageio_ffmpeg
import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "record_mpm_surface_test", Path(__file__).parents[1] / "scripts/record_mpm_surface.py"
)
record = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(record)


@pytest.mark.parametrize("fps", [25, 50, 60, 100, 200])
def test_streamed_frames_follow_physics_not_control_steps(monkeypatch, tmp_path, fps):
    modules = {}
    for name in ["isaaclab", "isaaclab.envs", "isaaclab.envs.manager_based_env",
                 "isaaclab.envs.utils", "isaaclab.envs.utils.video_recorder",
                 "isaaclab_newton", "isaaclab_newton.physics"]:
        modules[name] = ModuleType(name)
        monkeypatch.setitem(sys.modules, name, modules[name])
        parent, _, child = name.rpartition(".")
        if parent in modules:
            setattr(modules[parent], child, modules[name])

    class Manager:
        tick = 0

        @classmethod
        def _run_solver_substeps(cls, contacts):
            cls.tick += 1

        @classmethod
        def _mark_transforms_changed(cls):
            pass

    class NativeRecorder:
        def __init__(self, cfg, env):
            self._env = env
            self._clip_index = 0

        def _clip_path(self, index):
            return str(tmp_path / "video.mp4")

        def _frame_from_visualizer(self, *args):
            return np.full((2, 2, 3), Manager.tick, dtype=np.uint8)

    modules["isaaclab.envs.utils.video_recorder"].VideoRecorder = NativeRecorder
    modules["isaaclab_newton.physics"].NewtonManager = Manager
    frames = []

    def write_frames(*args, **kwargs):
        while True:
            frames.append((yield).copy())

    monkeypatch.setattr(imageio_ffmpeg, "write_frames", write_frames)

    class Env:
        cfg = SimpleNamespace(sim=SimpleNamespace(physics=SimpleNamespace(use_cuda_graph=False)))
        sim = SimpleNamespace(get_physics_dt=lambda: 0.005)

        def step(self):
            # One sim.step() contains four ticks; control frequency stays 50Hz.
            for _ in range(4):
                Manager._run_solver_substeps(None)

    spec = dict(seconds=0.2, fps=fps, supersample=1, width=2, height=2,
                visualizer="newton_gl", crf=18, preset="medium", output_dir=str(tmp_path))
    record.install_streaming_recorder(spec)
    env = Env()
    recorder = modules["isaaclab.envs.manager_based_env"].VideoRecorder(None, env)
    # Steps for initialization must not appear in the recording.
    Manager._run_solver_substeps(None)
    for _ in range(10):
        env.step()
        recorder.step()
    recorder.close()
    assert len(frames) == round(0.2 * fps)
    assert len(set(int(frame[0, 0, 0]) for frame in frames)) == len(frames)
    assert Manager.tick == 41
    stats = json.loads((tmp_path / "encoding_statistics.json").read_text())
    assert stats["duration"] == 0.2
    assert stats["physics_hz"] == 200


def test_sixty_fps_samples_on_existing_five_ms_grid():
    ticks = [tick for tick in range(1, 201) if record.frame_due(tick, 60, 200)]
    assert len(ticks) == 60
    assert set(np.diff(ticks)) == {3, 4}  # 15/20ms intervals, no synthetic frames.
