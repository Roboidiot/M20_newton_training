#!/usr/bin/env python3
"""Record an MPM policy with a sand surface or particles, isolated from training."""

from __future__ import annotations

import argparse
import atexit
import importlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_ENV = "M20_MPM_SURFACE_RECORDING"
TASK = "M20NewtonTraining-M20-M20-WalkOnSand"


def frame_due(tick: int, fps: int, physics_hz: int) -> bool:
    """Sample real physics states without duplicating frames or speeding playback."""
    return tick * fps // physics_hz > (tick - 1) * fps // physics_hz


def install_streaming_recorder(spec):
    """Replace the native RAM-buffered recorder only in this playback subprocess."""
    import imageio_ffmpeg

    import isaaclab.envs.manager_based_env as env_module
    from isaaclab.envs.utils.video_recorder import VideoRecorder
    from isaaclab_newton.physics import NewtonManager

    class PhysicsFrameRecorder(VideoRecorder):
        def __init__(self, cfg, env):
            super().__init__(cfg, env)
            self._writer = None
            self._armed = False
            self._ticks = 0
            self._written = 0
            self._finished = False
            self._physics_hz = round(1.0 / env.sim.get_physics_dt())
            if not math.isclose(self._physics_hz * env.sim.get_physics_dt(), 1.0):
                raise ValueError("Recording requires an integer physics frequency.")
            if spec["fps"] > self._physics_hz:
                raise ValueError("Output FPS must not exceed the physics frequency.")
            self._target_frames = round(spec["seconds"] * spec["fps"])
            original_env_step = env.step
            if env.cfg.sim.physics.use_cuda_graph:
                raise ValueError("Per-physics-step capture requires use_cuda_graph=False.")
            original_substeps = NewtonManager._run_solver_substeps.__func__

            def env_step(*args, **kwargs):
                # Initialization/reset settling steps outside env.step are excluded.
                self._armed = True
                try:
                    return original_env_step(*args, **kwargs)
                finally:
                    self._armed = False

            def solver_substeps(manager, contacts):
                # Newton batches four 5ms ticks inside sim.step(). Observe the
                # existing loop rather than changing its decimation or forces.
                result = original_substeps(manager, contacts)
                if self._armed and not self._finished:
                    self._ticks += 1
                    if frame_due(self._ticks, spec["fps"], self._physics_hz):
                        manager._mark_transforms_changed()
                        self._write_frame()
                return result

            env.step = env_step
            # This private hook is specific to the installed IsaacLab 3.0 Newton
            # implementation; it exists only for the life of this subprocess.
            NewtonManager._run_solver_substeps = classmethod(solver_substeps)
            atexit.register(self.close)

        def _write_frame(self):
            # Reuse native state synchronization, marker dispatch and capture.
            frame = self._frame_from_visualizer(spec["visualizer"], "")
            if frame is None:
                raise RuntimeError("The visualizer returned no video frame.")
            if spec["supersample"] > 1:
                import numpy as np
                from PIL import Image

                frame = np.asarray(Image.fromarray(frame).resize(
                    (spec["width"], spec["height"]), Image.Resampling.LANCZOS
                ))
            if frame.shape != (spec["height"], spec["width"], 3):
                raise RuntimeError(f"Unexpected rendered frame shape: {frame.shape}")
            if self._writer is None:
                self._path = self._clip_path(self._clip_index)
                self._writer = imageio_ffmpeg.write_frames(
                    self._path, (spec["width"], spec["height"]), fps=spec["fps"],
                    codec="libx264", pix_fmt_out="yuv420p", macro_block_size=1,
                    output_params=["-crf", str(spec["crf"]), "-preset", spec["preset"]],
                )
                self._writer.send(None)
            self._writer.send(frame)
            self._written += 1
            if self._written >= self._target_frames:
                self.close()

        def step(self):
            # Native playback still uses VideoRecorderCfg.video_length as its budget.
            pass

        def close(self):
            self._finished = True
            if self._writer is not None:
                self._writer.close()
                self._writer = None
                Path(spec["output_dir"], "encoding_statistics.json").write_text(json.dumps({
                    "path": self._path, "frames": self._written, "fps": spec["fps"],
                    "duration": self._written / spec["fps"], "physics_hz": self._physics_hz,
                }, indent=2) + "\n")
                print(f"[Video] Wrote {self._written} frames: {self._path}", flush=True)

    env_module.VideoRecorder = PhysicsFrameRecorder


def configure_recording():
    """Native playback callback: configure only this process's camera and command."""
    import gymnasium as gym

    importlib.import_module("M20_newton_training.tasks")
    from isaaclab_visualizers.newton import NewtonGLVisualizerCfg, NewtonRTXVisualizerCfg

    from isaaclab.envs.utils.video_recorder_cfg import VideoRecorderCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import replace

    from M20_newton_training.tasks.m20.mdp.single_axis_velocity import SingleAxisVelocityCommandCfg

    spec = json.loads(Path(os.environ[SPEC_ENV]).read_text())
    install_streaming_recorder(spec)
    entry = gym.spec(TASK).kwargs["env_cfg_entry_point"]
    module, symbol = entry.split(":")
    env_class = getattr(importlib.import_module(module), symbol)
    original_play_mode = env_class.play_mode

    def recording_play_mode(cfg):
        original_play_mode(cfg)
        cfg.viewer.eye = tuple(spec["eye"])
        cfg.viewer.lookat = tuple(spec["lookat"])
        command = cfg.commands.base_velocity
        cfg.commands.base_velocity = SingleAxisVelocityCommandCfg(
            asset_name=command.asset_name,
            ranges=replace(
                command.ranges,
                lin_vel_x=(spec["speed"], spec["speed"]),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(0.0, 0.0),
                heading=None,
            ),
            min_abs_command=0.01,
            heading_command=False,
            rel_standing_envs=0.0,
            resampling_time_range=(1.0e6, 1.0e6),
            debug_vis=False,
        )
        visualizer_class = NewtonGLVisualizerCfg if spec["renderer"] == "gl" else NewtonRTXVisualizerCfg
        renderer_options = {} if spec["renderer"] == "gl" else {
            "rtx_environment": "studio",
            "render_settings": {"omni:rtx:quality": ("Int", spec["rtx_quality"])},
        }
        cfg.sim.visualizer_cfgs = [visualizer_class(
            headless=True,
            window_width=spec["width"] * spec["supersample"],
            window_height=spec["height"] * spec["supersample"],
            eye=cfg.viewer.eye,
            lookat=cfg.viewer.lookat,
            focal_length=spec["focal_length"],
            show_particles=spec["sand_display"] == "particles",
            particle_color=(0.65, 0.49, 0.28),
            enable_picking=False,
            streaming_view=False,
            enable_shadows=True,
            light_color=(0.85, 0.85, 0.85),
            background_color=(0.78, 0.84, 0.90),
            **renderer_options,
        )]
        cfg.video_recorders = [VideoRecorderCfg(
            source=f"visualizer:{spec['visualizer']}",
            output_dir=spec["output_dir"],
            output_filename_prefix=f"mpm_sand_{spec['sand_display']}",
            video_length=spec["steps"],
            video_interval=0,
            fps=spec["fps"],
        )]

    env_class.play_mode = recording_play_mode

    def initialize_surface(sim):
        from isaaclab_newton.physics import NewtonManager

        model = NewtonManager.get_model()
        if not model.particle_count:
            raise RuntimeError("No MPM particles exist; select physics=newton_sand.")
        renderer = None
        if spec["sand_display"] == "surface":
            from isaaclab_visualizers.newton import ParticleSurfaceRenderer
            from newton.geometry import ParticleSurface

            surface = ParticleSurface(
                voxel_size=spec["surface_voxel"],
                kernel_radius=spec["surface_kernel"],
                threshold=0.25,
                smooth_lambda=0.0,
                anisotropic=False,
                field_smooth_iterations=1,
                mesh_smooth_iterations=1,
                max_grid_cells=spec["surface_max_cells"],
                world_count=max(model.world_count, 1),
                device=model.device,
            )
            renderer = ParticleSurfaceRenderer(
                sim.visualizers,
                surface,
                path="/Recording/SandSurface",
                color=(0.65, 0.49, 0.28),
                opacity=1.0,
                roughness=0.95,
                use_cuda_graph=False,
            )
        statistics = {
            "mode": spec["sand_display"], "particles": int(model.particle_count),
            "frames": 0, "max_triangles": 0,
        }

        # Headless playback renders on demand in render_rgb_array(), bypassing
        # SimulationContext.update_visualizers(). Rebuild before that actual frame.
        for visualizer in sim.visualizers:
            original_capture = visualizer.render_rgb_array

            def capture_surface(original_capture=original_capture):
                triangles = renderer.update() if renderer is not None else 0
                statistics["frames"] += 1
                statistics["max_triangles"] = max(statistics["max_triangles"], triangles)
                if statistics["frames"] == 1 or statistics["frames"] % 100 == 0:
                    print(
                        f"[Sand {spec['sand_display']}] frame={statistics['frames']} triangles={triangles}", flush=True
                    )
                return original_capture()

            visualizer.render_rgb_array = capture_surface
        atexit.register(lambda: Path(spec["output_dir"], "surface_statistics.json").write_text(
            json.dumps(statistics, indent=2) + "\n"
        ))
        print(f"[Sand {spec['sand_display']}] Displaying {model.particle_count} MPM particles.", flush=True)

    SimulationContext.add_reset_callback("m20_record_sand_surface", initialize_surface)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--speed", type=float, default=0.2)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=50, help="Real-state sampling FPS, 1–200; policy stays 50Hz.")
    parser.add_argument("--supersample", type=int, choices=(1, 2), default=1,
                        help="Render at 1x/2x each dimension, then Lanczos downsample.")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF, 0–51; lower is better/larger.")
    parser.add_argument("--preset", choices=("fast", "medium", "slow"), default="medium")
    parser.add_argument("--renderer", choices=("gl", "rtx"), default="gl")
    parser.add_argument("--sand-display", choices=("surface", "particles"), default="surface",
                        help="Display a reconstructed surface or native MPM particles without meshing.")
    parser.add_argument("--rtx-quality", type=int, default=32)
    parser.add_argument("--surface-voxel", type=float, default=0.02, help="Visual reconstruction voxel size in m.")
    parser.add_argument("--surface-kernel", type=float, default=0.045, help="Visual smoothing kernel radius in m.")
    parser.add_argument("--surface-max-cells", type=int, default=None)
    parser.add_argument("--eye", nargs=3, type=float, default=(2.8, -3.2, 2.5), metavar=("X", "Y", "Z"))
    parser.add_argument("--lookat", nargs=3, type=float, default=(0.0, 0.0, 0.25), metavar=("X", "Y", "Z"))
    parser.add_argument("--focal-length", type=float, default=12.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.renderer == "rtx" and importlib.util.find_spec("ovrtx") is None:
        parser.error("RTX rendering requires the optional ovrtx package; use --renderer gl in this environment.")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        parser.error(f"Checkpoint does not exist: {checkpoint}")
    if args.seconds <= 0 or args.speed < 0 or min(args.width, args.height) <= 0:
        parser.error("Duration and resolution must be positive; forward speed must be nonnegative.")
    if args.width % 2 or args.height % 2:
        parser.error("H.264 yuv420p requires even width and height.")
    if not 1 <= args.fps <= 200 or not 0 <= args.crf <= 51:
        parser.error("FPS must be 1–200 and CRF must be 0–51.")
    if min(args.surface_voxel, args.surface_kernel, args.focal_length, args.rtx_quality) <= 0:
        parser.error("Surface, focal length and RTX quality parameters must be positive.")
    if args.surface_max_cells is not None and args.surface_max_cells <= 0:
        parser.error("Surface cell capacity must be positive.")
    steps = round(args.seconds * 50)
    if steps < 1:
        parser.error("Duration must cover at least one 20ms control step.")
    if steps * args.fps % 50:
        parser.error("Choose a duration yielding an integer frame count at the requested FPS.")
    output = (args.output_dir or ROOT / "output" / "mpm_surface_video" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    output = output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    spec = {
        "checkpoint": str(checkpoint), "seconds": steps / 50, "steps": steps,
        "speed": args.speed, "width": args.width, "height": args.height,
        "fps": args.fps, "supersample": args.supersample, "crf": args.crf, "preset": args.preset,
        "renderer": args.renderer, "visualizer": "newton_gl" if args.renderer == "gl" else "newton_rtx",
        "sand_display": args.sand_display,
        "rtx_quality": args.rtx_quality, "surface_voxel": args.surface_voxel,
        "surface_kernel": args.surface_kernel,
        "surface_max_cells": args.surface_max_cells or math.ceil(750000 * max(1, (0.02 / args.surface_voxel)**3)),
        "eye": args.eye, "lookat": args.lookat, "focal_length": args.focal_length,
        "output_dir": str(output),
    }
    spec_path = output / "recording.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")
    command = [
        sys.executable, "-m", "isaaclab", "play",
        "--rl_library", "rsl_rl", "--task", TASK, "--num_envs", "1",
        "--viz", spec["visualizer"], "--checkpoint", str(checkpoint),
        "--video", "--video_length", str(steps),
        "--external_callback", "record_mpm_surface.configure_recording",
        "physics=newton_sand",
    ]
    (output / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    environment = os.environ.copy()
    # Select EGL before pyglet imports; mixing an X display with headless GL fails.
    environment.pop("DISPLAY", None)
    environment[SPEC_ENV] = str(spec_path)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONPATH"] = str(ROOT / "scripts") + os.pathsep + environment.get("PYTHONPATH", "")
    print(f"Recording {spec['seconds']} seconds to {output}", flush=True)
    with (output / "recording.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            status = process.wait()
            if status == 0:
                statistics = json.loads((output / "surface_statistics.json").read_text())
                if spec["sand_display"] == "surface" and statistics["max_triangles"] <= 0:
                    raise RuntimeError("Recording produced no reconstructed sand surface.")
                encoded = json.loads((output / "encoding_statistics.json").read_text())
                if encoded["frames"] != round(spec["seconds"] * spec["fps"]):
                    raise RuntimeError("Recording stopped before the requested frame count.")
            return status
        except KeyboardInterrupt:
            process.terminate()
            process.wait()
            raise


if __name__ == "__main__":
    raise SystemExit(main())
