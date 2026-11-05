"""Read-only integrity check: python -m sand_pinn_lab.collect.validate <run-directory>."""

import argparse
import json
from pathlib import Path

import numpy as np


def validate_run(directory):
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("schema_version", 1) >= 2:
        assert metadata["particle_state_saved"] is False
        assert not list(directory.rglob("particles_*.npz"))
        assert not list(directory.rglob("particle_properties.npz"))
        assert not list(directory.rglob("initial_sand.npz"))
        assert not list(directory.rglob("final_sand.npz"))
    total = 0
    complete = 0
    for episode in metadata["episodes"]:
        folder = directory / episode["directory"]
        manifest = json.loads((folder / "chunks.json").read_text())
        frames = 0
        for chunk in manifest["chunks"]:
            if metadata.get("schema_version", 1) >= 2:
                assert "particles_file" not in chunk
            with np.load(folder / chunk["file"], allow_pickle=False) as data:
                n = len(data["episode_time_s"])
                assert n == chunk["frames"], f"Chunk count: {folder / chunk['file']}"
                for key in data.files:
                    assert len(data[key]) == n and np.isfinite(data[key]).all(), key
                np.testing.assert_array_equal(data["frame_index"], np.arange(frames, frames + n))
                np.testing.assert_array_equal(data["episode_id"], episode["episode_id"])
                dt = metadata["physics_dt_s"]
                np.testing.assert_allclose(data["episode_time_start_s"], np.arange(frames, frames + n) * dt)
                np.testing.assert_allclose(data["episode_time_s"], np.arange(frames + 1, frames + n + 1) * dt)
                np.testing.assert_allclose(
                    data["total_mpm_force_w_N"], data["body_mpm_force_w_N"].sum(axis=1), rtol=1e-5, atol=1e-4
                )
                for key in ("body_pose_w", "mpm_proxy_body_pose_w"):
                    np.testing.assert_allclose(np.linalg.norm(data[key][..., 3:], axis=-1), 1, atol=1e-3)
                assert data["wheel_sinkage_reference_m"].shape == (n, 4)
                np.testing.assert_allclose(
                    data["wheel_sinkage_reference_m"],
                    data["reference_surface_height_w_m"][:, None] - data["wheel_bottom_w_m"][..., 2],
                    atol=1e-6,
                )
                frames += n
            if "particles_file" in chunk:
                with np.load(folder / chunk["particles_file"], allow_pickle=False) as particles:
                    assert np.all(np.diff(particles["episode_time_s"]) > 0)
                    assert np.all(particles["episode_time_s"] > chunk["time_start_s"])
                    assert np.all(particles["episode_time_s"] <= chunk["time_end_s"] + 1e-8)
                    for key in particles.files:
                        assert np.isfinite(particles[key]).all(), key
        assert frames == manifest["frames"] == episode["frames"]
        if episode["status"] == "complete":
            assert frames == round(metadata["episode_seconds"] / metadata["physics_dt_s"])
            complete += 1
        total += frames
    if metadata["status"] == "complete":
        assert complete == metadata["requested_episodes"]
    assert total == metadata["saved_frames"]
    return {
        "status": metadata["status"],
        "complete_episodes": complete,
        "frames": total,
        "recorded_seconds": total * metadata["physics_dt_s"],
        "integrity": "passed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(json.dumps(validate_run(args.directory), indent=2))


if __name__ == "__main__":
    main()
