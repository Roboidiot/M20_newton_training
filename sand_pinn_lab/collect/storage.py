"""Bounded-memory, atomic NPZ chunks. Completed chunks survive interruption."""

import json
from pathlib import Path

import numpy as np


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


class ChunkWriter:
    def __init__(self, directory, frames_per_chunk):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.frames_per_chunk = frames_per_chunk
        self.rows = []
        self.frames = 0
        self.chunks = []
        self.schema = None

    def append(self, row):
        for key, value in row.items():
            if not np.isfinite(value).all():
                raise FloatingPointError(f"Nonfinite {key} at frame {self.frames}")
        if self.schema is None:
            self.schema = {
                key: {"shape_per_frame": list(np.shape(value)), "dtype": str(np.asarray(value).dtype)}
                for key, value in row.items()
            }
            save_json(self.directory / "schema.json", self.schema)
        self.rows.append(row)
        self.frames += 1
        if len(self.rows) >= self.frames_per_chunk:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        index = len(self.chunks)
        filename = f"chunk_{index:05d}.npz"
        save_npz(self.directory / filename, **{key: np.stack([r[key] for r in self.rows]) for key in self.rows[0]})
        entry = {
            "file": filename,
            "frames": len(self.rows),
            "time_start_s": float(self.rows[0]["episode_time_start_s"]),
            "time_end_s": float(self.rows[-1]["episode_time_s"]),
        }
        self.chunks.append(entry)
        save_json(self.directory / "chunks.json", {"frames": self.frames, "chunks": self.chunks})
        self.rows.clear()
