"""Shared terrain-first artifact layout for the independent sand-proxy pipeline."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TERRAINS_ROOT = ROOT / "output/sand_proxy/terrains"
DEFAULT_TERRAIN = "m20_sand_v1"


def terrain_name(value: str) -> str:
    if not re.fullmatch(r"\w[\w.-]*", value):
        raise argparse.ArgumentTypeError("Terrain must be a single name, e.g. sand_soft_v1, not a path")
    return value


def add_terrain_argument(parser):
    parser.add_argument(
        "--terrain",
        type=terrain_name,
        help=f"Terrain directory name; infer from input path when possible, otherwise {DEFAULT_TERRAIN}",
    )


def inferred_terrain(path: str | Path | None) -> str | None:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
        for parent in (resolved, *resolved.parents):
            if parent.parent == TERRAINS_ROOT:
                return parent.name
    return None


def terrain_directory(name: str | None = None, source: str | Path | None = None) -> Path:
    """An explicit name wins; otherwise keep downstream outputs with their source terrain."""
    return TERRAINS_ROOT / terrain_name(name or inferred_terrain(source) or DEFAULT_TERRAIN)


def backbone_directory(data_directory: Path) -> Path:
    data_directory = Path(data_directory).resolve()
    terrain = inferred_terrain(data_directory)
    if terrain is not None and data_directory == terrain_directory(terrain) / "intruder":
        return terrain_directory(terrain) / "fit/backbone"
    # Explicit custom dataset paths continue to keep their outputs next to their data.
    return data_directory / "backbone_fit"
