"""Read-only normalization of the known duplicate trajectory-endpoint record."""

from __future__ import annotations

import numpy as np

from .config import COL, COLUMNS


def remove_repeated_terminal_sample(data: np.ndarray) -> tuple[np.ndarray, int]:
    """Keep the first endpoint sample; reject repeated/decreasing times elsewhere.

    Older collection used ceil(total / dt), which sometimes added a second solver
    step at the same clamped endpoint because total / dt rounded above an integer.
    Force values at that extra step need not be identical, so do not average them.
    The input array and the source NPZ remain unchanged.
    """
    if data.ndim != 2 or data.shape[1] != len(COLUMNS) or len(data) < 2 or not np.isfinite(data).all():
        raise ValueError("Invalid force-history data")
    repeats = np.flatnonzero(np.diff(data[:, COL["time"]]) <= 1e-9) + 1
    if not len(repeats):
        return data, 0
    if (
        len(repeats) != 1
        or repeats[0] != len(data) - 1
        or not np.isclose(data[-1, COL["time"]], data[-2, COL["time"]], atol=1e-9, rtol=0)
        or not np.allclose(data[-2:, COL["progress"]], 1, atol=1e-9, rtol=0)
        or not np.allclose(data[-1, 1:10], data[-2, 1:10], atol=1e-9, rtol=0)
    ):
        raise ValueError("Invalid time sequence: only one repeated identical trajectory endpoint is supported")
    return data[:-1], 1
