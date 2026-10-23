"""Positive-coefficient sliding contact response fitted to whole-robot MPM traces."""

import torch


def sliding_response(
    depth, velocity, parameters, *, stability_mass_kg=None, dt=0.0025, bearing_scale=1.0, drag_scale=1.0
):
    """Return yaw-frame force: passive horizontal/vertical drag plus residual bearing.

    Negative vertical forces during upward sliding are frictional drag. They do
    not imply tensile bearing or an adhesive normal spring.
    """
    p = parameters
    x, y, z = velocity.unbind(-1)
    cx = drag_scale * p["cx"] * (-torch.expm1(-depth / p["dx"])) / torch.sqrt(1 + (x / p["satx"]).square())
    cy = drag_scale * p["cy"] * (-torch.expm1(-depth / p["dy"])) / torch.sqrt(1 + (y / p["saty"]).square())
    power = depth.pow(p["nz"])
    cz = drag_scale * p["cz"] * power / (power + p["dz"] ** p["nz"])
    if stability_mass_kg is not None:
        limit = 0.5 * stability_mass_kg / dt
        cx, cy, cz = (torch.minimum(value, limit) for value in (cx, cy, cz))
    bearing = bearing_scale * p["bearing"] * (-torch.expm1(-depth / p["db"])) * torch.exp(-depth / p["soft"])
    return torch.stack([-cx * x, -cy * y, bearing - cz * z], dim=-1)
