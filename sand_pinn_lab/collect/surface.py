"""Particle-column surface estimates, with explicit masks for empty/outside cells."""

import numpy as np


def reference_surface(points, spacing):
    points = np.asarray(points)
    p = points[(np.abs(points[:, 0]) < 0.3) & (np.abs(points[:, 1]) < 0.3)]
    if not len(p) or not np.isfinite(p).all():
        raise ValueError("No finite sand particles in the central reference patch")
    _, inverse = np.unique(np.floor(p[:, :2] / (2 * spacing)).astype(int), axis=0, return_inverse=True)
    tops = np.full(inverse.max() + 1, -np.inf)
    np.maximum.at(tops, inverse, p[:, 2])
    return float(np.median(tops) + spacing / 2)


class LocalSurface:
    """Median of column tops in an annulus around each wheel, NOT a reconstructed interface."""

    def __init__(self, device, lower_xy=(-1.2, -1.2), upper_xy=(1.2, 1.2), cell=0.04, patch_size=13):
        import torch

        self.cell = cell
        self.size = np.ceil((np.asarray(upper_xy) - lower_xy) / cell).astype(int)
        self.lower = torch.tensor(lower_xy, dtype=torch.float32, device=device)
        offsets = torch.arange(-(patch_size // 2), patch_size // 2 + 1, device=device)
        self.offsets = torch.stack(torch.meshgrid(offsets, offsets, indexing="ij"), dim=-1)
        self.patch_size = patch_size

    def sample(self, particles, radius, wheel_xy, inner_radius=0.15, outer_radius=0.24):
        import torch

        if not torch.isfinite(particles).all():
            raise FloatingPointError("Nonfinite sand particle positions")
        indices = torch.floor((particles[:, :2] - self.lower) / self.cell).long()
        nx, ny = map(int, self.size)
        inside = ((indices >= 0) & (indices < indices.new_tensor([nx, ny]))).all(dim=1)
        tops = torch.full((nx * ny,), -torch.inf, device=particles.device)
        flat = indices[:, 0] * ny + indices[:, 1]
        tops.scatter_reduce_(0, flat[inside], (particles[:, 2] + radius)[inside], reduce="amax", include_self=True)
        center_index = torch.floor((wheel_xy - self.lower) / self.cell).long()
        patch_index = center_index[:, None, None, :] + self.offsets
        valid = ((patch_index >= 0) & (patch_index < patch_index.new_tensor([nx, ny]))).all(dim=-1)
        flat_patch = (patch_index[..., 0] * ny + patch_index[..., 1]).clamp(0, nx * ny - 1)
        patch = tops[flat_patch]
        valid &= torch.isfinite(patch)
        xy = self.lower + (patch_index + 0.5) * self.cell
        distance = torch.linalg.vector_norm(xy - wheel_xy[:, None, None, :], dim=-1)
        ring = valid & (distance >= inner_radius) & (distance <= outer_radius)
        samples = torch.where(ring, patch, torch.nan).flatten(1)
        count = ring.sum(dim=(1, 2))
        height = samples.nanmedian(dim=1).values
        height_valid = (count >= 3) & torch.isfinite(height)
        mid = self.patch_size // 2
        return {
            "wheel_surface_height_w_m": torch.where(height_valid, height, 0),
            "wheel_surface_valid": height_valid,
            "wheel_surface_column_count": count,
            "wheel_column_top_w_m": torch.where(valid[:, mid, mid], patch[:, mid, mid], 0),
            "wheel_column_top_valid": valid[:, mid, mid],
            "wheel_surface_patch_w_m": torch.where(valid, patch, 0),
            "wheel_surface_patch_valid": valid,
            "wheel_surface_patch_center_xy_m": self.lower + (center_index + 0.5) * self.cell,
            "sand_particles_outside_surface_grid": (~inside).sum(),
        }
