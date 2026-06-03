"""Voxel-grid framing + .npz writer.

Writes exactly the keys `piperouter_solver.grids.GridStack.load` expects, so the
extension never needs to import the solver package — the file on the shared dir is
the contract.
"""
from __future__ import annotations

import numpy as np


def frame_from_bounds(bounds_min, bounds_max, resolution: int):
    """Cubic cells sized from the longest axis / resolution, bounds padded to fit
    evenly. Mirrors piperouter_solver.models.GridFrame.from_bounds."""
    bmin = np.asarray(bounds_min, dtype=np.float64)
    bmax = np.asarray(bounds_max, dtype=np.float64)
    extent = bmax - bmin
    longest = float(np.max(extent))
    cell = longest / resolution
    res = np.maximum(1, np.ceil(extent / cell).astype(int))
    grid_extent = res * cell
    padding = (grid_extent - extent) * 0.5
    bmin_padded = bmin - padding
    return bmin_padded, float(cell), (int(res[0]), int(res[1]), int(res[2]))


def dilate_mask(mask, cells):
    """6-connectivity binary dilation by `cells` iterations, in pure numpy (Kit's
    Python may not ship scipy). Used to grow the occupancy overlay by the safety
    clearance so the debug cloud matches the keep-out volume the solver routes against.
    """
    out = np.asarray(mask, dtype=bool)
    for _ in range(int(cells)):
        d = out.copy()
        d[1:, :, :] |= out[:-1, :, :]
        d[:-1, :, :] |= out[1:, :, :]
        d[:, 1:, :] |= out[:, :-1, :]
        d[:, :-1, :] |= out[:, 1:, :]
        d[:, :, 1:] |= out[:, :, :-1]
        d[:, :, :-1] |= out[:, :, 1:]
        out = d
    return out


def save_grids(path, bounds_min, cell_size, res_xyz, occupancy, surface_dist,
               thermal, em) -> None:
    np.savez_compressed(
        path,
        bounds_min=np.asarray(bounds_min, dtype=np.float64),
        cell_size=np.float64(cell_size),
        res_xyz=np.asarray(res_xyz, dtype=np.int64),
        occupancy=np.asarray(occupancy, dtype=np.uint8),
        surface_dist=np.asarray(surface_dist, dtype=np.float32),
        thermal=np.asarray(thermal, dtype=np.float32),
        em=np.asarray(em, dtype=np.float32),
    )
