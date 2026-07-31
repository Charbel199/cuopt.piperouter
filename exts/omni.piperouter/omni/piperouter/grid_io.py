"""Voxel-grid framing and .npz writer.

Writes exactly the keys `piperouter_solver.grids.GridStack.load` expects. The file on
the shared directory is the contract between the two sides, so the extension never
imports the solver package.
"""
from __future__ import annotations

import numpy as np


def frame_from_bounds(bounds_min, bounds_max, resolution: int):
    """Frame cubic cells sized as longest axis / resolution, padding bounds to fit.

    Mirrors piperouter_solver.models.GridFrame.from_bounds; the two must agree.
    """
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
    """Dilate a binary mask by `cells` iterations with 6-connectivity.

    Pure numpy because Kit's Python may not ship scipy. Grows the occupancy overlay by
    the safety clearance so the debug cloud matches the keep-out volume the solver
    routes against.
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
               thermal, em, clearance_class=None, clearance_values=None) -> None:
    extra = {}
    if clearance_class is not None and clearance_values:
        # Per-object clearance: class grid (0 = untagged), plus metres per class id,
        # indexed 1-based.
        extra["clearance_class"] = np.asarray(clearance_class, dtype=np.uint8)
        extra["clearance_values"] = np.asarray(clearance_values, dtype=np.float64)
    # Uncompressed on purpose: the file lives on tmpfs (/dev/shm), so zlib would burn
    # CPU on both sides of the handoff for no I/O benefit.
    np.savez(
        path,
        bounds_min=np.asarray(bounds_min, dtype=np.float64),
        cell_size=np.float64(cell_size),
        res_xyz=np.asarray(res_xyz, dtype=np.int64),
        occupancy=np.asarray(occupancy, dtype=np.uint8),
        surface_dist=np.asarray(surface_dist, dtype=np.float32),
        thermal=np.asarray(thermal, dtype=np.float32),
        em=np.asarray(em, dtype=np.float32),
        **extra,
    )
