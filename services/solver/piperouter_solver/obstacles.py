"""Rasterize routed polylines into a boolean obstacle grid, so a wire being refined
can treat already-placed (locked) tubes as obstacles."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def rasterize_polylines(frame, routes) -> np.ndarray:
    """routes = [{"polyline": [[x,y,z], ...], "radius_m": float}]. Returns a bool grid
    (frame.res_xyz) with the swept tubes marked, each dilated by its radius."""
    nx, ny, nz = frame.res_xyz
    total = np.zeros((nx, ny, nz), dtype=bool)
    st = ndimage.generate_binary_structure(3, 1)
    for r in routes:
        poly = np.asarray(r.get("polyline", []), dtype=float)
        if poly.shape[0] == 0:
            continue
        m = np.zeros((nx, ny, nz), dtype=bool)
        samples = []
        if poly.shape[0] == 1:
            samples.append(poly[0])
        for a, b in zip(poly[:-1], poly[1:]):
            seg = b - a
            length = float(np.linalg.norm(seg))
            n = max(1, int(length / (frame.cell_size * 0.5)) + 1)
            for t in np.linspace(0.0, 1.0, n + 1):
                samples.append(a + seg * t)
        for p in samples:
            i, j, k = frame.world_to_grid(p)
            m[i, j, k] = True
        rad = float(r.get("radius_m", 0.0))
        d = int(rad / frame.cell_size + 0.5 + 1e-9)
        if d > 0:
            m = ndimage.binary_dilation(m, structure=st, iterations=d)
        total |= m
    return total
