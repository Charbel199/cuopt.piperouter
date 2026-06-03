"""Thermal / EM scalar fields, splatted from tagged prims into the voxel grid.

WHAT THESE ARE
--------------
Two of the four grids the router consumes (the other two are occupancy + surface
distance, produced by the voxelizer). Each is a dense float array, one value per cell:

  * thermal[i,j,k] = estimated temperature (°C) at that cell.
  * em[i,j,k]      = estimated EM field strength (0..1-ish) at that cell.

HOW THEY DRIVE ROUTING
----------------------
On the solver side (piperouter_solver):
  * thermal is BOTH a soft cost (warm cells cost more, scaled by the wire's "thermal"
    slider) AND a hard cutoff (cells hotter than the wire's max_temp_c are removed).
  * em is a soft cost scaled by the wire's "em" slider AND the wire type's
    em_sensitivity (a CAN signal cares about EM; an AC pipe doesn't).

THE PHYSICS (deliberately simple, phase-2 hook)
-----------------------------------------------
Each tagged source radiates its value, falling off LINEARLY to ambient over a
characteristic distance `falloff_m`:

      contribution(cell) = value * max(0, 1 - distance(cell, source) / falloff_m)

Thermal sources take the MAX over sources above an ambient floor (the hottest source
wins, like real radiant heat); EM sources SUM (multiple emitters add up). A real
conduction / EM solve can replace this without changing the grid contract.

IMPORTANT: each source carries its OWN falloff (see router_session) so a big hot
engine block heats a much larger region than a small connector — a single global
falloff made heat invisible once the sample scene was scaled up.
"""
from __future__ import annotations

import numpy as np


def splat_field(bounds_min, cell_size, res_xyz, sources, ambient=0.0, mode="max"):
    """Build a (ri,rj,rk) field by splatting point sources.

    Args:
        bounds_min: world-space origin of the grid (3,).
        cell_size: edge length of a (cubic) cell, world units.
        res_xyz: (ri, rj, rk) cell counts.
        sources: list of (world_pos (3,), value, falloff_m). `falloff_m` is the
            distance over which `value` decays linearly to zero contribution.
        ambient: baseline value every cell starts at.
        mode: "max" -> field = max(field, ambient + contribution)  (radiant heat);
              "sum" -> field = field + contribution                (additive EM).

    Returns:
        float32 array (ri, rj, rk).
    """
    ri, rj, rk = int(res_xyz[0]), int(res_xyz[1]), int(res_xyz[2])
    field = np.full((ri, rj, rk), ambient, dtype=np.float32)
    if not sources:
        # No tagged prims -> a uniform field (ambient everywhere). The solver's
        # normalize() turns a constant field into all-zeros, i.e. no effect.
        return field

    # Precompute the world-space CENTER coordinate of every cell along each axis,
    # then broadcast into a full (ri,rj,rk) coordinate grid. `indexing="ij"` keeps the
    # array axes aligned with (i,j,k) so X[i,j,k] is the x-coord of cell (i,j,k).
    xs = bounds_min[0] + (np.arange(ri) + 0.5) * cell_size
    ys = bounds_min[1] + (np.arange(rj) + 0.5) * cell_size
    zs = bounds_min[2] + (np.arange(rk) + 0.5) * cell_size
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    for pos, val, falloff in sources:
        falloff = max(float(falloff), 1e-6)  # guard divide-by-zero
        # Euclidean distance from every cell center to this source.
        d = np.sqrt((X - pos[0]) ** 2 + (Y - pos[1]) ** 2 + (Z - pos[2]) ** 2)
        # Linear falloff, clamped to [0,1]: 1 at the source, 0 at/after `falloff`.
        contrib = float(val) * np.clip(1.0 - d / falloff, 0.0, 1.0)
        if mode == "max":
            field = np.maximum(field, ambient + contrib)
        else:  # "sum"
            field = field + contrib
    return field.astype(np.float32)


def thermal_field(bounds_min, cell_size, res_xyz, sources, ambient=20.0):
    """Temperature grid. `sources` = [(world_pos, temp_c, falloff_m)].

    We subtract ambient before splatting and add it back inside splat_field so the
    field equals `temp_c` exactly at the source and decays to `ambient` (20 °C) far
    away — e.g. a 120 °C block reads 120 at its center, 20 a `falloff_m` away.
    """
    adjusted = [(pos, float(temp) - ambient, falloff) for pos, temp, falloff in sources]
    return splat_field(bounds_min, cell_size, res_xyz, adjusted, ambient=ambient, mode="max")


def em_field(bounds_min, cell_size, res_xyz, sources):
    """EM strength grid. `sources` = [(world_pos, strength, falloff_m)]. Contributions
    from multiple emitters add up; clamped to >= 0."""
    f = splat_field(bounds_min, cell_size, res_xyz, sources, ambient=0.0, mode="sum")
    return np.clip(f, 0.0, None).astype(np.float32)
