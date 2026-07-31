"""Thermal and EM scalar fields, splatted from tagged prims into the voxel grid.

These are two of the four grids the router consumes; the voxelizer produces the other
two, occupancy and surface distance. Each is a dense float array with one value per
cell: thermal[i,j,k] is an estimated temperature in °C, em[i,j,k] an estimated EM field
strength on a roughly 0..1 scale.

On the solver side, thermal acts both as a soft cost (warm cells cost more, scaled by
the wire's "thermal" slider) and as a hard cutoff (cells hotter than the wire's
max_temp_c are removed). EM is only a soft cost, scaled by the wire's "em" slider and by
the wire type's em_sensitivity, since a CAN signal cares about EM and an AC pipe does
not.

The physics is deliberately crude. Each tagged source radiates its value, falling off
linearly to ambient over a characteristic distance `falloff_m`:

      contribution(cell) = value * max(0, 1 - distance(cell, source) / falloff_m)

Thermal takes the maximum over sources above an ambient floor, so the hottest source
wins as radiant heat does; EM sums, since emitters add up. A real conduction or EM solve
can replace this without changing the grid contract.

Each source carries its own falloff (see router_session) so that a large hot engine
block heats a much wider region than a small connector does.
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
        mode: "max" -> field = max(field, ambient + contribution), radiant heat;
              "sum" -> field = field + contribution, additive EM.

    Returns:
        float32 array (ri, rj, rk).
    """
    ri, rj, rk = int(res_xyz[0]), int(res_xyz[1]), int(res_xyz[2])
    field = np.full((ri, rj, rk), ambient, dtype=np.float32)
    if not sources:
        # Uniform ambient everywhere. The solver's normalize() flattens a constant
        # field to zeros, so this has no effect on routing.
        return field

    # World-space centre coordinate of every cell along each axis, broadcast into a
    # full (ri,rj,rk) coordinate grid. `indexing="ij"` keeps the array axes aligned
    # with (i,j,k), so X[i,j,k] is the x coordinate of cell (i,j,k).
    xs = bounds_min[0] + (np.arange(ri) + 0.5) * cell_size
    ys = bounds_min[1] + (np.arange(rj) + 0.5) * cell_size
    zs = bounds_min[2] + (np.arange(rk) + 0.5) * cell_size
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

    for pos, val, falloff in sources:
        falloff = max(float(falloff), 1e-6)  # guard divide-by-zero
        d = np.sqrt((X - pos[0]) ** 2 + (Y - pos[1]) ** 2 + (Z - pos[2]) ** 2)
        # Linear falloff clamped to [0,1]: 1 at the source, 0 at and beyond `falloff`.
        contrib = float(val) * np.clip(1.0 - d / falloff, 0.0, 1.0)
        if mode == "max":
            field = np.maximum(field, ambient + contrib)
        else:  # "sum"
            field = field + contrib
    return field.astype(np.float32)


def thermal_field(bounds_min, cell_size, res_xyz, sources, ambient=20.0):
    """Build the temperature grid from [(world_pos, temp_c, falloff_m)] sources.

    Ambient is subtracted before splatting and added back inside splat_field, so the
    field equals `temp_c` exactly at the source and decays to `ambient`: a 120 °C block
    reads 120 at its centre and 20 one `falloff_m` away.
    """
    adjusted = [(pos, float(temp) - ambient, falloff) for pos, temp, falloff in sources]
    return splat_field(bounds_min, cell_size, res_xyz, adjusted, ambient=ambient, mode="max")


def em_field(bounds_min, cell_size, res_xyz, sources):
    """Build the EM strength grid from [(world_pos, strength, falloff_m)] sources.

    Emitter contributions add up, and the result is clamped to be non-negative.
    """
    f = splat_field(bounds_min, cell_size, res_xyz, sources, ambient=0.0, mode="sum")
    return np.clip(f, 0.0, None).astype(np.float32)


def _normalize(arr):
    a = arr.astype(np.float32)
    lo, hi = float(a.min()), float(a.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def soft_cost_field(surface_dist, thermal, em, wire_spec, weights):
    """Combine the soft-cost field for one wire and its slider weights.

    Mirrors the solver-side soft_cost_field, so the extension can visualise the routing
    terrain for a selected wire without a solver round-trip. `wire_spec` supplies
    'em_sensitivity'; `weights` holds 'surface', 'thermal' and 'em' on a 0..10 scale.
    """
    w_surface = float(weights.get("surface", 0.0))
    w_thermal = float(weights.get("thermal", 0.0))
    w_em = float(weights.get("em", 0.0))
    em_sens = float(wire_spec.get("em_sensitivity", 0.0))
    cost = (w_surface * _normalize(surface_dist)
            + w_thermal * _normalize(thermal)
            + w_em * em_sens * _normalize(em))
    return cost.astype(np.float32)
