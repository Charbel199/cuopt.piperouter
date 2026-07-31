from __future__ import annotations

import numpy as np


def neighbor_offsets(connectivity: int) -> list[tuple[int, int, int]]:
    """Integer neighbor offsets for 6-, 18-, or 26-connectivity."""
    if connectivity not in (6, 18, 26):
        raise ValueError(f"connectivity must be 6, 18 or 26, got {connectivity}")
    offs: list[tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                if dx == dy == dz == 0:
                    continue
                manhattan = abs(dx) + abs(dy) + abs(dz)
                if connectivity == 6 and manhattan != 1:
                    continue
                if connectivity == 18 and manhattan == 3:
                    continue
                offs.append((dx, dy, dz))
    return offs


def normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max to [0, 1]; a constant field maps to all zeros."""
    a = arr.astype(np.float32)
    lo = float(a.min())
    hi = float(a.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def soft_cost_field(stack, wire, weights: dict) -> np.ndarray:
    """Combine the soft constraint fields into one per-cell extra cost `S(cell)`.

    The lattice builder turns this into edge weight `step_len * (1 + S(dst)) + turn`, so
    a higher S(cell) makes the router prefer to route around that cell; it never forbids
    it, which is what the hard masks are for.

    Each field is normalized to [0,1] so the weights alone set relative importance,
    whatever the field's raw units:

      * surface_dist : distance to the nearest mesh surface, normalized so cells far from
        any surface cost more, pulling the route in to hug surfaces (where it can be
        clipped down instead of floating). Weight = `weights["surface"]`.
      * thermal      : temperature (deg C); hotter cells cost more. Weight =
        `weights["thermal"]`. Cells over the wire's rating are removed outright by
        melt_mask.
      * em           : EM field strength, costlier near emitters, multiplied by this wire
        type's `em_sensitivity` so an EM-immune pipe (sensitivity 0) ignores EM whatever
        the weight. Weight = `weights["em"]`.

    Bend cost is not here: it depends on the path's heading change rather than a single
    cell, so the lattice applies it as a turn-penalty scale.
    """
    w_surface = float(weights.get("surface", 0.0))
    w_thermal = float(weights.get("thermal", 0.0))
    w_em = float(weights.get("em", 0.0))
    # The normalized fields depend only on the stack (immutable per solve) and the
    # combined cost only on the three effective weights, so both are cached: route_all
    # calls this once or twice per wire and the min-max passes are expensive on big
    # grids. Treat the returned array as read-only.
    norm = stack.__dict__.get("_norm_fields")
    if norm is None:
        norm = {"surface": normalize(stack.surface_dist),
                "thermal": normalize(stack.thermal),
                "em": normalize(stack.em)}
        stack.__dict__["_norm_fields"] = norm
    key = (round(w_surface, 9), round(w_thermal, 9),
           round(w_em * float(wire.em_sensitivity), 9))
    cache = stack.__dict__.setdefault("_soft_cache", {})
    hit = cache.get(key)
    if hit is not None:
        return hit
    cost = (w_surface * norm["surface"]
            + w_thermal * norm["thermal"]
            + key[2] * norm["em"]).astype(np.float32)
    cache[key] = cost
    return cost


def melt_mask(stack, wire) -> np.ndarray:
    """Hard thermal constraint: True for cells hotter than the wire's `max_temp_c`.

    These cells are removed from the graph entirely, unlike the soft thermal cost in
    soft_cost_field which only discourages warm-but-survivable regions. Cached per
    temperature rating (thermal is immutable within a solve); treat the returned mask
    as read-only.
    """
    cache = stack.__dict__.setdefault("_melt_cache", {})
    key = round(float(wire.max_temp_c), 9)
    hit = cache.get(key)
    if hit is not None:
        return hit
    out = stack.thermal > wire.max_temp_c
    cache[key] = out
    return out


# Tuning constants for the bend (turn) penalty. The penalty comes back in metres of
# equivalent travel, so it is directly comparable to a step's length in the edge weight
#   edge = step_len * (1 + soft) + bend_weight * turn_penalty
# A turn through `angle` radians costs about _STRAIGHTNESS * angle cells of extra travel,
# plus a steep term when the turn is tighter than the wire's rated minimum bend radius.
# bend_weight scales all of it: 0 makes turns free, large forces straighter routes.
_STRAIGHTNESS = 0.6    # cells of cost per radian of turn (general straightness pull)
_SUB_RADIUS = 6.0      # cells of cost per unit of "tighter than allowed" deficit


def turn_penalty(
    h_in: tuple[int, int, int],
    h_out: tuple[int, int, int],
    min_bend_radius_mm: float,
    cell_size_mm: float,
) -> float:
    """Soft bend cost for turning from arrival heading `h_in` to departure heading `h_out`.

    The cost is path-dependent (it depends on how you arrived), which is why the lattice
    node carries the heading. It is always finite: a too-tight turn is expensive, not
    forbidden, so a route is always findable.

    Turning through `angle` radians over roughly one cell of travel implies a turn radius
    of about cell_size/angle. If that implied radius is below the wire's rated minimum
    bend radius, a steep penalty proportional to the shortfall is added. Straight travel
    costs nothing. Callers scale the result by the per-wire bend weight.
    """
    a = np.asarray(h_in, dtype=np.float64)
    b = np.asarray(h_out, dtype=np.float64)
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    cosang = float(np.clip(np.dot(a, b), -1.0, 1.0))
    angle = float(np.arccos(cosang))  # radians; 0 = same heading = straight
    if angle <= 1e-9:
        return 0.0
    # Implied arc radius of turning `angle` over ~one cell of travel.
    implied_radius_mm = cell_size_mm / angle
    penalty = _STRAIGHTNESS * angle  # general straightness pull (in cell units)
    if implied_radius_mm < min_bend_radius_mm:
        # Tighter than the cable can bend -> add a steep, proportional penalty.
        deficit = (min_bend_radius_mm - implied_radius_mm) / min_bend_radius_mm
        penalty += _SUB_RADIUS * deficit
    # Convert "cells of cost" to metres so it is comparable to a step's length.
    return penalty * (cell_size_mm / 1000.0)
