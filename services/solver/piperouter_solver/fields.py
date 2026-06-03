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
    """Combine the SOFT constraint fields into one per-cell extra cost `S(cell)`.

    The lattice builder turns this into edge weight `step_len * (1 + S(dst)) + turn`,
    so a higher S(cell) makes the router prefer to route AROUND that cell (it never
    forbids it — that's what the hard masks are for).

    Each field is normalized to [0,1] so the slider weights (0..10 in the UI) are the
    only thing that sets relative importance, regardless of the field's raw units:

      * surface_dist : distance to the nearest mesh surface. Normalized so cells far
        from any surface cost more -> the route is pulled in to hug surfaces (so it can
        be clipped down instead of floating). Weight = `weights["surface"]`.
      * thermal      : temperature (°C). Hotter cells cost more. Weight =
        `weights["thermal"]`. (Cells OVER the wire's rating are removed entirely by
        melt_mask — see below.)
      * em           : EM field strength. Costlier near emitters, but ALSO multiplied by
        this wire type's `em_sensitivity` so an EM-immune pipe (sensitivity 0) ignores
        EM no matter the slider. Weight = `weights["em"]`.

    (The 4th UI slider, "bend", is applied in the lattice as a turn-penalty scale, not
    here, because bend cost depends on the path's heading change, not a single cell.)
    """
    w_surface = float(weights.get("surface", 0.0))
    w_thermal = float(weights.get("thermal", 0.0))
    w_em = float(weights.get("em", 0.0))
    cost = (
        w_surface * normalize(stack.surface_dist)
        + w_thermal * normalize(stack.thermal)
        + w_em * wire.em_sensitivity * normalize(stack.em)
    )
    return cost.astype(np.float32)


def melt_mask(stack, wire) -> np.ndarray:
    """HARD thermal constraint: True for cells hotter than the wire's `max_temp_c`.

    These cells are removed from the graph entirely (the route physically cannot pass
    through a region that would melt the cable), as opposed to the soft thermal cost in
    soft_cost_field which merely discourages warm-but-survivable regions."""
    return stack.thermal > wire.max_temp_c


# Tuning constants for the bend (turn) penalty. The penalty is returned in "metres of
# equivalent travel" (it's multiplied by the cell size below) so it is directly
# comparable to a step's length in the edge weight
#   edge = step_len * (1 + soft) + bend_weight * turn_penalty
# A turn through `angle` radians therefore costs about _STRAIGHTNESS*angle CELLS of
# extra travel, with a steep extra term when the turn is tighter than the wire's rated
# minimum bend radius. The per-wire "bend" slider scales all of it (0 = turns free /
# fully relaxed, large = strongly forces straighter, gentler routes).
_STRAIGHTNESS = 0.6    # cells of cost per radian of turn (general straightness pull)
_SUB_RADIUS = 6.0      # cells of cost per unit of "tighter than allowed" deficit


def turn_penalty(
    h_in: tuple[int, int, int],
    h_out: tuple[int, int, int],
    min_bend_radius_mm: float,
    cell_size_mm: float,
) -> float:
    """SOFT bend cost for turning from arrival heading `h_in` to departure heading
    `h_out` at one cell.

    This is the "bend" constraint. It's path-dependent (the cost of a turn depends on
    how you arrived), which is exactly why the lattice node carries the heading. We
    return a cost, never infinity, so a route is always findable — a too-tight turn is
    expensive, not forbidden (that was the design choice: soft bend).

    Model: turning through `angle` radians over roughly one cell of travel implies a
    turn radius of about cell_size/angle. If that implied radius is below the wire's
    rated minimum bend radius, we add a steep penalty proportional to how far under it
    is. Straight travel (angle 0) costs nothing.

    The router_session/UI further multiplies this by the per-wire "bend" slider, so the
    expert can dial straightness up or down.
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
