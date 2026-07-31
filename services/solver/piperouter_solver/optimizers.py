"""Pluggable local optimizers: turn a coarse, collision-free polyline into a smooth final
cable shape. They are swappable for evaluation; the default is 'fibre', the fibre-neutre
least-squares smoother. All of them keep the fixed points (endpoints and waypoints) pinned
and never move a point into a blocked cell.

Signature: optimize(polyline, frame, blocked, wire, start_heading, end_heading,
                    strength, fixed_idx) -> list[np.ndarray]
  polyline      list of (3,) world points, in metres
  blocked       bool grid (wire-radius-dilated occupancy | melt | prior routes)
  strength      0 = pass through unchanged; higher = more iterations / smoother
"""
from __future__ import annotations

import logging

import numpy as np

from . import smoothing

_log = logging.getLogger("piperouter")


def _is_free(frame, blocked, p):
    i, j, k = frame.world_to_grid((float(p[0]), float(p[1]), float(p[2])))
    nx, ny, nz = blocked.shape
    if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
        return False
    return not blocked[i, j, k]


class NoOpLocal:
    """No smoothing: the raw grid path, as a baseline for comparison."""
    name = "none"

    def optimize(self, polyline, frame, blocked, wire, start_heading, end_heading,
                 strength, fixed_idx):
        return [np.asarray(p, dtype=np.float64) for p in polyline]


class FibreLocal:
    """Fibre-neutre least-squares smoothing, the default. A strength of 0 or fewer than
    three points returns the raw path."""
    name = "fibre"

    def optimize(self, polyline, frame, blocked, wire, start_heading, end_heading,
                 strength, fixed_idx):
        if strength <= 0.0 or len(polyline) < 3:
            return [np.asarray(p, dtype=np.float64) for p in polyline]
        return [np.asarray(p, dtype=np.float64)
                for p in smoothing.smooth_path(polyline, frame, blocked, wire,
                                               start_heading, end_heading, strength,
                                               fixed_idx=fixed_idx)]


class TrajOptLocal:
    """Trajectory optimization (CHOMP-style) against a signed clearance field.

    Each interior point follows two gradients: a smoothness term pulling toward the
    neighbour midpoint (curvature down), and an obstacle term climbing the
    distance-to-obstacle gradient whenever the point is closer than a safety margin. The
    clearance field is the Euclidean distance transform of the (wire-dilated plus
    prior-route) blocked grid, so it is a true SDF. Every move is collision-projected, so
    a point can never enter a blocked cell.
    """
    name = "trajopt"

    def optimize(self, polyline, frame, blocked, wire, start_heading, end_heading,
                 strength, fixed_idx):
        pts = [np.asarray(p, dtype=np.float64) for p in polyline]
        n = len(pts)
        if n < 3 or strength <= 0.0:
            return pts
        from scipy import ndimage
        cell = float(frame.cell_size)
        nx, ny, nz = blocked.shape
        # SDF: metres to the nearest blocked cell, plus its gradient (world axes == grid axes)
        dist = ndimage.distance_transform_edt(~blocked).astype(np.float64) * cell
        g0, g1, g2 = np.gradient(dist)
        margin = wire.radius_m + 0.5 * cell
        fixed = set(fixed_idx or [0, n - 1])
        iters = int(round(20 * min(max(strength, 0.1), 5.0)))
        lam, eta = 0.3, 0.6

        def cellof(p):
            return frame.world_to_grid((float(p[0]), float(p[1]), float(p[2])))

        for _ in range(iters):
            nxt = list(pts)
            for idx in range(1, n - 1):
                if idx in fixed:
                    continue
                p = pts[idx]
                step = lam * (0.5 * (pts[idx - 1] + pts[idx + 1]) - p)   # smoothness
                i, j, k = cellof(p)
                if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz and dist[i, j, k] < margin:
                    grad = np.array([g0[i, j, k], g1[i, j, k], g2[i, j, k]])
                    gn = float(np.linalg.norm(grad))
                    if gn > 1e-9:                                        # climb clearance
                        step = step + (grad / gn) * cell * eta
                cand = p + step
                if _is_free(frame, blocked, cand):
                    nxt[idx] = cand
            pts = nxt
        return pts


class ElasticRodLocal:
    """Discrete elastic-rod relaxation: bending and stretch, no twist.

    Springs hold each segment near its original length, preserving slack, and a bending
    force minimizes curvature with a stiffness derived from the cable's min-bend radius,
    so a stiff pipe straightens far more than a floppy wire. Every move is
    collision-projected against the blocked grid.
    """
    name = "elastic_rod"

    def optimize(self, polyline, frame, blocked, wire, start_heading, end_heading,
                 strength, fixed_idx):
        pts = [np.asarray(p, dtype=np.float64) for p in polyline]
        n = len(pts)
        if n < 3 or strength <= 0.0:
            return pts
        cell = float(frame.cell_size)
        fixed = set(fixed_idx or [0, n - 1])
        rest = [float(np.linalg.norm(pts[i + 1] - pts[i])) for i in range(n - 1)]
        # bending stiffness ~ min-bend radius in cells (clamped): stiffer rod => straighter
        stiff = min(max(wire.min_bend_radius_mm / 1000.0 / max(cell, 1e-6), 0.5), 6.0)
        iters = int(round(30 * min(max(strength, 0.1), 5.0)))
        k_stretch = 0.2
        k_bend = 0.05 * stiff
        for _ in range(iters):
            force = [np.zeros(3) for _ in range(n)]
            for i in range(n - 1):                      # stretch springs -> keep length
                d = pts[i + 1] - pts[i]
                L = float(np.linalg.norm(d))
                if L < 1e-9:
                    continue
                f = k_stretch * (L - rest[i]) * (d / L)
                force[i] += f
                force[i + 1] -= f
            for i in range(1, n - 1):                   # bending -> curvature * stiffness
                force[i] += k_bend * (0.5 * (pts[i - 1] + pts[i + 1]) - pts[i])
            nxt = list(pts)
            for i in range(1, n - 1):
                if i in fixed:
                    continue
                cand = pts[i] + force[i]
                if _is_free(frame, blocked, cand):
                    nxt[i] = cand
            pts = nxt
        return pts


LOCAL_OPTIMIZERS = {
    "fibre": FibreLocal,
    "none": NoOpLocal,
    "trajopt": TrajOptLocal,
    "elastic_rod": ElasticRodLocal,
}


def make_local(name):
    cls = LOCAL_OPTIMIZERS.get(name)
    if cls is None:
        _log.warning("[piperouter] unknown local optimizer %r (have: %s) -> using 'fibre'. "
                     "Rebuild/restart the solver if you expected this optimizer.",
                     name, ", ".join(LOCAL_OPTIMIZERS))
        cls = FibreLocal
    return cls()
