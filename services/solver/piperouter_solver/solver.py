from __future__ import annotations

import numpy as np

from . import fields, smoothing
from .backend import shortest_path
from .lattice import ExpandedLatticeBuilder, LatticeBuilder, diagnose_no_path
from .models import RouteRequest, RouteResult, SolveReport


class Solver:
    def __init__(self, builder: LatticeBuilder | None = None):
        self.builder = builder or ExpandedLatticeBuilder()

    def _solve_leg(self, stack, wire, weights, connectivity, start_cell, goal_cell,
                   extra_obstacles, clearance_m=0.0,
                   start_heading=None, goal_heading=None):
        g = self.builder.build(
            stack, wire, weights, connectivity, start_cell, goal_cell, extra_obstacles,
            clearance_m=clearance_m,
            start_heading=start_heading, goal_heading=goal_heading,
        )
        path, _cost = shortest_path(
            g.src, g.dst, g.weight, g.n_nodes, g.source_id, g.sink_id
        )
        if path is None:
            return None
        # map node path -> cell path, dropping source/sink and consecutive dupes
        cells: list[tuple[int, int, int]] = []
        for node in path:
            if node in (g.source_id, g.sink_id):
                continue
            c = g.cell_of(node)
            if not cells or cells[-1] != c:
                cells.append(c)
        return cells

    def route_one(self, stack, req: RouteRequest, extra_obstacles=None) -> RouteResult:
        frame = stack.frame
        waypts = [req.start, *req.waypoints, req.end]
        cell_seq = [frame.world_to_grid(p) for p in waypts]

        all_cells: list[tuple[int, int, int]] = []
        wp_cell_idx: list[int] = []   # index in all_cells of each waypoint's cell
        n_legs = len(cell_seq) - 1
        for li, (a, b) in enumerate(zip(cell_seq[:-1], cell_seq[1:])):
            sh = req.start_heading if li == 0 else None
            gh = req.end_heading if li == n_legs - 1 else None
            leg = self._solve_leg(
                stack, req.wire, req.weights, req.connectivity, a, b, extra_obstacles,
                clearance_m=req.clearance_m, start_heading=sh, goal_heading=gh,
            )
            if leg is None:
                reason = diagnose_no_path(
                    stack, req.wire, req.connectivity, a, b, extra_obstacles,
                    clearance_m=req.clearance_m, start_heading=sh, goal_heading=gh,
                )
                return RouteResult(wire_id=req.wire.id, status="no_path", reason=reason)
            if all_cells and leg and all_cells[-1] == leg[0]:
                leg = leg[1:]
            all_cells.extend(leg)
            if li < n_legs - 1 and all_cells:
                wp_cell_idx.append(len(all_cells) - 1)   # this leg ended at a waypoint

        # Build the world polyline. Anchor it to the ACTUAL endpoint markers (the start
        # cell isn't in all_cells — the source links to a NEIGHBOUR of it — and markers
        # sit at sub-cell positions, so otherwise the tube looks detached). Replace each
        # waypoint's cell centre with the EXACT waypoint marker, and mark start /
        # waypoints / end as hard points the smoother must pass through (waypoints are a
        # hard requirement; a free tangent there keeps the pass-through smooth).
        cell_world = [np.asarray(frame.grid_to_world(c), dtype=np.float64) for c in all_cells]
        wpset = set(wp_cell_idx)
        for k, ci in enumerate(wp_cell_idx):
            if k < len(req.waypoints):
                cell_world[ci] = np.asarray(req.waypoints[k], dtype=np.float64)

        pts = [np.asarray(req.start, dtype=np.float64)]
        flags = [True]
        for j, p in enumerate(cell_world):
            pts.append(p)
            flags.append(j in wpset)
        pts.append(np.asarray(req.end, dtype=np.float64))
        flags.append(True)

        polyline, fixed_flags = [], []
        for p, fx in zip(pts, flags):
            if polyline and float(np.linalg.norm(p - polyline[-1])) <= 1e-6:
                fixed_flags[-1] = fixed_flags[-1] or fx   # merge a near-duplicate
                continue
            polyline.append(p)
            fixed_flags.append(fx)
        fixed_idx = [i for i, fx in enumerate(fixed_flags) if fx]

        # Fibre-neutre smoothing (cuSolver least-squares), hard-safe against the
        # same prohibited voxels the lattice avoided. weights["smoothing"] == 0 -> off.
        strength = float(req.weights.get("smoothing", 1.0))
        if strength > 0.0 and len(polyline) >= 3:
            blocked = (
                stack.dilate_occupancy(req.wire.radius_m + req.clearance_m).astype(bool)
                | fields.melt_mask(stack, req.wire)
            )
            if extra_obstacles is not None:
                blocked |= np.asarray(extra_obstacles, dtype=bool)
            polyline = [np.asarray(p, dtype=np.float64)
                        for p in smoothing.smooth_path(
                            polyline, frame, blocked, req.wire,
                            req.start_heading, req.end_heading, strength,
                            fixed_idx=fixed_idx)]

        length = 0.0
        for p, q in zip(polyline[:-1], polyline[1:]):
            length += float(np.linalg.norm(np.asarray(q) - np.asarray(p)))
        return RouteResult(
            wire_id=req.wire.id, status="routed",
            polyline=polyline, length_m=length, cells=all_cells,
        )

    def route_all(self, stack, requests: list[RouteRequest]) -> SolveReport:
        """Priority-ordered greedy: earlier (lower-priority-number) routes become
        obstacles for later ones (spec §5, Route-All)."""
        ordered = sorted(requests, key=lambda r: r.priority)
        occupied = np.zeros(stack.frame.res_xyz, dtype=bool)
        results: list[RouteResult] = []
        for req in ordered:
            res = self.route_one(stack, req, extra_obstacles=occupied)
            if res.status == "routed":
                for c in res.cells:
                    occupied[c] = True
            results.append(res)
        return SolveReport(results=results)
