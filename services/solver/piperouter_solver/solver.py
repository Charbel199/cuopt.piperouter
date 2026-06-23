from __future__ import annotations

from collections import deque

import numpy as np

from . import optimizers, planners
from .lattice import ExpandedLatticeBuilder, LatticeBuilder, diagnose_no_path
from .models import RouteRequest, RouteResult, SolveReport

# 6-neighbour steps for the nearest-free BFS used to rescue a buried endpoint.
_BFS_STEPS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def _in_bounds(blocked, c):
    nx, ny, nz = blocked.shape
    return 0 <= c[0] < nx and 0 <= c[1] < ny and 0 <= c[2] < nz


def _nearest_free_cell(blocked, cell):
    """Nearest free cell to `cell` by breadth-first (6-connected) expansion through the
    blocked grid — used to relocate a START/END that's buried inside an obstacle to the
    closest open space. Returns the cell (the input cell if already free) or None if the
    whole grid is blocked. Clamps an out-of-bounds input into the grid first."""
    nx, ny, nz = blocked.shape
    start = (min(max(int(cell[0]), 0), nx - 1),
             min(max(int(cell[1]), 0), ny - 1),
             min(max(int(cell[2]), 0), nz - 1))
    if not blocked[start]:
        return start
    seen = {start}
    q = deque([start])
    while q:
        i, j, k = q.popleft()
        for di, dj, dk in _BFS_STEPS:
            n = (i + di, j + dj, k + dk)
            if _in_bounds(blocked, n) and n not in seen:
                if not blocked[n]:
                    return n           # first free cell reached = nearest (by hop count)
                seen.add(n)
                q.append(n)
    return None


class Solver:
    def __init__(self, builder: LatticeBuilder | None = None):
        self.builder = builder or ExpandedLatticeBuilder()

    def _solve_leg(self, planner, stack, wire, weights, connectivity, start_cell,
                   goal_cell, extra_obstacles, clearance_m=0.0,
                   start_heading=None, goal_heading=None):
        return planner.plan(stack, wire, weights, connectivity, start_cell, goal_cell,
                            extra_obstacles, clearance_m, start_heading, goal_heading)

    def route_one(self, stack, req: RouteRequest, extra_obstacles=None) -> RouteResult:
        frame = stack.frame
        planner = planners.make_global(getattr(req, "global_planner", "lattice"))
        optimizer = optimizers.make_local(getattr(req, "local_optimizer", "fibre"))
        waypts = [req.start, *req.waypoints, req.end]
        cell_seq = [frame.world_to_grid(p) for p in waypts]

        # Safety clearance, split into HARD vs a relaxable SHELL:
        #   hard  = mesh dilated by the wire radius — the tube physically can't intersect
        #           it. This is the ONLY thing that buries an endpoint (triggers relocation)
        #           and the only thing the route can never enter.
        #   shell = the extra safety-clearance band (radius .. radius+clearance). The route's
        #           interior keeps this clear, but it's WAIVED in a ball around each endpoint,
        #           so a connector sitting in the clearance band is still reachable and the
        #           tube can pass through those near-surface voxels to reach the terminal.
        # So clearance never pushes an endpoint off a surface; only a real mesh does.
        radius = req.wire.radius_m
        clr = float(req.clearance_m)
        cell = float(frame.cell_size)
        hard = stack.dilate_occupancy(radius).astype(bool)
        # relocation target avoids mesh+radius, melt and prior routes — but NOT the clearance
        # band (an endpoint is allowed to sit within clearance of a surface).
        reloc_blocked = planners.blocked_mask(stack, req.wire, 0.0, extra_obstacles)
        notes: list[str] = []
        start_world = np.asarray(req.start, dtype=np.float64)
        end_world = np.asarray(req.end, dtype=np.float64)

        def _rescue(cell_ijk, marker_world, label):
            c = tuple(int(v) for v in cell_ijk)
            if not (_in_bounds(hard, c) and hard[c]):    # buried in MESH+radius only
                return cell_ijk, marker_world, None
            free = _nearest_free_cell(reloc_blocked, cell_ijk)
            if free is None:
                return cell_ijk, marker_world, None      # nothing free anywhere -> planner fails
            w = np.asarray(frame.grid_to_world(free), dtype=np.float64)
            dist_mm = float(np.linalg.norm(w - marker_world)) * 1000.0
            return (free, w,
                    f"{label} is buried in an obstacle — routed to the nearest open point "
                    f"({dist_mm:.0f} mm away)")

        cell_seq[0], start_world, n0 = _rescue(cell_seq[0], start_world, "Start")
        cell_seq[-1], end_world, n1 = _rescue(cell_seq[-1], end_world, "End")
        for n in (n0, n1):
            if n:
                notes.append(n)

        # clearance shell with endpoint exemption (balls around the FINAL start/end cells)
        extra_arr = (np.asarray(extra_obstacles, dtype=bool) if extra_obstacles is not None
                     else np.zeros(hard.shape, dtype=bool))
        if clr > 0.0:
            shell = stack.dilate_occupancy(radius + clr).astype(bool) & ~hard
            er = (int(np.ceil(clr / cell)) + 1) if cell > 0 else 1
            nx, ny, nz = hard.shape
            for ci, cj, ck in (tuple(int(v) for v in cell_seq[0]),
                               tuple(int(v) for v in cell_seq[-1])):
                shell[max(0, ci - er):min(nx, ci + er + 1),
                      max(0, cj - er):min(ny, cj + er + 1),
                      max(0, ck - er):min(nz, ck + er + 1)] = False
        else:
            shell = np.zeros(hard.shape, dtype=bool)
        # The planner only adds mesh+radius+melt itself, so the clearance shell is folded
        # into its extra-obstacles and we call it with clearance_m=0. `blocked` (= the same
        # set) is what the relocation target and the smoother avoid.
        planner_extra = shell | extra_arr
        blocked = reloc_blocked | shell

        all_cells: list[tuple[int, int, int]] = []
        wp_cell_idx: list[int] = []   # index in all_cells of each waypoint's cell
        n_legs = len(cell_seq) - 1
        # Heading continuity across legs: each leg leaves a waypoint along the heading it
        # arrived with, so the bend penalty applies THROUGH the waypoint instead of a free
        # sharp turn there (legs are solved independently, so without this the join kinks).
        prev_arrival = None
        for li, (a, b) in enumerate(zip(cell_seq[:-1], cell_seq[1:])):
            sh = req.start_heading if li == 0 else prev_arrival
            gh = req.end_heading if li == n_legs - 1 else None
            leg = self._solve_leg(
                planner, stack, req.wire, req.weights, req.connectivity, a, b,
                planner_extra, clearance_m=0.0, start_heading=sh,
                goal_heading=gh,
            )
            if leg is None and li > 0 and sh is not None:
                # the continuity heading over-constrained this leg (e.g. a hairpin
                # waypoint that demands a >45 turn) — retry without it rather than fail.
                leg = self._solve_leg(
                    planner, stack, req.wire, req.weights, req.connectivity, a, b,
                    planner_extra, clearance_m=0.0, start_heading=None,
                    goal_heading=gh,
                )
            if leg is None:
                reason = diagnose_no_path(
                    stack, req.wire, req.connectivity, a, b, planner_extra,
                    clearance_m=0.0,
                    start_heading=(req.start_heading if li == 0 else None), goal_heading=gh,
                )
                return RouteResult(wire_id=req.wire.id, status="no_path", reason=reason)
            # arrival heading = direction of the leg's final step (feeds the next leg)
            if len(leg) >= 2:
                d = np.subtract(leg[-1], leg[-2])
                prev_arrival = tuple(int(np.sign(v)) for v in d)
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

        # Anchor to the markers — or, when an endpoint was buried, to the open point we
        # relocated it to (start_world / end_world), so the tube ends in free space.
        pts = [start_world]
        flags = [True]
        for j, p in enumerate(cell_world):
            pts.append(p)
            flags.append(j in wpset)
        pts.append(end_world)
        flags.append(True)

        polyline, fixed_flags = [], []
        for p, fx in zip(pts, flags):
            if polyline and float(np.linalg.norm(p - polyline[-1])) <= 1e-6:
                fixed_flags[-1] = fixed_flags[-1] or fx   # merge a near-duplicate
                continue
            polyline.append(p)
            fixed_flags.append(fx)
        fixed_idx = [i for i, fx in enumerate(fixed_flags) if fx]

        # Capture the pre-smoothing grid path (stair-stepped) for debug views.
        raw_polyline = [[float(x) for x in p] for p in polyline]

        # Local optimizer (default: fibre-neutre least-squares smoothing). Collision-safe
        # against the same prohibited voxels the planner avoided. weights["smoothing"]
        # is the strength knob (0 -> pass-through raw path for the smoothing-based ones).
        strength = float(req.weights.get("smoothing", 1.0))
        if len(polyline) >= 3:
            # same prohibited voxels the planner/relocation used
            polyline = optimizer.optimize(
                polyline, frame, blocked, req.wire,
                req.start_heading, req.end_heading, strength, fixed_idx)

        length = 0.0
        for p, q in zip(polyline[:-1], polyline[1:]):
            length += float(np.linalg.norm(np.asarray(q) - np.asarray(p)))
        return RouteResult(
            wire_id=req.wire.id, status="routed",
            polyline=polyline, length_m=length, cells=all_cells,
            raw_polyline=raw_polyline, note="; ".join(notes),
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
