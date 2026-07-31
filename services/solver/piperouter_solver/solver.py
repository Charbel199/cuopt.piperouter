from __future__ import annotations

from collections import deque

import numpy as np

from . import optimizers, planners
from .lattice import ExpandedLatticeBuilder, LatticeBuilder, diagnose_no_path
from .grids import dilate6
from .models import RouteRequest, RouteResult, SolveReport

# 6-neighbour steps for the nearest-free BFS used to rescue a buried endpoint.
_BFS_STEPS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

# Heading stub: a pinned start/end heading forces a straight run of cells along the
# heading before the route may bend (a connector's straight exit), sized from the cable's
# min bend radius and clamped to this range. Shortened where geometry blocks it.
_STUB_MIN_CELLS = 2
_STUB_MAX_CELLS = 6


def _in_bounds(blocked, c):
    nx, ny, nz = blocked.shape
    return 0 <= c[0] < nx and 0 <= c[1] < ny and 0 <= c[2] < nz


def _nearest_free_cell(blocked, cell):
    """Return the nearest free cell to `cell` by 6-connected breadth-first expansion.

    Used to relocate a start or end buried inside an obstacle to the closest open space.
    An out-of-bounds input is clamped into the grid first. Returns the input cell if it is
    already free, or None if the whole grid is blocked.
    """
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

        # Safety clearance splits into a hard part and a relaxable shell:
        #   hard  = mesh dilated by the wire radius. The tube cannot intersect it, so it is
        #           the only thing that buries an endpoint (triggering relocation) and the
        #           only thing the route may never enter.
        #   shell = the extra clearance band (radius .. radius+clearance). The route's
        #           interior keeps it clear, but it is waived in a ball around each
        #           endpoint, so a connector sitting in the band is still reachable.
        # Clearance therefore never pushes an endpoint off a surface; only real mesh does.
        radius = req.wire.radius_m
        clr = float(req.clearance_m)
        cell = float(frame.cell_size)
        rad_cells = int(radius / cell + 0.5 + 1e-9) if cell > 0 else 0
        hard = stack.dilate_occupancy(radius).astype(bool)
        # Prior routed wires arrive already dilated by their own radius; grow them by this
        # wire's radius too, so the two tube bodies stay r_prior + r_this apart rather than
        # only their centerlines.
        prior = None
        if extra_obstacles is not None:
            prior = np.asarray(extra_obstacles, dtype=bool)
            if rad_cells > 0 and prior.any():
                prior = dilate6(prior, rad_cells)
        # Relocation avoids mesh+radius, melt and prior tubes, but not the clearance band:
        # an endpoint is allowed to sit within clearance of a surface.
        reloc_blocked = planners.blocked_mask(stack, req.wire, 0.0, prior)
        notes: list[str] = []
        start_world = np.asarray(req.start, dtype=np.float64)
        end_world = np.asarray(req.end, dtype=np.float64)

        def _rescue(cell_ijk, marker_world, label):
            c = tuple(int(v) for v in cell_ijk)
            if not (_in_bounds(hard, c) and hard[c]):    # buried in mesh+radius only
                return cell_ijk, marker_world, None
            free = _nearest_free_cell(reloc_blocked, cell_ijk)
            if free is None:
                return cell_ijk, marker_world, None      # nothing free anywhere -> planner fails
            w = np.asarray(frame.grid_to_world(free), dtype=np.float64)
            dist_mm = float(np.linalg.norm(w - marker_world)) * 1000.0
            return (free, w,
                    f"{label} is buried in an obstacle - routed to the nearest open point "
                    f"({dist_mm:.0f} mm away)")

        cell_seq[0], start_world, n0 = _rescue(cell_seq[0], start_world, "Start")
        cell_seq[-1], end_world, n1 = _rescue(cell_seq[-1], end_world, "End")
        for n in (n0, n1):
            if n:
                notes.append(n)

        # Clearance shell, per object: tagged geometry keeps its own distance
        # (clearance_values by class), untagged geometry keeps the request default. The
        # shell is waived in a ball around every terminal the route must touch (start,
        # waypoints, end); a user-pinned point near a surface is as legitimate a terminal
        # as a connector, and without the waiver it turns the whole wire into no_path.
        cvals = list(getattr(stack, "clearance_values", ()) or ())
        has_cls = getattr(stack, "clearance_class", None) is not None and cvals
        max_clr = max([clr] + cvals) if (clr > 0.0 or cvals) else 0.0
        if max_clr > 0.0:
            if has_cls:
                shell = np.zeros(hard.shape, dtype=bool)
                if clr > 0.0:
                    shell |= stack.dilate_class(0, radius + clr)       # untagged: default
                for i, v in enumerate(cvals):
                    shell |= stack.dilate_class(i + 1, radius + float(v))  # tagged: its own
            else:
                shell = stack.dilate_occupancy(radius + clr).astype(bool).copy()
            shell &= ~hard
            # Waive the shell around each terminal by the minimum needed for reachability:
            # the BFS distance from the terminal to the nearest cell outside the band, +1.
            # A terminal outside every band gets no waiver, so a wire ending near a tagged
            # component still respects that component's clearance everywhere else.
            nx, ny, nz = hard.shape
            esc = shell | hard
            for c in cell_seq:                     # start, waypoints..., end (post-rescue)
                ci, cj, ck = (int(c[0]), int(c[1]), int(c[2]))
                if not (_in_bounds(shell, (ci, cj, ck)) and shell[ci, cj, ck]):
                    continue                       # terminal not inside a band -> no waiver
                out = _nearest_free_cell(esc, (ci, cj, ck))
                if out is None:
                    er = (int(np.ceil(max_clr / cell)) + 1) if cell > 0 else 1
                else:
                    er = max(abs(out[0] - ci), abs(out[1] - cj), abs(out[2] - ck)) + 1
                shell[max(0, ci - er):min(nx, ci + er + 1),
                      max(0, cj - er):min(ny, cj + er + 1),
                      max(0, ck - er):min(nz, ck + er + 1)] = False
        else:
            shell = None
        # The planner adds mesh+radius+melt itself, so the shell and prior tubes go in as
        # its extra obstacles and it runs with clearance_m=0. `blocked` is the same set,
        # and is what the smoother avoids. `prior` stays separate from the shell so no_path
        # diagnosis can tell "another wire" from "the clearance band".
        if shell is not None:
            planner_extra = shell if prior is None else (shell | prior)
            blocked = reloc_blocked | shell
        else:
            planner_extra = prior
            blocked = reloc_blocked

        # A pinned heading has to hold for a real distance, not one voxel, or a 90-degree
        # turn right after the first cell defeats it. March up to a min-bend-radius worth
        # of free cells along the heading, force them as the path's straight prefix (start)
        # or suffix (end), and route from the stub's far end, still heading-pinned there so
        # the hand-off into free routing stays gentle. The stub shortens wherever geometry
        # or clearance blocks the runway.
        def _stub_cells(anchor, heading, sign):
            if heading is None or cell <= 0:
                return []
            h = np.asarray(heading, dtype=np.float64)
            n = float(np.linalg.norm(h))
            if n <= 1e-9:
                return []
            h = h / n
            want = int(round((req.wire.min_bend_radius_mm / 1000.0) / cell))
            want = max(_STUB_MIN_CELLS, min(want, _STUB_MAX_CELLS))
            out: list[tuple[int, int, int]] = []
            a = np.asarray(anchor, dtype=np.float64)
            anchor_t = tuple(int(v) for v in anchor)
            for t in range(1, want + 1):
                c = tuple(int(round(v)) for v in (a + sign * h * t))
                if c == anchor_t or (out and c == out[-1]):
                    continue
                if not _in_bounds(blocked, c) or blocked[c]:
                    break
                out.append(c)
            return out

        start_stub = _stub_cells(cell_seq[0], req.start_heading, +1)
        # The arrival stub marches backward from the goal, against the travel direction.
        end_stub = _stub_cells(cell_seq[-1], req.end_heading, -1)
        orig_start = tuple(int(v) for v in cell_seq[0])
        orig_goal = tuple(int(v) for v in cell_seq[-1])
        if start_stub:
            cell_seq[0] = start_stub[-1]      # route from the stub's far end
        if end_stub:
            cell_seq[-1] = end_stub[-1]       # route to the arrival stub's far end

        all_cells: list[tuple[int, int, int]] = list(start_stub)
        wp_cell_idx: list[int] = []   # index in all_cells of each waypoint's cell
        n_legs = len(cell_seq) - 1
        # Heading continuity across legs: each leg leaves a waypoint along the heading it
        # arrived with, so the bend penalty applies through the waypoint instead of a free
        # sharp turn. Legs are solved independently, so without this the join kinks.
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
                # The continuity heading over-constrained this leg (e.g. a hairpin waypoint
                # demanding a >45 degree turn); retry without it rather than fail.
                leg = self._solve_leg(
                    planner, stack, req.wire, req.weights, req.connectivity, a, b,
                    planner_extra, clearance_m=0.0, start_heading=None,
                    goal_heading=gh,
                )
            if leg is None and li == 0 and start_stub:
                # The departure runway dead-ends (stub far end boxed in). Drop the stub and
                # route from the original start; the heading still gates the first step, so
                # this degrades rather than failing.
                start_stub = []
                cell_seq[0] = orig_start
                a = orig_start
                all_cells = []
                leg = self._solve_leg(
                    planner, stack, req.wire, req.weights, req.connectivity, a, b,
                    planner_extra, clearance_m=0.0, start_heading=sh, goal_heading=gh,
                )
            if leg is None and li == n_legs - 1 and end_stub:
                # same graceful degradation for a boxed-in arrival runway
                end_stub = []
                cell_seq[-1] = orig_goal
                b = orig_goal
                leg = self._solve_leg(
                    planner, stack, req.wire, req.weights, req.connectivity, a, b,
                    planner_extra, clearance_m=0.0, start_heading=sh, goal_heading=gh,
                )
            if leg is None:
                # Diagnose against prior wires and the real clearance, not the shell folded
                # into extra, so "overlaps another already-routed wire" is only said when a
                # wire is actually there and clearance-sealed cases name the clearance.
                reason = diagnose_no_path(
                    stack, req.wire, req.connectivity, a, b, prior,
                    clearance_m=clr,
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

        # Append the forced arrival run: effective goal -> ... -> the original goal cell.
        suffix_begin = None
        if end_stub:
            suffix_begin = len(all_cells)
            all_cells.extend(list(reversed(end_stub[:-1])) + [orig_goal])

        # Build the world polyline, anchored to the endpoint markers themselves: the start
        # cell isn't in all_cells (the source links to a neighbour of it) and markers sit
        # at sub-cell positions, so otherwise the tube looks detached. Each waypoint's cell
        # centre is replaced by the waypoint marker, and start/waypoints/end are marked as
        # hard points the smoother must pass through.
        cell_world = [np.asarray(frame.grid_to_world(c), dtype=np.float64) for c in all_cells]
        wpset = set(wp_cell_idx)
        for k, ci in enumerate(wp_cell_idx):
            if k < len(req.waypoints):
                cell_world[ci] = np.asarray(req.waypoints[k], dtype=np.float64)

        # Stub cells snap onto the heading ray from the marker (voxel centres can sit up to
        # half a cell off the arrow line). The stubs' far ends become pass-through-fixed
        # points so smoothing keeps the straight run straight.
        stub_fixed: set[int] = set()
        if start_stub:
            hs = np.asarray(req.start_heading, dtype=np.float64)
            hs = hs / np.linalg.norm(hs)
            for i in range(len(start_stub)):
                cell_world[i] = start_world + hs * cell * (i + 1)
            stub_fixed.add(len(start_stub) - 1)
        if suffix_begin is not None:
            he = np.asarray(req.end_heading, dtype=np.float64)
            he = he / np.linalg.norm(he)
            last = len(all_cells) - 1
            for idx in range(suffix_begin, len(all_cells)):
                cell_world[idx] = end_world - he * cell * (last - idx)
            stub_fixed.add(suffix_begin - 1)   # the leg's final cell = the stub's far end

        # Anchor to the markers, or, when an endpoint was buried, to the open point it was
        # relocated to (start_world / end_world), so the tube ends in free space.
        pts = [start_world]
        flags = [True]
        for j, p in enumerate(cell_world):
            pts.append(p)
            flags.append(j in wpset or j in stub_fixed)
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

        # Local optimizer (default: fibre-neutre least-squares smoothing), collision-safe
        # against the same prohibited voxels the planner and relocation used.
        # weights["smoothing"] is the strength knob; 0 passes the raw path through.
        strength = float(req.weights.get("smoothing", 1.0))
        if len(polyline) >= 3:
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
        """Route greedily in priority order; earlier routes become obstacles for later ones.

        Lower priority numbers route first. Each routed wire is marked dilated by its
        radius, i.e. its actual tube body rather than a 1-cell centerline; route_one
        additionally grows prior obstacles by the next wire's radius, so two tube bodies
        keep r_a + r_b apart instead of overlapping at fine resolutions.
        """
        ordered = sorted(requests, key=lambda r: r.priority)
        occupied = np.zeros(stack.frame.res_xyz, dtype=bool)
        cell = float(stack.frame.cell_size)
        results: list[RouteResult] = []
        for req in ordered:
            res = self.route_one(stack, req, extra_obstacles=occupied)
            if res.status == "routed" and res.cells:
                m = np.zeros(stack.frame.res_xyz, dtype=bool)
                for c in res.cells:
                    m[c] = True
                rc = int(req.wire.radius_m / cell + 0.5 + 1e-9) if cell > 0 else 0
                if rc > 0:
                    m = dilate6(m, rc)
                occupied |= m
            results.append(res)
        return SolveReport(results=results)
