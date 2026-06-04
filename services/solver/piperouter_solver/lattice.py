from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .fields import melt_mask, neighbor_offsets, soft_cost_field, turn_penalty

# Heading-pin cone: a pinned departure/arrival heading only admits neighbor
# offsets whose unit direction is within this half-angle of the pinned vector.
_HEADING_COS = float(np.cos(np.pi / 4.0))  # 45 degrees

# 6-connectivity neighbour offsets (used for endpoint reachability / freeing).
_NB6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def diagnose_no_path(stack, wire, connectivity, start_cell, goal_cell,
                     extra_obstacles=None, clearance_m=0.0,
                     start_heading=None, goal_heading=None):
    """Explain WHY a leg has no path, using the SAME hard masks build() applies.

    Mirrors how build() wires endpoints: the SOURCE links to free NEIGHBOURS of the
    start cell (a pinned heading restricts which neighbours count), and the goal must
    itself be free/freeable and approachable. So we classify, in order: start
    unreachable -> start blocker; start free but heading kills it -> heading; same for
    goal; otherwise 'no corridor'. Returns a one-line human-readable string."""
    frame = stack.frame
    nx, ny, nz = frame.res_xyz
    melt = melt_mask(stack, wire)
    mesh_r = stack.dilate_occupancy(wire.radius_m).astype(bool)
    extra = (np.asarray(extra_obstacles, dtype=bool) if extra_obstacles is not None
             else np.zeros((nx, ny, nz), dtype=bool))
    base = mesh_r | melt | extra
    clearance_m = float(clearance_m)
    if clearance_m > 0.0:
        blocked = stack.dilate_occupancy(wire.radius_m + clearance_m).astype(bool) | melt | extra
    else:
        blocked = base
    free_base = ~base
    thermal = stack.thermal
    offs = np.asarray(neighbor_offsets(connectivity), dtype=np.int64)
    offs_dir = offs / np.linalg.norm(offs, axis=1, keepdims=True)

    def _inb(c):
        return 0 <= c[0] < nx and 0 <= c[1] < ny and 0 <= c[2] < nz

    def _reachable(c):  # has a mesh-free 6-neighbour (build's endpoint-freeing test)
        for d in _NB6:
            n = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
            if _inb(n) and free_base[n]:
                return True
        return False

    def _neighbours(cell, heading, sign):
        """(has_free, has_free_in_cone) over the start/goal cell's neighbours. sign=+1
        for the start (departure offset), -1 for the goal (the cell it's entered FROM)."""
        c = np.asarray(cell, dtype=np.int64)
        hn = None
        if heading is not None:
            h = np.asarray(heading, dtype=np.float64)
            ln = np.linalg.norm(h)
            hn = h / ln if ln > 1e-9 else None
        has_free = has_cone = False
        for oi in range(len(offs)):
            n = tuple(int(v) for v in (c + sign * offs[oi]))
            if not _inb(n) or blocked[n]:
                continue
            has_free = True
            if hn is None or float(offs_dir[oi] @ hn) >= _HEADING_COS:
                has_cone = True
        return has_free, has_cone

    def _blocker(cell, label):
        """Name the dominant hard blocker AT the cell or its nearest blocked neighbour."""
        c = tuple(int(v) for v in cell)
        cand = [c] + [tuple(int(v) for v in (np.asarray(c) + o)) for o in offs]
        cand = [x for x in cand if _inb(x)]
        for x in cand:
            if melt[x] and not mesh_r[x] and not extra[x]:
                return (f"{label} is in a {thermal[x]:.0f}C zone, hotter than this "
                        f"{wire.kind}'s {wire.max_temp_c:.0f}C rating. Move it away from "
                        f"the heat source or pick a higher-temperature type.")
        for x in cand:
            if mesh_r[x]:
                return f"{label} is buried inside an obstacle (no open space around it)."
        for x in cand:
            if extra[x]:
                return f"{label} overlaps another already-routed wire."
        return (f"{label} sits inside the {clearance_m * 1000:.0f}mm safety-clearance "
                f"keep-out. Lower the safety clearance or move the endpoint out of it.")

    # --- start: source needs a free neighbour (within the heading cone if pinned) ---
    s_free, s_cone = _neighbours(start_cell, start_heading, +1)
    if not s_free:
        return _blocker(start_cell, "Start")
    if not s_cone:
        return ("Pinned start heading points straight into an obstacle. Clear that "
                "direction or set the start heading to None.")

    # --- goal: must itself be free/freeable AND approachable (cone if pinned) ---
    g = tuple(int(v) for v in goal_cell)
    g_ok = (not blocked[g]) or (base[g] and _reachable(g))
    g_free, g_cone = _neighbours(goal_cell, goal_heading, -1)
    if not g_ok or not g_free:
        return _blocker(goal_cell, "End")
    if not g_cone:
        return ("Pinned end heading points straight into an obstacle. Clear that "
                "direction or set the end heading to None.")

    return ("No clear corridor between the two ends — obstacles or the safety clearance "
            "seal every path. Lower the clearance, move a waypoint, or raise resolution.")


@dataclass
class LatticeGraph:
    src: np.ndarray            # int32 edge sources
    dst: np.ndarray            # int32 edge destinations
    weight: np.ndarray         # float32 edge weights
    n_nodes: int
    source_id: int
    sink_id: int
    heads: int                 # H = number of headings per cell
    ordinal_cells: np.ndarray  # (n_free, 3) int32: cell ijk for each free-cell ordinal

    def cell_of(self, node_id: int) -> tuple[int, int, int]:
        """Map a (cell x heading) node id back to its cell. Not valid for
        source/sink ids (callers skip those)."""
        c = self.ordinal_cells[node_id // self.heads]
        return (int(c[0]), int(c[1]), int(c[2]))


class LatticeBuilder(Protocol):
    def build(
        self, stack, wire, weights: dict, connectivity: int,
        start_cell, goal_cell, extra_obstacles, clearance_m,
    ) -> LatticeGraph: ...


class ExpandedLatticeBuilder:
    """Option 1: full (cell x heading) state expansion (spec §5), vectorized.

    State node = (free-cell ordinal) * H + heading_index, where heading_index is
    the index of the offset by which the cell was *entered*. Turn cost between an
    entry heading and an exit heading is path-dependent, which is why the heading
    lives in the node. Edges, source links, and sink links are assembled with numpy
    so the build scales to car-sized grids.
    """

    def build(
        self, stack, wire, weights: dict, connectivity: int,
        start_cell, goal_cell, extra_obstacles=None, clearance_m=0.0,
        start_heading=None, goal_heading=None,
    ) -> LatticeGraph:
        frame = stack.frame
        nx, ny, nz = frame.res_xyz
        offs = np.asarray(neighbor_offsets(connectivity), dtype=np.int64)  # (H,3)
        H = len(offs)

        # --- hard constraints: forbidden cells ---
        # Occupancy is grown by the wire's body radius PLUS the safety clearance, so
        # the route keeps (radius + clearance) away from any mesh. clearance 0 keeps
        # only the wire body out of meshes (it may run flush against a surface).
        clearance_m = float(clearance_m)
        extra = np.asarray(extra_obstacles, dtype=bool) if extra_obstacles is not None else None
        melt = melt_mask(stack, wire)

        # `base` = blocked by the MESH itself (+ wire radius, melt, other routes).
        # `blocked` additionally grows by the safety clearance.
        base = stack.dilate_occupancy(wire.radius_m).astype(bool) | melt
        if extra is not None:
            base |= extra
        if clearance_m > 0.0:
            blocked = stack.dilate_occupancy(wire.radius_m + clearance_m).astype(bool) | melt
            if extra is not None:
                blocked |= extra
        else:
            blocked = base.copy()

        start_cell = tuple(int(v) for v in start_cell)
        goal_cell = tuple(int(v) for v in goal_cell)

        # Endpoints are fixed terminals. Free a blocked endpoint ONLY if it is blocked
        # by the mesh AND is reachable from open space (a connector sitting ON a
        # surface has at least one free neighbour). Do NOT free an endpoint that is:
        #   - buried deep in solid / in an entirely-blocked scene (no free neighbour), or
        #   - blocked only by the safety-clearance band (clearance too large here).
        # Such endpoints stay blocked, so routing correctly returns no_path instead of
        # inventing a path through solid geometry.
        free_base = ~base
        _NB6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

        def _reachable(c):
            i, j, k = c
            for dx, dy, dz in _NB6:
                ni, nj, nk = i + dx, j + dy, k + dz
                if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz and free_base[ni, nj, nk]:
                    return True
            return False

        for c in (start_cell, goal_cell):
            if not blocked[c]:
                continue                      # already in free space
            if base[c] and _reachable(c):
                blocked[c] = False            # connector on a surface -> connect to it
            # else: buried in solid, or clearance-only -> leave blocked -> no_path
        free = ~blocked

        # --- compact free-cell ordinals (C-order, matching np.argwhere) ---
        flat_free = free.reshape(-1)
        n_free = int(flat_free.sum())
        cell_ord_flat = np.full(flat_free.size, -1, dtype=np.int64)
        cell_ord_flat[flat_free] = np.arange(n_free, dtype=np.int64)
        cell_ord = cell_ord_flat.reshape(nx, ny, nz)
        ordinal_cells = np.argwhere(free).astype(np.int32)  # (n_free, 3)

        source_id = n_free * H
        sink_id = n_free * H + 1
        n_nodes = n_free * H + 2

        soft = soft_cost_field(stack, wire, weights)
        cell_mm = frame.cell_size * 1000.0
        step_len = frame.cell_size * np.sqrt((offs ** 2).sum(axis=1))  # (H,)

        # turn-penalty lookup: turn_lut[h_in, h_out], scaled by the soft "bend" weight
        # (0 = allow sharp turns, >1 = prefer straighter/gentler routes). Bend radius
        # is intrinsic via the wire's min_bend_radius; this slider scales how hard we
        # weigh it.
        bend_w = float(weights.get("bend", 1.0))
        turn_lut = np.zeros((H, H), dtype=np.float64)
        for a in range(H):
            for b in range(H):
                turn_lut[a, b] = bend_w * turn_penalty(
                    tuple(int(v) for v in offs[a]),
                    tuple(int(v) for v in offs[b]),
                    wire.min_bend_radius_mm, cell_mm,
                )

        arange_h = np.arange(H, dtype=np.int64)
        src_parts: list[np.ndarray] = []
        dst_parts: list[np.ndarray] = []
        w_parts: list[np.ndarray] = []

        # --- bulk edges: every valid move (a -> b via offset oi), broadcast over
        #     the H entry headings of a ---
        for oi in range(H):
            dx, dy, dz = (int(offs[oi, 0]), int(offs[oi, 1]), int(offs[oi, 2]))
            sx0, sx1 = max(0, -dx), nx - max(0, dx)
            sy0, sy1 = max(0, -dy), ny - max(0, dy)
            sz0, sz1 = max(0, -dz), nz - max(0, dz)
            if sx0 >= sx1 or sy0 >= sy1 or sz0 >= sz1:
                continue
            a_free = free[sx0:sx1, sy0:sy1, sz0:sz1]
            b_free = free[sx0 + dx:sx1 + dx, sy0 + dy:sy1 + dy, sz0 + dz:sz1 + dz]
            valid = a_free & b_free
            if not valid.any():
                continue
            a_ord = cell_ord[sx0:sx1, sy0:sy1, sz0:sz1][valid]
            b_ord = cell_ord[sx0 + dx:sx1 + dx, sy0 + dy:sy1 + dy, sz0 + dz:sz1 + dz][valid]
            soft_b = soft[sx0 + dx:sx1 + dx, sy0 + dy:sy1 + dy, sz0 + dz:sz1 + dz][valid]
            base = step_len[oi] * (1.0 + soft_b.astype(np.float64))  # (M,)
            m = a_ord.shape[0]
            # each of the H entry headings of a -> the single dst node (b, oi)
            src_parts.append((a_ord[:, None] * H + arange_h[None, :]).reshape(-1))
            dst_node = (b_ord * H + oi)[:, None]  # (M, 1)
            dst_parts.append(np.broadcast_to(dst_node, (m, H)).reshape(-1))
            w_parts.append((base[:, None] + turn_lut[:, oi][None, :]).reshape(-1))

        # Unit direction of each neighbor offset, for heading-pin cone tests.
        offs_norm = np.linalg.norm(offs, axis=1, keepdims=True)
        offs_dir = offs / offs_norm  # (H,3) unit vectors

        def _cone_mask(heading):
            if heading is None:
                return np.ones(H, dtype=bool)
            h = np.asarray(heading, dtype=np.float64)
            n = np.linalg.norm(h)
            if n <= 1e-9:
                return np.ones(H, dtype=bool)
            return (offs_dir @ (h / n)) >= _HEADING_COS

        allowed_src = _cone_mask(start_heading)
        allowed_goal = _cone_mask(goal_heading)

        # --- source -> first cell (no turn penalty; start has no entry heading) ---
        si, sj, sk = start_cell
        for oi in range(H):
            if not allowed_src[oi]:
                continue
            dx, dy, dz = (int(offs[oi, 0]), int(offs[oi, 1]), int(offs[oi, 2]))
            ni, nj, nk = si + dx, sj + dy, sk + dz
            if 0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz and free[ni, nj, nk]:
                b_ord = int(cell_ord[ni, nj, nk])
                base = step_len[oi] * (1.0 + float(soft[ni, nj, nk]))
                src_parts.append(np.array([source_id], dtype=np.int64))
                dst_parts.append(np.array([b_ord * H + oi], dtype=np.int64))
                w_parts.append(np.array([base], dtype=np.float64))

        # --- goal heading-nodes within the arrival cone -> sink (weight 0) ---
        g_ord = int(cell_ord[goal_cell])
        if g_ord >= 0:
            hs = np.nonzero(allowed_goal)[0].astype(np.int64)
            if hs.size:
                src_parts.append(g_ord * H + hs)
                dst_parts.append(np.full(hs.size, sink_id, dtype=np.int64))
                w_parts.append(np.zeros(hs.size, dtype=np.float64))

        if src_parts:
            src = np.concatenate(src_parts).astype(np.int32)
            dst = np.concatenate(dst_parts).astype(np.int32)
            weight = np.concatenate(w_parts).astype(np.float32)
        else:
            src = np.zeros(0, dtype=np.int32)
            dst = np.zeros(0, dtype=np.int32)
            weight = np.zeros(0, dtype=np.float32)

        return LatticeGraph(
            src=src, dst=dst, weight=weight, n_nodes=n_nodes,
            source_id=source_id, sink_id=sink_id, heads=H,
            ordinal_cells=ordinal_cells,
        )
