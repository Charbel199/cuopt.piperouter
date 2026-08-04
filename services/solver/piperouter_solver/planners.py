"""Global planners: a coarse, collision-free path of grid cells from a start cell to a
goal cell.

Planners are interchangeable so they can be compared against each other (bench_algos.py);
'lattice' is the heading-aware default, the rest are other families (plain grid A*,
Eikonal cost-to-go descent, sampling-based RRT-Connect).

Every planner returns a list of (i,j,k) cells from start to goal (no source/sink nodes)
or None, and searches only cells that are free after dilating the occupancy by the wire
radius plus clearance and removing over-temperature (melt) cells.
"""
from __future__ import annotations

import heapq
import logging

import numpy as np

from . import fields, stencil
from .backend import shortest_path
from .grids import _xp
from .lattice import ExpandedLatticeBuilder

_log = logging.getLogger("piperouter")


def blocked_mask(stack, wire, clearance_m, extra_obstacles):
    """Cells the wire may not occupy: mesh+clearance dilation, melt cells, prior routes."""
    blocked = (stack.dilate_occupancy(wire.radius_m + clearance_m).astype(bool)
               | fields.melt_mask(stack, wire))
    if extra_obstacles is not None:
        blocked = blocked | np.asarray(extra_obstacles, dtype=bool)
    return blocked


def _nearest_free(blocked, cell, max_r=2):
    """Return the closest free cell to `cell` within max_r, or None.

    Endpoints often sit just inside the clearance halo, so a small snap is enough."""
    nx, ny, nz = blocked.shape
    ci, cj, ck = (int(cell[0]), int(cell[1]), int(cell[2]))
    best, best_d = None, 1e18
    for di in range(-max_r, max_r + 1):
        for dj in range(-max_r, max_r + 1):
            for dk in range(-max_r, max_r + 1):
                i, j, k = ci + di, cj + dj, ck + dk
                if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz and not blocked[i, j, k]:
                    d = di * di + dj * dj + dk * dk
                    if d < best_d:
                        best, best_d = (i, j, k), d
    return best


class LatticeGlobal:
    """Heading-expanded lattice + SSSP. Bend-aware, and the default planner."""
    name = "lattice"

    def __init__(self):
        self._builder = ExpandedLatticeBuilder()

    def plan(self, stack, wire, weights, connectivity, start_cell, goal_cell,
             extra_obstacles, clearance_m, start_heading, goal_heading):
        g = self._builder.build(
            stack, wire, weights, connectivity, start_cell, goal_cell, extra_obstacles,
            clearance_m=clearance_m, start_heading=start_heading, goal_heading=goal_heading)
        path, _cost = shortest_path(g.src, g.dst, g.weight, g.n_nodes,
                                    g.source_id, g.sink_id)
        if path is None:
            return None
        cells = []
        for node in path:
            if node in (g.source_id, g.sink_id):
                continue
            c = g.cell_of(node)
            if not cells or cells[-1] != c:
                cells.append(c)
        return cells


class _GridPlannerBase:
    """Shared setup for the plain-cell planners (no heading state)."""

    def _prep(self, stack, wire, weights, connectivity, start_cell, goal_cell,
              extra_obstacles, clearance_m):
        blocked = blocked_mask(stack, wire, clearance_m, extra_obstacles)
        a = tuple(int(v) for v in start_cell)
        b = tuple(int(v) for v in goal_cell)
        nx, ny, nz = blocked.shape
        if not (0 <= a[0] < nx and 0 <= a[1] < ny and 0 <= a[2] < nz):
            return None
        # Same endpoint rule as the lattice: route when a free neighbour is one step away
        # (a connector sitting on or just inside a surface), but reject an endpoint buried
        # deeper than that rather than fabricating a route by relocating it far away.
        if blocked[a]:
            a = _nearest_free(blocked, a, max_r=1)
        if blocked[b]:
            b = _nearest_free(blocked, b, max_r=1)
        if a is None or b is None:
            return None
        soft = fields.soft_cost_field(stack, wire, weights)
        offs = fields.neighbor_offsets(connectivity)
        cell = float(stack.frame.cell_size)
        step = [cell * float(np.linalg.norm(o)) for o in offs]
        return blocked, a, b, soft, offs, step

    @staticmethod
    def _neighbors(c, offs, blocked):
        nx, ny, nz = blocked.shape
        for oi, (dx, dy, dz) in enumerate(offs):
            i, j, k = c[0] + dx, c[1] + dy, c[2] + dz
            if 0 <= i < nx and 0 <= j < ny and 0 <= k < nz and not blocked[i, j, k]:
                yield oi, (i, j, k)


class AStarGlobal(_GridPlannerBase):
    """Plain uniform-grid A*, no heading expansion.

    Cheap because there is no H^2 edge blow-up, but it leans on the local optimizer for
    smoothness. Edge cost = step_len * (1 + soft[neighbor])."""
    name = "astar"

    def plan(self, stack, wire, weights, connectivity, start_cell, goal_cell,
             extra_obstacles, clearance_m, start_heading=None, goal_heading=None):
        prep = self._prep(stack, wire, weights, connectivity, start_cell, goal_cell,
                          extra_obstacles, clearance_m)
        if prep is None:
            return None
        blocked, a, b, soft, offs, step = prep
        cell = float(stack.frame.cell_size)
        bb = np.asarray(b, dtype=float)

        def h(c):
            return cell * float(np.linalg.norm(np.asarray(c, dtype=float) - bb))

        g = {a: 0.0}
        came = {}
        pq = [(h(a), a)]
        while pq:
            _f, c = heapq.heappop(pq)
            if c == b:
                path = [c]
                while c in came:
                    c = came[c]
                    path.append(c)
                return path[::-1]
            gc = g[c]
            for oi, nb in self._neighbors(c, offs, blocked):
                ng = gc + step[oi] * (1.0 + float(soft[nb]))
                if ng < g.get(nb, 1e18):
                    g[nb] = ng
                    came[nb] = c
                    heapq.heappush(pq, (ng + h(nb), nb))
        return None


class FMMGlobal(_GridPlannerBase):
    """Eikonal fast marching.

    Solves |∇T| = slowness for the arrival-time field T from the goal using the Godunov
    upwind quadratic update, which is sub-cell accurate and has less grid bias than
    graph-Dijkstra, then gradient-descends T from the start. slowness = 1 + soft, so the
    front advances slower through costly cells and the descent skirts them."""
    name = "fmm"

    def plan(self, stack, wire, weights, connectivity, start_cell, goal_cell,
             extra_obstacles, clearance_m, start_heading=None, goal_heading=None):
        prep = self._prep(stack, wire, weights, connectivity, start_cell, goal_cell,
                          extra_obstacles, clearance_m)
        if prep is None:
            return None
        blocked, a, b, soft, offs, step = prep
        nx, ny, nz = blocked.shape
        cell = float(stack.frame.cell_size)
        slow = (1.0 + soft).astype(np.float64)        # cost per metre
        INF = 1e18
        T = np.full(blocked.shape, INF)
        frozen = np.zeros(blocked.shape, dtype=bool)
        FACES = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))

        def godunov(i, j, k):
            f = slow[i, j, k] * cell                  # one-cell traversal cost
            mins = []
            for axis, (dp, dm) in enumerate(((1, -1),) * 3):
                pa, pb = list((i, j, k)), list((i, j, k))
                pa[axis] += 1
                pb[axis] -= 1
                m = INF
                for (ii, jj, kk) in (pa, pb):
                    if 0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz and frozen[ii, jj, kk]:
                        m = min(m, T[ii, jj, kk])
                if m < INF:
                    mins.append(m)
            mins.sort()
            t = mins[0] + f                            # 1-axis solution
            if len(mins) >= 2 and t > mins[1]:
                a2, b2 = mins[0], mins[1]
                s = a2 + b2
                disc = s * s - 2.0 * (a2 * a2 + b2 * b2 - f * f)
                if disc >= 0:
                    t = 0.5 * (s + disc ** 0.5)
                if len(mins) >= 3 and t > mins[2]:
                    a3, b3, c3 = mins
                    s = a3 + b3 + c3
                    disc = s * s - 3.0 * (a3 * a3 + b3 * b3 + c3 * c3 - f * f)
                    if disc >= 0:
                        t = (s + disc ** 0.5) / 3.0
            return t

        T[b] = 0.0
        heap = [(0.0, b)]
        while heap:
            tval, c = heapq.heappop(heap)
            if frozen[c]:
                continue
            frozen[c] = True
            ci, cj, ck = c
            for dx, dy, dz in FACES:
                nb = (ci + dx, cj + dy, ck + dz)
                if not (0 <= nb[0] < nx and 0 <= nb[1] < ny and 0 <= nb[2] < nz):
                    continue
                if blocked[nb] or frozen[nb]:
                    continue
                nt = godunov(*nb)
                if nt < T[nb]:
                    T[nb] = nt
                    heapq.heappush(heap, (nt, nb))
                if a == nb and frozen[a]:
                    pass
            if frozen[a]:
                break
        if T[a] >= INF:
            return None
        # Descend T from start to goal, stepping over the full connectivity.
        path = [a]
        c = a
        seen = {a}
        while c != b and len(path) < blocked.size:
            best, best_c = T[c], None
            for _oi, nb in self._neighbors(c, offs, blocked):
                if nb not in seen and T[nb] < best:
                    best, best_c = T[nb], nb
            if best_c is None:
                return None
            path.append(best_c)
            seen.add(best_c)
            c = best_c
        return path


class RRTGlobal(_GridPlannerBase):
    """RRT-Connect on the free voxel grid. The RNG is seeded, so runs are deterministic.

    Output is jagged and relies on the local optimizer to smooth it."""
    name = "rrt"

    def __init__(self, seed=12345, step_cells=6, max_iter=4000):
        self.seed = seed
        self.step_cells = step_cells
        self.max_iter = max_iter

    def _steer(self, frm, to, blocked):
        """Walk from `frm` toward `to` up to step_cells, stopping before any blocked cell.

        Returns the last reachable cell, which may be `frm` itself."""
        frm = np.asarray(frm, dtype=float)
        to = np.asarray(to, dtype=float)
        d = to - frm
        dist = float(np.linalg.norm(d))
        if dist < 1e-9:
            return tuple(int(v) for v in frm)
        n = min(self.step_cells, int(np.ceil(dist)))
        last = tuple(int(round(v)) for v in frm)
        for s in range(1, n + 1):
            p = frm + d * (s / max(dist, 1e-9))
            c = tuple(int(round(v)) for v in p)
            if not (0 <= c[0] < blocked.shape[0] and 0 <= c[1] < blocked.shape[1]
                    and 0 <= c[2] < blocked.shape[2]) or blocked[c]:
                break
            last = c
        return last

    def plan(self, stack, wire, weights, connectivity, start_cell, goal_cell,
             extra_obstacles, clearance_m, start_heading=None, goal_heading=None):
        prep = self._prep(stack, wire, weights, connectivity, start_cell, goal_cell,
                          extra_obstacles, clearance_m)
        if prep is None:
            return None
        blocked, a, b, _soft, _offs, _step = prep
        nx, ny, nz = blocked.shape
        rng = np.random.RandomState(self.seed)
        # Two trees, each a node list plus a parent index.
        ta, pa = [a], [-1]
        tb, pb = [b], [-1]

        def nearest(tree, c):
            arr = np.asarray(tree, dtype=float)
            return int(np.argmin(((arr - np.asarray(c, dtype=float)) ** 2).sum(axis=1)))

        def extend(tree, parent, target):
            ni = nearest(tree, target)
            new = self._steer(tree[ni], target, blocked)
            if new != tree[ni]:
                tree.append(new)
                parent.append(ni)
                return len(tree) - 1
            return None

        for _ in range(self.max_iter):
            sample = (rng.randint(nx), rng.randint(ny), rng.randint(nz))
            if blocked[sample]:
                continue
            gi = extend(ta, pa, sample)
            if gi is None:
                continue
            # Try to connect tree B to the new node of A.
            ci = extend(tb, pb, ta[gi])
            if ci is not None and tb[ci] == ta[gi]:
                # Joined: reconstruct a->join, then join->b.
                left = []
                k = gi
                while k != -1:
                    left.append(ta[k]); k = pa[k]
                left.reverse()
                right = []
                k = ci
                while k != -1:
                    right.append(tb[k]); k = pb[k]
                # `right` runs join..b, so drop its duplicate join node.
                return left + right[1:]
            ta, pa, tb, pb = tb, pb, ta, pa   # swap trees (RRT-Connect)
        return None


def _raster_cells(a, b):
    """Cells along the straight segment a->b, inclusive (3D DDA).

    The caller is responsible for both ends and the segment between them being free."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    steps = int(np.max(np.abs(b - a))) or 1
    out = []
    for s in range(steps + 1):
        p = a + (b - a) * (s / steps)
        c = tuple(int(round(v)) for v in p)
        if not out or out[-1] != c:
            out.append(c)
    return out


class _CSRAdj:
    """Read-only leaf adjacency in CSR storage (node id -> array of neighbour ids).

    Supports only the `.get(lid, default)` access pattern the planners use."""
    __slots__ = ("indptr", "indices")

    def __init__(self, indptr, indices):
        self.indptr = indptr
        self.indices = indices

    def get(self, lid, default=()):
        lid = int(lid)
        if lid < 0 or lid + 1 >= self.indptr.size:
            return default
        i0, i1 = int(self.indptr[lid]), int(self.indptr[lid + 1])
        return self.indices[i0:i1] if i1 > i0 else default


def leaf_adjacency(leaf_of, n_leaves):
    """Face-adjacency of the octree's free leaves, as a `_CSRAdj`.

    Compares each axis's neighbouring slabs (`leaf_of[:-1]` vs `leaf_of[1:]`) to find
    boundary cells whose two sides belong to different free leaves, dedups the (lo,hi)
    pairs and groups the unique edges into CSR. Array ops throughout, on cupy when
    PIPEROUTER_GPU_BUILD=1."""
    xp = _xp()
    lof = xp.asarray(leaf_of)
    pa, pb = [], []
    for x, y in ((lof[:-1, :, :], lof[1:, :, :]),
                 (lof[:, :-1, :], lof[:, 1:, :]),
                 (lof[:, :, :-1], lof[:, :, 1:])):
        m = (x >= 0) & (y >= 0) & (x != y)
        pa.append(x[m].ravel())
        pb.append(y[m].ravel())
    a = xp.concatenate(pa).astype(xp.int64)
    if a.size == 0:
        return _CSRAdj(np.zeros(max(n_leaves, 0) + 1, dtype=np.int64),
                       np.empty(0, dtype=np.int64))
    b = xp.concatenate(pb).astype(xp.int64)
    lo = xp.minimum(a, b)
    hi = xp.maximum(a, b)
    keys = xp.unique(lo * np.int64(n_leaves) + hi)     # unique undirected leaf-pairs
    u = keys // n_leaves
    v = keys % n_leaves
    src = xp.concatenate([u, v])
    dst = xp.concatenate([v, u])
    order = xp.argsort(src)
    dst = dst[order]
    counts = xp.bincount(src, minlength=n_leaves)
    indptr = xp.zeros(n_leaves + 1, dtype=xp.int64)
    indptr[1:] = xp.cumsum(counts)
    if xp is not np:
        dst = xp.asnumpy(dst)
        indptr = xp.asnumpy(indptr)
    return _CSRAdj(indptr, dst)


def octree_leaves(blocked):
    """Subdivide `blocked` into octree free leaves.

    Returns (ranges, leaf_of): `ranges` is an (n, 6) array of (i0,i1,j0,j1,k0,k1)
    free-leaf boxes and leaf_of[i,j,k] is a leaf id, or -1 for a blocked cell. Geometry
    only, independent of soft costs and prior routes, so it can be cached per scene.

    Each subdivision level classifies all of its boxes at once: a padded 3D integral
    image (summed-area table) turns a box's any()/all() test into an 8-corner gather, and
    children are emitted with array ops. Python cost is therefore per level (~log n)
    rather than per box."""
    blocked = np.asarray(blocked, dtype=bool)
    nx, ny, nz = blocked.shape
    leaf_of = np.full(blocked.shape, -1, dtype=np.int32)
    xp = _xp()
    # Padded integral image: P[i,j,k] = count of blocked cells in blocked[:i,:j,:k].
    # int32 is safe (max count = total cells < 2^31 for any grid we can hold in RAM).
    P = xp.zeros((nx + 1, ny + 1, nz + 1), dtype=np.int32)
    P[1:, 1:, 1:] = xp.asarray(blocked)
    # In-place cumsum along each axis (the zero padding at index 0 is preserved).
    xp.cumsum(P, axis=0, out=P)
    xp.cumsum(P, axis=1, out=P)
    xp.cumsum(P, axis=2, out=P)

    # Boxes at the current level, as columns (n, 6): i0,i1,j0,j1,k0,k1. The whole
    # classify/subdivide loop runs on xp; only each level's free boxes come back to the
    # host, for leaf painting and the ranges array.
    level_free = []
    n_leaves = 0
    boxes = xp.asarray(np.array([[0, nx, 0, ny, 0, nz]], dtype=np.int32))
    while boxes.size:
        i0, i1, j0, j1, k0, k1 = (boxes[:, c] for c in range(6))
        cnt = (P[i1, j1, k1] - P[i0, j1, k1] - P[i1, j0, k1] - P[i1, j1, k0]
               + P[i0, j0, k1] + P[i0, j1, k0] + P[i1, j0, k0] - P[i0, j0, k0])
        vol = (i1 - i0).astype(xp.int64) * (j1 - j0) * (k1 - k0)
        free = cnt == 0
        mixed = (cnt > 0) & (cnt < vol)          # fully-blocked boxes are dropped

        if bool(free.any()):
            fb = boxes[free]
            fb = xp.asnumpy(fb) if xp is not np else fb
            base = n_leaves
            n_leaves += len(fb)
            level_free.append(fb)
            single = (fb[:, 1] - fb[:, 0] == 1) & (fb[:, 3] - fb[:, 2] == 1) \
                     & (fb[:, 5] - fb[:, 4] == 1)
            # Most leaves on big grids are single cells; paint those in one scatter.
            sb = fb[single]
            leaf_of[sb[:, 0], sb[:, 2], sb[:, 4]] = base + np.flatnonzero(single)
            for off, box in zip(np.flatnonzero(~single), fb[~single]):
                a0, a1, b0, b1, c0, c1 = box
                leaf_of[a0:a1, b0:b1, c0:c1] = base + off

        if not bool(mixed.any()):
            break
        mb = boxes[mixed]
        i0, i1, j0, j1, k0, k1 = (mb[:, c] for c in range(6))
        mi, mj, mk = (i0 + i1) // 2, (j0 + j1) // 2, (k0 + k1) // 2
        sx, sy, sz = i1 - i0 > 1, j1 - j0 > 1, k1 - k0 > 1
        children = []
        for bx in (0, 1):
            ca0 = i0 if bx == 0 else mi
            ca1 = xp.where(sx, mi, i1) if bx == 0 else i1
            for by in (0, 1):
                cb0 = j0 if by == 0 else mj
                cb1 = xp.where(sy, mj, j1) if by == 0 else j1
                for bz in (0, 1):
                    cc0 = k0 if bz == 0 else mk
                    cc1 = xp.where(sz, mk, k1) if bz == 0 else k1
                    # There is no upper half along a size-1 axis.
                    valid = xp.ones(len(mb), dtype=bool)
                    if bx:
                        valid = valid & sx
                    if by:
                        valid = valid & sy
                    if bz:
                        valid = valid & sz
                    if bool(valid.any()):
                        children.append(xp.stack(
                            [xp.broadcast_to(ca0, valid.shape)[valid],
                             xp.broadcast_to(ca1, valid.shape)[valid],
                             xp.broadcast_to(cb0, valid.shape)[valid],
                             xp.broadcast_to(cb1, valid.shape)[valid],
                             xp.broadcast_to(cc0, valid.shape)[valid],
                             xp.broadcast_to(cc1, valid.shape)[valid]], axis=1))
        boxes = xp.concatenate(children, axis=0) if children else \
            xp.asarray(np.empty((0, 6), dtype=np.int32))
    ranges = (np.concatenate(level_free, axis=0).astype(np.int32) if level_free
              else np.empty((0, 6), dtype=np.int32))
    return ranges, leaf_of


def corridor_geometry(ranges, adj):
    """Return (leaf centres, geometric length of every CSR edge) for the corridor A*.

    Wire-independent, so it is cached alongside the scene octree. Centres are in cell
    coordinates, not metres."""
    r = np.asarray(ranges, dtype=np.float64).reshape(-1, 6)
    centers = np.stack([(r[:, 0] + r[:, 1] - 1) * 0.5,
                        (r[:, 2] + r[:, 3] - 1) * 0.5,
                        (r[:, 4] + r[:, 5] - 1) * 0.5], axis=1)
    d = centers[adj.indices] - centers[np.repeat(np.arange(len(centers)),
                                                 np.diff(adj.indptr))]
    elen = np.sqrt((d * d).sum(axis=1))
    return centers, elen


def octree_corridor(ranges, leaf_of, adj, start_cell, goal_cell, leaf_soft=None,
                    geo=None):
    """Coarse A* through free-leaf centres, start->goal, rasterized to a dense cell list.

    Returns the corridor cells or None. This is only a heuristic corridor for
    octree_lattice; the fine lattice does the real cost and collision routing inside the
    band around it. `geo` is `corridor_geometry(ranges, adj)`, cached per scene octree;
    per-edge weights and the per-leaf heuristic are one vectorized pass each so the A*
    loop touches only floats and allocates nothing per neighbour."""
    nx, ny, nz = leaf_of.shape

    def clamp(c):
        return (min(max(int(c[0]), 0), nx - 1), min(max(int(c[1]), 0), ny - 1),
                min(max(int(c[2]), 0), nz - 1))

    a, b = clamp(start_cell), clamp(goal_cell)
    sa, sb = int(leaf_of[a]), int(leaf_of[b])
    if sa < 0 or sb < 0:
        return None
    if geo is None:
        geo = corridor_geometry(ranges, adj)
    centers, elen = geo
    indptr, indices = adj.indptr, adj.indices
    # Bias each edge by the destination leaf's soft cost so the corridor heads toward
    # cheap cells (hugging surfaces, avoiding heat/EM) rather than always taking the
    # geometric shortest path through open air.
    w = elen * (1.0 + leaf_soft[indices]) if leaf_soft is not None else elen
    diff = centers - centers[sb]
    hval = np.sqrt((diff * diff).sum(axis=1))

    g = {sa: 0.0}
    came = {}
    pq = [(float(hval[sa]), sa)]
    found = False
    while pq:
        _f, lid = heapq.heappop(pq)
        if lid == sb:
            found = True
            break
        gc = g[lid]
        for e in range(int(indptr[lid]), int(indptr[lid + 1])):
            nb = int(indices[e])
            ng = gc + float(w[e])
            if ng < g.get(nb, 1e18):
                g[nb] = ng
                came[nb] = lid
                heapq.heappush(pq, (ng + float(hval[nb]), nb))
    if not found:
        return None
    chain = [sb]
    while chain[-1] in came:
        chain.append(came[chain[-1]])
    chain.reverse()
    waypts = [np.asarray(a, float)] + [centers[l] for l in chain] + [np.asarray(b, float)]
    cells = []
    for p, q in zip(waypts[:-1], waypts[1:]):
        for c in _raster_cells(p, q):
            if (0 <= c[0] < nx and 0 <= c[1] < ny and 0 <= c[2] < nz
                    and (not cells or cells[-1] != c)):
                cells.append(c)
    return cells or None


def band_mask(shape, pts, r):
    """Union of clipped (2r+1)-cube boxes centred on `pts`, as a bool mask of `shape`.

    Built as a scatter plus a separable box dilation (cupy when PIPEROUTER_GPU_BUILD=1).
    Points may lie outside the grid, since heading rays walk off it, and are clipped."""
    nx, ny, nz = shape
    r = int(r)
    p = np.asarray(pts, dtype=np.int64).reshape(-1, 3)
    keep = ((p[:, 0] >= -r) & (p[:, 0] < nx + r)
            & (p[:, 1] >= -r) & (p[:, 1] < ny + r)
            & (p[:, 2] >= -r) & (p[:, 2] < nz + r))
    p = p[keep] + r
    if len(p) == 0:
        return np.zeros(shape, dtype=bool)
    xp = _xp()
    grid = xp.zeros((nx + 2 * r, ny + 2 * r, nz + 2 * r), dtype=bool)
    px = xp.asarray(p)
    grid[px[:, 0], px[:, 1], px[:, 2]] = True
    for axis in range(3):
        out = grid.copy()
        for shift in range(1, r + 1):
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis] = slice(shift, None)
            hi[axis] = slice(None, -shift)
            out[tuple(lo)] |= grid[tuple(hi)]
            out[tuple(hi)] |= grid[tuple(lo)]
        grid = out
    band = grid[r:r + nx, r:r + ny, r:r + nz]
    return xp.asnumpy(band) if xp is not np else band


def leaf_soft_means(stack, rad_cells, soft, ranges, leaf_of):
    """Per-leaf mean of the soft-cost field.

    Cached on the stack per (octree radius, soft-field object). The soft field is itself
    cached per weights on the stack, so its id() is a stable key for one solve and wires
    sharing weights reuse the result. The denominator comes from box volumes and is
    exact, since a leaf is exactly its box."""
    cache = stack.__dict__.setdefault("_leaf_soft_cache", {})
    key = (int(rad_cells), id(soft))
    hit = cache.get(key)
    if hit is not None:
        return hit
    n = len(ranges)
    xp = _xp()
    flat = xp.asarray(leaf_of).ravel()
    m = flat >= 0
    sums = xp.bincount(flat[m], weights=xp.asarray(soft).ravel()[m], minlength=n)
    if xp is not np:
        sums = xp.asnumpy(sums)
    r = np.asarray(ranges, dtype=np.float64).reshape(-1, 6)
    vols = (r[:, 1] - r[:, 0]) * (r[:, 3] - r[:, 2]) * (r[:, 5] - r[:, 4])
    out = sums / np.maximum(vols, 1.0)
    cache[key] = out
    return out


class OctreeGlobal(_GridPlannerBase):
    """Adaptive-resolution planner over an octree of the grid.

    A box is kept whole when it is uniformly free or uniformly blocked and split
    otherwise, so open air becomes a few big leaves and only surfaces and narrow gaps get
    fine ones. A* runs over the free-leaf adjacency graph. The straight line between two
    face-adjacent free leaves is collision-free, so the leaf-centre chain rasterizes back
    to cells directly."""
    name = "octree"

    def plan(self, stack, wire, weights, connectivity, start_cell, goal_cell,
             extra_obstacles, clearance_m, start_heading=None, goal_heading=None):
        prep = self._prep(stack, wire, weights, connectivity, start_cell, goal_cell,
                          extra_obstacles, clearance_m)
        if prep is None:
            return None
        blocked, a, b, soft, _offs, _step = prep
        nx, ny, nz = blocked.shape
        cell = float(stack.frame.cell_size)

        # --- octree leaves + cell->leaf map for free leaves ---
        leaf_of = np.full(blocked.shape, -1, dtype=np.int64)
        leaves = []   # (cx, cy, cz) centre in float cell coords, then the mean soft cost
        stack_nodes = [(0, nx, 0, ny, 0, nz)]
        while stack_nodes:
            i0, i1, j0, j1, k0, k1 = stack_nodes.pop()
            sub = blocked[i0:i1, j0:j1, k0:k1]
            if not sub.any():                                   # uniformly free -> leaf
                lid = len(leaves)
                leaf_of[i0:i1, j0:j1, k0:k1] = lid
                leaves.append((0.5 * (i0 + i1 - 1), 0.5 * (j0 + j1 - 1),
                               0.5 * (k0 + k1 - 1),
                               float(soft[i0:i1, j0:j1, k0:k1].mean())))
                continue
            if sub.all():                                       # uniformly blocked -> drop
                continue
            mi, mj, mk = (i0 + i1) // 2, (j0 + j1) // 2, (k0 + k1) // 2
            xs = [(i0, i1)] if i1 - i0 <= 1 else [(i0, mi), (mi, i1)]
            ys = [(j0, j1)] if j1 - j0 <= 1 else [(j0, mj), (mj, j1)]
            zs = [(k0, k1)] if k1 - k0 <= 1 else [(k0, mk), (mk, k1)]
            for xa, xb in xs:
                for ya, yb in ys:
                    for za, zb in zs:
                        stack_nodes.append((xa, xb, ya, yb, za, zb))

        sa, sb = leaf_of[a], leaf_of[b]
        if sa < 0 or sb < 0:
            return None

        # --- adjacency: each free leaf -> its face-adjacent free leaves ---
        adj = leaf_adjacency(leaf_of, len(leaves))

        # --- A* over leaf centres ---
        def ctr(lid):
            return np.array(leaves[lid][:3])
        goal_c = ctr(sb)

        def h(lid):
            return cell * float(np.linalg.norm(ctr(lid) - goal_c))

        g = {sa: 0.0}
        came = {}
        pq = [(h(sa), sa)]
        found = False
        while pq:
            _f, lid = heapq.heappop(pq)
            if lid == sb:
                found = True
                break
            for nb in adj.get(lid, ()):
                w = cell * float(np.linalg.norm(ctr(nb) - ctr(lid))) * (1.0 + leaves[nb][3])
                ng = g[lid] + w
                if ng < g.get(nb, 1e18):
                    g[nb] = ng
                    came[nb] = lid
                    heapq.heappush(pq, (ng + h(nb), nb))
        if not found:
            return None

        chain = [sb]
        while chain[-1] in came:
            chain.append(came[chain[-1]])
        chain.reverse()
        # Leaf centres -> dense free cells, bracketed by the start and goal cells.
        waypts = [np.asarray(a, dtype=float)] + [ctr(l) for l in chain] + [np.asarray(b, dtype=float)]
        cells = []
        for p, q in zip(waypts[:-1], waypts[1:]):
            for c in _raster_cells(p, q):
                if (0 <= c[0] < nx and 0 <= c[1] < ny and 0 <= c[2] < nz
                        and not blocked[c] and (not cells or cells[-1] != c)):
                    cells.append(c)
        return cells or None


class MedialGlobal(_GridPlannerBase):
    """Clearance-seeking planner, in the flavour of a medial axis.

    Runs A* with an edge cost that penalizes cells with a small Euclidean distance to the
    nearest obstacle, so the route is pulled onto the high-clearance spine of free
    space."""
    name = "medial"

    def plan(self, stack, wire, weights, connectivity, start_cell, goal_cell,
             extra_obstacles, clearance_m, start_heading=None, goal_heading=None):
        prep = self._prep(stack, wire, weights, connectivity, start_cell, goal_cell,
                          extra_obstacles, clearance_m)
        if prep is None:
            return None
        from scipy import ndimage
        blocked, a, b, soft, offs, step = prep
        cell = float(stack.frame.cell_size)
        dist = ndimage.distance_transform_edt(~blocked).astype(np.float64)   # cells to obstacle
        dmax = float(dist.max()) or 1.0
        bb = np.asarray(b, dtype=float)
        w_clear = 4.0           # how hard to hug the clearance spine

        def h(c):
            return cell * float(np.linalg.norm(np.asarray(c, dtype=float) - bb))

        g = {a: 0.0}
        came = {}
        pq = [(h(a), a)]
        while pq:
            _f, c = heapq.heappop(pq)
            if c == b:
                path = [c]
                while c in came:
                    c = came[c]
                    path.append(c)
                return path[::-1]
            gc = g[c]
            for oi, nb in self._neighbors(c, offs, blocked):
                clear_pen = 1.0 + w_clear * (1.0 - dist[nb] / dmax)   # nearer obstacle, costlier
                ng = gc + step[oi] * (1.0 + float(soft[nb])) * clear_pen
                if ng < g.get(nb, 1e18):
                    g[nb] = ng
                    came[nb] = c
                    heapq.heappush(pq, (ng + h(nb), nb))
        return None


# Edge budget for the full-lattice fallback. The exhaustive (cell x heading) graph has
# roughly n_free * H^2 edges; past 1e9 the build stalls for minutes on CPU and runs a
# 95 GB GPU out of memory under cuGraph. Above the budget the fallback is skipped and the
# wire reports no_path rather than taking the whole solve down with it.
_FULL_FALLBACK_EDGE_BUDGET = 1.0e9


class OctreeLatticeGlobal(_GridPlannerBase):
    """Hierarchical octree plus heading lattice.

    The octree finds a cheap coarse corridor, then the bend-aware lattice runs only in a
    band of fine cells around it: cells outside the band are masked as obstacles, so the
    lattice expands a fraction of the volume. Min-bend routing is unchanged; the octree
    only prunes the open-air search. Falls back to the full lattice when the band is too
    tight, so coverage is never worse than `lattice`."""
    name = "octree_lattice"

    def __init__(self, band_cells=4):
        self.band = int(band_cells)
        self._lat = LatticeGlobal()

    def _scene_octree(self, stack, wire, clearance_m):
        """Geometry-only octree (leaves, leaf_of, adjacency), cached on the stack.

        Keyed by (wire radius + clearance) so a Route All reuses it across wires.
        Deliberately excludes prior routes; the fine lattice pass enforces those, which is
        what keeps this cache valid for a whole scene."""
        cell = float(stack.frame.cell_size)
        rad_cells = int(round((wire.radius_m + clearance_m) / cell)) if cell > 0 else 0
        cache = stack.__dict__.setdefault("_octree_cache", {})
        if rad_cells not in cache:
            blocked = stack.dilate_occupancy(wire.radius_m + clearance_m).astype(bool)
            ranges, leaf_of = octree_leaves(blocked)
            adj = leaf_adjacency(leaf_of, len(ranges))
            cache[rad_cells] = (ranges, leaf_of, adj, corridor_geometry(ranges, adj))
        return cache[rad_cells]

    def plan(self, stack, wire, weights, connectivity, start_cell, goal_cell,
             extra_obstacles, clearance_m, start_heading, goal_heading):
        ranges, leaf_of, adj, geo = self._scene_octree(stack, wire, clearance_m)
        # Per-leaf mean soft cost (surface-hug, thermal, EM) so the coarse corridor is
        # pulled toward the cells the route actually wants.
        soft = fields.soft_cost_field(stack, wire, weights)
        cell = float(stack.frame.cell_size)
        rad_cells = int(round((wire.radius_m + clearance_m) / cell)) if cell > 0 else 0
        leaf_soft = leaf_soft_means(stack, rad_cells, soft, ranges, leaf_of)
        corridor = octree_corridor(ranges, leaf_of, adj,
                                   tuple(int(v) for v in start_cell),
                                   tuple(int(v) for v in goal_cell),
                                   leaf_soft=leaf_soft, geo=geo)
        if corridor:
            nx, ny, nz = stack.occupancy.shape
            base_pts = list(corridor) + [tuple(int(v) for v in start_cell),
                                         tuple(int(v) for v in goal_cell)]

            # The corridor is computed heading-blind, so a pinned departure or arrival
            # heading can point out of its band and force an immediate kink back, or a
            # fallback. Cover the heading's runway too: a ray of cells leaving the start
            # along start_heading, and one approaching the goal along goal_heading.
            def _heading_ray(pts, cell_ijk, heading, sign, r):
                if heading is None:
                    return
                h = np.asarray(heading, dtype=np.float64)
                n = np.linalg.norm(h)
                if n <= 1e-9:
                    return
                h = h / n
                c = np.asarray(cell_ijk, dtype=np.float64)
                for t in range(1, 3 * r + 1):
                    p = c + sign * h * t
                    pts.append(tuple(int(round(v)) for v in p))

            # The corridor octree is geometry-only, so under congestion (many prior routes
            # near the same connectors) already-routed wires can saturate the default
            # band. Widen the band before considering the full lattice: a 3x band still
            # costs band-graph money, whereas the full lattice costs whole-grid money and
            # is infeasible at high resolution.
            for r in (self.band, 3 * self.band):
                pts = list(base_pts)
                _heading_ray(pts, start_cell, start_heading, +1, r)
                _heading_ray(pts, goal_cell, goal_heading, -1, r)
                band = band_mask((nx, ny, nz), pts, r)   # corridor dilated into a band
                # Restrict the lattice to the band: cells outside it become obstacles,
                # and prior routes (extra_obstacles) stay obstacles, so the final route is
                # collision-correct against the other wires even though the cached octree
                # never saw them.
                outside = ~band
                if extra_obstacles is not None:
                    outside = outside | np.asarray(extra_obstacles, dtype=bool)
                cells = self._lat.plan(stack, wire, weights, connectivity, start_cell,
                                       goal_cell, outside, clearance_m,
                                       start_heading, goal_heading)
                if cells is not None:
                    return cells
                _log.info("[piperouter] %s: band r=%d failed for wire %s - escalating",
                          self.name, r, wire.id)
        # No corridor, or every band too tight: fall back to the full lattice (with prior
        # routes), but only when that graph is buildable at all.
        blocked = blocked_mask(stack, wire, clearance_m, extra_obstacles)
        n_free = int(blocked.size - int(blocked.sum()))
        est_edges = float(n_free) * float(connectivity) * float(connectivity)
        if est_edges > _FULL_FALLBACK_EDGE_BUDGET:
            _log.warning("[piperouter] %s: full-lattice fallback skipped for wire %s "
                         "(~%.1fB edges > budget) - reporting no_path",
                         self.name, wire.id, est_edges / 1e9)
            return None
        return self._lat.plan(stack, wire, weights, connectivity, start_cell, goal_cell,
                              extra_obstacles, clearance_m, start_heading, goal_heading)


class DenseGlobal(_GridPlannerBase):
    """Dense heading lattice with no corridor, solved without building the graph.

    Same objective and same edge weights as `lattice`, but the (cell x heading) graph is
    relaxed in place as array operations instead of materialized, so memory is O(nodes)
    rather than O(edges). That is what makes a dense search possible at all on a
    car-sized grid: at 26-connectivity the explicit edge list for a 250x177x58 scene is
    ~19 GiB before the solver's own copies, against ~0.25 GiB for the distance array.

    Dropping the corridor is a quality decision, not just a performance one. The coarse
    corridor is heading-blind and confines the fine search to a band around it, which
    flattens soft-cost detours; measured on a real scene this planner found cheaper
    routes than `octree_lattice` on every pair tried, by 2 to 37 percent.

    The cost is time: it is far slower than the pruned planner, and on CPU it is not
    practical. Use it when route quality matters more than latency.
    """
    name = "dense"

    def plan(self, stack, wire, weights, connectivity, start_cell, goal_cell,
             extra_obstacles, clearance_m, start_heading=None, goal_heading=None):
        prep = self._prep(stack, wire, weights, connectivity, start_cell, goal_cell,
                          extra_obstacles, clearance_m)
        if prep is None:
            return None
        blocked, a, b, soft, offs, _step = prep
        lut = stencil.build_turn_lut(offs, wire.min_bend_radius_mm,
                                     float(stack.frame.cell_size) * 1000.0,
                                     float(weights.get("bend", 1.0)))
        cells, cost = stencil.solve(~blocked, soft, float(stack.frame.cell_size),
                                    offs, lut, a, b)
        if cells is None or not np.isfinite(cost):
            return None
        return cells


GLOBAL_PLANNERS = {
    "lattice": LatticeGlobal,
    "astar": AStarGlobal,
    "fmm": FMMGlobal,
    "rrt": RRTGlobal,
    "octree": OctreeGlobal,
    "medial": MedialGlobal,
    "octree_lattice": OctreeLatticeGlobal,
    "dense": DenseGlobal,
}


def make_global(name):
    cls = GLOBAL_PLANNERS.get(name)
    if cls is None:
        _log.warning("[piperouter] unknown global planner %r (have: %s) -> using 'lattice'. "
                     "Rebuild/restart the solver if you expected this planner.",
                     name, ", ".join(GLOBAL_PLANNERS))
        cls = LatticeGlobal
    return cls()
