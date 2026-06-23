"""Pluggable GLOBAL planners — find a coarse, collision-free path of grid cells from a
start cell to a goal cell. Swappable for evaluation (see bench_algos.py); the default
'lattice' is the proven heading-aware expanded-lattice + SSSP, the others are different
families (plain grid A*, Eikonal-style cost-to-go descent, sampling-based RRT-Connect)
for comparison.

Every planner returns a list of (i,j,k) cells (start..goal, no source/sink) or None, and
every planner is COLLISION-SAFE: it searches only cells that are free after dilating the
occupancy by the wire radius + clearance and removing over-temperature (melt) cells.
"""
from __future__ import annotations

import heapq
import logging

import numpy as np

from . import fields
from .backend import shortest_path
from .lattice import ExpandedLatticeBuilder

_log = logging.getLogger("piperouter")


def blocked_mask(stack, wire, clearance_m, extra_obstacles):
    """Cells the wire may NOT occupy: mesh+clearance dilation, melt cells, prior routes."""
    blocked = (stack.dilate_occupancy(wire.radius_m + clearance_m).astype(bool)
               | fields.melt_mask(stack, wire))
    if extra_obstacles is not None:
        blocked = blocked | np.asarray(extra_obstacles, dtype=bool)
    return blocked


def _nearest_free(blocked, cell, max_r=2):
    """Snap a start/goal cell to the closest free cell within max_r (endpoints often sit
    just inside the clearance halo). Returns the cell or None."""
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
    """Heading-expanded lattice + SSSP (the production default — bend-aware)."""
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
        # Match the lattice's endpoint rule: route if there's an IMMEDIATE free neighbour
        # (a connector sitting on/just inside a surface), but reject a DEEPLY buried
        # endpoint (no free cell within one step) instead of fabricating a route by
        # relocating it far away.
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
    """Plain uniform-grid A* (no heading expansion). Cheap (no H^2 blow-up); leans on the
    local optimizer for smoothness. Edge cost = step_len * (1 + soft[neighbor])."""
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
    """True Eikonal Fast Marching: solve |∇T| = slowness for the arrival-time field T from
    the goal with the Godunov upwind quadratic update (sub-cell accurate, less grid bias
    than graph-Dijkstra), then gradient-descend T from the start. slowness = 1 + soft, so
    the front advances slower through costly cells and the descent skirts them."""
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
        # gradient descent on T from start to goal (use the full connectivity for steps)
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
    """Sampling-based RRT-Connect on the free voxel grid. Seeded RNG -> deterministic.
    Represents the sampling family; output is jagged and relies on the local optimizer."""
    name = "rrt"

    def __init__(self, seed=12345, step_cells=6, max_iter=4000):
        self.seed = seed
        self.step_cells = step_cells
        self.max_iter = max_iter

    def _steer(self, frm, to, blocked):
        """Walk from `frm` toward `to` up to step_cells, stopping before any blocked cell.
        Returns the last reachable cell (or frm)."""
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
        # two trees: nodes list + parent index
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
            # try to connect tree B to the new node of A
            ci = extend(tb, pb, ta[gi])
            if ci is not None and tb[ci] == ta[gi]:
                # joined — reconstruct a -> join, then join -> b
                left = []
                k = gi
                while k != -1:
                    left.append(ta[k]); k = pa[k]
                left.reverse()
                right = []
                k = ci
                while k != -1:
                    right.append(tb[k]); k = pb[k]
                # right currently goes join..b; drop the duplicate join node
                return left + right[1:]
            ta, pa, tb, pb = tb, pb, ta, pa   # swap trees (RRT-Connect)
        return None


def _raster_cells(a, b):
    """Cells along the straight segment a->b (inclusive), 3D DDA. Both ends free + the
    segment staying in free space is the caller's guarantee."""
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


def leaf_adjacency(leaf_of, n_leaves):
    """Face-adjacency of octree free leaves, VECTORIZED. Compares each axis's
    neighbouring slabs (`leaf_of[:-1] vs leaf_of[1:]`) to find boundary cells whose two
    sides belong to different free leaves, dedups the (lo,hi) pairs, and builds the
    adjacency from the UNIQUE edges. O(grid) in numpy + O(edges) in Python — replaces the
    old O(free-voxels) Python loop that made octree_lattice blow up at high resolution."""
    adj = {}
    pa, pb = [], []
    for x, y in ((leaf_of[:-1, :, :], leaf_of[1:, :, :]),
                 (leaf_of[:, :-1, :], leaf_of[:, 1:, :]),
                 (leaf_of[:, :, :-1], leaf_of[:, :, 1:])):
        m = (x >= 0) & (y >= 0) & (x != y)
        if m.any():
            pa.append(x[m].ravel())
            pb.append(y[m].ravel())
    if not pa:
        return adj
    a = np.concatenate(pa).astype(np.int64)
    b = np.concatenate(pb).astype(np.int64)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    keys = np.unique(lo * np.int64(n_leaves) + hi)     # unique undirected leaf-pairs
    for k in keys:
        u, v = int(k // n_leaves), int(k % n_leaves)
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def octree_leaves(blocked):
    """Subdivide `blocked` into octree free leaves. Returns (ranges, leaf_of): ranges =
    list of (i0,i1,j0,j1,k0,k1) free-leaf boxes; leaf_of[i,j,k] = leaf id or -1. Geometry
    only — independent of soft costs / prior routes, so it can be cached per scene."""
    blocked = np.asarray(blocked, dtype=bool)
    nx, ny, nz = blocked.shape
    leaf_of = np.full(blocked.shape, -1, dtype=np.int64)
    ranges = []
    nodes = [(0, nx, 0, ny, 0, nz)]
    while nodes:
        i0, i1, j0, j1, k0, k1 = nodes.pop()
        sub = blocked[i0:i1, j0:j1, k0:k1]
        if not sub.any():
            lid = len(ranges)
            leaf_of[i0:i1, j0:j1, k0:k1] = lid
            ranges.append((i0, i1, j0, j1, k0, k1))
            continue
        if sub.all():
            continue
        mi, mj, mk = (i0 + i1) // 2, (j0 + j1) // 2, (k0 + k1) // 2
        xs = [(i0, i1)] if i1 - i0 <= 1 else [(i0, mi), (mi, i1)]
        ys = [(j0, j1)] if j1 - j0 <= 1 else [(j0, mj), (mj, j1)]
        zs = [(k0, k1)] if k1 - k0 <= 1 else [(k0, mk), (mk, k1)]
        for xa, xb in xs:
            for ya, yb in ys:
                for za, zb in zs:
                    nodes.append((xa, xb, ya, yb, za, zb))
    return ranges, leaf_of


def octree_corridor(ranges, leaf_of, adj, start_cell, goal_cell, leaf_soft=None):
    """Coarse A* (plain distance) through free-leaf centres start->goal, rasterized to a
    dense cell list. Returns the corridor cells or None. Used as a cheap HEURISTIC corridor
    for octree_lattice; the fine lattice does the real cost/collision routing in the band."""
    nx, ny, nz = leaf_of.shape

    def clamp(c):
        return (min(max(int(c[0]), 0), nx - 1), min(max(int(c[1]), 0), ny - 1),
                min(max(int(c[2]), 0), nz - 1))

    a, b = clamp(start_cell), clamp(goal_cell)
    sa, sb = int(leaf_of[a]), int(leaf_of[b])
    if sa < 0 or sb < 0:
        return None

    def ctr(lid):
        r = ranges[lid]
        return np.array([0.5 * (r[0] + r[1] - 1), 0.5 * (r[2] + r[3] - 1),
                         0.5 * (r[4] + r[5] - 1)])

    goal_c = ctr(sb)
    g = {sa: 0.0}
    came = {}
    pq = [(float(np.linalg.norm(ctr(sa) - goal_c)), sa)]
    found = False
    while pq:
        _f, lid = heapq.heappop(pq)
        if lid == sb:
            found = True
            break
        for nb in adj.get(lid, ()):
            # distance, biased by the destination leaf's soft cost so the corridor heads
            # toward cheap cells (e.g. hugs surfaces, avoids heat/EM) instead of always
            # taking the open-air geometric shortest — which is what defeated surface-hug.
            step = float(np.linalg.norm(ctr(nb) - ctr(lid)))
            soft_nb = float(leaf_soft[nb]) if leaf_soft is not None else 0.0
            ng = g[lid] + step * (1.0 + soft_nb)
            if ng < g.get(nb, 1e18):
                g[nb] = ng
                came[nb] = lid
                heapq.heappush(pq, (ng + float(np.linalg.norm(ctr(nb) - goal_c)), nb))
    if not found:
        return None
    chain = [sb]
    while chain[-1] in came:
        chain.append(came[chain[-1]])
    chain.reverse()
    waypts = [np.asarray(a, float)] + [ctr(l) for l in chain] + [np.asarray(b, float)]
    cells = []
    for p, q in zip(waypts[:-1], waypts[1:]):
        for c in _raster_cells(p, q):
            if (0 <= c[0] < nx and 0 <= c[1] < ny and 0 <= c[2] < nz
                    and (not cells or cells[-1] != c)):
                cells.append(c)
    return cells or None


class OctreeGlobal(_GridPlannerBase):
    """Adaptive-resolution planner. Recursively subdivides the grid into an octree: a leaf
    is kept whole when it is uniformly free or uniformly blocked, and split otherwise — so
    open air is a few big cells and only surfaces/narrow gaps get fine cells. A* runs over
    the (much smaller) free-leaf adjacency graph; the path between two face-adjacent free
    leaves is collision-free, so we rasterize leaf-centre to leaf-centre back to cells."""
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

        # --- build octree leaves (ranges) + cell->leaf map for free leaves ---
        leaf_of = np.full(blocked.shape, -1, dtype=np.int64)
        leaves = []   # (cx, cy, cz center-cell-coords float, soft_avg)
        stack_nodes = [(0, nx, 0, ny, 0, nz)]
        while stack_nodes:
            i0, i1, j0, j1, k0, k1 = stack_nodes.pop()
            sub = blocked[i0:i1, j0:j1, k0:k1]
            if not sub.any():                                   # uniformly FREE -> leaf
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

        # --- adjacency: each free leaf -> set of face-adjacent free leaves (vectorized) ---
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
        # leaf centres -> dense free cells (start cell, centres, goal cell)
        waypts = [np.asarray(a, dtype=float)] + [ctr(l) for l in chain] + [np.asarray(b, dtype=float)]
        cells = []
        for p, q in zip(waypts[:-1], waypts[1:]):
            for c in _raster_cells(p, q):
                if (0 <= c[0] < nx and 0 <= c[1] < ny and 0 <= c[2] < nz
                        and not blocked[c] and (not cells or cells[-1] != c)):
                    cells.append(c)
        return cells or None


class MedialGlobal(_GridPlannerBase):
    """Clearance-seeking corridor planner (medial-axis flavour). Uses the Euclidean
    distance-to-obstacle field and runs A* with an edge cost that penalizes low-clearance
    cells, so the route is pulled onto the high-clearance 'spine' of free space — maximally
    clear of hot/EM/mesh, like routing along the medial axis."""
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
        w_clear = 4.0           # how strongly to hug the clearance spine

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
                clear_pen = 1.0 + w_clear * (1.0 - dist[nb] / dmax)   # near obstacle -> costlier
                ng = gc + step[oi] * (1.0 + float(soft[nb])) * clear_pen
                if ng < g.get(nb, 1e18):
                    g[nb] = ng
                    came[nb] = c
                    heapq.heappush(pq, (ng + h(nb), nb))
        return None


class OctreeLatticeGlobal(_GridPlannerBase):
    """Hierarchical octree + heading lattice. The octree finds a cheap coarse corridor;
    then the FULL bend-aware lattice runs only in a band of fine cells around that corridor
    (cells outside the band are masked as obstacles, so the lattice expands a fraction of
    the volume — ~10x fewer nodes at high res). This keeps the lattice's min-bend / gentle
    routing exactly, while the octree prunes the open-air search. Falls back to the full
    lattice if the band is too tight, so it is never worse than `lattice` on coverage."""
    name = "octree_lattice"

    def __init__(self, band_cells=4):
        self.band = int(band_cells)
        self._lat = LatticeGlobal()

    def _scene_octree(self, stack, wire, clearance_m):
        """Geometry-only octree (leaves + leaf_of + adjacency), CACHED on the stack per
        (wire-radius + clearance) so a Route All reuses it across wires instead of
        rebuilding per wire. Prior routes are NOT in here — they're enforced by the fine
        lattice pass below — which is why this cache stays valid for the whole scene."""
        cell = float(stack.frame.cell_size)
        rad_cells = int(round((wire.radius_m + clearance_m) / cell)) if cell > 0 else 0
        cache = stack.__dict__.setdefault("_octree_cache", {})
        if rad_cells not in cache:
            geo = stack.dilate_occupancy(wire.radius_m + clearance_m).astype(bool)
            ranges, leaf_of = octree_leaves(geo)
            cache[rad_cells] = (ranges, leaf_of, leaf_adjacency(leaf_of, len(ranges)))
        return cache[rad_cells]

    def plan(self, stack, wire, weights, connectivity, start_cell, goal_cell,
             extra_obstacles, clearance_m, start_heading, goal_heading):
        ranges, leaf_of, adj = self._scene_octree(stack, wire, clearance_m)
        # per-leaf mean soft cost (surface-hug / thermal / EM) so the coarse corridor is
        # pulled toward the cells the route actually wants — vectorized, leaves are cached.
        soft = fields.soft_cost_field(stack, wire, weights)
        flat = leaf_of.ravel()
        m = flat >= 0
        n = len(ranges)
        sums = np.bincount(flat[m], weights=soft.ravel()[m], minlength=n)
        cnts = np.bincount(flat[m], minlength=n)
        leaf_soft = sums / np.maximum(cnts, 1)
        corridor = octree_corridor(ranges, leaf_of, adj,
                                   tuple(int(v) for v in start_cell),
                                   tuple(int(v) for v in goal_cell), leaf_soft=leaf_soft)
        if corridor:
            nx, ny, nz = stack.occupancy.shape
            band = np.zeros((nx, ny, nz), dtype=bool)
            r = self.band
            pts = list(corridor) + [tuple(int(v) for v in start_cell),
                                    tuple(int(v) for v in goal_cell)]
            for ci, cj, ck in pts:                       # dilate the corridor into a band
                band[max(0, ci - r):min(nx, ci + r + 1),
                     max(0, cj - r):min(ny, cj + r + 1),
                     max(0, ck - r):min(nz, ck + r + 1)] = True
            # lattice restricted to the band: cells OUTSIDE the band become obstacles, AND
            # prior routes (extra_obstacles) stay obstacles — so the final route is
            # collision-correct against the other wires even though the cached octree wasn't.
            outside = ~band
            if extra_obstacles is not None:
                outside = outside | np.asarray(extra_obstacles, dtype=bool)
            cells = self._lat.plan(stack, wire, weights, connectivity, start_cell,
                                   goal_cell, outside, clearance_m,
                                   start_heading, goal_heading)
            if cells is not None:
                return cells
        # no corridor / band too tight -> full lattice (with prior routes); never worse
        return self._lat.plan(stack, wire, weights, connectivity, start_cell, goal_cell,
                              extra_obstacles, clearance_m, start_heading, goal_heading)


GLOBAL_PLANNERS = {
    "lattice": LatticeGlobal,
    "astar": AStarGlobal,
    "fmm": FMMGlobal,
    "rrt": RRTGlobal,
    "octree": OctreeGlobal,
    "medial": MedialGlobal,
    "octree_lattice": OctreeLatticeGlobal,
}


def make_global(name):
    cls = GLOBAL_PLANNERS.get(name)
    if cls is None:
        _log.warning("[piperouter] unknown global planner %r (have: %s) -> using 'lattice'. "
                     "Rebuild/restart the solver if you expected this planner.",
                     name, ", ".join(GLOBAL_PLANNERS))
        cls = LatticeGlobal
    return cls()
