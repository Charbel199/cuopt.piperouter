"""Reconstruct the octree_lattice planner's structure for VISUALIZATION (pxr-free,
headless-testable).

Mirrors `piperouter_solver.planners.OctreeGlobal` / `OctreeLatticeGlobal`: subdivide the
blocked grid into an octree (a node is a leaf when uniformly free, dropped when uniformly
blocked, split otherwise), run A* over the free-leaf adjacency for a coarse corridor, then
dilate that corridor into the band the fine heading-lattice would actually search. We
rebuild it from the occupancy grid the extension already holds, so the viz needs no solver
round-trip. It's the same algorithm, so the leaves are identical; the corridor uses a plain
distance cost (the planner adds soft costs) so treat the corridor/band as representative.
"""
from __future__ import annotations

import heapq

import numpy as np

_DIRS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def build_octree(blocked):
    """Return (leaves, leaf_of). leaves = list of (i0,i1,j0,j1,k0,k1) free-leaf ranges;
    leaf_of[i,j,k] = leaf id or -1 (blocked/dropped)."""
    blocked = np.asarray(blocked, dtype=bool)
    nx, ny, nz = blocked.shape
    leaf_of = np.full(blocked.shape, -1, dtype=np.int64)
    leaves = []
    nodes = [(0, nx, 0, ny, 0, nz)]
    while nodes:
        i0, i1, j0, j1, k0, k1 = nodes.pop()
        sub = blocked[i0:i1, j0:j1, k0:k1]
        if not sub.any():                                   # uniformly FREE -> leaf
            lid = len(leaves)
            leaf_of[i0:i1, j0:j1, k0:k1] = lid
            leaves.append((i0, i1, j0, j1, k0, k1))
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
                    nodes.append((xa, xb, ya, yb, za, zb))
    return leaves, leaf_of


def leaf_adjacency(leaf_of, n_leaves):
    """Vectorized face-adjacency of free leaves (mirrors planners.leaf_adjacency): compare
    neighbouring slabs per axis, dedup the (lo,hi) leaf pairs. O(grid) in numpy instead of
    the old O(free-voxels) Python loop."""
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
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    for k in np.unique(lo * np.int64(n_leaves) + hi):
        u, v = int(k // n_leaves), int(k % n_leaves)
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def _center(leaf):
    return np.array([0.5 * (leaf[0] + leaf[1] - 1), 0.5 * (leaf[2] + leaf[3] - 1),
                     0.5 * (leaf[4] + leaf[5] - 1)])


def _seg_cells(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    steps = max(2, int(np.linalg.norm(q - p)) * 2)
    out = []
    for t in np.linspace(0.0, 1.0, steps):
        c = tuple(int(round(v)) for v in (p + (q - p) * t))
        if not out or out[-1] != c:
            out.append(c)
    return out


def corridor_and_band(blocked, leaves, leaf_of, start_cell, goal_cell, band=4):
    """Coarse octree A* corridor (start->goal through free-leaf centres) + the dilated band
    of fine cells the heading-lattice would search. Returns (corridor_cells, band_mask) or
    (None, None) if either endpoint isn't in a free leaf / no corridor."""
    blocked = np.asarray(blocked, dtype=bool)
    nx, ny, nz = blocked.shape

    def clamp(c):
        return (min(max(int(c[0]), 0), nx - 1), min(max(int(c[1]), 0), ny - 1),
                min(max(int(c[2]), 0), nz - 1))

    a, b = clamp(start_cell), clamp(goal_cell)
    sa, sb = int(leaf_of[a]), int(leaf_of[b])
    if sa < 0 or sb < 0:
        return None, None

    adj = leaf_adjacency(leaf_of, len(leaves))

    goal_c = _center(leaves[sb])

    def h(lid):
        return float(np.linalg.norm(_center(leaves[lid]) - goal_c))

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
            w = float(np.linalg.norm(_center(leaves[nb]) - _center(leaves[lid])))
            ng = g[lid] + w
            if ng < g.get(nb, 1e18):
                g[nb] = ng
                came[nb] = lid
                heapq.heappush(pq, (ng + h(nb), nb))
    if not found:
        return None, None

    chain = [sb]
    while chain[-1] in came:
        chain.append(came[chain[-1]])
    chain.reverse()
    waypts = [np.asarray(a, float)] + [_center(leaves[l]) for l in chain] + [np.asarray(b, float)]
    corridor = []
    for p, q in zip(waypts[:-1], waypts[1:]):
        for c in _seg_cells(p, q):
            if 0 <= c[0] < nx and 0 <= c[1] < ny and 0 <= c[2] < nz and (not corridor or corridor[-1] != c):
                corridor.append(c)

    r = int(band)
    band_mask = np.zeros((nx, ny, nz), dtype=bool)
    for ci, cj, ck in corridor + [a, b]:
        band_mask[max(0, ci - r):min(nx, ci + r + 1),
                  max(0, cj - r):min(ny, cj + r + 1),
                  max(0, ck - r):min(nz, ck + r + 1)] = True
    return corridor, band_mask
