"""Shortest path over the heading-expanded lattice without ever building the graph.

The lattice node is (cell, arrival heading) and every edge weight is a function of the
soft-cost field and the turn table, so the graph is fully implicit: it can be relaxed in
place instead of materialized. That turns the memory cost from O(edges) into O(nodes),
which is the difference between 19 GiB and 0.25 GiB on a 250x177x58 grid at
26-connectivity, and it is what makes a dense search feasible at all.

The relaxation is a Bellman-Ford sweep expressed as array operations: for each departure
offset, reduce over the arrival headings through the turn table, shift the result by the
offset, add the entry cost, and keep the elementwise minimum. Every sweep is dense
regular work over a 4D array, which is what the GPU is for; on the CPU the same code
runs under numpy and is correct but far slower.
"""
from __future__ import annotations

import numpy as np

from .fields import neighbor_offsets, turn_penalty
from .grids import _xp

INF = np.float32(np.inf)


def _shift_into(xp, src, off, nx, ny, nz, fill=INF, dtype=None):
    """Return `dst` with dst[c] = src[c - off], out-of-grid entries set to `fill`.

    This moves a per-cell quantity one step ALONG `off`, so a value sitting at a cell
    lands on the cell that offset reaches.
    """
    dx, dy, dz = off
    dst = xp.full((nx, ny, nz), fill, dtype=dtype or xp.float32)
    sx0, sx1 = max(0, -dx), nx - max(0, dx)
    sy0, sy1 = max(0, -dy), ny - max(0, dy)
    sz0, sz1 = max(0, -dz), nz - max(0, dz)
    if sx0 >= sx1 or sy0 >= sy1 or sz0 >= sz1:
        return dst
    dst[sx0 + dx:sx1 + dx, sy0 + dy:sy1 + dy, sz0 + dz:sz1 + dz] = \
        src[sx0:sx1, sy0:sy1, sz0:sz1]
    return dst


def _intermediate_offsets(off):
    """Face cells a 2D edge-diagonal squeezes between; empty for face and 3D moves.

    Mirrors the lattice builder's relaxed no-corner-cutting rule: 2D diagonals must keep
    both face neighbours free, full 3D corner moves stay unrestricted so the router can
    still thread tight openings.
    """
    axes = [i for i in range(3) if off[i] != 0]
    if len(axes) != 2:
        return []
    return [tuple(int(off[ax]) if ax == a else 0 for ax in range(3)) for a in axes]


def solve(free, soft, cell_size, offsets, turn_lut, start_cell, goal_cell,
          max_sweeps=None, xp=None):
    """Least-cost path from start_cell to goal_cell over the implicit lattice.

    `free` is the boolean free-space mask, `soft` the per-cell soft cost S(v) and
    `turn_lut[h_in, h_out]` the already-scaled bend penalty in metres. Edge weight
    matches the materialized builder exactly:
        step_len[o] * (1 + S[dst]) + turn_lut[h_in, o]

    Returns (cells, cost) with `cells` the cell path including both endpoints, or
    (None, inf) when the goal is unreachable.
    """
    xp = xp or _xp()
    nx, ny, nz = free.shape
    H = len(offsets)
    offs = np.asarray(offsets, dtype=np.int64)
    step_len = cell_size * np.sqrt((offs ** 2).sum(axis=1)).astype(np.float32)

    free_x = xp.asarray(free)
    soft_x = xp.asarray(soft, dtype=xp.float32)
    turn_x = xp.asarray(np.asarray(turn_lut, dtype=np.float32))

    # Entry cost of landing on each cell via offset o, +inf where the move is illegal.
    entry = []
    for o in range(H):
        e = step_len[o] * (xp.float32(1.0) + soft_x)
        legal = free_x & _shift_into(xp, free_x.astype(xp.float32), tuple(offs[o]),
                                     nx, ny, nz).astype(bool)
        for mid in _intermediate_offsets(tuple(offs[o])):
            legal &= _shift_into(xp, free_x.astype(xp.float32), mid,
                                 nx, ny, nz).astype(bool)
        entry.append(xp.where(legal, e.astype(xp.float32), INF))

    # dist[h] = best cost to stand on a cell having arrived along offset h.
    dist = xp.full((H, nx, ny, nz), INF, dtype=xp.float32)
    pred = xp.full((H, nx, ny, nz), -1, dtype=xp.int8)

    # Seed: leave the start cell along every legal offset. The start itself carries no
    # arrival heading, so this is the one step with no turn cost.
    si, sj, sk = (int(v) for v in start_cell)
    for o in range(H):
        ni, nj, nk = si + int(offs[o][0]), sj + int(offs[o][1]), sk + int(offs[o][2])
        if not (0 <= ni < nx and 0 <= nj < ny and 0 <= nk < nz):
            continue
        w = float(entry[o][ni, nj, nk])
        if np.isfinite(w):
            dist[o, ni, nj, nk] = xp.float32(w)

    gi, gj, gk = (int(v) for v in goal_cell)
    sweeps = max_sweeps or (nx + ny + nz)
    used = 0
    for _ in range(sweeps):
        used += 1
        changed = False
        for o in range(H):
            # cheapest way to be standing anywhere, then turn onto offset o.
            # The (H, nx, ny, nz) temporary is the peak allocation of the sweep, so it
            # is formed once and reduced twice rather than rebuilt for the argmin.
            t = dist + turn_x[:, o][:, None, None, None]
            best_in = t.min(axis=0)
            src_h = t.argmin(axis=0)
            del t
            cand = _shift_into(xp, best_in, tuple(offs[o]), nx, ny, nz) + entry[o]
            better = cand < dist[o]
            if bool(better.any()):
                changed = True
                dist[o] = xp.where(better, cand, dist[o])
                # -1 fill keeps this integral; casting an inf sentinel would be UB
                shifted_h = _shift_into(xp, src_h.astype(xp.int8), tuple(offs[o]),
                                        nx, ny, nz, fill=-1, dtype=xp.int8)
                pred[o] = xp.where(better, shifted_h, pred[o])
        if not changed:
            break

    solve.last_sweeps = used          # diagnostic: how far Bellman-Ford had to go
    goal_costs = dist[:, gi, gj, gk]
    h_star = int(goal_costs.argmin())
    cost = float(goal_costs[h_star])
    if not np.isfinite(cost):
        return None, float("inf")

    # Walk back: standing at `cell` having arrived along `h` came from cell - offs[h],
    # where the arrival heading was pred[h, cell].
    pred_h = xp.asnumpy(pred) if xp is not np else pred
    cells = []
    cur, h = (gi, gj, gk), h_star
    guard = nx * ny * nz + 10
    while guard > 0:
        guard -= 1
        cells.append(cur)
        prev = (cur[0] - int(offs[h][0]), cur[1] - int(offs[h][1]),
                cur[2] - int(offs[h][2]))
        if prev == (si, sj, sk):
            cells.append(prev)
            break
        ph = int(pred_h[h, cur[0], cur[1], cur[2]])
        if ph < 0:
            break
        cur, h = prev, ph
    cells.reverse()
    return cells, cost


def build_turn_lut(offsets, min_bend_radius_mm, cell_size_mm, bend_weight):
    """Turn table in metres, scaled quadratically by the bend weight, as the lattice
    builder does so the two agree edge for edge."""
    H = len(offsets)
    scale = float(bend_weight) ** 2
    lut = np.zeros((H, H), dtype=np.float64)
    for a in range(H):
        for b in range(H):
            lut[a, b] = scale * turn_penalty(tuple(offsets[a]), tuple(offsets[b]),
                                             min_bend_radius_mm, cell_size_mm)
    return lut


def offsets_for(connectivity):
    return [tuple(o) for o in neighbor_offsets(connectivity)]
