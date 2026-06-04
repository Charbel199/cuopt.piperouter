"""Fibre-neutre smoothing.

Turn the stair-stepped grid polyline into a smooth, collision-safe curve by
minimizing a global least-squares energy:

    min_P   w_data * sum_i ||P_i - G_i||^2  +  w_curv * sum_i ||P_{i-1} - 2 P_i + P_{i+1}||^2

with the endpoints soft-fixed to the connector points, optional tangency rows
pinning the first/last segment direction to a requested heading, and an outer
loop that re-pins any point landing in a prohibited voxel back to its (safe)
grid anchor and re-solves. The energy is separable per axis (x/y/z), so it is
three independent banded SPD systems solved via the normal equations.

Backend mirrors backend.py: cuSolver via cupy on GPU, scipy.sparse on CPU.
"""
from __future__ import annotations

import os

import numpy as np

_W_DATA = 1.0       # default fidelity weight (stay near the grid path)
_W_FIX = 1.0e6      # endpoints: effectively fixed
_W_TAN = 1.0e4      # tangency point (second / second-to-last): strongly pinned
_W_PIN = 1.0e6      # a point re-pinned because it entered a prohibited voxel
_MAX_ITERS = 8


def _unit(v):
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def _densify(G, max_seg):
    """Resample so each segment <= max_seg. Returns (points (M,3), idx_map) where
    idx_map[i] is the output index of original point G[i] (kept exactly), so callers
    can re-locate hard-fixed points (endpoints, waypoints) after densification."""
    G = np.asarray(G, dtype=np.float64)
    out = [G[0]]
    idx_map = [0]
    for a, b in zip(G[:-1], G[1:]):
        seg = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(seg / max_seg))) if max_seg > 0 else 1
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
        idx_map.append(len(out) - 1)
    return np.asarray(out, dtype=np.float64), idx_map


def _in_blocked(P, frame, blocked):
    """Boolean mask: which points P fall inside a prohibited voxel."""
    inv = (np.asarray(P, dtype=np.float64) - frame.bounds_min) / frame.cell_size
    idx = np.floor(inv).astype(int)
    nx, ny, nz = frame.res_xyz
    idx[:, 0] = np.clip(idx[:, 0], 0, nx - 1)
    idx[:, 1] = np.clip(idx[:, 1], 0, ny - 1)
    idx[:, 2] = np.clip(idx[:, 2], 0, nz - 1)
    return blocked[idx[:, 0], idx[:, 1], idx[:, 2]]


def _solve_normal(A, B):
    """Least-squares solve of A x = B (B has 3 columns) via the normal equations.
    Returns (N,3) ndarray.

    CPU (scipy) by default — and on purpose. The smoothing system is tiny (a few hundred
    points), so scipy solves it in microseconds. The GPU (cupy / cuSOLVER) path is OFF by
    default because running cupy in the SAME long-lived server process as cuGraph
    intermittently crashed the worker (a CUDA-context / RMM-allocator interaction), with
    no real speedup at this size. Set PIPEROUTER_GPU_SMOOTH=1 to opt back into the GPU
    solve (the cuGraph routing always stays on the GPU regardless)."""
    if os.environ.get("PIPEROUTER_GPU_SMOOTH") == "1":
        try:
            import cupy as cp
            import cupyx.scipy.sparse as csp
            import cupyx.scipy.sparse.linalg as csla

            Ag = csp.csr_matrix(A.astype(np.float64))
            AtA = (Ag.T @ Ag).tocsr()
            Bg = cp.asarray(B, dtype=cp.float64)
            AtB = Ag.T @ Bg
            out = cp.empty((A.shape[1], 3), dtype=cp.float64)
            for j in range(3):
                out[:, j] = csla.spsolve(AtA, AtB[:, j])
            return cp.asnumpy(out)
        except Exception:
            pass

    from scipy.sparse.linalg import spsolve

    AtA = (A.T @ A).tocsc()
    AtB = A.T @ B
    out = np.empty((A.shape[1], 3), dtype=np.float64)
    for j in range(3):
        out[:, j] = spsolve(AtA, AtB[:, j])
    return out


def _build_ls(anchors, pin_mask, fixed, w_curv, h0, hN, d):
    """Assemble sparse A (R,N) and dense B (R,3) for the per-axis LS problem.

    `fixed` = indices pinned hard IN PLACE (endpoints + waypoints): the curve must pass
    exactly through them, but their tangent is free, so curvature minimisation makes the
    pass-through smooth (continuous tangent)."""
    from scipy.sparse import csr_matrix

    N = len(anchors)
    wd = np.full(N, _W_DATA, dtype=np.float64)
    for i in fixed:
        wd[i] = _W_FIX
    if h0 is not None:
        wd[1] = max(wd[1], _W_TAN)
    if hN is not None:
        wd[N - 2] = max(wd[N - 2], _W_TAN)
    wd[pin_mask] = np.maximum(wd[pin_mask], _W_PIN)

    targets = anchors.copy()                 # fixed points pin to their own anchor
    if h0 is not None and 1 not in fixed:
        targets[1] = anchors[0] + d * h0     # P1 = start + d*h0  -> leaves along h0
    if hN is not None and (N - 2) not in fixed:
        targets[N - 2] = anchors[-1] - d * hN  # P_{N-2} = end - d*hN -> enters along hN

    rows, cols, vals = [], [], []
    B = []
    r = 0
    # data rows
    for i in range(N):
        sw = np.sqrt(wd[i])
        rows.append(r); cols.append(i); vals.append(sw)
        B.append(sw * targets[i]); r += 1
    # curvature rows (2nd difference) on interior points
    swc = np.sqrt(w_curv)
    if swc > 0:
        for i in range(1, N - 1):
            rows += [r, r, r]; cols += [i - 1, i, i + 1]
            vals += [swc, -2.0 * swc, swc]
            B.append([0.0, 0.0, 0.0]); r += 1

    A = csr_matrix((vals, (rows, cols)), shape=(r, N))
    return A, np.asarray(B, dtype=np.float64)


def smooth_path(G, frame, blocked, wire, start_heading, end_heading, strength,
                fixed_idx=None):
    """Return a smoothed, hard-safe polyline (list of (3,) points).

    G: grid polyline (world points). strength: >0 smoothing weight (0 = off).
    blocked: bool occupancy (mesh+radius+clearance+melt) the curve must avoid.
    fixed_idx: indices in G the curve must pass through EXACTLY (waypoints). The two
    endpoints (0 and len(G)-1) are always fixed. Fixed points keep a free tangent, so
    the curve sweeps smoothly THROUGH them rather than kinking.
    """
    G = np.asarray(G, dtype=np.float64)
    strength = float(strength)
    if strength <= 0.0 or len(G) < 3:
        return [tuple(float(x) for x in p) for p in G]

    anchors, idx_map = _densify(G, 0.5 * frame.cell_size)
    N = len(anchors)
    if N < 3:
        return [tuple(float(x) for x in p) for p in anchors]

    # densified indices that are hard-fixed (endpoints + any requested waypoints)
    want = set(fixed_idx or ())
    want |= {0, len(G) - 1}
    fixed = sorted({idx_map[i] for i in want if 0 <= i < len(G)})
    fixed_set = set(fixed)

    h0, hN = _unit(start_heading), _unit(end_heading)
    d = float(frame.cell_size)
    # stiffer pipes (bigger min bend) round more; scale curvature by strength.
    w_curv = strength * (1.0 + wire.min_bend_radius_mm / 100.0)

    pin_mask = np.zeros(N, dtype=bool)
    P = anchors
    for _ in range(_MAX_ITERS):
        A, B = _build_ls(anchors, pin_mask, fixed_set, w_curv, h0, hN, d)
        P = _solve_normal(A, B)
        inside = _in_blocked(P, frame, blocked)
        inside[fixed] = False                 # never re-pin hard points (ends/waypoints)
        new = inside & ~pin_mask
        if not new.any():
            break
        pin_mask |= inside                    # pin offenders back to their safe anchor

    # Final guarantee: anything still inside collapses to its safe grid anchor.
    inside = _in_blocked(P, frame, blocked)
    inside[fixed] = False
    P[inside] = anchors[inside]
    return [tuple(float(x) for x in p) for p in P]
