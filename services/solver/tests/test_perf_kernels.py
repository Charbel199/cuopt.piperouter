"""Equivalence tests for the performance kernels that replaced SciPy/recursive code:

* grids.dilate6            — shifted-OR 6-connected dilation == scipy binary_dilation
* planners.octree_leaves   — integral-image, level-vectorized subdivision produces the
                             SAME partition as the original recursive implementation
* fields caches            — normalized fields / soft cost / melt masks computed once
"""
import numpy as np
import pytest
from scipy import ndimage

from piperouter_solver import fields, planners
from piperouter_solver.grids import GridStack, dilate6
from piperouter_solver.models import GridFrame, WireType


def _rand_blocked(shape, p, seed):
    rng = np.random.default_rng(seed)
    return rng.random(shape) < p


# ---------------------------------------------------------------- dilate6
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_dilate6_matches_scipy(seed, k):
    m = _rand_blocked((23, 17, 9), 0.08, seed)
    ref = (ndimage.binary_dilation(m, structure=ndimage.generate_binary_structure(3, 1),
                                   iterations=k) if k > 0 else m)
    out = dilate6(m, k)
    assert out.dtype == bool
    assert np.array_equal(out, ref)


def test_dilate6_edges_and_extremes():
    m = np.zeros((5, 5, 5), dtype=bool)
    m[0, 0, 0] = True                       # corner voxel dilates into the volume
    out = dilate6(m, 1)
    assert out.sum() == 4 and out[1, 0, 0] and out[0, 1, 0] and out[0, 0, 1]
    full = np.ones((4, 4, 4), dtype=bool)
    assert np.array_equal(dilate6(full, 2), full)
    empty = np.zeros((4, 4, 4), dtype=bool)
    assert not dilate6(empty, 3).any()


# ---------------------------------------------------------------- octree
def _octree_reference(blocked):
    """The original recursive implementation, kept verbatim as the reference."""
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


@pytest.mark.parametrize("shape,p,seed", [
    ((37, 23, 19), 0.05, 1), ((16, 16, 16), 0.3, 2),
    ((33, 9, 5), 0.5, 3), ((7, 7, 7), 0.0, 4), ((6, 6, 6), 1.0, 5),
    ((11, 13, 3), 0.15, 6),
])
def test_octree_leaves_same_partition(shape, p, seed):
    blocked = _rand_blocked(shape, p, seed)
    r_ref, lof_ref = _octree_reference(blocked)
    r_new, lof_new = planners.octree_leaves(blocked)
    # identical SET of leaf boxes (ids may be assigned in a different order)
    assert sorted(map(tuple, r_ref)) == sorted(map(tuple, r_new))
    # identical free/blocked coverage
    assert np.array_equal(lof_ref >= 0, lof_new >= 0)
    # leaf_of must be consistent with the emitted ranges
    for lid, (i0, i1, j0, j1, k0, k1) in enumerate(r_new):
        assert (lof_new[i0:i1, j0:j1, k0:k1] == lid).all()


# ---------------------------------------------------------------- field caches
def _stack():
    frame = GridFrame(bounds_min=np.zeros(3), cell_size=0.1, res_xyz=(12, 10, 8))
    shape = frame.res_xyz
    rng = np.random.default_rng(0)
    return GridStack(frame=frame, occupancy=np.zeros(shape, dtype=np.uint8),
                     surface_dist=rng.random(shape).astype(np.float32) * 3,
                     thermal=(20 + rng.random(shape) * 200).astype(np.float32),
                     em=rng.random(shape).astype(np.float32))


def _wire(sens=0.5, tmax=120.0):
    return WireType(id="w", label="w", kind="wire", outer_diameter_mm=4.0,
                    min_bend_radius_mm=20.0, cost_per_m=1.0, mass_per_m_kg=0.1,
                    max_temp_c=tmax, em_sensitivity=sens, color=(1, 1, 0))


def test_soft_cost_field_cached_and_correct():
    s = _stack()
    w = {"surface": 1.5, "thermal": 2.0, "em": 3.0}
    a = fields.soft_cost_field(s, _wire(), w)
    b = fields.soft_cost_field(s, _wire(), w)
    assert a is b                                     # cache hit: same object
    expected = (1.5 * fields.normalize(s.surface_dist)
                + 2.0 * fields.normalize(s.thermal)
                + 3.0 * 0.5 * fields.normalize(s.em)).astype(np.float32)
    assert np.allclose(a, expected, atol=1e-6)
    c = fields.soft_cost_field(s, _wire(sens=0.9), w)  # different susceptibility
    assert c is not a and not np.allclose(c, a)


def test_melt_mask_cached_and_correct():
    s = _stack()
    a = fields.melt_mask(s, _wire(tmax=120.0))
    b = fields.melt_mask(s, _wire(tmax=120.0))
    assert a is b
    assert np.array_equal(a, s.thermal > 120.0)
    c = fields.melt_mask(s, _wire(tmax=60.0))
    assert c is not a


def test_leaf_adjacency_matches_reference():
    blocked = _rand_blocked((21, 17, 13), 0.2, 7)
    ranges, leaf_of = planners.octree_leaves(blocked)
    ref = {}
    for x, y in ((leaf_of[:-1], leaf_of[1:]), (leaf_of[:, :-1], leaf_of[:, 1:]),
                 (leaf_of[:, :, :-1], leaf_of[:, :, 1:])):
        m = (x >= 0) & (y >= 0) & (x != y)
        for u, v in zip(x[m].ravel(), y[m].ravel()):
            ref.setdefault(int(u), set()).add(int(v))
            ref.setdefault(int(v), set()).add(int(u))
    adj = planners.leaf_adjacency(leaf_of, len(ranges))
    for lid in range(len(ranges)):
        assert set(map(int, adj.get(lid, ()))) == ref.get(lid, set())
    assert adj.get(-1) == () and adj.get(len(ranges) + 5) == ()


# ---------------------------------------------------------------- band mask
def test_band_mask_matches_slice_loop():
    shape = (23, 17, 11)
    rng = np.random.default_rng(8)
    # in-grid points plus out-of-range ones (heading rays walk off the grid)
    pts = [tuple(int(v) for v in rng.integers(-6, 28, 3)) for _ in range(60)]
    pts += [(-2, 5, 5), (25, 16, 10), (-9, -9, -9), (22, 18, 12)]
    for r in (1, 3, 4):
        ref = np.zeros(shape, dtype=bool)
        nx, ny, nz = shape
        for ci, cj, ck in pts:
            # intended clipped-box semantics; the production loop this replaced used
            # min(n, c+r+1) as the stop, which goes NEGATIVE for points more than r+1
            # below the low edge and silently slice-wraps, painting a spurious stripe -
            # band_mask fixes that, so the reference clamps the stop at 0 too
            ref[max(0, ci - r):max(0, min(nx, ci + r + 1)),
                max(0, cj - r):max(0, min(ny, cj + r + 1)),
                max(0, ck - r):max(0, min(nz, ck + r + 1))] = True
        out = planners.band_mask(shape, pts, r)
        assert out.dtype == bool and np.array_equal(out, ref)
    assert not planners.band_mask(shape, [], 4).any()
    assert not planners.band_mask(shape, [(-99, 0, 0)], 4).any()


# ---------------------------------------------------------------- leaf soft cache
def test_leaf_soft_cached_and_correct():
    blocked = _rand_blocked((14, 12, 10), 0.2, 9)
    ranges, leaf_of = planners.octree_leaves(blocked)
    rng = np.random.default_rng(10)
    soft = rng.random(blocked.shape).astype(np.float32)

    class FakeStack:
        pass
    st = FakeStack()
    a = planners.leaf_soft_means(st, 3, soft, ranges, leaf_of)
    b = planners.leaf_soft_means(st, 3, soft, ranges, leaf_of)
    assert a is b                                     # cached per (rad_cells, field)
    c = planners.leaf_soft_means(st, 4, soft, ranges, leaf_of)
    assert c is not a
    # reference: per-leaf mean via bincount, exactly the old inline computation
    flat = leaf_of.ravel()
    m = flat >= 0
    n = len(ranges)
    sums = np.bincount(flat[m], weights=soft.ravel()[m], minlength=n)
    cnts = np.bincount(flat[m], minlength=n)
    assert np.allclose(a, sums / np.maximum(cnts, 1), atol=1e-9)


# ---------------------------------------------------------------- band escalation
def _mini_stack(n=20):
    frame = GridFrame(bounds_min=np.zeros(3), cell_size=0.01, res_xyz=(n, n, n))
    z = np.zeros((n, n, n), dtype=np.float32)
    return GridStack(frame=frame, occupancy=np.zeros((n, n, n), dtype=np.uint8),
                     surface_dist=z.copy(), thermal=z.copy(), em=z.copy())


def test_octree_lattice_escalates_band_before_full_lattice(monkeypatch):
    s = _mini_stack()
    g = planners.OctreeLatticeGlobal(band_cells=2)
    calls = []

    def fake_lat_plan(stack, wire, weights, connectivity, start_cell, goal_cell,
                      extra_obstacles, clearance_m, start_heading, goal_heading):
        blocked_cells = int(np.asarray(extra_obstacles).sum())
        calls.append(blocked_cells)
        return None if len(calls) == 1 else [(1, 1, 1), (5, 5, 5)]

    monkeypatch.setattr(g._lat, "plan", fake_lat_plan)
    out = g.plan(s, _wire(), {}, 26, (2, 2, 2), (15, 15, 15), None, 0.0, None, None)
    assert out == [(1, 1, 1), (5, 5, 5)]
    # second attempt used a WIDER band -> fewer outside-band blocked cells
    assert len(calls) == 2 and calls[1] < calls[0]


def test_octree_lattice_full_fallback_guarded_on_huge_grids(monkeypatch):
    s = _mini_stack(n=120)   # 1.7M free cells -> ~1.2B expanded edges at 26-conn
    g = planners.OctreeLatticeGlobal(band_cells=2)
    monkeypatch.setattr(planners, "octree_corridor", lambda *a, **k: None)

    def boom(*a, **k):
        raise AssertionError("full-lattice fallback must not run on a huge grid")

    monkeypatch.setattr(g._lat, "plan", boom)
    out = g.plan(s, _wire(), {}, 26, (2, 2, 2), (100, 100, 100), None, 0.0, None, None)
    assert out is None


def test_octree_lattice_full_fallback_still_runs_on_small_grids(monkeypatch):
    s = _mini_stack(n=20)
    g = planners.OctreeLatticeGlobal(band_cells=2)
    monkeypatch.setattr(planners, "octree_corridor", lambda *a, **k: None)
    sentinel = [(0, 0, 0)]
    monkeypatch.setattr(g._lat, "plan", lambda *a, **k: sentinel)
    assert g.plan(s, _wire(), {}, 26, (2, 2, 2), (15, 15, 15),
                  None, 0.0, None, None) is sentinel
