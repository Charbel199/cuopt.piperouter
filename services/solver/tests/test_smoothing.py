import numpy as np

from piperouter_solver.models import GridFrame, WireType
from piperouter_solver import smoothing


def _frame():
    return GridFrame(bounds_min=np.zeros(3), cell_size=0.1, res_xyz=(20, 20, 3))


def _wire():
    return WireType(id="w", label="w", kind="wire", outer_diameter_mm=10.0,
                    min_bend_radius_mm=50.0, cost_per_m=1.0, mass_per_m_kg=0.1,
                    max_temp_c=200.0, em_sensitivity=0.0, color=(1, 0, 0))


def _max_turn(poly):
    """Largest heading change at any single vertex (radians). Smoothing spreads a
    sharp corner across many gentle vertices, so this drops even though the TOTAL
    turning (endpoint-determined) stays ~constant."""
    poly = np.asarray(poly, float)
    segs = np.diff(poly, axis=0)
    n = np.linalg.norm(segs, axis=1, keepdims=True)
    segs = segs[(n > 1e-9).ravel()] / n[n > 1e-9][:, None]
    worst = 0.0
    for a, b in zip(segs[:-1], segs[1:]):
        worst = max(worst, float(np.arccos(np.clip(a @ b, -1.0, 1.0))))
    return worst


def test_strength_zero_returns_input_unchanged():
    f = _frame()
    blocked = np.zeros(f.res_xyz, dtype=bool)
    G = [f.grid_to_world((i, 5, 1)) for i in range(10)]
    out = smoothing.smooth_path(G, f, blocked, _wire(), None, None, 0.0)
    assert np.allclose(np.asarray(out), np.asarray(G))


def test_endpoints_preserved():
    f = _frame()
    blocked = np.zeros(f.res_xyz, dtype=bool)
    G = ([f.grid_to_world((i, 5, 1)) for i in range(6)]
         + [f.grid_to_world((5, j, 1)) for j in range(6, 12)])  # L-shape
    out = np.asarray(smoothing.smooth_path(G, f, blocked, _wire(), None, None, 5.0))
    assert np.allclose(out[0], G[0])
    assert np.allclose(out[-1], G[-1])


def test_smoothing_reduces_turning_on_a_corner():
    f = _frame()
    blocked = np.zeros(f.res_xyz, dtype=bool)
    G = ([f.grid_to_world((i, 5, 1)) for i in range(6)]
         + [f.grid_to_world((5, j, 1)) for j in range(6, 12)])  # sharp 90deg corner
    out = smoothing.smooth_path(G, f, blocked, _wire(), None, None, 5.0)
    # the sharp ~90deg vertex is spread into many gentle ones
    assert _max_turn(out) < _max_turn(G) - 1e-3


def test_smoothed_curve_stays_out_of_blocked():
    f = _frame()
    blocked = np.zeros(f.res_xyz, dtype=bool)
    blocked[5, 6:, :] = True   # a wall the corner could cut into if smoothed naively
    # path hugs the inside of the corner around the wall
    G = ([f.grid_to_world((i, 5, 1)) for i in range(5)]
         + [f.grid_to_world((4, j, 1)) for j in range(6, 12)])
    out = np.asarray(smoothing.smooth_path(G, f, blocked, _wire(), None, None, 8.0))
    # no INTERIOR point lands in a prohibited voxel (endpoints are terminals)
    for p in out[1:-1]:
        idx = f.world_to_grid(p)
        assert not blocked[idx]


def test_waypoint_is_passed_through_exactly():
    f = _frame()
    blocked = np.zeros(f.res_xyz, dtype=bool)
    # an L-shaped path; the corner point is a hard waypoint the curve must hit
    G = ([f.grid_to_world((i, 5, 1)) for i in range(6)]
         + [f.grid_to_world((5, j, 1)) for j in range(6, 12)])
    wp_idx = 5   # the corner vertex
    wp = np.asarray(G[wp_idx], float)
    out = np.asarray(smoothing.smooth_path(G, f, blocked, _wire(), None, None, 8.0,
                                           fixed_idx=[wp_idx]))
    # some point on the smoothed curve coincides with the waypoint (hard pass-through)
    dmin = float(np.min(np.linalg.norm(out - wp, axis=1)))
    assert dmin < 1e-6


def test_waypoint_free_means_curve_may_miss_it():
    # without pinning, strong smoothing cuts the corner (so pinning is what guarantees it)
    f = _frame()
    blocked = np.zeros(f.res_xyz, dtype=bool)
    G = ([f.grid_to_world((i, 5, 1)) for i in range(6)]
         + [f.grid_to_world((5, j, 1)) for j in range(6, 12)])
    wp = np.asarray(G[5], float)
    out = np.asarray(smoothing.smooth_path(G, f, blocked, _wire(), None, None, 8.0))
    dmin = float(np.min(np.linalg.norm(out - wp, axis=1)))
    assert dmin > 1e-6   # the rounded curve pulls away from the un-pinned corner


def test_tangency_pins_first_segment_to_heading():
    f = _frame()
    blocked = np.zeros(f.res_xyz, dtype=bool)
    G = ([f.grid_to_world((2, j, 1)) for j in range(5, 9)]      # starts going +Y
         + [f.grid_to_world((i, 8, 1)) for i in range(3, 10)])
    out = np.asarray(smoothing.smooth_path(
        G, f, blocked, _wire(), (1.0, 0.0, 0.0), None, 5.0))  # force leave +X
    d = out[1] - out[0]
    d = d / (np.linalg.norm(d) + 1e-12)
    assert d[0] > 0.9   # first segment points along +X


def test_degenerate_short_path_is_safe():
    f = _frame()
    blocked = np.zeros(f.res_xyz, dtype=bool)
    G = [f.grid_to_world((2, 5, 1)), f.grid_to_world((3, 5, 1))]
    out = smoothing.smooth_path(G, f, blocked, _wire(), None, None, 5.0)
    assert len(out) >= 2
    assert np.allclose(out[0], G[0]) and np.allclose(out[-1], G[-1])
