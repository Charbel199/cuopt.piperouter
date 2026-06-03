import time

import numpy as np

from piperouter_solver.grids import GridStack
from piperouter_solver.lattice import ExpandedLatticeBuilder
from piperouter_solver.models import GridFrame, WireType


def _big_stack(n=24):
    frame = GridFrame(bounds_min=np.zeros(3), cell_size=0.05, res_xyz=(n, n, n))
    shape = (n, n, n)
    return GridStack(
        frame=frame,
        occupancy=np.zeros(shape, dtype=np.uint8),
        surface_dist=np.full(shape, 1.0, dtype=np.float32),
        thermal=np.full(shape, 20.0, dtype=np.float32),
        em=np.zeros(shape, dtype=np.float32),
    )


def _wire():
    return WireType(
        id="t", label="t", kind="wire", outer_diameter_mm=8.0,
        min_bend_radius_mm=40.0, cost_per_m=1.0, mass_per_m_kg=0.1,
        max_temp_c=200.0, em_sensitivity=0.0, color=(1.0, 0.0, 0.0),
    )


def test_vectorized_build_scales_to_24cubed_26conn():
    s = _big_stack(24)
    n = 24
    t0 = time.perf_counter()
    g = ExpandedLatticeBuilder().build(
        s, _wire(), weights={"surface": 1.0}, connectivity=26,
        start_cell=(0, 0, 0), goal_cell=(n - 1, n - 1, n - 1), extra_obstacles=None,
    )
    elapsed = time.perf_counter() - t0
    # 24^3 free cells * 26 headings + 2 virtual nodes
    assert g.n_nodes == 24 ** 3 * 26 + 2
    assert g.src.size > 1_000_000          # real edge volume, built vectorized
    assert g.src.shape == g.dst.shape == g.weight.shape
    # vectorized build must be fast; the old python-loop builder took ~minutes here
    assert elapsed < 10.0, f"lattice build too slow: {elapsed:.1f}s"
