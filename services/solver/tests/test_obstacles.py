import numpy as np

from piperouter_solver.models import GridFrame
from piperouter_solver.obstacles import rasterize_polylines


def _frame():
    return GridFrame(bounds_min=np.zeros(3), cell_size=0.1, res_xyz=(10, 10, 3))


def test_marks_polyline_cells():
    frame = _frame()
    poly = [frame.grid_to_world((i, 5, 1)) for i in range(10)]
    mask = rasterize_polylines(frame, [{"polyline": poly, "radius_m": 0.0}])
    assert mask[:, 5, 1].all()
    assert not mask[:, 0, 1].any()


def test_radius_widens_band():
    frame = _frame()
    poly = [frame.grid_to_world((i, 5, 1)) for i in range(10)]
    thin = rasterize_polylines(frame, [{"polyline": poly, "radius_m": 0.0}])
    fat = rasterize_polylines(frame, [{"polyline": poly, "radius_m": 0.15}])
    assert fat.sum() > thin.sum()
    assert fat[5, 4, 1] and fat[5, 6, 1]  # widened in y


def test_empty_routes_all_false():
    frame = _frame()
    assert not rasterize_polylines(frame, []).any()
    assert not rasterize_polylines(frame, [{"polyline": [], "radius_m": 0.1}]).any()
