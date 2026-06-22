import numpy as np

from piperouter_solver.models import RouteRequest, WireType
from piperouter_solver.solver import Solver


def _wire(**kw):
    base = dict(id="w", label="w", kind="wire", outer_diameter_mm=10.0,
                min_bend_radius_mm=50.0, cost_per_m=1.0, mass_per_m_kg=0.1,
                max_temp_c=200.0, em_sensitivity=1.0, color=(1.0, 0.0, 0.0))
    base.update(kw)
    return WireType(**base)


def test_buried_start_relocates_and_routes_with_note(empty_stack):
    s = empty_stack                          # 10x10x3, 0.1 m cells, open
    s.occupancy[1, 1, 1] = 1                  # bury the START cell inside a solid block
    frame = s.frame
    start = frame.grid_to_world((1, 1, 1))   # exactly on the buried cell
    end = frame.grid_to_world((8, 8, 1))
    res = Solver().route_one(s, RouteRequest(wire=_wire(), start=start, end=end,
                                             weights={"smoothing": 1.0}))
    # it routes instead of failing...
    assert res.status == "routed"
    # ...flags the relocation...
    assert res.note and "buried" in res.note.lower()
    # ...and the tube's first point is NOT inside the blocked cell anymore
    i, j, k = frame.world_to_grid(tuple(res.polyline[0]))
    assert not s.occupancy[i, j, k]


def test_open_endpoints_route_with_no_note(empty_stack):
    frame = empty_stack.frame
    res = Solver().route_one(empty_stack, RouteRequest(
        wire=_wire(), start=frame.grid_to_world((1, 1, 1)),
        end=frame.grid_to_world((8, 8, 1)), weights={"smoothing": 1.0}))
    assert res.status == "routed"
    assert res.note == ""                    # nothing buried -> no warning (no behavior change)
