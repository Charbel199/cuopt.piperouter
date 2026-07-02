"""Per-object clearance (clearance-class grid): tagged geometry keeps ITS distance,
untagged geometry keeps the request default."""
import numpy as np

from piperouter_solver.grids import GridStack
from piperouter_solver.models import GridFrame, RouteRequest, WireType
from piperouter_solver.solver import Solver


def _wire():
    return WireType(id="w", label="w", kind="wire", outer_diameter_mm=4.0,
                    min_bend_radius_mm=20.0, cost_per_m=1.0, mass_per_m_kg=0.1,
                    max_temp_c=200.0, em_sensitivity=0.0, color=(1.0, 1.0, 0.0))


def _stack_two_walls():
    """A corridor between two wall slabs: y=4 (TAGGED, big clearance) and y=16
    (untagged). The route runs along x between them."""
    frame = GridFrame(bounds_min=np.zeros(3), cell_size=0.01, res_xyz=(30, 21, 8))
    occ = np.zeros(frame.res_xyz, np.uint8)
    occ[:, 4, :] = 1     # tagged wall
    occ[:, 16, :] = 1    # untagged wall
    cls = np.zeros(frame.res_xyz, np.uint8)
    cls[:, 4, :] = 1     # class 1 = the tagged wall
    return GridStack(frame=frame, occupancy=occ,
                     surface_dist=np.full(frame.res_xyz, 5.0, np.float32),
                     thermal=np.full(frame.res_xyz, 20.0, np.float32),
                     em=np.zeros(frame.res_xyz, np.float32),
                     clearance_class=cls, clearance_values=(0.06,))   # 6 cells


def test_tagged_wall_repels_route_untagged_does_not():
    s = _stack_two_walls()
    # endpoints midway between the walls (y=10); zero DEFAULT clearance
    res = Solver().route_one(s, RouteRequest(
        wire=_wire(), start=tuple(s.frame.grid_to_world((1, 10, 4))),
        end=tuple(s.frame.grid_to_world((28, 10, 4))), connectivity=26, clearance_m=0.0,
        weights={"smoothing": 0.0, "bend": 0.0}))
    assert res.status == "routed"
    # judge the MID-SPAN only: the terminals sit at y=10 and are legitimately exempted
    # from the shell (terminal reachability), so cells near x=1 / x=28 may be at y=10.
    mid = [c for c in res.cells if 9 <= c[0] <= 20]
    assert mid, "route has no mid-span cells"
    ys = [c[1] for c in mid]
    # tagged wall at y=4 with 0.06 m (6 cells) clearance -> mid-span stays y >= 11
    assert min(ys) >= 11, f"route entered the tagged wall's clearance band (min y={min(ys)})"
    # untagged wall at y=16 has NO clearance -> approaching it is allowed
    assert max(ys) <= 15


def test_default_clearance_still_applies_to_untagged():
    s = _stack_two_walls()
    # default clearance 0.03 (3 cells) applies to the UNTAGGED wall only (class 0)
    res = Solver().route_one(s, RouteRequest(
        wire=_wire(), start=tuple(s.frame.grid_to_world((1, 10, 4))),
        end=tuple(s.frame.grid_to_world((28, 10, 4))), connectivity=26, clearance_m=0.03,
        weights={"smoothing": 0.0, "bend": 0.0}))
    assert res.status == "routed"
    mid = [c for c in res.cells if 9 <= c[0] <= 20]
    assert mid
    ys = [c[1] for c in mid]
    assert min(ys) >= 11          # tagged wall: its own 6-cell clearance
    assert max(ys) <= 12          # untagged wall at 16: default 3 cells -> stay <= 12


def test_grid_roundtrip_carries_clearance_classes(tmp_path):
    s = _stack_two_walls()
    p = tmp_path / "stack.npz"
    s.save(p)
    s2 = GridStack.load(p)
    assert s2.clearance_class is not None
    assert np.array_equal(s2.clearance_class, s.clearance_class)
    assert s2.clearance_values == (0.06,)


def test_old_grids_without_classes_still_load(tmp_path):
    frame = GridFrame(bounds_min=np.zeros(3), cell_size=0.1, res_xyz=(5, 5, 5))
    s = GridStack(frame=frame, occupancy=np.zeros(frame.res_xyz, np.uint8),
                  surface_dist=np.full(frame.res_xyz, 5.0, np.float32),
                  thermal=np.full(frame.res_xyz, 20.0, np.float32),
                  em=np.zeros(frame.res_xyz, np.float32))
    p = tmp_path / "stack.npz"
    s.save(p)
    s2 = GridStack.load(p)
    assert s2.clearance_class is None and s2.clearance_values == ()
