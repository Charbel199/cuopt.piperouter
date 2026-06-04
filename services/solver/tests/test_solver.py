import numpy as np

from piperouter_solver.models import RouteRequest, SolveReport, WireType
from piperouter_solver.solver import Solver


def _wire(min_bend=50.0, max_temp=200.0, em_sens=0.0):
    return WireType(
        id="w1", label="w1", kind="wire", outer_diameter_mm=10.0,
        min_bend_radius_mm=min_bend, cost_per_m=1.0, mass_per_m_kg=0.1,
        max_temp_c=max_temp, em_sensitivity=em_sens, color=(1.0, 0.0, 0.0),
    )


def test_routes_straight_line_in_open_grid(empty_stack):
    s = empty_stack
    req = RouteRequest(
        wire=_wire(), start=tuple(s.frame.grid_to_world((0, 5, 1))),
        end=tuple(s.frame.grid_to_world((9, 5, 1))), connectivity=6,
    )
    res = Solver().route_one(s, req)
    assert res.status == "routed"
    assert len(res.polyline) >= 2
    # length is roughly the straight span (~0.8 m). The default smoothing pass keeps a
    # straight line straight but introduces sub-micron numerical variation, so the
    # lower bound has a hair of slack below the exact span.
    assert 0.79 <= res.length_m <= 1.2


def test_route_detours_around_a_wall(wall_stack):
    s = wall_stack  # wall at x=5 for all y except y=0
    req = RouteRequest(
        wire=_wire(), start=tuple(s.frame.grid_to_world((0, 5, 1))),
        end=tuple(s.frame.grid_to_world((9, 5, 1))), connectivity=26,
    )
    res = Solver().route_one(s, req)
    assert res.status == "routed"
    # no routed cell sits inside the wall
    assert all(not (c[0] == 5 and c[1] >= 1) for c in res.cells)


def test_unroutable_when_fully_walled(empty_stack):
    s = empty_stack
    s.occupancy[5, :, :] = 1  # complete wall, no gap
    req = RouteRequest(
        wire=_wire(), start=tuple(s.frame.grid_to_world((0, 5, 1))),
        end=tuple(s.frame.grid_to_world((9, 5, 1))), connectivity=26,
    )
    res = Solver().route_one(s, req)
    assert res.status == "no_path"


def test_clearance_widens_obstacles(empty_stack):
    s = empty_stack
    s.occupancy[5, 1:, :] = 1  # wall at x=5 with a one-cell gap at y=0
    start = tuple(s.frame.grid_to_world((0, 5, 1)))
    end = tuple(s.frame.grid_to_world((9, 5, 1)))
    # clearance 0: thin wire squeezes through the gap
    flush = Solver().route_one(s, RouteRequest(wire=_wire(), start=start, end=end,
                                               connectivity=26, clearance_m=0.0))
    # clearance 0.25 m (~2-3 cells) grows the wall and seals the gap
    safe = Solver().route_one(s, RouteRequest(wire=_wire(), start=start, end=end,
                                              connectivity=26, clearance_m=0.25))
    assert flush.status == "routed"
    assert safe.status == "no_path"


def test_entirely_blocked_scene_is_no_path(empty_stack):
    s = empty_stack
    s.occupancy[:, :, :] = 1  # the whole scene is solid -> nothing routable
    start = tuple(s.frame.grid_to_world((2, 2, 1)))
    end = tuple(s.frame.grid_to_world((7, 7, 1)))
    res = Solver().route_one(s, RouteRequest(wire=_wire(), start=start, end=end,
                                             connectivity=26))
    assert res.status == "no_path"   # endpoints buried, no free neighbour -> no route


def test_endpoint_on_surface_still_routes(empty_stack):
    s = empty_stack
    s.occupancy[5, 5, 1] = 1  # a single occupied cell; start sits ON it (a connector)
    start = tuple(s.frame.grid_to_world((5, 5, 1)))
    end = tuple(s.frame.grid_to_world((0, 0, 1)))
    res = Solver().route_one(s, RouteRequest(wire=_wire(), start=start, end=end,
                                             connectivity=26))
    assert res.status == "routed"    # freed because it has open neighbours


def test_route_interior_never_enters_clearance_zone():
    # the route's interior cells must all be OUTSIDE the (radius+clearance) keep-out —
    # clearance is hard. (Endpoints may sit on a surface, so they're excluded.)
    from piperouter_solver.grids import GridStack
    from piperouter_solver.models import GridFrame
    frame = GridFrame(bounds_min=np.zeros(3), cell_size=0.1, res_xyz=(20, 20, 3))
    occ = np.zeros((20, 20, 3), np.uint8)
    occ[8:12, 6:14, :] = 1  # a block the route must go around
    s = GridStack(frame=frame, occupancy=occ,
                  surface_dist=np.full((20, 20, 3), 5.0, np.float32),
                  thermal=np.full((20, 20, 3), 20.0, np.float32),
                  em=np.zeros((20, 20, 3), np.float32))
    wire = _wire()
    clr = 0.3
    res = Solver().route_one(s, RouteRequest(
        wire=wire, start=tuple(frame.grid_to_world((1, 10, 1))),
        end=tuple(frame.grid_to_world((18, 10, 1))), connectivity=26, clearance_m=clr))
    assert res.status == "routed"
    blocked = s.dilate_occupancy(wire.radius_m + clr).astype(bool)
    for c in res.cells[1:-1]:           # exclude the two endpoints
        assert not blocked[c], f"route cell {c} is inside the clearance keep-out"


def test_clearance_burying_endpoint_fails(empty_stack):
    s = empty_stack
    s.occupancy[5, :, :] = 1  # wall at x=5; both endpoints on the x<5 side
    start = tuple(s.frame.grid_to_world((4, 5, 1)))  # one cell from the wall
    end = tuple(s.frame.grid_to_world((0, 5, 1)))
    # clearance 0: start (x=4) is free -> routes on the near side
    assert Solver().route_one(s, RouteRequest(wire=_wire(), start=start, end=end,
                                              connectivity=26, clearance_m=0.0)).status == "routed"
    # clearance 0.25 m dilates the wall ~3 cells, burying the start in the clearance
    # band (not the mesh) -> endpoint not force-freed -> no_path
    assert Solver().route_one(s, RouteRequest(wire=_wire(), start=start, end=end,
                                              connectivity=26, clearance_m=0.25)).status == "no_path"


def test_melt_cutoff_blocks_hot_corridor(empty_stack):
    s = empty_stack
    s.thermal[5, :, :] = 300.0  # hot wall exceeds rating -> hard removed
    req = RouteRequest(
        wire=_wire(max_temp=100.0),
        start=tuple(s.frame.grid_to_world((0, 5, 1))),
        end=tuple(s.frame.grid_to_world((9, 5, 1))), connectivity=26,
    )
    res = Solver().route_one(s, req)
    assert res.status == "no_path"


def test_waypoint_is_passed_through(empty_stack):
    s = empty_stack
    wp = s.frame.grid_to_world((5, 9, 1))  # force the route up to y=9
    req = RouteRequest(
        wire=_wire(), start=tuple(s.frame.grid_to_world((0, 0, 1))),
        end=tuple(s.frame.grid_to_world((9, 0, 1))),
        waypoints=[tuple(wp)], connectivity=26,
    )
    res = Solver().route_one(s, req)
    assert res.status == "routed"
    assert any(c[1] == 9 for c in res.cells)  # route reached the waypoint row


def test_surface_hug_pulls_route_toward_surface(empty_stack):
    s = empty_stack
    # make the y=0 edge "near a surface" (low distance), interior far
    s.surface_dist[:, :, :] = 5.0
    s.surface_dist[:, 0, :] = 0.0
    start = tuple(s.frame.grid_to_world((0, 3, 1)))
    end = tuple(s.frame.grid_to_world((9, 3, 1)))
    hug = Solver().route_one(
        s, RouteRequest(wire=_wire(), start=start, end=end,
                        weights={"surface": 5.0}, connectivity=26))
    plain = Solver().route_one(
        s, RouteRequest(wire=_wire(), start=start, end=end,
                        weights={"surface": 0.0}, connectivity=26))
    assert hug.status == plain.status == "routed"
    # with surface weight the path's mean y is pulled lower (toward y=0 surface)
    mean_y_hug = np.mean([c[1] for c in hug.cells])
    mean_y_plain = np.mean([c[1] for c in plain.cells])
    assert mean_y_hug <= mean_y_plain


def test_route_all_orders_by_priority_and_avoids_earlier(empty_stack):
    s = empty_stack
    # two wires both want the same corridor at y=5; the second must divert
    a = RouteRequest(
        wire=_wire(), start=tuple(s.frame.grid_to_world((0, 5, 1))),
        end=tuple(s.frame.grid_to_world((9, 5, 1))), connectivity=26, priority=0)
    b = RouteRequest(
        wire=_wire(), start=tuple(s.frame.grid_to_world((0, 5, 2))),
        end=tuple(s.frame.grid_to_world((9, 5, 2))), connectivity=26, priority=1)
    report = Solver().route_all(s, [b, a])  # pass out of order on purpose
    assert isinstance(report, SolveReport)
    assert report.routed == 2
    first_cells = set(report.results[0].cells)   # priority 0 routed first
    second_cells = set(report.results[1].cells)
    assert first_cells.isdisjoint(second_cells)  # no shared cells


def test_no_path_reason_thermal(empty_stack):
    s = empty_stack
    s.thermal[0:3, :, :] = 300.0   # a hot slab around the start, no cool neighbour
    start = tuple(s.frame.grid_to_world((0, 5, 1)))
    end = tuple(s.frame.grid_to_world((9, 5, 1)))
    res = Solver().route_one(s, RouteRequest(wire=_wire(max_temp=90.0), start=start,
                                             end=end, connectivity=26))
    assert res.status == "no_path"
    assert "rating" in res.reason and "300C" in res.reason   # thermal explanation


def test_no_path_reason_buried_in_geometry(empty_stack):
    s = empty_stack
    s.occupancy[:, :, :] = 1       # whole scene solid
    start = tuple(s.frame.grid_to_world((2, 2, 1)))
    end = tuple(s.frame.grid_to_world((7, 7, 1)))
    res = Solver().route_one(s, RouteRequest(wire=_wire(), start=start, end=end,
                                             connectivity=26))
    assert res.status == "no_path"
    assert "buried" in res.reason


def test_no_path_reason_no_corridor(empty_stack):
    s = empty_stack
    s.occupancy[5, :, :] = 1       # full wall: both endpoints free but disconnected
    start = tuple(s.frame.grid_to_world((0, 5, 1)))
    end = tuple(s.frame.grid_to_world((9, 5, 1)))
    res = Solver().route_one(s, RouteRequest(wire=_wire(), start=start, end=end,
                                             connectivity=26))
    assert res.status == "no_path"
    assert "corridor" in res.reason


def test_no_path_reason_impossible_heading(empty_stack):
    s = empty_stack
    start = tuple(s.frame.grid_to_world((0, 5, 1)))  # on the -X boundary
    end = tuple(s.frame.grid_to_world((9, 5, 1)))
    res = Solver().route_one(s, RouteRequest(
        wire=_wire(), start=start, end=end, connectivity=26,
        start_heading=(-1.0, 0.0, 0.0)))             # must leave -X off the edge
    assert res.status == "no_path"
    assert "heading" in res.reason


def test_routed_wire_has_no_reason(empty_stack):
    s = empty_stack
    start = tuple(s.frame.grid_to_world((0, 5, 1)))
    end = tuple(s.frame.grid_to_world((9, 5, 1)))
    res = Solver().route_one(s, RouteRequest(wire=_wire(), start=start, end=end,
                                             connectivity=26))
    assert res.status == "routed" and res.reason == ""


def test_pinned_start_heading_forces_first_move_direction(empty_stack):
    s = empty_stack
    start = tuple(s.frame.grid_to_world((2, 5, 1)))
    end = tuple(s.frame.grid_to_world((8, 5, 1)))  # natural path is +X
    req = RouteRequest(
        wire=_wire(), start=start, end=end, connectivity=26,
        weights={"smoothing": 0.0},          # keep grid path for a deterministic check
        start_heading=(0.0, 1.0, 0.0),       # force leaving +Y
    )
    res = Solver().route_one(s, req)
    assert res.status == "routed"
    # the source connects to a NEIGHBOR of the start cell, so cells[0] is already the
    # first move; pinned +Y means it must leave with y greater than the start cell's y.
    start_cell = s.frame.world_to_grid(start)
    assert res.cells[0][1] > start_cell[1]


def test_impossible_heading_is_no_path(empty_stack):
    s = empty_stack
    start = tuple(s.frame.grid_to_world((0, 5, 1)))  # on the -X boundary
    end = tuple(s.frame.grid_to_world((8, 5, 1)))
    req = RouteRequest(
        wire=_wire(), start=start, end=end, connectivity=26,
        weights={"smoothing": 0.0},
        start_heading=(-1.0, 0.0, 0.0),      # must leave -X, but x=0 is the edge
    )
    res = Solver().route_one(s, req)
    assert res.status == "no_path"
