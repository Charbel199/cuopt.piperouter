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


def test_clearance_does_not_relocate_a_near_surface_endpoint(empty_stack):
    s = empty_stack
    s.occupancy[5, :, :] = 1  # wall at x=5; start one cell away on the x<5 side
    start = tuple(s.frame.grid_to_world((4, 5, 1)))
    end = tuple(s.frame.grid_to_world((0, 5, 1)))
    # clearance 0.25 m puts the start inside the wall's clearance BAND (but not the mesh).
    # Clearance must NOT relocate it (only the mesh does) — it routes from the real start,
    # passing through the near-surface clearance voxels, with no relocation note.
    res = Solver().route_one(s, RouteRequest(wire=_wire(), start=start, end=end,
                                             connectivity=26, clearance_m=0.25))
    assert res.status == "routed"
    assert res.note == ""                                   # clearance never pushes endpoints
    assert np.allclose(res.polyline[0], start, atol=2 * s.frame.cell_size)  # starts AT start


def test_mesh_buried_endpoint_still_relocates(empty_stack):
    s = empty_stack
    s.occupancy[5, 5, 1] = 1                                # bury the START in the MESH
    start = tuple(s.frame.grid_to_world((5, 5, 1)))
    end = tuple(s.frame.grid_to_world((9, 5, 1)))
    res = Solver().route_one(s, RouteRequest(wire=_wire(), start=start, end=end,
                                             connectivity=26, clearance_m=0.25))
    assert res.status == "routed"
    assert "buried" in res.note.lower()                     # mesh burial -> relocate + note


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


def _cuts_edge(a, b, occ):
    """True if a->b is a 2D EDGE diagonal (exactly two non-zero components) that
    squeezes between two occupied face cells. The relaxed rule forbids exactly this;
    3D corner moves are intentionally allowed, so they are not checked here."""
    off = tuple(int(b[i] - a[i]) for i in range(3))
    axes = [i for i in range(3) if off[i] != 0]
    if len(axes) != 2:
        return False
    for ax in axes:
        c = tuple(a[i] + (off[i] if i == ax else 0) for i in range(3))
        if occ[c]:
            return True
    return False


def test_no_edge_corner_cutting_around_block(empty_stack):
    s = empty_stack
    # the two cells a 2D (4,4)->(5,5) edge diagonal would squeeze between
    s.occupancy[5, 4, 1] = 1
    s.occupancy[4, 5, 1] = 1
    start = tuple(s.frame.grid_to_world((4, 4, 1)))
    end = tuple(s.frame.grid_to_world((5, 5, 1)))
    res = Solver().route_one(s, RouteRequest(wire=_wire(), start=start, end=end,
                                             connectivity=26, weights={"smoothing": 0.0}))
    assert res.status == "routed"
    occ = s.occupancy.astype(bool)
    start_cell = s.frame.world_to_grid(start)
    full = [tuple(int(v) for v in start_cell)] + [tuple(c) for c in res.cells]
    for a, b in zip(full[:-1], full[1:]):
        assert not _cuts_edge(a, b, occ), f"step {a}->{b} cuts a 2D edge"


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


def test_no_path_reason_corridor_blocked_by_heat(empty_stack):
    s = empty_stack
    s.thermal[5, :, :] = 300.0   # a hot wall (melt) separates two cool endpoints
    start = tuple(s.frame.grid_to_world((0, 5, 1)))
    end = tuple(s.frame.grid_to_world((9, 5, 1)))
    res = Solver().route_one(s, RouteRequest(wire=_wire(max_temp=90.0), start=start,
                                             end=end, connectivity=26))
    assert res.status == "no_path"
    # the 'no corridor' message must call out the heat, not be generic
    assert "rating" in res.reason or "heat" in res.reason


def test_routed_polyline_connects_to_markers(empty_stack):
    s = empty_stack
    start = tuple(s.frame.grid_to_world((1, 5, 1)))
    end = tuple(s.frame.grid_to_world((8, 5, 1)))
    res = Solver().route_one(s, RouteRequest(wire=_wire(), start=start, end=end,
                                             connectivity=26, weights={"smoothing": 0.0}))
    assert res.status == "routed"
    # the polyline must START at the start marker and END at the end marker exactly
    assert np.allclose(res.polyline[0], start)
    assert np.allclose(res.polyline[-1], end)


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


def test_route_all_keeps_tube_bodies_apart(empty_stack):
    # Two OD-24mm pipes with endpoints 2 cells (20mm) apart: their tubes need 24mm of
    # centerline separation, so the second route must BOW AWAY mid-span instead of
    # running parallel and interpenetrating (the old code only blocked the 1-cell
    # centerline, so tubes overlapped at fine resolutions).
    import numpy as np
    from piperouter_solver.grids import GridStack
    from piperouter_solver.models import GridFrame

    frame = GridFrame(bounds_min=np.zeros(3), cell_size=0.01, res_xyz=(40, 40, 40))
    s = GridStack(frame=frame, occupancy=np.zeros(frame.res_xyz, np.uint8),
                  surface_dist=np.full(frame.res_xyz, 5.0, np.float32),
                  thermal=np.full(frame.res_xyz, 20.0, np.float32),
                  em=np.zeros(frame.res_xyz, np.float32))
    fat = WireType(id="fat", label="fat", kind="pipe", outer_diameter_mm=24.0,
                   inner_diameter_mm=20.0, min_bend_radius_mm=60.0, cost_per_m=1.0,
                   mass_per_m_kg=0.1, max_temp_c=200.0, em_sensitivity=0.0,
                   color=(0.2, 0.4, 0.9))
    reqA = RouteRequest(wire=fat, start=(0.02, 0.20, 0.20), end=(0.38, 0.20, 0.20),
                        weights={"bend": 1.0, "smoothing": 0.0}, priority=0)
    reqB = RouteRequest(wire=fat, start=(0.02, 0.22, 0.20), end=(0.38, 0.22, 0.20),
                        weights={"bend": 1.0, "smoothing": 0.0}, priority=1)
    rep = Solver().route_all(s, [reqA, reqB])
    assert [r.status for r in rep.results] == ["routed", "routed"]
    A = np.asarray(rep.results[0].polyline)
    B = np.asarray(rep.results[1].polyline)
    interior = B[3:-3]                      # terminals are user-fixed; judge the run
    dmin = min(float(np.linalg.norm(a - b)) for a in A for b in interior)
    assert dmin >= 0.024 - 1e-6, f"tube bodies overlap: {dmin*1000:.1f}mm < 24mm"


def test_waypoint_in_clearance_band_still_routes(empty_stack):
    # A waypoint 3 cells above a floor, inside a 6-cell clearance band: the shell must be
    # waived around WAYPOINTS like endpoints, so the route passes through it (was no_path).
    import numpy as np
    from piperouter_solver.grids import GridStack
    from piperouter_solver.models import GridFrame

    frame = GridFrame(bounds_min=np.zeros(3), cell_size=0.01, res_xyz=(40, 40, 40))
    s = GridStack(frame=frame, occupancy=np.zeros(frame.res_xyz, np.uint8),
                  surface_dist=np.full(frame.res_xyz, 5.0, np.float32),
                  thermal=np.full(frame.res_xyz, 20.0, np.float32),
                  em=np.zeros(frame.res_xyz, np.float32))
    s.occupancy[:, :, 8] = 1                # floor slab
    wp = (0.205, 0.205, 0.115)              # ~3 cells above the floor
    res = Solver().route_one(s, RouteRequest(
        wire=_wire(), start=(0.05, 0.20, 0.30), end=(0.35, 0.20, 0.30),
        waypoints=[wp], weights={"bend": 1.0, "smoothing": 1.0}, clearance_m=0.06))
    assert res.status == "routed"
    d = min(float(np.linalg.norm(np.asarray(p) - np.asarray(wp))) for p in res.polyline)
    assert d < 0.015, f"route does not pass through the waypoint ({d*1000:.0f}mm away)"


def test_no_path_reason_names_clearance_not_phantom_wire(empty_stack):
    # A corridor sealed by CLEARANCE alone (no other wires) must not be blamed on
    # "another already-routed wire" (the shell used to be folded into extra_obstacles).
    import numpy as np
    from piperouter_solver.grids import GridStack
    from piperouter_solver.models import GridFrame

    frame = GridFrame(bounds_min=np.zeros(3), cell_size=0.01, res_xyz=(40, 40, 40))
    s = GridStack(frame=frame, occupancy=np.zeros(frame.res_xyz, np.uint8),
                  surface_dist=np.full(frame.res_xyz, 5.0, np.float32),
                  thermal=np.full(frame.res_xyz, 20.0, np.float32),
                  em=np.zeros(frame.res_xyz, np.float32))
    s.occupancy[20, :, :] = 1
    s.occupancy[20, 20, 20] = 0             # 1-cell hole, sealed once clearance applies
    res = Solver().route_one(s, RouteRequest(
        wire=_wire(), start=(0.05, 0.20, 0.20), end=(0.35, 0.20, 0.20), clearance_m=0.05))
    assert res.status == "no_path"
    assert "already-routed" not in res.reason
    assert "clearance" in res.reason.lower()
