import numpy as np
from pxr import UsdGeom

from omni.piperouter import scene_ops, wire_library
from omni.piperouter.router_session import RouterSession


def test_full_pipeline_routes_around_cube(solver_server, cube_stage):
    base, grid_dir = solver_server
    stage = cube_stage  # box at (0.5,0.5,0.5), half 0.2

    session = RouterSession(grid_dir=grid_dir, solver_url=base)
    # pad generously so there is room to route around the cube
    session.voxelize_scene(stage, "t", resolution=24, pad_frac=1.0)

    spec = wire_library.as_spec(
        wire_library.by_id(wire_library.load_wire_library(), "sig_can"))
    wires = [{
        "name": "harness_0", "spec": spec,
        "start": [0.0, 0.5, 0.5], "end": [1.0, 0.5, 0.5],
        "weights": {}, "connectivity": 6, "priority": 0,
    }]
    results, bom = session.route_all(stage, "t", wires)

    assert results[0]["status"] == "routed"
    assert bom[0]["length_m"] > 0.0
    assert bom[0]["cost"] > 0.0
    # a tube was authored into the stage
    crv = UsdGeom.BasisCurves(
        stage.GetPrimAtPath(scene_ops.ROUTES_SCOPE + "/harness_0"))
    assert crv
    pts = np.array([[p[0], p[1], p[2]] for p in crv.GetPointsAttr().Get()])
    # the route must leave the y=0.5/z=0.5 centre line to clear the cube
    assert np.ptp(pts[:, 1]) > 0.05 or np.ptp(pts[:, 2]) > 0.05


def test_clearance_affects_route_all(solver_server):
    # two boxes with a ~0.6 m gap; a wire through the gap routes at clearance 0 but the
    # gap seals at a large clearance -> proves clearance flows through route_all.
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    scene_ops.author_box_mesh(stage, "/World/a", (1.0, 0.3, 0.5), (0.6, 0.4, 1.2))
    scene_ops.author_box_mesh(stage, "/World/b", (1.0, 1.3, 0.5), (0.6, 0.4, 1.2))
    start, end = (0.2, 0.8, 0.5), (1.8, 0.8, 0.5)
    scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/g_start", start, radius=0.05)
    scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/g_end", end, radius=0.05)

    base, grid_dir = solver_server
    session = RouterSession(grid_dir=grid_dir, solver_url=base)
    spec = wire_library.as_spec(wire_library.by_id(wire_library.load_wire_library(), "sig_can"))

    def route(clearance):
        session.voxelize_scene(stage, f"c{clearance}", resolution=40)
        wire = {"name": "g", "spec": spec, "start": list(start), "end": list(end),
                "connectivity": 6, "clearance_m": clearance}
        results, _ = session.route_all(stage, f"c{clearance}", [wire])
        return results[0]["status"]

    assert route(0.0) == "routed"     # gap is open
    assert route(0.5) == "no_path"    # clearance seals the 0.6 m gap


def test_grid_includes_markers_beyond_geometry(solver_server, cube_stage):
    base, grid_dir = solver_server
    stage = cube_stage  # geometry roughly y in [0.3, 0.7]
    far = (0.5, 2.0, 0.5)  # well beyond the geometry in +y
    scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/t_wp", far, radius=0.05)
    session = RouterSession(grid_dir=grid_dir, solver_url=base)
    session.voxelize_scene(stage, "t", resolution=24)
    gbmin, cell, res = session.frame
    ymax = gbmin[1] + res[1] * cell
    assert gbmin[1] <= far[1] <= ymax        # the far marker is inside the grid
    assert ymax > 1.5                        # bounds actually grew toward it


def test_far_waypoint_is_reached(solver_server, cube_stage):
    base, grid_dir = solver_server
    stage = cube_stage
    start, end, wp = (0.1, 0.1, 0.5), (0.9, 0.1, 0.5), (0.5, 2.0, 0.5)
    for nm, p in (("ww_start", start), ("ww_end", end), ("ww_wp0", wp)):
        scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/{nm}", p, radius=0.05)
    session = RouterSession(grid_dir=grid_dir, solver_url=base)
    session.voxelize_scene(stage, "t", resolution=28)

    spec = wire_library.as_spec(wire_library.by_id(wire_library.load_wire_library(), "sig_can"))
    wire = {"name": "ww", "spec": spec, "start": list(start), "end": list(end),
            "waypoints": [list(wp)], "connectivity": 6}
    res, _bom = session.refine_wire(stage, "t", wire, locked_wires=[])
    assert res["status"] == "routed"
    # the route must actually reach the far waypoint, not clamp to the old edge (~0.7)
    assert max(p[1] for p in res["polyline"]) > 1.5


def test_refine_wire_with_waypoint_and_locked_obstacle(solver_server, cube_stage):
    base, grid_dir = solver_server
    stage = cube_stage  # box at (0.5,0.5,0.5), half 0.2; route in the free y=0.1 plane

    session = RouterSession(grid_dir=grid_dir, solver_url=base)
    session.voxelize_scene(stage, "t", resolution=24, pad_frac=1.0)

    spec = wire_library.as_spec(
        wire_library.by_id(wire_library.load_wire_library(), "sig_can"))

    # (a) a +z waypoint pulls the route up, well clear of the cube
    wire = {"name": "w_a", "spec": spec, "start": [0.1, 0.1, 0.5],
            "end": [0.9, 0.1, 0.5], "waypoints": [[0.5, 0.1, 0.95]],
            "weights": {}, "connectivity": 6}
    res, bom = session.refine_wire(stage, "t", wire, locked_wires=[])
    assert res["status"] == "routed"
    assert max(p[2] for p in res["polyline"]) > 0.8   # reached the waypoint band
    assert bom["length_m"] > 0.0

    # (b) lock w_a, then route w_b along the same line -> must avoid w_a's tube
    locked = [{"spec": spec, "polyline": res["polyline"]}]
    wire_b = dict(wire, name="w_b")
    res_b, _ = session.refine_wire(stage, "t", wire_b, locked_wires=locked)
    assert res_b["status"] == "routed"
    a_cells = {(round(p[0], 3), round(p[1], 3), round(p[2], 3)) for p in res["polyline"]}
    b_cells = {(round(p[0], 3), round(p[1], 3), round(p[2], 3)) for p in res_b["polyline"]}
    # b cannot reuse a's locked interior cells (endpoints may coincide)
    assert len(a_cells & b_cells) < len(a_cells)
