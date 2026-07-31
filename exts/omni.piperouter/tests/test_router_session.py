import numpy as np
from pxr import UsdGeom

from omni.piperouter import scene_ops, wire_library
from omni.piperouter.router_session import RouterSession


def test_full_pipeline_routes_around_cube(solver_server, cube_stage):
    base, grid_dir = solver_server
    stage = cube_stage  # box at (0.5,0.5,0.5), half 0.2

    session = RouterSession(grid_dir=grid_dir, solver_url=base)
    # Pad generously so there is room to route around the cube.
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
    # A tube was authored into the stage.
    crv = UsdGeom.BasisCurves(
        stage.GetPrimAtPath(scene_ops.ROUTES_SCOPE + "/harness_0"))
    assert crv
    pts = np.array([[p[0], p[1], p[2]] for p in crv.GetPointsAttr().Get()])
    # Start and end are collinear through the cube, so the route has to leave the
    # y=0.5/z=0.5 centre line to clear it.
    assert np.ptp(pts[:, 1]) > 0.05 or np.ptp(pts[:, 2]) > 0.05


def test_clearance_affects_route_all(solver_server):
    # Two boxes with a ~0.6 m gap. A wire through the gap routes at clearance 0 and
    # fails once the clearance band seals the gap, which pins clearance reaching the
    # solver through route_all.
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
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
        # Clearance is fixed at voxelize time, so each value needs its own session id.
        session.voxelize_scene(stage, f"c{clearance}", resolution=40, clearance_m=clearance)
        wire = {"name": "g", "spec": spec, "start": list(start), "end": list(end),
                "connectivity": 6}
        results, _ = session.route_all(stage, f"c{clearance}", [wire])
        return results[0]["status"]

    assert route(0.0) == "routed"     # gap is open
    assert route(0.5) == "no_path"    # clearance seals the 0.6 m gap


def test_no_path_reason_flows_through_http(solver_server):
    # A wall spanning the whole cross-section partitions the bay, and pad_frac=0 leaves
    # no free border to sneak around, so this is a genuine no_path. The reason must
    # survive the solver schema and the HTTP round-trip into the BOM row.
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    scene_ops.author_box_mesh(stage, "/World/wall", (1.0, 0.5, 0.5), (0.2, 2.0, 2.0))
    start, end = (0.2, 0.5, 0.5), (1.8, 0.5, 0.5)  # opposite sides of the wall, both open
    scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/b_start", start, radius=0.05)
    scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/b_end", end, radius=0.05)

    base, grid_dir = solver_server
    session = RouterSession(grid_dir=grid_dir, solver_url=base)
    session.voxelize_scene(stage, "nb", resolution=32, pad_frac=0.0)
    spec = wire_library.as_spec(wire_library.by_id(wire_library.load_wire_library(), "sig_can"))
    wire = {"name": "blocked", "spec": spec, "start": list(start), "end": list(end),
            "connectivity": 26}
    results, bom = session.route_all(stage, "nb", [wire])

    assert results[0]["status"] == "no_path"
    assert results[0]["reason"]            # reason present in the HTTP result
    assert bom[0]["reason"]                # and carried into the BOM row


def test_relocation_note_flows_through_http(solver_server, cube_stage):
    # The end sits buried in the cube centre and the start in open space, with
    # pad_frac=1.0 leaving room to route. The buried end relocates to the nearest open
    # point, and the non-fatal note must survive the HTTP round-trip into the result.
    base, grid_dir = solver_server
    stage = cube_stage  # solid box at (0.5,0.5,0.5), half 0.2
    session = RouterSession(grid_dir=grid_dir, solver_url=base)
    session.voxelize_scene(stage, "rb", resolution=24, pad_frac=1.0)
    spec = wire_library.as_spec(wire_library.by_id(wire_library.load_wire_library(), "sig_can"))
    wire = {"name": "relocated", "spec": spec, "start": [0.0, 0.5, 0.5],
            "end": [0.5, 0.5, 0.5], "connectivity": 26}   # end at the cube centre
    results, _bom = session.route_all(stage, "rb", [wire])

    assert results[0]["status"] == "routed"          # rescued instead of failing
    assert "buried" in results[0].get("note", "").lower()   # note survived the round-trip


def test_selected_algorithm_flows_through_http(solver_server, cube_stage):
    # The global/local algorithm choice must survive route dict -> schema ->
    # RouteRequest -> solver dispatch and still produce a route.
    base, grid_dir = solver_server
    stage = cube_stage
    session = RouterSession(grid_dir=grid_dir, solver_url=base)
    session.global_planner, session.local_optimizer = "astar", "trajopt"
    session.voxelize_scene(stage, "algo", resolution=24)
    spec = wire_library.as_spec(wire_library.by_id(wire_library.load_wire_library(), "sig_can"))
    wire = {"name": "a", "spec": spec, "start": [0.1, 0.1, 0.5], "end": [0.9, 0.1, 0.5],
            "connectivity": 18}
    results, _bom = session.route_all(stage, "algo", [wire])
    assert results[0]["status"] == "routed"


def test_clearance_not_baked_into_occupancy(cube_stage):
    # Occupancy holds the raw mesh so the solver can tell mesh from clearance halo, so
    # occ must come out identical whatever the clearance. The value is only remembered
    # and sent to the solver per route, where it acts as a relaxable band.
    session = RouterSession()   # compute_grids is local; no solver needed
    g0 = session.compute_grids(cube_stage, resolution=24, clearance_m=0.0)
    g1 = session.compute_grids(cube_stage, resolution=24, clearance_m=0.3)
    assert int(g0[3].sum()) == int(g1[3].sum())          # occupancy unchanged by clearance
    assert session.last_clearance_m == 0.3               # but the value is remembered


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
    assert ymax > 1.5                        # bounds grew past the geometry to reach it


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
    # The route must reach the far waypoint, not stop at the geometry bounds (y~0.7).
    assert max(p[1] for p in res["polyline"]) > 1.5


def test_refine_wire_with_waypoint_and_locked_obstacle(solver_server, cube_stage):
    base, grid_dir = solver_server
    stage = cube_stage  # box at (0.5,0.5,0.5), half 0.2; route in the free y=0.1 plane

    session = RouterSession(grid_dir=grid_dir, solver_url=base)
    session.voxelize_scene(stage, "t", resolution=24, pad_frac=1.0)

    spec = wire_library.as_spec(
        wire_library.by_id(wire_library.load_wire_library(), "sig_can"))

    # (a) A +z waypoint pulls the route up, well clear of the cube.
    wire = {"name": "w_a", "spec": spec, "start": [0.1, 0.1, 0.5],
            "end": [0.9, 0.1, 0.5], "waypoints": [[0.5, 0.1, 0.95]],
            "weights": {}, "connectivity": 6}
    res, bom = session.refine_wire(stage, "t", wire, locked_wires=[])
    assert res["status"] == "routed"
    assert max(p[2] for p in res["polyline"]) > 0.8   # reached the waypoint band
    assert bom["length_m"] > 0.0

    # (b) Lock w_a, then route w_b along the same line: it must avoid w_a's tube.
    locked = [{"spec": spec, "polyline": res["polyline"]}]
    wire_b = dict(wire, name="w_b")
    res_b, _ = session.refine_wire(stage, "t", wire_b, locked_wires=locked)
    assert res_b["status"] == "routed"
    a_cells = {(round(p[0], 3), round(p[1], 3), round(p[2], 3)) for p in res["polyline"]}
    b_cells = {(round(p[0], 3), round(p[1], 3), round(p[2], 3)) for p in res_b["polyline"]}
    # b cannot reuse a's locked interior cells, though endpoints may coincide.
    assert len(a_cells & b_cells) < len(a_cells)


def test_bundle_trunk_and_branches_routed(solver_server):
    """Two wires share a bundle trunk; each gets a stitched full polyline."""
    from pxr import Usd
    from omni.piperouter.router_session import RouterSession

    base, grid_dir = solver_server
    stage = Usd.Stage.CreateInMemory()
    from pxr import UsdGeom as _UG
    _UG.SetStageMetersPerUnit(stage, 1.0)
    _UG.Xform.Define(stage, "/World")
    # An open scene: no obstacles except the ground plane.
    scene_ops.author_box_mesh(stage, "/World/ground",
                              (2.0, 2.0, -0.05), (6.0, 6.0, 0.05))
    scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/bA_merge",
                           (1.0, 1.0, 0.5), radius=0.05)
    scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/bA_split",
                           (3.0, 1.0, 0.5), radius=0.05)

    session = RouterSession(grid_dir=grid_dir, solver_url=base)
    session.voxelize_scene(stage, "bundle_t", resolution=32, pad_frac=0.5)

    types = wire_library.load_wire_library()
    spec = wire_library.as_spec(wire_library.by_id(types, "sig_can"))

    wires = [
        {"name": "w0", "spec": spec, "start": [0.0, 0.5, 0.5],
         "end": [4.0, 0.5, 0.5], "bundle_id": "bA", "weights": {}, "connectivity": 18},
        {"name": "w1", "spec": spec, "start": [0.0, 1.5, 0.5],
         "end": [4.0, 1.5, 0.5], "bundle_id": "bA", "weights": {}, "connectivity": 18},
    ]
    bundles_list = [{
        "id": "bA", "name": "bundle A", "kind": "wire",
        "members": ["w0", "w1"],
        "merge_marker": f"{scene_ops.MARKERS_SCOPE}/bA_merge",
        "split_marker": f"{scene_ops.MARKERS_SCOPE}/bA_split",
    }]

    results, bom = session.route_all_with_bundles(
        stage, "bundle_t", wires, bundles_list)

    by_id = {r["wire_id"]: r for r in results}
    assert by_id["w0"]["status"] == "routed", by_id["w0"].get("reason")
    assert by_id["w1"]["status"] == "routed", by_id["w1"].get("reason")
    # Each member gets branch + trunk + branch stitched into one polyline.
    assert len(by_id["w0"]["polyline"]) >= 3
    assert len(by_id["w1"]["polyline"]) >= 3
    # The shared trunk gets its own BOM row.
    trunk_bom = [b for b in bom if "trunk" in b["wire_id"]]
    assert len(trunk_bom) == 1
    assert trunk_bom[0]["length_m"] > 0.0
    # and its own tube in the stage
    trunk_prim = stage.GetPrimAtPath(
        f"{scene_ops.ROUTES_SCOPE}/bundle_bA_trunk")
    assert trunk_prim and trunk_prim.IsValid()


def test_bundle_trunk_passes_through_its_waypoint(solver_server):
    """A waypoint on a bundle detours the shared trunk, not just the member branches.

    The trunk must bend through the waypoint instead of taking the straight
    merge->split line.
    """
    from pxr import Usd
    from omni.piperouter.router_session import RouterSession

    base, grid_dir = solver_server
    stage = Usd.Stage.CreateInMemory()
    from pxr import UsdGeom as _UG
    _UG.SetStageMetersPerUnit(stage, 1.0)
    _UG.Xform.Define(stage, "/World")
    scene_ops.author_box_mesh(stage, "/World/ground",
                              (2.0, 2.0, -0.05), (6.0, 6.0, 0.05))
    # Merge and split are both at y=1, so an undetoured trunk runs straight along y=1.
    scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/bW_merge",
                           (1.0, 1.0, 0.5), radius=0.05)
    scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/bW_split",
                           (3.0, 1.0, 0.5), radius=0.05)
    # A waypoint pulled well off that line in +y, so the trunk has to bend to reach it.
    wp = (2.0, 2.2, 0.5)
    scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/bW_wp0", wp, radius=0.05)

    session = RouterSession(grid_dir=grid_dir, solver_url=base)
    session.voxelize_scene(stage, "bw", resolution=40, pad_frac=0.5)

    spec = wire_library.as_spec(wire_library.by_id(wire_library.load_wire_library(), "sig_can"))
    wires = [{"name": "w0", "spec": spec, "start": [0.0, 0.5, 0.5],
              "end": [4.0, 0.5, 0.5], "bundle_id": "bW", "weights": {}, "connectivity": 18},
             {"name": "w1", "spec": spec, "start": [0.0, 1.5, 0.5],
              "end": [4.0, 1.5, 0.5], "bundle_id": "bW", "weights": {}, "connectivity": 18}]
    bundles_list = [{
        "id": "bW", "name": "bundle W", "kind": "wire", "members": ["w0", "w1"],
        "merge_marker": f"{scene_ops.MARKERS_SCOPE}/bW_merge",
        "split_marker": f"{scene_ops.MARKERS_SCOPE}/bW_split",
        "waypoints": [f"{scene_ops.MARKERS_SCOPE}/bW_wp0"],
    }]

    results, _bom = session.route_all_with_bundles(stage, "bw", wires, bundles_list)
    trunk = next(r for r in results if r["wire_id"] == "bundle_bW_trunk")
    assert trunk["status"] == "routed"
    poly = np.array(trunk["polyline"])
    # Nearest approach to the waypoint, in metres (mpu=1.0, so metres == stage units).
    dmin = float(np.linalg.norm(poly - np.array(wp), axis=1).min())
    assert dmin < 0.25, f"trunk nearest approach to waypoint was {dmin:.3f} m"
    # and it genuinely left the straight y=1 line
    assert float(poly[:, 1].max()) > 1.8


def test_bundled_member_honours_its_own_waypoint_before_the_trunk(solver_server):
    """A waypoint at slot 0 of a bundled wire pulls its branch on the way into the trunk.

    Slot 0 means "before the wire's bundle", so the waypoint belongs to the member's
    entry branch and must not be dropped when the wire is folded into the trunk.
    """
    from pxr import Usd
    from omni.piperouter.router_session import RouterSession

    base, grid_dir = solver_server
    stage = Usd.Stage.CreateInMemory()
    from pxr import UsdGeom as _UG
    _UG.SetStageMetersPerUnit(stage, 1.0)
    _UG.Xform.Define(stage, "/World")
    scene_ops.author_box_mesh(stage, "/World/ground", (2.0, 2.0, -0.05), (6.0, 6.0, 0.05))
    scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/bM_merge", (2.0, 1.0, 0.5), radius=0.05)
    scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/bM_split", (3.5, 1.0, 0.5), radius=0.05)

    session = RouterSession(grid_dir=grid_dir, solver_url=base)
    session.voxelize_scene(stage, "bm", resolution=40, pad_frac=0.5)
    spec = wire_library.as_spec(wire_library.by_id(wire_library.load_wire_library(), "sig_can"))

    # w0 carries a waypoint at slot 0 (before the bundle), pulled off to +y; w1 has none.
    wp = [0.7, 2.4, 0.5]
    wires = [
        {"name": "w0", "spec": spec, "start": [0.0, 0.5, 0.5], "end": [4.0, 0.5, 0.5],
         "weights": {}, "connectivity": 18, "waypoints": [wp], "waypoint_slots": [0]},
        {"name": "w1", "spec": spec, "start": [0.0, 1.5, 0.5], "end": [4.0, 1.5, 0.5],
         "weights": {}, "connectivity": 18},
    ]
    bundles_list = [{
        "id": "bM", "name": "bundle M", "kind": "wire", "members": ["w0", "w1"],
        "merge_marker": f"{scene_ops.MARKERS_SCOPE}/bM_merge",
        "split_marker": f"{scene_ops.MARKERS_SCOPE}/bM_split",
    }]

    results, _bom = session.route_all_with_bundles(stage, "bm", wires, bundles_list)
    by_id = {r["wire_id"]: r for r in results}
    assert by_id["w0"]["status"] == "routed", by_id["w0"].get("reason")
    poly = np.array(by_id["w0"]["polyline"])
    # w0's full polyline passes near its waypoint; w1 has none and must not detour there.
    assert float(np.linalg.norm(poly - np.array(wp), axis=1).min()) < 0.3
    assert float(np.array(by_id["w1"]["polyline"])[:, 1].max()) < 2.0


def test_wire_in_two_bundles_routes_through_both_trunks(solver_server):
    """A wire in two bundles visits B1_merge->B1_split, then B2_merge->B2_split.

    Neither bundle may be skipped, and the leg into B2 must start from B1's split
    rather than from the wire's original start.
    """
    from pxr import Usd
    from omni.piperouter.router_session import RouterSession

    base, grid_dir = solver_server
    stage = Usd.Stage.CreateInMemory()
    from pxr import UsdGeom as _UG
    _UG.SetStageMetersPerUnit(stage, 1.0)
    _UG.Xform.Define(stage, "/World")
    scene_ops.author_box_mesh(stage, "/World/ground",
                              (3.0, 1.0, -0.05), (8.0, 4.0, 0.05))

    # Two bundles in sequence along X: B1 spans x=1..2, B2 spans x=4..5.
    for name, pos in (("bB1_merge", (1.0, 1.0, 0.5)),
                      ("bB1_split", (2.0, 1.0, 0.5)),
                      ("bB2_merge", (4.0, 1.0, 0.5)),
                      ("bB2_split", (5.0, 1.0, 0.5))):
        scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/{name}",
                               pos, radius=0.04)

    session = RouterSession(grid_dir=grid_dir, solver_url=base)
    session.voxelize_scene(stage, "mb", resolution=32, pad_frac=0.5)

    types = wire_library.load_wire_library()
    spec = wire_library.as_spec(wire_library.by_id(types, "sig_can"))
    wires = [
        {"name": "wa", "spec": spec, "start": [0.0, 0.8, 0.5],
         "end": [6.5, 0.8, 0.5], "bundle_id": "", "weights": {}, "connectivity": 18},
        {"name": "wb", "spec": spec, "start": [0.0, 1.2, 0.5],
         "end": [6.5, 1.2, 0.5], "bundle_id": "", "weights": {}, "connectivity": 18},
    ]
    bundles_list = [
        {"id": "bB1", "name": "B1", "kind": "wire",
         "members": ["wa", "wb"], "weights": {},
         "merge_marker": f"{scene_ops.MARKERS_SCOPE}/bB1_merge",
         "split_marker": f"{scene_ops.MARKERS_SCOPE}/bB1_split"},
        {"id": "bB2", "name": "B2", "kind": "wire",
         "members": ["wa", "wb"], "weights": {},
         "merge_marker": f"{scene_ops.MARKERS_SCOPE}/bB2_merge",
         "split_marker": f"{scene_ops.MARKERS_SCOPE}/bB2_split"},
    ]

    results, bom = session.route_all_with_bundles(stage, "mb", wires, bundles_list)
    by_id = {r["wire_id"]: r for r in results}

    assert by_id["wa"]["status"] == "routed", by_id["wa"].get("reason")
    assert by_id["wb"]["status"] == "routed", by_id["wb"].get("reason")
    # One trunk tube and one trunk BOM row per bundle.
    assert stage.GetPrimAtPath(f"{scene_ops.ROUTES_SCOPE}/bundle_bB1_trunk").IsValid()
    assert stage.GetPrimAtPath(f"{scene_ops.ROUTES_SCOPE}/bundle_bB2_trunk").IsValid()
    trunk_boms = [b for b in bom if "trunk" in b["wire_id"]]
    assert len(trunk_boms) == 2
    # Each wire's polyline spans branch + B1 + link + B2 + branch.
    for wn in ("wa", "wb"):
        assert len(by_id[wn]["polyline"]) >= 4


def test_clearance_tag_builds_class_grid(cube_stage):
    # A per-object clearance tag produces a clearance-class grid: class ids on that
    # prim's voxels, plus the per-class distances in metres the solver must respect.
    prim = cube_stage.GetPrimAtPath("/World/Obstacle")
    scene_ops.write_tags(prim, clearance_m=0.15)
    session = RouterSession()
    _gb, _cell, _res, occ, _sd, _th, _em = session.compute_grids(cube_stage, resolution=24)
    cc = session.last_clearance_classes
    assert cc is not None
    cls_grid, vals = cc
    assert vals == [0.15]
    tagged = cls_grid == 1
    assert tagged.any()                         # the cube's voxels are classed
    assert (tagged & (occ.astype(bool))).sum() == tagged.sum()  # only occupied cells classed


def test_untagged_scene_has_no_class_grid(cube_stage):
    session = RouterSession()
    session.compute_grids(cube_stage, resolution=24)
    assert session.last_clearance_classes is None
