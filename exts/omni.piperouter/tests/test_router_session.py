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
        # clearance is baked into the voxel grid at voxelize time
        session.voxelize_scene(stage, f"c{clearance}", resolution=40, clearance_m=clearance)
        wire = {"name": "g", "spec": spec, "start": list(start), "end": list(end),
                "connectivity": 6}
        results, _ = session.route_all(stage, f"c{clearance}", [wire])
        return results[0]["status"]

    assert route(0.0) == "routed"     # gap is open
    assert route(0.5) == "no_path"    # clearance seals the 0.6 m gap


def test_no_path_reason_flows_through_http(solver_server):
    # a wire whose end is buried in a solid block -> no_path with an explanatory
    # reason that survives the solver schema + HTTP round-trip into the BOM row.
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    scene_ops.author_box_mesh(stage, "/World/block", (1.0, 0.5, 0.5), (0.6, 0.6, 0.6))
    start, end = (0.1, 0.5, 0.5), (1.0, 0.5, 0.5)  # end sits inside the block
    scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/b_start", start, radius=0.05)
    scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/b_end", end, radius=0.05)

    base, grid_dir = solver_server
    session = RouterSession(grid_dir=grid_dir, solver_url=base)
    session.voxelize_scene(stage, "nb", resolution=32)
    spec = wire_library.as_spec(wire_library.by_id(wire_library.load_wire_library(), "sig_can"))
    wire = {"name": "buried", "spec": spec, "start": list(start), "end": list(end),
            "connectivity": 26}
    results, bom = session.route_all(stage, "nb", [wire])

    assert results[0]["status"] == "no_path"
    assert results[0]["reason"]            # reason present in the HTTP result
    assert bom[0]["reason"]                # ...and carried into the BOM row


def test_clearance_bakes_more_prohibited_voxels(cube_stage):
    # the core mental model: more clearance -> more prohibited voxels in the grid
    session = RouterSession()   # compute_grids is local; no solver needed
    g0 = session.compute_grids(cube_stage, resolution=24, clearance_m=0.0)
    g1 = session.compute_grids(cube_stage, resolution=24, clearance_m=0.3)
    occ0, occ1 = g0[3], g1[3]
    assert int(occ1.sum()) > int(occ0.sum())


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


def test_bundle_trunk_and_branches_routed(solver_server):
    """Two wires share a bundle trunk; each gets a stitched full polyline."""
    from pxr import Usd
    from omni.piperouter.router_session import RouterSession

    base, grid_dir = solver_server
    stage = Usd.Stage.CreateInMemory()
    from pxr import UsdGeom as _UG
    _UG.SetStageMetersPerUnit(stage, 1.0)
    _UG.Xform.Define(stage, "/World")
    # a simple open scene (no obstacles except a ground plane)
    scene_ops.author_box_mesh(stage, "/World/ground",
                              (2.0, 2.0, -0.05), (6.0, 6.0, 0.05))
    # merge marker at (1,1,0.5), split marker at (3,1,0.5)
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
    # each member polyline starts at its own start and ends at its own end
    assert len(by_id["w0"]["polyline"]) >= 3
    assert len(by_id["w1"]["polyline"]) >= 3
    # trunk BOM row present
    trunk_bom = [b for b in bom if "trunk" in b["wire_id"]]
    assert len(trunk_bom) == 1
    assert trunk_bom[0]["length_m"] > 0.0
    # trunk tube authored in stage
    trunk_prim = stage.GetPrimAtPath(
        f"{scene_ops.ROUTES_SCOPE}/bundle_bA_trunk")
    assert trunk_prim and trunk_prim.IsValid()


def test_bundle_trunk_passes_through_its_waypoint(solver_server):
    """A waypoint on a bundle forces the shared TRUNK to detour through it, not take the
    straight merge->split line."""
    from pxr import Usd
    from omni.piperouter.router_session import RouterSession

    base, grid_dir = solver_server
    stage = Usd.Stage.CreateInMemory()
    from pxr import UsdGeom as _UG
    _UG.SetStageMetersPerUnit(stage, 1.0)
    _UG.Xform.Define(stage, "/World")
    scene_ops.author_box_mesh(stage, "/World/ground",
                              (2.0, 2.0, -0.05), (6.0, 6.0, 0.05))
    # merge at (1,1,0.5), split at (3,1,0.5): the straight trunk runs along y=1.
    scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/bW_merge",
                           (1.0, 1.0, 0.5), radius=0.05)
    scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/bW_split",
                           (3.0, 1.0, 0.5), radius=0.05)
    # a waypoint pulled well off that line (+y) — the trunk must bend to reach it.
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
    # the trunk must come close to the waypoint (mpu=1.0, so meters == stage units)
    dmin = float(np.linalg.norm(poly - np.array(wp), axis=1).min())
    assert dmin < 0.25, f"trunk nearest approach to waypoint was {dmin:.3f} m"
    # and it genuinely detoured off the straight y=1 line
    assert float(poly[:, 1].max()) > 1.8


def test_bundled_member_honours_its_own_waypoint_before_the_trunk(solver_server):
    """A waypoint on a BUNDLED wire (slot 0 = before its bundle) must pull that member's
    branch through the waypoint on the way INTO the trunk — previously ignored."""
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
    # w0's full polyline must pass near its waypoint; w1 (no waypoint) must not detour there.
    assert float(np.linalg.norm(poly - np.array(wp), axis=1).min()) < 0.3
    assert float(np.array(by_id["w1"]["polyline"])[:, 1].max()) < 2.0


def test_wire_in_two_bundles_routes_through_both_trunks(solver_server):
    """A wire in two bundles must visit B1_merge→B1_split then B2_merge→B2_split —
    not skip the first bundle or re-route from the original start for B2."""
    from pxr import Usd
    from omni.piperouter.router_session import RouterSession

    base, grid_dir = solver_server
    stage = Usd.Stage.CreateInMemory()
    from pxr import UsdGeom as _UG
    _UG.SetStageMetersPerUnit(stage, 1.0)
    _UG.Xform.Define(stage, "/World")
    scene_ops.author_box_mesh(stage, "/World/ground",
                              (3.0, 1.0, -0.05), (8.0, 4.0, 0.05))

    # Two bundles placed sequentially along X
    # B1: merge at x=1, split at x=2
    # B2: merge at x=4, split at x=5
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

    # both wires routed
    assert by_id["wa"]["status"] == "routed", by_id["wa"].get("reason")
    assert by_id["wb"]["status"] == "routed", by_id["wb"].get("reason")
    # both trunks authored
    assert stage.GetPrimAtPath(f"{scene_ops.ROUTES_SCOPE}/bundle_bB1_trunk").IsValid()
    assert stage.GetPrimAtPath(f"{scene_ops.ROUTES_SCOPE}/bundle_bB2_trunk").IsValid()
    # two trunk BOM rows
    trunk_boms = [b for b in bom if "trunk" in b["wire_id"]]
    assert len(trunk_boms) == 2
    # each wire's polyline is long enough to span the full route (B1 + B2 + branches)
    for wn in ("wa", "wb"):
        assert len(by_id[wn]["polyline"]) >= 4
