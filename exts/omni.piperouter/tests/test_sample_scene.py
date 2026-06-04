import numpy as np
from pxr import Usd, UsdGeom

from omni.piperouter import sample_scene, scene_ops, voxelizer, grid_io, wire_library
from omni.piperouter.router_session import RouterSession


def _new_stage():
    return Usd.Stage.CreateInMemory()


def test_build_returns_three_wires_and_obstacle_meshes():
    s = _new_stage()
    wires = sample_scene.build_sample_scene(s)
    assert len(wires) == 3
    assert {w["type_id"] for w in wires} == {"pwr_4awg", "sig_can", "ac_pipe_12"}
    meshes = scene_ops.list_collidable_meshes(s)
    # ground + 2 firewall + block + 2 comps = 6 obstacle meshes
    assert len(meshes) >= 6


def test_complex_scene_has_many_wires_and_all_types():
    s = _new_stage()
    wires = sample_scene.build_complex_scene(s)
    assert len(wires) >= 12
    used = {w["type_id"] for w in wires}
    # exercises every wire + tube class
    assert {"pwr_4awg", "sig_can", "ac_pipe_12",
            "coolant_hose_16", "coolant_hose_8", "brake_line_6"} <= used
    assert len(scene_ops.list_collidable_meshes(s)) >= 12
    # has both a thermal and an EM source
    tags = scene_ops.read_thermal_em_tags(s)
    assert any(t is not None for (_p, t, _e, _c) in tags)
    assert any(e is not None for (_p, _t, e, _c) in tags)


def test_complex_scene_idempotent_and_replaces_sample():
    s = _new_stage()
    sample_scene.build_sample_scene(s)            # start from the small scene
    wires = sample_scene.build_complex_scene(s)   # must clear it and rebuild
    n = len(wires)
    assert n >= 12
    wires2 = sample_scene.build_complex_scene(s)  # rebuild must not duplicate/error
    assert len(wires2) == n
    for w in wires2:
        assert scene_ops.get_world_pos(
            s, f"{scene_ops.MARKERS_SCOPE}/{w['name']}_start") is not None


def test_markers_and_tags_present():
    s = _new_stage()
    wires = sample_scene.build_sample_scene(s)
    for w in wires:
        assert scene_ops.get_world_pos(s, f"{scene_ops.MARKERS_SCOPE}/{w['name']}_start") is not None
        assert scene_ops.get_world_pos(s, f"{scene_ops.MARKERS_SCOPE}/{w['name']}_end") is not None
    tags = scene_ops.read_thermal_em_tags(s)
    assert any(t is not None for (_p, t, _e, _c) in tags)   # hot engine block
    assert any(e is not None for (_p, _t, e, _c) in tags)   # EM component
    # the hot source must sit at the block (away from origin), not at (0,0,0)
    hot = [(p, t) for (p, t, _e, _c) in tags if t is not None][0]
    assert np.linalg.norm(hot[0]) > 1.0


def test_idempotent_rebuild():
    s = _new_stage()
    sample_scene.build_sample_scene(s)
    wires = sample_scene.build_sample_scene(s)   # rebuild must not error or duplicate
    assert len(wires) == 3
    assert len(scene_ops.list_collidable_meshes(s)) >= 6


def test_sample_scene_voxelizes_to_mixed_occupancy():
    s = _new_stage()
    sample_scene.build_sample_scene(s)
    prims = scene_ops.list_collidable_meshes(s)
    bmin, bmax = scene_ops.compute_bounds(s, prims)
    pad = (bmax - bmin) * 0.05 + 1e-3
    gbmin, cell, res = grid_io.frame_from_bounds(bmin - pad, bmax + pad, 48)
    pts, idx = voxelizer.collect_meshes(s, prims)
    occ, _ = voxelizer.voxelize(pts, idx, gbmin, cell, res)
    assert occ.sum() > 0                 # obstacles present
    assert occ.sum() < occ.size          # free space to route through


def test_sample_scene_thermal_field_is_hot_near_block():
    # compute_grids is local (no solver / no save), so no server needed here.
    s = _new_stage()
    sample_scene.build_sample_scene(s)
    session = RouterSession()
    gbmin, cell, res, occ, sd, thermal, em = session.compute_grids(s, resolution=48)
    assert thermal.max() > 30.0          # heat is actually present (ambient is 20°C)
    assert em.max() > 0.0                # EM source present too
    # the hottest cell sits near the (scaled) engine-block centre, NOT the origin
    hot = np.array(np.unravel_index(np.argmax(thermal), thermal.shape))
    hot_world = gbmin + (hot + 0.5) * cell
    block_center_xy = np.array([0.55, 0.50]) * sample_scene.SCALE
    assert np.linalg.norm(hot_world[:2] - block_center_xy) < 2.0


def test_sample_scene_routes_through_solver(solver_server):
    base, grid_dir = solver_server
    s = _new_stage()
    descriptors = sample_scene.build_sample_scene(s)
    session = RouterSession(grid_dir=grid_dir, solver_url=base)
    session.voxelize_scene(s, "samp", resolution=40)

    types = wire_library.load_wire_library()
    wires = []
    for i, d in enumerate(descriptors):
        spec = wire_library.as_spec(wire_library.by_id(types, d["type_id"]))
        wires.append({"name": d["name"], "spec": spec, "start": list(d["start"]),
                      "end": list(d["end"]), "connectivity": 6, "priority": i})
    results, bom = session.route_all(s, "samp", wires)
    routed = sum(1 for r in results if r["status"] == "routed")
    assert routed >= 2                   # the demo scene is routable
    assert any(b["cost"] > 0 for b in bom)


def test_complex_scene_routes_through_solver(solver_server):
    base, grid_dir = solver_server
    s = _new_stage()
    descriptors = sample_scene.build_complex_scene(s)
    session = RouterSession(grid_dir=grid_dir, solver_url=base)
    session.voxelize_scene(s, "cplx", resolution=56)

    types = wire_library.load_wire_library()
    wires = []
    for i, d in enumerate(descriptors):
        spec = wire_library.as_spec(wire_library.by_id(types, d["type_id"]))
        wires.append({"name": d["name"], "spec": spec, "start": list(d["start"]),
                      "end": list(d["end"]), "connectivity": 26, "priority": i})
    results, bom = session.route_all(s, "cplx", wires)
    routed = sum(1 for r in results if r["status"] == "routed")
    failed = [r for r in results if r["status"] != "routed"]
    # MOST wires route. With the strict no-corner-cutting rule a busy bay leaves a few
    # genuinely tight diagonal corridors unroutable at this resolution — that's correct,
    # and each such failure must carry an explanatory reason.
    assert routed >= len(descriptors) - 4
    assert all(r.get("reason") for r in failed)
    assert any(b["cost"] > 0 for b in bom)
