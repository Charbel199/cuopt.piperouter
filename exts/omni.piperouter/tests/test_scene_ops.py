import numpy as np
from pxr import Usd, UsdGeom

from omni.piperouter import scene_ops


def _stage():
    s = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(s, "/World")
    return s


def test_marker_roundtrips_position():
    s = _stage()
    scene_ops.spawn_marker(s, "/World/PipeRouter/markers/w0_start", (1.0, 2.0, 3.0))
    pos = scene_ops.get_world_pos(s, "/World/PipeRouter/markers/w0_start")
    assert np.allclose(pos, [1.0, 2.0, 3.0])


def test_waypoint_marker_is_wireframe_and_roundtrips_position():
    s = _stage()
    path = "/World/PipeRouter/markers/w0_wp0"
    scene_ops.spawn_waypoint_marker(s, path, (1.0, 2.0, 3.0), radius=0.05, segments=24)
    # it's a wireframe gizmo (BasisCurves: 3 closed rings), not a solid sphere
    crv = UsdGeom.BasisCurves(s.GetPrimAtPath(path))
    assert crv
    assert list(crv.GetCurveVertexCountsAttr().Get()) == [24, 24, 24]
    # readable/draggable like a normal marker
    assert np.allclose(scene_ops.get_world_pos(s, path), [1.0, 2.0, 3.0])


def test_bend_heatmap_flags_sharp_corner_on_coarse_path():
    # a 90 deg corner over long (0.5 m) segments, min bend 55 mm. The chord-based radius
    # would read this coarse corner as a gentle arc; resampling to the bend radius must
    # still flag it RED (this was the "no hotspots with smoothing 0" bug).
    s = _stage()
    poly = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.5, 0.5, 0.0)]
    scene_ops.author_bend_heatmap(s, "t", poly, min_bend_radius_mm=55.0)
    crv = UsdGeom.BasisCurves(s.GetPrimAtPath(f"{scene_ops.DEBUG_SCOPE}/bend_t"))
    cols = crv.GetDisplayColorAttr().Get()
    reds = [c for c in cols if c[0] > 0.9 and c[1] < 0.2]
    assert reds, "sharp corner should produce at least one red (below-min-bend) vertex"


def test_hide_and_show_route_toggles_visibility():
    s = _stage()
    scene_ops.author_tube(s, f"{scene_ops.ROUTES_SCOPE}/w0", [(0, 0, 0), (1, 0, 0)], 0.01)
    scene_ops.author_tube(s, f"{scene_ops.ROUTES_SCOPE}/w0_seg0", [(0, 0, 0), (1, 0, 0)], 0.01)
    scene_ops.author_tube(s, f"{scene_ops.ROUTES_SCOPE}/w1", [(0, 0, 0), (1, 0, 0)], 0.01)

    scene_ops.hide_route(s, "w0")
    vis = lambda p: UsdGeom.Imageable(s.GetPrimAtPath(p)).GetVisibilityAttr().Get()
    assert vis(f"{scene_ops.ROUTES_SCOPE}/w0") == UsdGeom.Tokens.invisible
    assert vis(f"{scene_ops.ROUTES_SCOPE}/w0_seg0") == UsdGeom.Tokens.invisible  # branch segs too
    assert vis(f"{scene_ops.ROUTES_SCOPE}/w1") != UsdGeom.Tokens.invisible       # other wires untouched

    scene_ops.set_all_routes_visible(s)
    assert vis(f"{scene_ops.ROUTES_SCOPE}/w0") == UsdGeom.Tokens.inherited


def test_author_tube_creates_curve_with_points():
    s = _stage()
    poly = [(0, 0, 0), (1, 0, 0), (1, 1, 0)]
    scene_ops.author_tube(s, "/World/PipeRouter/routes/w0", poly, 0.01, (0.8, 0.1, 0.1))
    crv = UsdGeom.BasisCurves(s.GetPrimAtPath("/World/PipeRouter/routes/w0"))
    assert crv
    assert len(crv.GetPointsAttr().Get()) == 3
    assert list(crv.GetCurveVertexCountsAttr().Get()) == [3]


def test_author_colored_points_per_point_color():
    s = _stage()
    # rendered as BasisCurves stubs (2 verts/point) so the RTX viewport actually draws them
    p = scene_ops.author_colored_points(
        s, scene_ops.DEBUG_SCOPE + "/thermal",
        [(0, 0, 0), (1, 0, 0)], [(1, 0, 0), (0, 0, 1)], size=0.05)
    assert p.GetPrim().IsA(UsdGeom.BasisCurves)
    assert list(p.GetCurveVertexCountsAttr().Get()) == [2, 2]   # one stub per point
    assert len(p.GetPointsAttr().Get()) == 4                    # 2 verts per point
    assert len(p.GetDisplayColorAttr().Get()) == 4              # one color per vertex
    assert str(p.GetDisplayColorPrimvar().GetInterpolation()) == "vertex"


def test_clear_routes_removes_scope():
    s = _stage()
    scene_ops.author_tube(s, "/World/PipeRouter/routes/w0", [(0, 0, 0), (1, 0, 0)], 0.01)
    assert s.GetPrimAtPath("/World/PipeRouter/routes/w0").IsValid()
    scene_ops.clear_routes(s)
    assert not s.GetPrimAtPath("/World/PipeRouter/routes/w0").IsValid()


def test_author_box_mesh_is_a_mesh_with_8_points():
    s = _stage()
    m = scene_ops.author_box_mesh(s, "/World/Box", (1, 1, 1), (0.4, 0.4, 0.4))
    assert m.GetPrim().IsA(UsdGeom.Mesh)
    assert len(m.GetPointsAttr().Get()) == 8
    assert list(m.GetFaceVertexCountsAttr().Get()) == [4, 4, 4, 4, 4, 4]


def test_points_overlay_and_clear():
    s = _stage()
    scene_ops.author_points(s, scene_ops.DEBUG_SCOPE + "/occ",
                            [(0, 0, 0), (1, 1, 1)], size=0.02)
    crv = UsdGeom.BasisCurves(s.GetPrimAtPath(scene_ops.DEBUG_SCOPE + "/occ"))
    assert list(crv.GetCurveVertexCountsAttr().Get()) == [2, 2]   # one stub per point
    assert len(crv.GetPointsAttr().Get()) == 4
    scene_ops.clear_debug(s)
    assert not s.GetPrimAtPath(scene_ops.DEBUG_SCOPE).IsValid()


def test_tag_write_then_read():
    s = _stage()
    prim = UsdGeom.Xform.Define(s, "/World/Hot").GetPrim()
    UsdGeom.XformCommonAPI(prim).SetTranslate((0.5, 0.0, 0.0))
    scene_ops.write_tags(prim, temp_c=130.0, em=0.4)
    tags = scene_ops.read_thermal_em_tags(s)
    assert len(tags) == 1
    pos, temp, em, _char = tags[0]
    assert abs(temp - 130.0) < 1e-6
    assert abs(em - 0.4) < 1e-6
    assert np.allclose(pos, [0.5, 0.0, 0.0])


def test_list_and_clear_tags():
    s = _stage()
    p1 = UsdGeom.Xform.Define(s, "/World/Hot").GetPrim()
    p2 = UsdGeom.Xform.Define(s, "/World/Emitter").GetPrim()
    scene_ops.write_tags(p1, temp_c=120.0)
    scene_ops.write_tags(p2, em=0.7)
    tags = {t["path"]: t for t in scene_ops.list_tagged_prims(s)}
    assert abs(tags["/World/Hot"]["temp_c"] - 120.0) < 1e-4
    assert abs(tags["/World/Emitter"]["em"] - 0.7) < 1e-4
    scene_ops.clear_tags(p1)
    paths = {t["path"] for t in scene_ops.list_tagged_prims(s)}
    assert "/World/Hot" not in paths and "/World/Emitter" in paths


def test_tag_source_uses_box_center_not_origin():
    # author_box_mesh bakes geometry into world points with an identity xform, so the
    # heat source must come from the bbox CENTRE, not the (0,0,0) xform translation.
    s = _stage()
    box = scene_ops.author_box_mesh(s, "/World/Hot", center=(5.0, 5.0, 1.0),
                                    size=(2.0, 2.0, 2.0))
    scene_ops.write_tags(box.GetPrim(), temp_c=120.0)
    (center, temp, _em, char) = scene_ops.read_thermal_em_tags(s)[0]
    assert temp == 120.0
    assert np.allclose(center, [5.0, 5.0, 1.0], atol=1e-3)   # NOT the origin
    assert char > 1.0                                        # ~half the bbox diagonal


def test_tagging_an_instance_proxy_tags_exactly_that_prim():
    # Greyed-out prims in instanced CAD are INSTANCE PROXIES - USD forbids authoring
    # attributes on them, so proxy tags go into a path-keyed registry (customData on the
    # PipeRouter root). The tag applies to EXACTLY the selected prim: the same part in a
    # SIBLING instance stays untagged, and nothing is written on the parent/instance.
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/Protos/part")
    UsdGeom.Mesh.Define(stage, "/Protos/part/Mesh")
    for name in ("a", "b"):
        inst = UsdGeom.Xform.Define(stage, f"/World/{name}").GetPrim()
        inst.GetReferences().AddInternalReference("/Protos/part")
        inst.SetInstanceable(True)

    proxies = [p for p in stage.Traverse(Usd.TraverseInstanceProxies())
               if p.IsA(UsdGeom.Mesh) and p.IsInstanceProxy()]
    proxy_a = next(p for p in proxies if str(p.GetPath()).startswith("/World/a"))
    proxy_b = next(p for p in proxies if str(p.GetPath()).startswith("/World/b"))

    # tagging the PROXY must not raise, and must not author on the instance root
    scene_ops.write_tags(proxy_a, clearance_m=0.05, temp_c=90.0)
    root_a = stage.GetPrimAtPath("/World/a")
    assert not root_a.GetAttribute(scene_ops.CLEARANCE_ATTR).HasAuthoredValue()

    # the reader resolves the proxy's own path from the registry
    assert abs(scene_ops.clearance_for_prim(proxy_a) - 0.05) < 1e-9
    # the SIBLING instance's identical part is NOT tagged
    assert scene_ops.clearance_for_prim(proxy_b) is None

    # it shows in the tag list, feeds the thermal reader, and clears cleanly
    listed = {t["path"]: t for t in scene_ops.list_tagged_prims(stage)}
    assert str(proxy_a.GetPath()) in listed
    assert listed[str(proxy_a.GetPath())]["temp_c"] == 90.0
    assert any(t == 90.0 for (_c, t, _e, _s) in scene_ops.read_thermal_em_tags(stage))
    scene_ops.clear_tags(proxy_a)
    assert scene_ops.clearance_for_prim(proxy_a) is None
    assert str(proxy_a.GetPath()) not in {t["path"] for t in scene_ops.list_tagged_prims(stage)}
