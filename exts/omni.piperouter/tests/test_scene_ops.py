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
    p = scene_ops.author_colored_points(
        s, scene_ops.DEBUG_SCOPE + "/thermal",
        [(0, 0, 0), (1, 0, 0)], [(1, 0, 0), (0, 0, 1)], size=0.05)
    assert len(p.GetPointsAttr().Get()) == 2
    assert len(p.GetDisplayColorAttr().Get()) == 2     # one color per point
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
    pts = UsdGeom.Points(s.GetPrimAtPath(scene_ops.DEBUG_SCOPE + "/occ"))
    assert len(pts.GetPointsAttr().Get()) == 2
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
