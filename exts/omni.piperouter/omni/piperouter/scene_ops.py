"""USD stage I/O: collidable meshes, bounds, draggable markers, tube authoring, and
thermal/EM tag read/write. pxr-only so it is testable headlessly with usd-core."""
from __future__ import annotations

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

PIPEROUTER_ROOT = "/World/PipeRouter"
ROUTES_SCOPE = PIPEROUTER_ROOT + "/routes"
MARKERS_SCOPE = PIPEROUTER_ROOT + "/markers"
DEBUG_SCOPE = PIPEROUTER_ROOT + "/debug"
TEMP_ATTR = "piperouter:temp_c"
EM_ATTR = "piperouter:em_strength"


def list_collidable_meshes(stage, exclude_prefixes=(PIPEROUTER_ROOT,)):
    out = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        if any(path.startswith(p) for p in exclude_prefixes):
            continue
        out.append(prim)
    return out


def compute_bounds(stage, prims):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                              [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    bmin = np.array([np.inf, np.inf, np.inf])
    bmax = np.array([-np.inf, -np.inf, -np.inf])
    for prim in prims:
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        mn, mx = rng.GetMin(), rng.GetMax()
        bmin = np.minimum(bmin, [mn[0], mn[1], mn[2]])
        bmax = np.maximum(bmax, [mx[0], mx[1], mx[2]])
    if not np.all(np.isfinite(bmin)):
        return None
    return bmin, bmax


def spawn_marker(stage, path, position, color=(0.1, 0.9, 0.1), radius=0.03):
    """A single draggable Sphere prim (no parent/child split) so the prim you move in
    the viewport is exactly the one we read back."""
    sph = UsdGeom.Sphere.Define(stage, path)
    # reuse the existing translate op if the marker already exists (e.g. on rebuild)
    xf = UsdGeom.Xformable(sph)
    ops = [o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    op = ops[0] if ops else xf.AddTranslateOp()
    op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    sph.GetRadiusAttr().Set(float(radius))
    sph.GetDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    return sph


def get_world_pos(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    t = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
    return np.array([t[0], t[1], t[2]])


def marker_positions(stage):
    """World positions of every marker (start/end/waypoint) under MARKERS_SCOPE, so
    the voxel grid can be framed to include them (markers dragged beyond the scene
    geometry must still be inside the grid, or routing to them clamps to the edge)."""
    root = stage.GetPrimAtPath(MARKERS_SCOPE)
    if not root or not root.IsValid():
        return []
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    out = []
    for child in root.GetChildren():
        t = cache.GetLocalToWorldTransform(child).ExtractTranslation()
        out.append(np.array([t[0], t[1], t[2]]))
    return out


def author_tube(stage, path, polyline, diameter_m, color=(0.8, 0.1, 0.1)):
    crv = UsdGeom.BasisCurves.Define(stage, path)
    pts = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in polyline]
    crv.GetPointsAttr().Set(pts)
    crv.GetCurveVertexCountsAttr().Set([len(pts)])
    crv.GetTypeAttr().Set(UsdGeom.Tokens.linear)
    crv.GetWidthsAttr().Set([float(diameter_m)] * len(pts))
    crv.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    crv.GetDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    return crv


def clear_routes(stage):
    prim = stage.GetPrimAtPath(ROUTES_SCOPE)
    if prim and prim.IsValid():
        stage.RemovePrim(prim.GetPath())


def author_box_mesh(stage, path, center, size, color=(0.5, 0.5, 0.5)):
    """Author an axis-aligned box as a real UsdGeom.Mesh (the voxelizer collects
    Meshes, not Cube prims). `size` is full extents."""
    mesh = UsdGeom.Mesh.Define(stage, path)
    cx, cy, cz = (float(c) for c in center)
    hx, hy, hz = (float(s) / 2.0 for s in size)
    pts = [
        (cx - hx, cy - hy, cz - hz), (cx + hx, cy - hy, cz - hz),
        (cx + hx, cy + hy, cz - hz), (cx - hx, cy + hy, cz - hz),
        (cx - hx, cy - hy, cz + hz), (cx + hx, cy - hy, cz + hz),
        (cx + hx, cy + hy, cz + hz), (cx - hx, cy + hy, cz + hz),
    ]
    mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in pts])
    mesh.GetFaceVertexCountsAttr().Set([4, 4, 4, 4, 4, 4])
    mesh.GetFaceVertexIndicesAttr().Set([
        0, 3, 2, 1,  4, 5, 6, 7,  0, 1, 5, 4,
        2, 3, 7, 6,  1, 2, 6, 5,  0, 4, 7, 3,
    ])
    mesh.GetDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    return mesh


def author_points(stage, path, points, size=0.01, color=(0.2, 0.6, 1.0)):
    """A UsdGeom.Points cloud (used for the occupancy debug overlay)."""
    pts_prim = UsdGeom.Points.Define(stage, path)
    pts = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points]
    pts_prim.GetPointsAttr().Set(pts)
    pts_prim.GetWidthsAttr().Set([float(size)] * len(pts))
    pts_prim.GetDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    return pts_prim


def author_colored_points(stage, path, points, colors, size=0.02):
    """Like author_points but with a PER-POINT color (vertex interpolation), used for
    the thermal/EM debug clouds where each cell is tinted by its field value."""
    pts_prim = UsdGeom.Points.Define(stage, path)
    pts_prim.GetPointsAttr().Set([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2]))
                                  for p in points])
    pts_prim.GetWidthsAttr().Set([float(size)] * len(points))
    pts_prim.GetDisplayColorAttr().Set(
        [Gf.Vec3f(float(c[0]), float(c[1]), float(c[2])) for c in colors])
    # vertex interpolation = one color per point (else USD expects a single constant)
    pts_prim.GetDisplayColorPrimvar().SetInterpolation(UsdGeom.Tokens.vertex)
    return pts_prim


def clear_debug(stage):
    prim = stage.GetPrimAtPath(DEBUG_SCOPE)
    if prim and prim.IsValid():
        stage.RemovePrim(prim.GetPath())


def read_thermal_em_tags(stage):
    """Find every prim tagged with a temperature and/or EM strength, and report WHERE
    it is and HOW BIG it is so the field builder can splat heat/EM from the right place.

    Returns: list of (center, temp_c|None, em|None, char_size) where
        center    = WORLD bounding-box CENTRE of the tagged prim (3,) ndarray.
        temp_c    = authored °C value, or None.
        em        = authored EM strength, or None.
        char_size = half the bbox diagonal, i.e. a characteristic radius of the object.

    WHY the bbox centre and not the xform translation:
        author_box_mesh() bakes geometry into world-space points with an IDENTITY
        transform, so the prim's xform translation is (0,0,0). Using it would splat ALL
        heat at the world origin instead of at the object. The world bbox centre is
        correct whether the geometry is baked into points OR positioned by an xform.
    """
    out = []
    # BBoxCache computes the world-space, axis-aligned bounds of a prim's geometry.
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                             [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    xform = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in stage.Traverse():
        t = prim.GetAttribute(TEMP_ATTR)
        e = prim.GetAttribute(EM_ATTR)
        has_t = bool(t) and t.IsValid() and t.HasAuthoredValue()
        has_e = bool(e) and e.IsValid() and e.HasAuthoredValue()
        if not (has_t or has_e):
            continue

        rng = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            # Prim has no geometry of its own (e.g. a bare Xform) — fall back to its
            # world translation, with a zero characteristic size.
            tr = xform.GetLocalToWorldTransform(prim).ExtractTranslation()
            center = np.array([tr[0], tr[1], tr[2]], dtype=float)
            char_size = 0.0
        else:
            mn, mx = rng.GetMin(), rng.GetMax()
            lo = np.array([mn[0], mn[1], mn[2]], dtype=float)
            hi = np.array([mx[0], mx[1], mx[2]], dtype=float)
            center = 0.5 * (lo + hi)
            char_size = 0.5 * float(np.linalg.norm(hi - lo))  # half the bbox diagonal

        out.append((center,
                    float(t.Get()) if has_t else None,
                    float(e.Get()) if has_e else None,
                    char_size))
    return out


def write_tags(prim, temp_c=None, em=None):
    if temp_c is not None:
        prim.CreateAttribute(TEMP_ATTR, Sdf.ValueTypeNames.Float).Set(float(temp_c))
    if em is not None:
        prim.CreateAttribute(EM_ATTR, Sdf.ValueTypeNames.Float).Set(float(em))


def list_tagged_prims(stage):
    """Every prim carrying a thermal and/or EM tag: [{path, temp_c|None, em|None}]."""
    out = []
    for prim in stage.Traverse():
        t = prim.GetAttribute(TEMP_ATTR)
        e = prim.GetAttribute(EM_ATTR)
        has_t = bool(t) and t.IsValid() and t.HasAuthoredValue()
        has_e = bool(e) and e.IsValid() and e.HasAuthoredValue()
        if has_t or has_e:
            out.append({"path": str(prim.GetPath()),
                        "temp_c": float(t.Get()) if has_t else None,
                        "em": float(e.Get()) if has_e else None})
    return out


def clear_tags(prim):
    """Remove the thermal/EM tags from a prim."""
    for attr in (TEMP_ATTR, EM_ATTR):
        a = prim.GetAttribute(attr)
        if a and a.IsValid() and a.HasAuthoredValue():
            prim.RemoveProperty(attr)
