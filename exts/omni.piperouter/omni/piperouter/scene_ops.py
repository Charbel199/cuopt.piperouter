"""USD stage I/O: collidable meshes, bounds, draggable markers, tube authoring, and
thermal/EM tag read/write. pxr-only so it is testable headlessly with usd-core."""
from __future__ import annotations

import json

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

PIPEROUTER_ROOT = "/World/PipeRouter"
ROUTES_SCOPE = PIPEROUTER_ROOT + "/routes"
MARKERS_SCOPE = PIPEROUTER_ROOT + "/markers"
DEBUG_SCOPE = PIPEROUTER_ROOT + "/debug"
TEMP_ATTR = "piperouter:temp_c"
EM_ATTR = "piperouter:em_strength"
SESSION_KEY = "piperouterSession"   # customData key holding the embedded panel session


def write_session(stage, data: dict):
    """Embed the panel session dict (as JSON) in customData on the PipeRouter root prim,
    so it travels with the stage on Save / usdz export."""
    root = UsdGeom.Scope.Define(stage, PIPEROUTER_ROOT)
    root.GetPrim().SetCustomDataByKey(SESSION_KEY, json.dumps(data))


def read_session(stage):
    """Return the embedded session dict, or None if the stage has no PipeRouter session."""
    prim = stage.GetPrimAtPath(PIPEROUTER_ROOT)
    if not prim or not prim.IsValid():
        return None
    raw = prim.GetCustomDataByKey(SESSION_KEY)
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def list_collidable_meshes(stage, exclude_prefixes=(PIPEROUTER_ROOT,)):
    """Return all UsdGeom.Mesh prims outside the PipeRouter scope.

    Handles drag-dropped USD assets (Xform payloads) which arrive as unloaded
    payloads — grayed-out in the Stage panel. We force-load every unloaded prim
    before traversing so their mesh contents become visible.

    Also handles INSTANCED assets (CAD imports usually instance every repeated part:
    the geometry lives once under /Prototypes and the scene holds instanceable
    references to it — Omniverse shows those meshes greyed out as read-only instance
    proxies). A plain stage.Traverse() skips inside instances entirely, so we traverse
    with Usd.TraverseInstanceProxies() to see the proxy meshes. Their points (prototype
    geometry) and world transforms (per-instance placement) read normally.
    """
    # Walk every prim (including unloaded payload roots) and explicitly load any
    # that haven't been loaded yet. TraverseAll() visits inactive/unloaded roots
    # that stage.Traverse() would skip.
    try:
        from pxr import Usd as _Usd
        for prim in stage.TraverseAll():
            if stage.GetLoadRules().GetEffectiveRuleForPath(
                    prim.GetPath()) == _Usd.StageLoadRules.NoneRule:
                try:
                    stage.Load(prim.GetPath())
                except Exception:
                    pass
    except Exception:
        # Fallback: blanket load everything under root
        try:
            stage.Load("/")
        except Exception:
            pass

    out = []
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        if any(path.startswith(p) for p in exclude_prefixes):
            continue
        out.append(prim)

    try:
        import logging as _log
        _log.getLogger("piperouter").debug(
            "[piperouter] list_collidable_meshes: found %d mesh(es)", len(out))
    except Exception:
        pass
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


def spawn_marker(stage, path, position, color=(0.1, 0.9, 0.1), radius=0.03, opacity=1.0):
    """A single draggable Sphere prim (no parent/child split) so the prim you move in
    the viewport is exactly the one we read back. opacity < 1.0 makes it see-through
    (e.g. waypoints, so they don't hide the geometry they sit on)."""
    sph = UsdGeom.Sphere.Define(stage, path)
    # reuse the existing translate op if the marker already exists (e.g. on rebuild)
    xf = UsdGeom.Xformable(sph)
    ops = [o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    op = ops[0] if ops else xf.AddTranslateOp()
    op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    sph.GetRadiusAttr().Set(float(radius))
    sph.GetDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    sph.GetDisplayOpacityAttr().Set([float(opacity)])
    return sph


def spawn_waypoint_marker(stage, path, position, color=(0.1, 0.5, 0.9), radius=0.05,
                          segments=24):
    """A see-through wireframe gizmo (three orthogonal rings) used for waypoints, so the
    routed wire and the geometry behind it stay visible — unlike a solid sphere, and
    unlike displayOpacity which the RTX viewport ignores without a translucent material.

    Draggable and readable exactly like spawn_marker: the translate op lives on the prim
    itself, so marker_positions()/get_world_pos() pick it up unchanged."""
    crv = UsdGeom.BasisCurves.Define(stage, path)
    xf = UsdGeom.Xformable(crv)
    ops = [o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    op = ops[0] if ops else xf.AddTranslateOp()
    op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    r = float(radius)
    th = np.linspace(0.0, 2.0 * np.pi, int(segments), endpoint=False)
    pts = []
    for ax in range(3):  # rings in the XY, XZ and YZ planes
        for t in th:
            c, s = r * float(np.cos(t)), r * float(np.sin(t))
            if ax == 0:
                pts.append(Gf.Vec3f(c, s, 0.0))
            elif ax == 1:
                pts.append(Gf.Vec3f(c, 0.0, s))
            else:
                pts.append(Gf.Vec3f(0.0, c, s))
    n = int(segments)
    crv.GetPointsAttr().Set(pts)
    crv.GetCurveVertexCountsAttr().Set([n, n, n])
    crv.GetTypeAttr().Set(UsdGeom.Tokens.linear)
    crv.GetWrapAttr().Set(UsdGeom.Tokens.periodic)   # closes each ring into a loop
    crv.GetWidthsAttr().Set([r * 0.1] * (n * 3))
    crv.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    crv.GetDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    return crv


def get_world_pos(stage, path):
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    t = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
    return np.array([t[0], t[1], t[2]])


def get_world_axis(stage, path, local_axis=(1.0, 0.0, 0.0)):
    """World-space unit vector of a prim's `local_axis` (rotation only, scale stripped).
    Reads a marker's heading from how the user rotated it. None if the prim is missing
    or the axis degenerates."""
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return None
    try:
        cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        v = cache.GetLocalToWorldTransform(prim).TransformDir(
            Gf.Vec3d(*[float(x) for x in local_axis]))
        n = float(v.GetLength())
        if n < 1e-9:
            return None
        return np.array([v[0] / n, v[1] / n, v[2] / n])
    except Exception:
        return None


def aim_quat(direction, local_axis=(1.0, 0.0, 0.0)):
    """Quaternion rotating `local_axis` onto world `direction` (shortest arc), for
    pointing a marker's heading arrow along a chosen vector."""
    d = Gf.Vec3d(*[float(x) for x in direction])
    if d.GetLength() < 1e-9:
        return Gf.Quatf(1.0)
    a = Gf.Vec3d(*[float(x) for x in local_axis]).GetNormalized()
    return Gf.Quatf(Gf.Rotation(a, d.GetNormalized()).GetQuat())


def _author_heading_arrow(stage, path, length, color, incoming=False):
    """An arrow along the parent's LOCAL +X (shaft + two head strokes) as a child prim,
    so it inherits the marker's translate+orient - rotating the marker aims the arrow.

    incoming=False (START): drawn FROM the marker outward - "the cable leaves this way".
    incoming=True  (END):   drawn approaching the marker with the tip AT it - "the cable
    arrives along this arrow". Both point in local +X (the travel direction the solver
    reads); only where the strokes sit relative to the marker differs."""
    crv = UsdGeom.BasisCurves.Define(stage, path)
    L = float(length)
    h = L * 0.25
    if incoming:
        pts = [Gf.Vec3f(-L, 0.0, 0.0), Gf.Vec3f(0.0, 0.0, 0.0),
               Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(-h, h * 0.6, 0.0),
               Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(-h, -h * 0.6, 0.0)]
    else:
        pts = [Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(L, 0.0, 0.0),
               Gf.Vec3f(L, 0.0, 0.0), Gf.Vec3f(L - h, h * 0.6, 0.0),
               Gf.Vec3f(L, 0.0, 0.0), Gf.Vec3f(L - h, -h * 0.6, 0.0)]
    crv.GetPointsAttr().Set(pts)
    crv.GetCurveVertexCountsAttr().Set([2, 2, 2])
    crv.GetTypeAttr().Set(UsdGeom.Tokens.linear)
    crv.GetWrapAttr().Set(UsdGeom.Tokens.nonperiodic)
    crv.GetWidthsAttr().Set([L * 0.06] * 6)
    crv.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    crv.GetDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    return crv


def set_marker_direction(stage, path, direction=None, show=True,
                         color=(0.95, 0.8, 0.15), incoming=False):
    """Show/aim a heading arrow on a start/end marker.

    Ensures the marker carries an xformOp:orient (so Kit's rotate manipulator can spin
    it) and an arrow child at `{path}/dir` along local +X. `direction` (world vector)
    re-aims the orient; None keeps the current rotation. show=False removes the arrow
    (the orient op stays, harmless). Returns True if the marker exists."""
    prim = stage.GetPrimAtPath(path)
    if not prim or not prim.IsValid():
        return False
    xf = UsdGeom.Xformable(prim)
    ops = [o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeOrient]
    op = ops[0] if ops else xf.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
    if direction is not None:
        op.Set(aim_quat(direction))
    elif not ops:
        op.Set(Gf.Quatf(1.0))
    arrow_path = f"{path}/dir"
    if show:
        # arrow sized from the marker sphere so it reads at scene scale
        r = 0.03
        try:
            attr = UsdGeom.Sphere(prim).GetRadiusAttr()
            if attr and attr.HasValue():
                r = float(attr.Get())
        except Exception:
            pass
        _author_heading_arrow(stage, arrow_path, r * 3.5, color, incoming=incoming)
    else:
        ap = stage.GetPrimAtPath(arrow_path)
        if ap and ap.IsValid():
            stage.RemovePrim(ap.GetPath())
    return True


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


def set_all_routes_visible(stage):
    """Make every authored route tube visible again (undo any debug-view hide)."""
    routes = stage.GetPrimAtPath(ROUTES_SCOPE)
    if not routes or not routes.IsValid():
        return
    for prim in routes.GetChildren():
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)


def hide_route(stage, wire_name):
    """Hide a wire's final tube(s) so a per-wire debug view (cells, grid-vs-smooth,
    cost terrain, bend heatmap) isn't occluded by the cable. Matches the wire's own
    tube and any bundle branch segments (<name>_seg<i>)."""
    routes = stage.GetPrimAtPath(ROUTES_SCOPE)
    if not routes or not routes.IsValid():
        return
    for prim in routes.GetChildren():
        nm = prim.GetName()
        if nm == wire_name or nm.startswith(wire_name + "_seg"):
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)


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


_BOX_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
              (0, 4), (1, 5), (2, 6), (3, 7))


def author_box_wireframe(stage, path, boxes, colors=None, color=(0.3, 0.7, 1.0),
                         width=0.01):
    """See-through wireframe boxes (12 edges each) as ONE BasisCurves — used to show the
    octree leaves without occluding the scene. `boxes` = list of (min_xyz, max_xyz) in
    STAGE units; `colors` (optional) = per-box RGB."""
    crv = UsdGeom.BasisCurves.Define(stage, path)
    pts, counts, disp = [], [], []
    for bi, (mn, mx) in enumerate(boxes):
        x0, y0, z0 = (float(v) for v in mn)
        x1, y1, z1 = (float(v) for v in mx)
        corners = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
                   (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        col = colors[bi] if colors is not None else color
        cv = Gf.Vec3f(float(col[0]), float(col[1]), float(col[2]))
        for a, b in _BOX_EDGES:
            pts.append(Gf.Vec3f(*corners[a]))
            pts.append(Gf.Vec3f(*corners[b]))
            counts.append(2)
            disp.append(cv)               # one colour per edge (uniform interpolation)
    crv.GetPointsAttr().Set(pts)
    crv.GetCurveVertexCountsAttr().Set(counts)
    crv.GetTypeAttr().Set(UsdGeom.Tokens.linear)
    crv.GetWrapAttr().Set(UsdGeom.Tokens.nonperiodic)
    crv.GetWidthsAttr().Set([float(width)] * len(pts))
    crv.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    pv = crv.GetDisplayColorPrimvar()
    pv.Set(disp)
    pv.SetInterpolation(UsdGeom.Tokens.uniform)
    return crv


def _author_blob_cloud(stage, path, points, size, colors=None, color=(0.2, 0.6, 1.0)):
    """Render a point cloud as short fat BasisCurves stubs — one tiny 2-vertex linear
    curve (width = size) per point. We use curves, NOT UsdGeom.Points, because the RTX
    viewport renders curves reliably while it routinely fails to draw Points at all
    (that's why the debug dot-clouds were invisible). Per-point color is supported via
    vertex-interpolated displayColor (2 verts per point)."""
    pts_in = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
    n = len(pts_in)
    crv = UsdGeom.BasisCurves.Define(stage, path)
    # Very short stub (length << width) so the round end-caps dominate and each point
    # reads as a CIRCLE/sphere, not an elongated pill.
    off = float(size) * 0.08
    verts = []
    for x, y, z in pts_in:
        verts.append(Gf.Vec3f(x - off, y, z))
        verts.append(Gf.Vec3f(x + off, y, z))
    crv.GetPointsAttr().Set(verts)
    crv.GetCurveVertexCountsAttr().Set([2] * n)
    crv.GetTypeAttr().Set(UsdGeom.Tokens.linear)
    crv.GetWidthsAttr().Set([float(size)] * (2 * n))
    crv.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    if colors is not None:
        col = []
        for c in colors:
            v = Gf.Vec3f(float(c[0]), float(c[1]), float(c[2]))
            col.append(v)
            col.append(v)   # one color per vertex, 2 verts per point
        crv.GetDisplayColorAttr().Set(col)
        crv.GetDisplayColorPrimvar().SetInterpolation(UsdGeom.Tokens.vertex)
    else:
        crv.GetDisplayColorAttr().Set(
            [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    return crv


def author_points(stage, path, points, size=0.01, color=(0.2, 0.6, 1.0)):
    """A debug dot cloud (used for the occupancy overlay + per-wire cells)."""
    return _author_blob_cloud(stage, path, points, size, colors=None, color=color)


def author_colored_points(stage, path, points, colors, size=0.02):
    """Like author_points but with a PER-POINT color, used for the thermal/EM/cost
    debug clouds where each cell is tinted by its field value."""
    return _author_blob_cloud(stage, path, points, size, colors=colors)


def author_wire_cells(stage, wire_name, cells, gbmin, cell_size, color=(0.8, 0.1, 0.1),
                      cap=100_000):
    """Point cloud of the voxel cells the router claimed for this wire, in the wire's color."""
    import numpy as np
    if not cells:
        return
    ijk = np.asarray(cells, dtype=np.float64)
    centres = np.asarray(gbmin, dtype=np.float64) + (ijk + 0.5) * float(cell_size)
    step = max(1, len(centres) // cap)
    centres = centres[::step]
    author_points(stage, f"{DEBUG_SCOPE}/cells_{wire_name}", centres,
                  size=float(cell_size) * 0.5, color=color)


def author_raw_path(stage, wire_name, raw_polyline, color=(0.8, 0.8, 0.0), width=0.02):
    """BasisCurves for the stair-stepped grid path BEFORE smoothing. width is in STAGE
    units (callers scale by 1/metersPerUnit so it stays visible in cm/mm stages)."""
    if not raw_polyline or len(raw_polyline) < 2:
        return
    crv = UsdGeom.BasisCurves.Define(stage, f"{DEBUG_SCOPE}/raw_{wire_name}")
    pts = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in raw_polyline]
    crv.GetPointsAttr().Set(pts)
    crv.GetCurveVertexCountsAttr().Set([len(pts)])
    crv.GetTypeAttr().Set(UsdGeom.Tokens.linear)
    crv.GetWidthsAttr().Set([float(width)] * len(pts))
    crv.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    crv.GetDisplayColorAttr().Set([Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])


def author_bend_heatmap(stage, wire_name, polyline, min_bend_radius_mm, cap=5_000,
                         pos_scale=1.0, width=None):
    """BasisCurves coloured green/yellow/red by local curvature:
       green = radius > 1.5× min_bend, yellow = 0.5-1.5×, red = below limit.

    The polyline is in METERS (solver space) so the curvature physics is correct;
    pos_scale converts the AUTHORED point positions back to stage units (1/metersPerUnit)
    without disturbing the radius computation. width (STAGE units) sets the tube thickness —
    pass the wire's real display diameter so the heatmap is to-scale (default ~legacy)."""
    import numpy as np
    pts = [np.asarray(p, dtype=np.float64) for p in polyline]
    if len(pts) < 3:
        return

    # Resample to a fixed step (one min-bend-radius) so curvature is measured over a
    # CONSISTENT arc length instead of the grid cell size. Without this a sharp corner on
    # a coarse grid spreads its turn over a big cell and reads as a gentle arc (big chord
    # -> big implied radius) so it never flags red. Long segments get subdivided; original
    # vertices (the actual corners) are preserved, and already-dense smooth paths are left
    # as-is, so a real smooth curve still reads its true radius.
    min_bend_m = max(float(min_bend_radius_mm) / 1000.0, 1e-4)
    ds = min_bend_m
    rs = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        seg = b - a
        L = float(np.linalg.norm(seg))
        if L < 1e-9:
            continue
        n = max(1, int(np.ceil(L / ds)))
        for k in range(1, n + 1):
            rs.append(a + seg * (k / n))
    pts = rs
    if len(pts) < 3:
        return

    seg_colors = []
    # compute curvature radius at each interior (resampled) vertex
    for i in range(len(pts)):
        if i == 0 or i == len(pts) - 1:
            seg_colors.append((0.1, 0.85, 0.1))   # endpoints default green
            continue
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        ab, bc = b - a, c - b
        n_ab, n_bc = np.linalg.norm(ab), np.linalg.norm(bc)
        if n_ab < 1e-9 or n_bc < 1e-9:
            seg_colors.append((0.1, 0.85, 0.1))
            continue
        cos_a = float(np.clip(np.dot(ab / n_ab, bc / n_bc), -1.0, 1.0))
        angle = np.arccos(cos_a)          # turning angle in radians
        chord = float(n_ab + n_bc) / 2.0
        if angle < 1e-6:
            r_mm = 1e9
        else:
            r_mm = (chord / (2.0 * np.sin(angle / 2.0))) * 1000.0
        ratio = r_mm / max(float(min_bend_radius_mm), 1.0)
        if ratio >= 1.5:
            seg_colors.append((0.1, 0.85, 0.1))  # green — well within spec
        elif ratio >= 0.8:
            seg_colors.append((0.9, 0.7, 0.0))   # yellow — near limit
        else:
            seg_colors.append((0.95, 0.1, 0.1))  # red — violating min bend

    step = max(1, len(pts) // cap)
    pts_sub = pts[::step]
    cols_sub = seg_colors[::step]
    ps = float(pos_scale)
    gf_pts = [Gf.Vec3f(float(p[0]) * ps, float(p[1]) * ps, float(p[2]) * ps) for p in pts_sub]
    gf_cols = [Gf.Vec3f(*c) for c in cols_sub]
    crv = UsdGeom.BasisCurves.Define(stage, f"{DEBUG_SCOPE}/bend_{wire_name}")
    crv.GetPointsAttr().Set(gf_pts)
    crv.GetCurveVertexCountsAttr().Set([len(gf_pts)])
    crv.GetTypeAttr().Set(UsdGeom.Tokens.linear)
    w = float(width) if width is not None else 0.03 * ps
    crv.GetWidthsAttr().Set([w] * len(gf_pts))
    crv.SetWidthsInterpolation(UsdGeom.Tokens.vertex)
    crv.GetDisplayColorAttr().Set(gf_cols)
    crv.GetDisplayColorPrimvar().SetInterpolation(UsdGeom.Tokens.vertex)


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
