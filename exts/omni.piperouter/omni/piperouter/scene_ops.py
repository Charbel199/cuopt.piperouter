"""USD stage I/O.

Collidable meshes, bounds, draggable markers, tube authoring and thermal/EM tag
read/write. Uses pxr only, so it can be tested headless against usd-core.
"""
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
CLEARANCE_ATTR = "piperouter:clearance_m"   # per-object safety clearance, in metres
SESSION_KEY = "piperouterSession"   # customData key holding the embedded panel session


def write_session(stage, data: dict):
    """Embed the panel session dict as JSON in customData on the PipeRouter root prim.

    Storing it on the stage is what makes it travel with a Save or a usdz export.
    """
    root = UsdGeom.Scope.Define(stage, PIPEROUTER_ROOT)
    root.GetPrim().SetCustomDataByKey(SESSION_KEY, json.dumps(data))


def read_session(stage):
    """Return the embedded session dict, or None if the stage carries no session."""
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
    """Return every UsdGeom.Mesh prim outside the PipeRouter scope.

    Two USD traversal traps are handled here. A drag-dropped asset arrives as an
    unloaded payload, greyed out in the Stage panel, so every unloaded prim is loaded
    before traversing or its meshes are invisible to us.

    CAD imports also tend to instance every repeated part: the geometry lives once under
    /Prototypes and the scene holds instanceable references to it, which Omniverse shows
    as read-only instance proxies. A plain stage.Traverse() never descends into an
    instance, so this uses Usd.TraverseInstanceProxies(). Proxy points (prototype
    geometry) and world transforms (per-instance placement) then read normally.
    """
    # TraverseAll() reaches inactive and unloaded roots that stage.Traverse() skips,
    # which is what lets the loop below load them.
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
        # Fall back to loading everything under the root.
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
    """Author a draggable marker as a single Sphere prim.

    Deliberately one prim rather than a parent/child pair, so the prim the user moves in
    the viewport is exactly the one read back. An opacity below 1.0 makes the marker
    see-through, which keeps waypoints from hiding the geometry they sit on.
    """
    sph = UsdGeom.Sphere.Define(stage, path)
    # Reuse the existing translate op when the marker is being rebuilt.
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
    """Author a waypoint gizmo as three orthogonal wireframe rings.

    A wireframe keeps the routed wire and the geometry behind it visible, which a solid
    sphere does not. displayOpacity is not an option here: the RTX viewport ignores it
    without a translucent material.

    The translate op lives on the prim itself, as in spawn_marker, so marker_positions()
    and get_world_pos() read it unchanged.
    """
    crv = UsdGeom.BasisCurves.Define(stage, path)
    xf = UsdGeom.Xformable(crv)
    ops = [o for o in xf.GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate]
    op = ops[0] if ops else xf.AddTranslateOp()
    op.Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))

    r = float(radius)
    th = np.linspace(0.0, 2.0 * np.pi, int(segments), endpoint=False)
    pts = []
    for ax in range(3):  # one ring each in the XY, XZ and YZ planes
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
    """Return the world-space unit vector of a prim's `local_axis`, or None.

    Rotation only, with scale stripped, which is how a marker's heading is read back
    from the way the user rotated it. None means the prim is missing or the axis
    degenerated.
    """
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
    """Return the shortest-arc quaternion taking `local_axis` onto world `direction`.

    Used to point a marker's heading arrow along a chosen vector.
    """
    d = Gf.Vec3d(*[float(x) for x in direction])
    if d.GetLength() < 1e-9:
        return Gf.Quatf(1.0)
    a = Gf.Vec3d(*[float(x) for x in local_axis]).GetNormalized()
    return Gf.Quatf(Gf.Rotation(a, d.GetNormalized()).GetQuat())


def _author_heading_arrow(stage, path, length, color, incoming=False):
    """Author an arrow (a shaft and two head strokes) along the parent's local +X.

    It is a child prim, so it inherits the marker's translate and orient and rotating
    the marker aims the arrow.

    A start marker (incoming=False) draws the arrow outward from the marker: the cable
    leaves this way. An end marker (incoming=True) draws it approaching the marker with
    the tip on it: the cable arrives along the arrow. Both point along local +X, the
    travel direction the solver reads; only the placement of the strokes differs.
    """
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
    """Show or aim the heading arrow on a start or end marker.

    Guarantees the marker carries an xformOp:orient, without which Kit's rotate
    manipulator cannot spin it, plus an arrow child at `{path}/dir` along local +X. A
    world-vector `direction` re-aims the orient, while None keeps the current rotation.
    show=False removes the arrow and leaves the orient op in place. Returns True when
    the marker exists.
    """
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
        # Size the arrow from the marker sphere so it reads at scene scale.
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
    """Return the world position of every marker under MARKERS_SCOPE.

    The voxel grid is framed to include these. A marker dragged beyond the scene
    geometry still has to fall inside the grid, or routing to it clamps at the edge.
    """
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
    """Make every authored route tube visible again, undoing any debug-view hide."""
    routes = stage.GetPrimAtPath(ROUTES_SCOPE)
    if not routes or not routes.IsValid():
        return
    for prim in routes.GetChildren():
        UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)


def hide_route(stage, wire_name):
    """Hide a wire's final tubes so a per-wire debug view is not occluded by the cable.

    Matches the wire's own tube and any bundle branch segments, named <name>_seg<i>.
    """
    routes = stage.GetPrimAtPath(ROUTES_SCOPE)
    if not routes or not routes.IsValid():
        return
    for prim in routes.GetChildren():
        nm = prim.GetName()
        if nm == wire_name or nm.startswith(wire_name + "_seg"):
            UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)


def author_box_mesh(stage, path, center, size, color=(0.5, 0.5, 0.5)):
    """Author an axis-aligned box as a real UsdGeom.Mesh, `size` being the full extents.

    It has to be a Mesh: the voxelizer collects Meshes and ignores Cube prims.
    """
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
    """Author see-through wireframe boxes, 12 edges each, as one BasisCurves prim.

    Used to show the octree leaves without occluding the scene. `boxes` is a list of
    (min_xyz, max_xyz) in stage units, and the optional `colors` gives per-box RGB.
    """
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
            disp.append(cv)               # one colour per edge, hence uniform below
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
    """Render a point cloud as short fat BasisCurves stubs, one per point.

    Each point becomes a two-vertex linear curve of width `size`. Curves rather than
    UsdGeom.Points because the RTX viewport draws curves reliably and routinely fails to
    draw Points at all. Per-point colour works through vertex-interpolated displayColor,
    with two vertices per point.
    """
    pts_in = [(float(p[0]), float(p[1]), float(p[2])) for p in points]
    n = len(pts_in)
    crv = UsdGeom.BasisCurves.Define(stage, path)
    # Keeping the stub much shorter than it is wide lets the round end caps dominate, so
    # each point reads as a sphere rather than an elongated pill.
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
            col.append(v)   # one colour per vertex, two vertices per point
        crv.GetDisplayColorAttr().Set(col)
        crv.GetDisplayColorPrimvar().SetInterpolation(UsdGeom.Tokens.vertex)
    else:
        crv.GetDisplayColorAttr().Set(
            [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))])
    return crv


def author_points(stage, path, points, size=0.01, color=(0.2, 0.6, 1.0)):
    """Author a debug dot cloud, as used by the occupancy overlay and per-wire cells."""
    return _author_blob_cloud(stage, path, points, size, colors=None, color=color)


def author_colored_points(stage, path, points, colors, size=0.02):
    """Author a dot cloud with a per-point colour.

    Used by the thermal, EM and cost clouds, where each cell is tinted by its field
    value.
    """
    return _author_blob_cloud(stage, path, points, size, colors=colors)


def author_wire_cells(stage, wire_name, cells, gbmin, cell_size, color=(0.8, 0.1, 0.1),
                      cap=100_000):
    """Author a point cloud of the voxel cells the router claimed for this wire."""
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
    """Author the stair-stepped grid path as it is before smoothing.

    `width` is in stage units; callers scale by 1/metersPerUnit so the curve stays
    visible on centimetre and millimetre stages.
    """
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
    """Author the route as a curve coloured by local bend radius.

    Green means a radius comfortably above min_bend, yellow near the limit, red below
    it.

    The polyline must be in metres, i.e. solver space, so the curvature comes out right.
    `pos_scale` converts only the authored point positions back to stage units,
    1/metersPerUnit, leaving the radius computation alone. `width` is in stage units;
    pass the wire's real display diameter to keep the heatmap to scale.
    """
    import numpy as np
    pts = [np.asarray(p, dtype=np.float64) for p in polyline]
    if len(pts) < 3:
        return

    # Resample to a fixed step of one min-bend-radius so curvature is measured over a
    # consistent arc length rather than over the grid cell size. Otherwise a sharp corner
    # on a coarse grid spreads its turn across a big cell and reads as a gentle arc,
    # since a big chord implies a big radius, and never flags red. Long segments are
    # subdivided while the original vertices, which are the real corners, are kept;
    # already-dense smooth paths pass through unchanged and keep their true radius.
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
    # Curvature radius at each interior resampled vertex.
    for i in range(len(pts)):
        if i == 0 or i == len(pts) - 1:
            seg_colors.append((0.1, 0.85, 0.1))   # endpoints default to green
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
            seg_colors.append((0.1, 0.85, 0.1))  # green: well within spec
        elif ratio >= 0.8:
            seg_colors.append((0.9, 0.7, 0.0))   # yellow: near the limit
        else:
            seg_colors.append((0.95, 0.1, 0.1))  # red: below the minimum bend radius

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
    """Return the position and size of every prim tagged with a temperature or EM value.

    Each entry is (center, temp_c or None, em or None, char_size), where center is the
    world bounding-box centre as a (3,) ndarray and char_size is half the bbox diagonal,
    a characteristic radius the field builder uses as a falloff distance.

    The centre comes from the bounding box rather than the xform translation because
    author_box_mesh() bakes geometry into world-space points under an identity
    transform, leaving the prim's translation at the origin. The bbox centre is right
    either way, whether the geometry is baked into points or placed by an xform.
    """
    out = []
    # BBoxCache computes world-space, axis-aligned bounds of a prim's geometry.
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
            # No geometry of its own, as with a bare Xform, so fall back to the world
            # translation and a zero characteristic size.
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

    # Instance-proxy tags from the registry. The stage.Traverse() above skips proxies, so
    # nothing is double-counted, and reading a proxy's bbox or xform is allowed; only
    # authoring on one is not.
    for path, entry in read_proxy_tags(stage).items():
        if entry.get("temp_c") is None and entry.get("em") is None:
            continue
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            continue
        rng = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        if rng.IsEmpty():
            tr = xform.GetLocalToWorldTransform(prim).ExtractTranslation()
            center = np.array([tr[0], tr[1], tr[2]], dtype=float)
            char_size = 0.0
        else:
            mn, mx = rng.GetMin(), rng.GetMax()
            lo = np.array([mn[0], mn[1], mn[2]], dtype=float)
            hi = np.array([mx[0], mx[1], mx[2]], dtype=float)
            center = 0.5 * (lo + hi)
            char_size = 0.5 * float(np.linalg.norm(hi - lo))
        out.append((center, entry.get("temp_c"), entry.get("em"), char_size))
    return out


PROXY_TAGS_KEY = "piperouterProxyTags"   # customData: {proxy path -> {temp_c, em, clearance_m}}


def read_proxy_tags(stage):
    """Return the tag registry for instance-proxy prims.

    USD forbids authoring attributes on a proxy, since its geometry lives once in a
    shared prototype, so these tags are keyed by path in customData on the PipeRouter
    root. Keying by path means a tag applies to the selected prim alone, not to its
    parent or its instance, and the same part in another instance stays untagged.
    """
    prim = stage.GetPrimAtPath(PIPEROUTER_ROOT)
    if not prim or not prim.IsValid():
        return {}
    raw = prim.GetCustomDataByKey(PROXY_TAGS_KEY)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _write_proxy_tags(stage, tags: dict):
    root = UsdGeom.Scope.Define(stage, PIPEROUTER_ROOT)
    root.GetPrim().SetCustomDataByKey(PROXY_TAGS_KEY, json.dumps(tags))


def write_tags(prim, temp_c=None, em=None, clearance_m=None):
    if prim.IsInstanceProxy():
        stage = prim.GetStage()
        tags = read_proxy_tags(stage)
        entry = tags.get(str(prim.GetPath()), {})
        if temp_c is not None:
            entry["temp_c"] = float(temp_c)
        if em is not None:
            entry["em"] = float(em)
        if clearance_m is not None:
            entry["clearance_m"] = float(clearance_m)
        tags[str(prim.GetPath())] = entry
        _write_proxy_tags(stage, tags)
        return prim
    if temp_c is not None:
        prim.CreateAttribute(TEMP_ATTR, Sdf.ValueTypeNames.Float).Set(float(temp_c))
    if em is not None:
        prim.CreateAttribute(EM_ATTR, Sdf.ValueTypeNames.Float).Set(float(em))
    if clearance_m is not None:
        prim.CreateAttribute(CLEARANCE_ATTR, Sdf.ValueTypeNames.Float).Set(float(clearance_m))
    return prim


def list_tagged_prims(stage):
    """Return every prim carrying a thermal, EM or clearance tag.

    Each entry is {path, temp_c, em, clearance_m}, with None for the absent tags.
    """
    out = []
    for prim in stage.Traverse():
        t = prim.GetAttribute(TEMP_ATTR)
        e = prim.GetAttribute(EM_ATTR)
        c = prim.GetAttribute(CLEARANCE_ATTR)
        has_t = bool(t) and t.IsValid() and t.HasAuthoredValue()
        has_e = bool(e) and e.IsValid() and e.HasAuthoredValue()
        has_c = bool(c) and c.IsValid() and c.HasAuthoredValue()
        if has_t or has_e or has_c:
            out.append({"path": str(prim.GetPath()),
                        "temp_c": float(t.Get()) if has_t else None,
                        "em": float(e.Get()) if has_e else None,
                        "clearance_m": float(c.Get()) if has_c else None})
    # Instance-proxy tags live in the registry; skip entries whose prim has gone.
    for path, entry in read_proxy_tags(stage).items():
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            out.append({"path": path,
                        "temp_c": entry.get("temp_c"),
                        "em": entry.get("em"),
                        "clearance_m": entry.get("clearance_m")})
    return out


def clearance_for_prim(prim, proxy_tags=None):
    """Return the effective per-object clearance for a mesh, or None if untagged.

    Takes CLEARANCE_ATTR from the prim itself or its nearest tagged ancestor, since
    users usually tag the component Xform whose meshes sit below it. For instance
    proxies it takes the registry entry matching the prim's own path or an ancestor's.

    Pass `proxy_tags` from read_proxy_tags(stage) when calling in a loop; leaving it None
    re-reads the registry on every call.
    """
    if proxy_tags is None:
        try:
            proxy_tags = read_proxy_tags(prim.GetStage())
        except Exception:
            proxy_tags = {}
    p = prim
    while p and p.IsValid():
        entry = proxy_tags.get(str(p.GetPath()))
        if entry and entry.get("clearance_m") is not None:
            return float(entry["clearance_m"])
        a = p.GetAttribute(CLEARANCE_ATTR)
        if a and a.IsValid() and a.HasAuthoredValue():
            return float(a.Get())
        p = p.GetParent()
    return None


def clear_tags(prim):
    """Remove the thermal, EM and clearance tags from a prim.

    An instance proxy clears its registry entry instead, mirroring write_tags.
    """
    if prim.IsInstanceProxy():
        stage = prim.GetStage()
        tags = read_proxy_tags(stage)
        if tags.pop(str(prim.GetPath()), None) is not None:
            _write_proxy_tags(stage, tags)
        return
    for attr in (TEMP_ATTR, EM_ATTR, CLEARANCE_ATTR):
        a = prim.GetAttribute(attr)
        if a and a.IsValid() and a.HasAuthoredValue():
            prim.RemoveProperty(attr)
