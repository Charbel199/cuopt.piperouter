import math
import re

import numpy as np
from pxr import UsdGeom, Gf, Vt, Usd


PIPE_PALETTE = [
    {"tube": (0.18, 0.50, 0.92), "start": (0.30, 0.75, 1.00), "end": (0.10, 0.30, 0.65)},
    {"tube": (0.92, 0.55, 0.18), "start": (1.00, 0.72, 0.30), "end": (0.62, 0.35, 0.10)},
    {"tube": (0.65, 0.22, 0.85), "start": (0.82, 0.42, 1.00), "end": (0.40, 0.12, 0.55)},
    {"tube": (0.20, 0.80, 0.45), "start": (0.35, 1.00, 0.55), "end": (0.10, 0.50, 0.25)},
]

# engine bay layout (X = left-right, Y = up, Z = front-back, front of car = -Z)
# Scale: 1 unit ~ 1 cm.  Bay is roughly 120 wide, 70 deep, 50 tall.
DEFAULT_OBSTACLES = [
    {"name": "EngineBlock",  "center": (0, 12, 8),     "size": (22, 20, 24)},
    {"name": "ValveCover",   "center": (0, 24, 8),     "size": (18, 4, 20)},
    {"name": "Radiator",     "center": (0, 12, -28),   "size": (40, 18, 3)},
    {"name": "Battery",      "center": (32, 8, -8),    "size": (10, 12, 12)},
    {"name": "ACCompressor", "center": (-28, 6, 14),   "size": (8, 8, 8)},
    {"name": "Alternator",   "center": (24, 6, 18),    "size": (7, 7, 7)},
    {"name": "Firewall",     "center": (0, 14, 30),    "size": (70, 28, 2)},
    {"name": "FenderLeft",   "center": (-40, 10, 0),   "size": (2, 18, 60)},
    {"name": "FenderRight",  "center": (40, 10, 0),    "size": (2, 18, 60)},
]

DEFAULT_PIPES = [
    # coolant hose: above-left of radiator to gap behind engine
    {"name": "Coolant",  "start": (-16, 28, -22), "end": (6, 28, 24)},
    # oil line: right gap beside engine to in front of radiator
    {"name": "OilLine",  "start": (18, 2, 0),     "end": (22, 4, -35)},
    # AC line: left gap above compressor to in front of radiator
    {"name": "ACLine",   "start": (-20, 16, 6),   "end": (-12, 4, -35)},
]


def create_sample_scene(stage):
    # I asked claude to create a sample car scene (it looks very bad)
    clear_all(stage)

    _ensure_xform(stage, "/World")
    _ensure_xform(stage, "/World/Obstacles")
    _ensure_xform(stage, "/World/Markers")

    obstacle_colors = {
        "EngineBlock":    Gf.Vec3f(0.35, 0.35, 0.38),
        "ValveCover":     Gf.Vec3f(0.25, 0.25, 0.28),
        "Radiator":       Gf.Vec3f(0.12, 0.12, 0.14),
        "Battery":        Gf.Vec3f(0.20, 0.20, 0.25),
        "ACCompressor":   Gf.Vec3f(0.30, 0.30, 0.32),
        "Alternator":     Gf.Vec3f(0.32, 0.32, 0.35),
        "Firewall":       Gf.Vec3f(0.42, 0.42, 0.44),
        "FenderLeft":     Gf.Vec3f(0.44, 0.44, 0.46),
        "FenderRight":    Gf.Vec3f(0.44, 0.44, 0.46),
    }
    default_color = Gf.Vec3f(0.35, 0.35, 0.38)

    for obs in DEFAULT_OBSTACLES:
        path = f"/World/Obstacles/{obs['name']}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        color = obstacle_colors.get(obs["name"], default_color)
        cube.GetDisplayColorAttr().Set(Vt.Vec3fArray([color]))
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*obs["center"]))
        xf.AddScaleOp().Set(Gf.Vec3f(*obs["size"]))

    for i, pipe in enumerate(DEFAULT_PIPES):
        pal = PIPE_PALETTE[i % len(PIPE_PALETTE)]
        _create_sphere(
            stage, f"/World/Markers/{pipe['name']}_Start",
            pipe["start"], Gf.Vec3f(*pal["start"]), 3.0,
        )
        _create_sphere(
            stage, f"/World/Markers/{pipe['name']}_End",
            pipe["end"], Gf.Vec3f(*pal["end"]), 3.0,
        )


def get_pipe_markers(stage):
    root = stage.GetPrimAtPath("/World/Markers")
    if not root or not root.IsValid():
        return []

    starts = {}
    ends = {}
    for child in root.GetChildren():
        name = child.GetName()
        m = re.match(r"^(.+)_Start$", name)
        if m:
            pos = _read_translate(stage, child.GetPath().pathString)
            if pos:
                starts[m.group(1)] = pos
            continue
        m = re.match(r"^(.+)_End$", name)
        if m:
            pos = _read_translate(stage, child.GetPath().pathString)
            if pos:
                ends[m.group(1)] = pos

    pipes = []
    for i, key in enumerate(sorted(starts.keys())):
        if key in ends:
            pipes.append({
                "name": key,
                "start": starts[key],
                "end": ends[key],
                "index": i,
            })
    return pipes


def get_obstacle_bounds(stage):
    root = stage.GetPrimAtPath("/World/Obstacles")
    if not root or not root.IsValid():
        return []

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])

    # whole-subtree AABB (catches deeply nested meshes)
    root_bbox = cache.ComputeWorldBound(root)
    root_rng = root_bbox.ComputeAlignedRange()
    if root_rng.IsEmpty():
        return []

    bounds = []
    lo = root_rng.GetMin()
    hi = root_rng.GetMax()
    bounds.append(((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])))

    # also add per-child AABBs for finer bounding-box fallback
    for child in root.GetChildren():
        bbox = cache.ComputeWorldBound(child)
        rng = bbox.ComputeAlignedRange()
        if rng.IsEmpty():
            continue
        lo = rng.GetMin()
        hi = rng.GetMax()
        bounds.append(((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])))

    return bounds


def voxelize_obstacles(stage, grid, clearance=0.0):
    import carb

    root = stage.GetPrimAtPath("/World/Obstacles")
    if not root or not root.IsValid():
        return 0

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    total_verts = 0

    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        points = UsdGeom.Mesh(prim).GetPointsAttr().Get()
        if not points:
            continue

        world_xform = xform_cache.GetLocalToWorldTransform(prim)

        for p in points:
            wp = world_xform.Transform(Gf.Vec3d(p))
            pt = np.array([wp[0], wp[1], wp[2]])
            grid.mark_box(pt - clearance, pt + clearance)

        total_verts += len(points)

    carb.log_info(
        f"[voxelize] {total_verts} vertices, "
        f"{100.0 * grid.occupied.sum() / grid.resolution**3:.1f}% occupied"
    )
    return total_verts


def fill_interior(grid):
    from collections import deque

    ri, rj, rk = grid.res_xyz
    exterior = np.zeros((ri, rj, rk), dtype=bool)
    queue = deque()

    # seed from all six boundary faces
    for i in range(ri):
        for j in range(rj):
            for idx in [(i, j, 0), (i, j, rk - 1)]:
                if not grid.occupied[idx] and not exterior[idx]:
                    exterior[idx] = True
                    queue.append(idx)
    for i in range(ri):
        for k in range(rk):
            for idx in [(i, 0, k), (i, rj - 1, k)]:
                if not grid.occupied[idx] and not exterior[idx]:
                    exterior[idx] = True
                    queue.append(idx)
    for j in range(rj):
        for k in range(rk):
            for idx in [(0, j, k), (ri - 1, j, k)]:
                if not grid.occupied[idx] and not exterior[idx]:
                    exterior[idx] = True
                    queue.append(idx)

    # 6-connected BFS
    neighbors = [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]
    while queue:
        ci, cj, ck = queue.popleft()
        for di, dj, dk in neighbors:
            ni, nj, nk = ci + di, cj + dj, ck + dk
            if 0 <= ni < ri and 0 <= nj < rj and 0 <= nk < rk:
                if not exterior[ni, nj, nk] and not grid.occupied[ni, nj, nk]:
                    exterior[ni, nj, nk] = True
                    queue.append((ni, nj, nk))

    grid.occupied |= ~exterior


def create_tube_mesh(stage, path_points, pipe_name="Pipe", radius=2.0,
                     color=(0.18, 0.50, 0.92), segments=12):
    """Sweep a circular cross-section along *path_points* to create a tube."""
    prim_path = f"/World/Pipes/{pipe_name}"
    _remove_prim(stage, prim_path)
    _ensure_xform(stage, "/World")
    _ensure_xform(stage, "/World/Pipes")

    pts = np.array(path_points)
    n = len(pts)
    if n < 2:
        return

    tangents = _compute_tangents(pts)
    normals, binormals = _parallel_transport(tangents)

    verts = []
    for i in range(n):
        c, nm, bn = pts[i], normals[i], binormals[i]
        for j in range(segments):
            a = 2.0 * math.pi * j / segments
            v = c + radius * (math.cos(a) * nm + math.sin(a) * bn)
            verts.append(Gf.Vec3f(float(v[0]), float(v[1]), float(v[2])))

    indices = []
    counts = []
    for i in range(n - 1):
        for j in range(segments):
            jn = (j + 1) % segments
            indices.extend([
                i * segments + j,
                (i + 1) * segments + j,
                (i + 1) * segments + jn,
                i * segments + jn,
            ])
            counts.append(4)

    # caps
    ci = len(verts)
    verts.append(Gf.Vec3f(float(pts[0][0]), float(pts[0][1]), float(pts[0][2])))
    for j in range(segments):
        jn = (j + 1) % segments
        indices.extend([ci, jn, j])
        counts.append(3)

    ci = len(verts)
    verts.append(Gf.Vec3f(float(pts[-1][0]), float(pts[-1][1]), float(pts[-1][2])))
    base = (n - 1) * segments
    for j in range(segments):
        jn = (j + 1) % segments
        indices.extend([ci, base + j, base + jn])
        counts.append(3)

    mesh = UsdGeom.Mesh.Define(stage, prim_path)
    mesh.GetPointsAttr().Set(Vt.Vec3fArray(verts))
    mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray(indices))
    mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray(counts))
    mesh.GetSubdivisionSchemeAttr().Set("none")
    mesh.GetDoubleSidedAttr().Set(True)
    mesh.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))


def mark_tube_obstacle(grid, path_points, radius):
    for pt in path_points:
        grid.mark_box(pt - radius, pt + radius)


def count_bends(path_points, threshold_deg=15.0):
    if len(path_points) < 3:
        return 0
    pts = np.array(path_points)
    bends = 0
    cos_thresh = math.cos(math.radians(threshold_deg))
    for i in range(1, len(pts) - 1):
        d1 = pts[i] - pts[i - 1]
        d2 = pts[i + 1] - pts[i]
        n1 = np.linalg.norm(d1)
        n2 = np.linalg.norm(d2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_a = np.dot(d1, d2) / (n1 * n2)
        if cos_a < cos_thresh:
            bends += 1
    return bends


def path_length(path_points):
    pts = np.array(path_points)
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def clear_pipes(stage):
    _remove_prim(stage, "/World/Pipes")


def clear_all(stage):
    for p in ("/World/Pipes", "/World/Pipe", "/World/Markers", "/World/Obstacles"):
        _remove_prim(stage, p)


def _ensure_xform(stage, path):
    if not stage.GetPrimAtPath(path).IsValid():
        UsdGeom.Xform.Define(stage, path)


def _remove_prim(stage, path):
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        stage.RemovePrim(path)


def _create_sphere(stage, path, pos, color, radius):
    sp = UsdGeom.Sphere.Define(stage, path)
    sp.GetRadiusAttr().Set(radius)
    sp.GetDisplayColorAttr().Set(Vt.Vec3fArray([color]))
    UsdGeom.Xformable(sp.GetPrim()).AddTranslateOp().Set(Gf.Vec3d(*pos))


def _read_translate(stage, prim_path):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        return None
    xf = UsdGeom.Xformable(prim)
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            v = op.Get()
            return (float(v[0]), float(v[1]), float(v[2]))
    return None


def _compute_tangents(pts):
    n = len(pts)
    t = np.zeros_like(pts)
    t[0] = pts[1] - pts[0]
    t[-1] = pts[-1] - pts[-2]
    for i in range(1, n - 1):
        t[i] = pts[i + 1] - pts[i - 1]
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return t / norms


def _parallel_transport(tangents):
    n = len(tangents)
    normals = np.zeros((n, 3))
    binormals = np.zeros((n, 3))

    t0 = tangents[0]
    abs_t = np.abs(t0)
    if abs_t[0] <= abs_t[1] and abs_t[0] <= abs_t[2]:
        ref = np.array([1.0, 0.0, 0.0])
    elif abs_t[1] <= abs_t[2]:
        ref = np.array([0.0, 1.0, 0.0])
    else:
        ref = np.array([0.0, 0.0, 1.0])

    normals[0] = np.cross(t0, ref)
    normals[0] /= max(np.linalg.norm(normals[0]), 1e-12)
    binormals[0] = np.cross(t0, normals[0])
    binormals[0] /= max(np.linalg.norm(binormals[0]), 1e-12)

    for i in range(1, n):
        b = np.cross(tangents[i - 1], tangents[i])
        bn = np.linalg.norm(b)
        if bn < 1e-10:
            normals[i] = normals[i - 1]
        else:
            b /= bn
            angle = math.acos(np.clip(np.dot(tangents[i - 1], tangents[i]), -1, 1))
            normals[i] = _rodrigues(normals[i - 1], b, angle)
            nm = np.linalg.norm(normals[i])
            if nm > 1e-12:
                normals[i] /= nm
        binormals[i] = np.cross(tangents[i], normals[i])
        nm = np.linalg.norm(binormals[i])
        if nm > 1e-12:
            binormals[i] /= nm

    return normals, binormals


def _rodrigues(v, axis, angle):
    c = math.cos(angle)
    s = math.sin(angle)
    return v * c + np.cross(axis, v) * s + axis * np.dot(axis, v) * (1 - c)
