import math
import os
import re

import numpy as np
from pxr import UsdGeom, Gf, Vt, Usd, Sdf

PIPE_PALETTE = [
    {"tube": (0.18, 0.50, 0.92), "start": (0.30, 0.75, 1.00), "end": (0.10, 0.30, 0.65)},
    {"tube": (0.92, 0.55, 0.18), "start": (1.00, 0.72, 0.30), "end": (0.62, 0.35, 0.10)},
    {"tube": (0.65, 0.22, 0.85), "start": (0.82, 0.42, 1.00), "end": (0.40, 0.12, 0.55)},
    {"tube": (0.20, 0.80, 0.45), "start": (0.35, 1.00, 0.55), "end": (0.10, 0.50, 0.25)},
]

# scene 1: simple cubes
SIMPLE_OBSTACLES = [
    {"name": "Block_A", "center": (0, 15, 0),    "size": (25, 30, 25), "color": (0.35, 0.35, 0.38)},
    {"name": "Block_B", "center": (-25, 25, 15), "size": (10, 25, 20), "color": (0.40, 0.40, 0.42)},
    {"name": "Block_C", "center": (20, 10, -15), "size": (15, 20, 15), "color": (0.30, 0.30, 0.33)},
    {"name": "Block_D", "center": (0, 40, -10),  "size": (30, 8, 12),  "color": (0.38, 0.38, 0.40)},
    {"name": "Block_E", "center": (-10, 0, -20), "size": (12, 10, 12), "color": (0.33, 0.33, 0.36)},
]

SIMPLE_PIPES = [
    {"name": "Pipe_01", "start": (-45, 10, 35),  "end": (45, 35, -30)},
    {"name": "Pipe_02", "start": (-45, 40, -30),  "end": (45, 5, 35)},
    {"name": "Pipe_03", "start": (-45, -10, 0),   "end": (45, 50, 0)},
]

# scene 2: engine bay (loads USDC from assets/)
ENGINE_BAY_ASSET = "engine_bay.usdc"

ENGINE_BAY_PIPES = [
    {"name": "Coolant", "start": (110, 149, -22), "end": (-83, 264, 24)},
    {"name": "OilLine", "start": (249, 83, 23),   "end": (-148, 321, -35)},
    {"name": "ACLine",  "start": (-98, 245, 100), "end": (135, 4, -35)},
]


def _get_asset_path(filename):
    ext_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(ext_dir, "assets", filename)


def create_simple_scene(stage):
    clear_all(stage)
    _ensure_xform(stage, "/World")
    _ensure_xform(stage, "/World/Obstacles")
    _ensure_xform(stage, "/World/Markers")

    for obs in SIMPLE_OBSTACLES:
        path = f"/World/Obstacles/{obs['name']}"
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        cube.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*obs["color"])]))
        xf = UsdGeom.Xformable(cube.GetPrim())
        xf.AddTranslateOp().Set(Gf.Vec3d(*obs["center"]))
        xf.AddScaleOp().Set(Gf.Vec3f(*obs["size"]))

    _create_pipe_markers(stage, SIMPLE_PIPES, sphere_radius=3.0)


def create_engine_bay_scene(stage):
    clear_all(stage)
    _ensure_xform(stage, "/World")
    _ensure_xform(stage, "/World/Obstacles")
    _ensure_xform(stage, "/World/Markers")

    asset_path = _get_asset_path(ENGINE_BAY_ASSET)
    if os.path.exists(asset_path):
        asset_stage = Usd.Stage.Open(asset_path)
        if asset_stage:
            dst_layer = stage.GetRootLayer()
            src_layer = asset_stage.GetRootLayer()

            # match the asset's stage settings so geometry looks identical
            UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(asset_stage))
            UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(asset_stage))

            # copy each root prim directly under /World/Obstacles
            for child in asset_stage.GetPseudoRoot().GetChildren():
                src_path = child.GetPath()
                dst_path = Sdf.Path(f"/World/Obstacles{src_path}")
                Sdf.CopySpec(src_layer, src_path, dst_layer, dst_path)
    else:
        import carb
        carb.log_warn(f"[omni.cuopt] asset not found: {asset_path}")

    _create_pipe_markers(stage, ENGINE_BAY_PIPES, sphere_radius=4.0)


def _create_pipe_markers(stage, pipes, sphere_radius=3.0):
    for i, pipe in enumerate(pipes):
        pal = PIPE_PALETTE[i % len(PIPE_PALETTE)]
        _create_sphere(
            stage, f"/World/Markers/{pipe['name']}_Start",
            pipe["start"], Gf.Vec3f(*pal["start"]), sphere_radius,
        )
        _create_sphere(
            stage, f"/World/Markers/{pipe['name']}_End",
            pipe["end"], Gf.Vec3f(*pal["end"]), sphere_radius,
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

    root_bbox = cache.ComputeWorldBound(root)
    root_rng = root_bbox.ComputeAlignedRange()
    if root_rng.IsEmpty():
        return []

    bounds = []
    lo = root_rng.GetMin()
    hi = root_rng.GetMax()
    bounds.append(((lo[0], lo[1], lo[2]), (hi[0], hi[1], hi[2])))

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

    carb.log_warn(
        f"[omni.cuopt] cpu voxelize: {total_verts} vertices, "
        f"{100.0 * grid.occupied.sum() / grid.occupied.size:.1f}% occupied"
    )
    return total_verts


def fill_interior(grid):
    from collections import deque

    ri, rj, rk = grid.res_xyz
    exterior = np.zeros((ri, rj, rk), dtype=bool)
    queue = deque()

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
