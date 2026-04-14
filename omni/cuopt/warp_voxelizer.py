"""GPU-accelerated mesh voxelization using NVIDIA Warp.

Uses wp.mesh_query_point_sign_normal to classify each grid cell as:
  - inside the mesh (signed distance < 0)
  - within clearance of the surface
  - free space

Runs entirely on GPU in a single kernel launch.
"""

import time as _time

import carb
import numpy as np
import warp as wp
from pxr import UsdGeom, Gf, Usd


@wp.kernel
def _sdf_voxelize(
    mesh_id: wp.uint64,
    grid_origin: wp.vec3,
    cell_size: float,
    rj: int,
    rk: int,
    clearance: float,
    max_dist: float,
    occupied: wp.array(dtype=wp.int32),
):
    i, j, k = wp.tid()

    center = wp.vec3(
        grid_origin[0] + (float(i) + 0.5) * cell_size,
        grid_origin[1] + (float(j) + 0.5) * cell_size,
        grid_origin[2] + (float(k) + 0.5) * cell_size,
    )

    face = int(0)
    bary_u = float(0.0)
    bary_v = float(0.0)
    sign = float(0.0)

    found = wp.mesh_query_point_sign_normal(
        mesh_id, center, max_dist, sign, face, bary_u, bary_v
    )

    if found:
        if sign < 0.0:
            occupied[i * rj * rk + j * rk + k] = 1
        else:
            closest = wp.mesh_eval_position(mesh_id, face, bary_u, bary_v)
            dist = wp.length(center - closest)
            if dist < clearance:
                occupied[i * rj * rk + j * rk + k] = 1


def voxelize_obstacles_gpu(stage, grid, clearance=0.0):
    # voxelize all meshes under /World/Obstacles using Warp SDF queries

    root = stage.GetPrimAtPath("/World/Obstacles")
    if not root or not root.IsValid():
        return 0

    all_points, all_indices = _collect_meshes(stage, root)
    if len(all_points) == 0:
        return 0

    points_wp = wp.array(all_points, dtype=wp.vec3, device="cuda:0")
    indices_wp = wp.array(all_indices, dtype=wp.int32, device="cuda:0")
    mesh = wp.Mesh(points=points_wp, indices=indices_wp)

    ri, rj, rk = int(grid.res_xyz[0]), int(grid.res_xyz[1]), int(grid.res_xyz[2])
    origin = wp.vec3(float(grid.bounds_min[0]),
                     float(grid.bounds_min[1]),
                     float(grid.bounds_min[2]))
    cs = float(grid.cell_size[0])

    max_dist = clearance * 2.0 + cs

    occupied_flat = wp.zeros(ri * rj * rk, dtype=wp.int32, device="cuda:0")

    wp.launch(
        _sdf_voxelize,
        dim=(ri, rj, rk),
        inputs=[mesh.id, origin, cs, rj, rk, clearance, max_dist, occupied_flat],
        device="cuda:0",
    )
    wp.synchronize()

    result = occupied_flat.numpy().reshape((ri, rj, rk))
    grid.occupied |= result.astype(bool)

    occupied_pct = 100.0 * grid.occupied.sum() / grid.occupied.size
    carb.log_warn(
        f"[omni.cuopt] warp: {ri}x{rj}x{rk} grid, {len(all_points)} verts, "
        f"{len(all_indices)//3} tris, {occupied_pct:.1f}% occupied"
    )

    return 1


def _collect_meshes(stage, root):
    # gather all mesh geometry under root into merged world-space arrays

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())

    point_chunks = []
    index_chunks = []
    vert_offset = 0

    for prim in Usd.PrimRange(root):
        if not prim.IsA(UsdGeom.Mesh):
            continue

        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        if not points or not face_indices or not face_counts:
            continue

        t = _time.time()

        # bulk vertex transform
        xf = xform_cache.GetLocalToWorldTransform(prim)
        m = np.array([list(xf.GetRow(r)) for r in range(4)], dtype=np.float64)
        pts = np.array(points, dtype=np.float64)
        homo = np.column_stack([pts, np.ones(len(pts))])
        world = (homo @ m)[:, :3]
        point_chunks.append(world.astype(np.float32))

        # vectorized fan triangulation
        fi = np.array(face_indices, dtype=np.int32)
        fc = np.array(face_counts, dtype=np.int32)
        tri_indices = _fan_triangulate(fi, fc, vert_offset)
        index_chunks.append(tri_indices)

        carb.log_warn(
            f"[omni.cuopt] mesh {prim.GetName()}: "
            f"{len(points)} verts, {len(tri_indices)//3} tris, "
            f"{_time.time() - t:.3f}s"
        )

        vert_offset += len(points)

    if not point_chunks:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)

    return np.concatenate(point_chunks), np.concatenate(index_chunks)


def _fan_triangulate(face_indices, face_counts, vert_offset):
    # vectorized fan triangulation

    fc = face_counts.astype(np.int64)
    n_tris = int(np.sum(fc - 2))

    # face start offsets
    face_starts = np.zeros(len(fc) + 1, dtype=np.int64)
    np.cumsum(fc, out=face_starts[1:])

    # repeat face index for each triangle it produces
    tris_per_face = (fc - 2).astype(np.int64)
    face_ids = np.repeat(np.arange(len(fc)), tris_per_face)

    # triangle index within each face (0, 1, ..., count-3)
    tri_local = np.arange(n_tris) - np.repeat(
        np.concatenate([[0], np.cumsum(tris_per_face[:-1])]), tris_per_face
    )

    starts = face_starts[face_ids]
    v0 = face_indices[starts] + vert_offset
    v1 = face_indices[starts + tri_local + 1] + vert_offset
    v2 = face_indices[starts + tri_local + 2] + vert_offset

    result = np.empty(n_tris * 3, dtype=np.int32)
    result[0::3] = v0
    result[1::3] = v1
    result[2::3] = v2
    return result
