"""GPU mesh voxelization with Warp.

One kernel classifies each cell via a signed-distance query against the merged
scene mesh: occupied if inside (sign < 0) or within `clearance` of the surface, and
emits the distance-to-surface field in the same pass (so no scipy dependency). Runs
on CUDA when available, else on Warp's CPU device (keeps it testable without a GPU).
"""
from __future__ import annotations

import numpy as np
from pxr import Usd, UsdGeom

# Warp is imported lazily (see _ensure_warp) so this module - and therefore the
# whole extension - loads even in a Kit app that hasn't enabled omni.warp. Warp is
# only required when voxelize() actually runs, and then we raise a clear error.
_wp = None
_kernel = None


def _ensure_warp():
    global _wp, _kernel
    if _wp is not None:
        return _wp, _kernel
    try:
        import warp as wp
    except Exception as exc:  # pragma: no cover - depends on host app
        raise RuntimeError(
            "NVIDIA Warp is required for voxelization but is not available in this "
            "Kit app. Enable the 'omni.warp.core' extension (it ships with Isaac Sim "
            "and is available in the registry for USD Composer)."
        ) from exc
    wp.init()

    @wp.kernel
    def _voxelize_kernel(
        mesh_id: wp.uint64,
        origin: wp.vec3,
        cell_size: float,
        rj: int,
        rk: int,
        clearance: float,
        surface_band: float,
        max_dist: float,
        occupied: wp.array(dtype=wp.int32),
        dist_out: wp.array(dtype=wp.float32),
    ):
        i, j, k = wp.tid()
        center = wp.vec3(
            origin[0] + (float(i) + 0.5) * cell_size,
            origin[1] + (float(j) + 0.5) * cell_size,
            origin[2] + (float(k) + 0.5) * cell_size,
        )
        idx = i * rj * rk + j * rk + k
        # Winding-number sign test: robust inside/outside for closed and MULTI-component
        # meshes. (The older sign_normal test uses the nearest face's normal, which
        # misclassifies points between two objects / near concavities as "inside",
        # producing occupancy artifacts in empty gaps.)
        query = wp.mesh_query_point_sign_winding_number(mesh_id, center, max_dist, 2.0, 0.5)
        if query.result:
            cp = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
            d = wp.length(center - cp)
            dist_out[idx] = d
            if query.sign < 0.0:
                occupied[idx] = 1            # cell centre inside a solid (>=cell-thick) volume
            elif d < surface_band:
                occupied[idx] = 1            # surface passes THROUGH this cell -> watertight
            elif d < clearance:
                occupied[idx] = 1            # within requested safety clearance of the surface
        else:
            dist_out[idx] = max_dist

    _wp, _kernel = wp, _voxelize_kernel
    return _wp, _kernel


def _pick_device(wp, device):
    if device:
        return device
    try:
        if wp.is_cuda_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def _fan_triangulate(face_indices, face_counts, vert_offset):
    fc = face_counts.astype(np.int64)
    n_tris = int(np.sum(fc - 2))
    if n_tris <= 0:
        return np.zeros(0, dtype=np.int32)
    face_starts = np.zeros(len(fc) + 1, dtype=np.int64)
    np.cumsum(fc, out=face_starts[1:])
    tris_per_face = (fc - 2).astype(np.int64)
    face_ids = np.repeat(np.arange(len(fc)), tris_per_face)
    tri_local = np.arange(n_tris) - np.repeat(
        np.concatenate([[0], np.cumsum(tris_per_face[:-1])]), tris_per_face
    )
    starts = face_starts[face_ids]
    v0 = face_indices[starts] + vert_offset
    v1 = face_indices[starts + tri_local + 1] + vert_offset
    v2 = face_indices[starts + tri_local + 2] + vert_offset
    out = np.empty(n_tris * 3, dtype=np.int32)
    out[0::3] = v0
    out[1::3] = v1
    out[2::3] = v2
    return out


def collect_meshes(stage, prims):
    """Merge UsdGeom.Mesh prims into world-space (points f32 (N,3), indices i32)."""
    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    point_chunks = []
    index_chunks = []
    vert_offset = 0
    for prim in prims:
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        points = mesh.GetPointsAttr().Get()
        fi = mesh.GetFaceVertexIndicesAttr().Get()
        fc = mesh.GetFaceVertexCountsAttr().Get()
        if not points or not fi or not fc:
            continue
        xf = xform_cache.GetLocalToWorldTransform(prim)
        m = np.array([list(xf.GetRow(r)) for r in range(4)], dtype=np.float64)
        pts = np.array(points, dtype=np.float64)
        homo = np.column_stack([pts, np.ones(len(pts))])
        world = (homo @ m)[:, :3]
        point_chunks.append(world.astype(np.float32))
        index_chunks.append(
            _fan_triangulate(np.array(fi, dtype=np.int32), np.array(fc, dtype=np.int32), vert_offset)
        )
        vert_offset += len(points)
    if not point_chunks:
        return np.zeros((0, 3), dtype=np.float32), np.zeros(0, dtype=np.int32)
    return np.concatenate(point_chunks), np.concatenate(index_chunks)


def voxelize(points, indices, bounds_min, cell_size, res_xyz,
             clearance=None, surface_band=None, max_dist=None, device=None):
    """Return (occupancy uint8 (ri,rj,rk), surface_dist float32 (ri,rj,rk))."""
    ri, rj, rk = int(res_xyz[0]), int(res_xyz[1]), int(res_xyz[2])
    if clearance is None:
        # 0 = beyond the watertight surface band (below), add no extra safety margin
        # here. Safety clearance is added later as a dilation (solver + overlay), so it
        # visibly grows outward.
        clearance = 0.0
    if surface_band is None:
        # Mark cells the surface passes THROUGH so thin shells (door panels, sheet metal)
        # are watertight even when no cell centre lands inside them. A continuous wall's
        # nearest cell-centre is at most 0.5*cell away (axis-aligned worst case; tilted
        # planes are closer), so 0.6*cell guarantees every column crossing the wall has
        # an occupied cell - no straight-line tunnelling - while adding only a thin shell
        # (not the ~1-cell dilation a full half-diagonal band would impose on every solid).
        surface_band = 0.6 * float(cell_size)
    if max_dist is None:
        max_dist = 8.0 * float(cell_size)
    if len(points) == 0 or len(indices) == 0:
        return (np.zeros((ri, rj, rk), dtype=np.uint8),
                np.full((ri, rj, rk), max_dist, dtype=np.float32))

    wp, _voxelize_kernel = _ensure_warp()
    device = _pick_device(wp, device)
    pts = wp.array(np.asarray(points, dtype=np.float32), dtype=wp.vec3, device=device)
    idx = wp.array(np.asarray(indices, dtype=np.int32), dtype=wp.int32, device=device)
    mesh = wp.Mesh(points=pts, indices=idx)
    origin = wp.vec3(float(bounds_min[0]), float(bounds_min[1]), float(bounds_min[2]))
    occ_flat = wp.zeros(ri * rj * rk, dtype=wp.int32, device=device)
    dist_flat = wp.zeros(ri * rj * rk, dtype=wp.float32, device=device)
    wp.launch(
        _voxelize_kernel,
        dim=(ri, rj, rk),
        inputs=[mesh.id, origin, float(cell_size), rj, rk,
                float(clearance), float(surface_band), float(max_dist),
                occ_flat, dist_flat],
        device=device,
    )
    wp.synchronize()
    occ = occ_flat.numpy().reshape((ri, rj, rk)).astype(np.uint8)
    sd = dist_flat.numpy().reshape((ri, rj, rk)).astype(np.float32)
    return occ, sd
