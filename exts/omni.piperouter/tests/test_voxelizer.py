import numpy as np

from omni.piperouter import grid_io, scene_ops, voxelizer


def test_voxelize_marks_cube_and_clears_far(cube_stage):
    prims = scene_ops.list_collidable_meshes(cube_stage)
    bmin, bmax = scene_ops.compute_bounds(cube_stage, prims)
    pad = (bmax - bmin) * 0.5 + 0.1
    gbmin, cell, res = grid_io.frame_from_bounds(bmin - pad, bmax + pad, 24)
    pts, idx = voxelizer.collect_meshes(cube_stage, prims)
    occ, sd = voxelizer.voxelize(pts, idx, gbmin, cell, res)

    assert occ.sum() > 0           # the cube produced occupied cells
    assert occ[0, 0, 0] == 0       # far corner is free
    # surface distance is smaller near the cube centre than at the far corner
    ci = tuple(((np.array([0.5, 0.5, 0.5]) - gbmin) / cell).astype(int))
    assert sd[ci] < sd[0, 0, 0]


def test_empty_mesh_list_is_all_free():
    gbmin, cell, res = grid_io.frame_from_bounds([0, 0, 0], [1, 1, 1], 8)
    occ, sd = voxelizer.voxelize(np.zeros((0, 3), np.float32), np.zeros(0, np.int32),
                                 gbmin, cell, res)
    assert occ.sum() == 0
    assert occ.shape == res
