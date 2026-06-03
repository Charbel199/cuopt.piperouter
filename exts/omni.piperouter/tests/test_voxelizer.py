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


def test_no_occupancy_in_gap_between_two_boxes():
    # winding-number sign test must keep the EMPTY gap between two boxes free
    # (the old normal-based test produced false "inside" occupancy there).
    from pxr import Usd, UsdGeom
    s = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(s, "/World")
    scene_ops.author_box_mesh(s, "/World/a", (0.3, 0.5, 0.5), (0.2, 0.4, 0.4))
    scene_ops.author_box_mesh(s, "/World/b", (0.7, 0.5, 0.5), (0.2, 0.4, 0.4))
    prims = scene_ops.list_collidable_meshes(s)
    bmin, bmax = scene_ops.compute_bounds(s, prims)
    pad = (bmax - bmin) * 0.2 + 0.05
    gb, cell, res = grid_io.frame_from_bounds(bmin - pad, bmax + pad, 40)
    pts, idx = voxelizer.collect_meshes(s, prims)
    occ, _ = voxelizer.voxelize(pts, idx, gb, cell, res)

    def cell_at(p):
        return tuple(((np.array(p) - gb) / cell).astype(int))

    assert occ[cell_at((0.5, 0.5, 0.5))] == 0   # empty gap between the boxes -> FREE
    assert occ[cell_at((0.3, 0.5, 0.5))] == 1   # inside box A -> occupied
    assert occ[cell_at((0.7, 0.5, 0.5))] == 1   # inside box B -> occupied


def test_empty_mesh_list_is_all_free():
    gbmin, cell, res = grid_io.frame_from_bounds([0, 0, 0], [1, 1, 1], 8)
    occ, sd = voxelizer.voxelize(np.zeros((0, 3), np.float32), np.zeros(0, np.int32),
                                 gbmin, cell, res)
    assert occ.sum() == 0
    assert occ.shape == res
