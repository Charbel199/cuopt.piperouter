import numpy as np

from omni.piperouter import octree_viz


def test_leaves_partition_free_space_and_adapt():
    occ = np.zeros((16, 16, 16), dtype=bool)
    occ[7:9, :, :] = True                       # a solid wall
    leaves, leaf_of = octree_viz.build_octree(occ)
    # Every free cell lands in exactly one leaf; blocked cells land in none.
    assert (leaf_of[~occ] >= 0).all()
    assert (leaf_of[occ] == -1).all()
    # Adaptive: big leaves in open air, 1-cell leaves hugging the wall.
    sizes = [max(l[1] - l[0], l[3] - l[2], l[5] - l[4]) for l in leaves]
    assert max(sizes) >= 4
    assert min(sizes) == 1


def test_corridor_band_connects_and_is_a_small_fraction():
    occ = np.zeros((16, 16, 16), dtype=bool)
    occ[7:9, 0:14, :] = True                    # wall with a gap at high y
    leaves, leaf_of = octree_viz.build_octree(occ)
    corr, band = octree_viz.corridor_and_band(occ, leaves, leaf_of,
                                              (2, 7, 7), (13, 7, 7), band=2)
    assert corr and band is not None
    for c in corr:                              # band contains the whole corridor
        assert band[c]
    assert not band.all()                       # but stays a fraction of the grid
    assert band.sum() < occ.size


def test_endpoint_in_blocked_leaf_returns_none():
    occ = np.zeros((12, 12, 12), dtype=bool)
    occ[5:7, 5:7, 5:7] = True
    leaves, leaf_of = octree_viz.build_octree(occ)
    corr, band = octree_viz.corridor_and_band(occ, leaves, leaf_of,
                                              (5, 5, 5), (1, 1, 1), band=2)  # start blocked
    assert corr is None and band is None
