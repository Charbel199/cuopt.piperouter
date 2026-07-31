import numpy as np

from omni.piperouter import grid_io


def test_frame_matches_solver_framing():
    bmin, cell, res = grid_io.frame_from_bounds([0, 0, 0], [2, 1, 1], 4)
    assert cell == 0.5
    assert res == (4, 2, 2)


def test_dilate_mask_grows_by_iterations():
    m = np.zeros((7, 7, 7), dtype=bool)
    m[3, 3, 3] = True
    assert grid_io.dilate_mask(m, 0).sum() == 1          # no growth
    assert grid_io.dilate_mask(m, 1).sum() == 7          # center + 6 face neighbors
    assert grid_io.dilate_mask(m, 2).sum() > 7           # grows further


def test_npz_is_loadable_by_solver_gridstack(tmp_path):
    # The extension writes the grid; the solver package must read it back unchanged.
    from piperouter_solver.grids import GridStack

    bmin, cell, res = grid_io.frame_from_bounds([0, 0, 0], [1, 1, 1], 5)
    shape = res
    occ = np.zeros(shape, np.uint8)
    occ[2, 2, 2] = 1
    sd = np.full(shape, 0.3, np.float32)
    thermal = np.full(shape, 21.0, np.float32)
    em = np.zeros(shape, np.float32)
    p = tmp_path / "stack.npz"
    grid_io.save_grids(p, bmin, cell, res, occ, sd, thermal, em)

    stack = GridStack.load(p)
    assert stack.frame.res_xyz == res
    assert stack.frame.cell_size == cell
    assert stack.occupancy[2, 2, 2] == 1
    assert np.allclose(stack.thermal, 21.0)
