import numpy as np

from piperouter_solver.models import GridFrame


def test_frame_from_bounds_cubic_cells_and_resolution():
    frame = GridFrame.from_bounds([0.0, 0.0, 0.0], [2.0, 1.0, 1.0], resolution=4)
    # longest axis = 2.0, resolution 4 -> cell_size 0.5
    assert frame.cell_size == 0.5
    # res per axis: ceil(extent / cell): x=4, y=2, z=2
    assert frame.res_xyz == (4, 2, 2)


def test_world_to_grid_and_back_roundtrip_center():
    frame = GridFrame.from_bounds([0.0, 0.0, 0.0], [2.0, 2.0, 2.0], resolution=4)
    idx = frame.world_to_grid([0.6, 0.6, 0.6])
    world = frame.grid_to_world(idx)
    # grid_to_world returns the cell center; re-binning must land in the same cell
    assert frame.world_to_grid(world) == idx


def test_world_to_grid_clamps_out_of_bounds():
    frame = GridFrame.from_bounds([0.0, 0.0, 0.0], [1.0, 1.0, 1.0], resolution=2)
    assert frame.world_to_grid([-5.0, -5.0, -5.0]) == (0, 0, 0)
    nx, ny, nz = frame.res_xyz
    assert frame.world_to_grid([5.0, 5.0, 5.0]) == (nx - 1, ny - 1, nz - 1)


from piperouter_solver.grids import GridStack


def test_stack_shapes_match_frame(empty_stack):
    nx, ny, nz = empty_stack.frame.res_xyz
    assert empty_stack.occupancy.shape == (nx, ny, nz)


def test_dilate_occupancy_grows_blocked_region_by_radius(empty_stack):
    s = empty_stack
    s.occupancy[5, 5, 1] = 1
    # radius 0.15 m at cell_size 0.1 m -> ceil(1.5) = 2 cells of growth
    dilated = s.dilate_occupancy(radius_m=0.15)
    assert dilated[5, 5, 1] == 1
    assert dilated[3, 5, 1] == 1  # 2 cells away in x is now blocked
    assert dilated[2, 5, 1] == 0  # 3 cells away is still free


def test_save_load_roundtrip(empty_stack, tmp_path):
    p = tmp_path / "session.npz"
    empty_stack.save(p)
    loaded = GridStack.load(p)
    assert loaded.frame.res_xyz == empty_stack.frame.res_xyz
    assert loaded.frame.cell_size == empty_stack.frame.cell_size
    assert np.array_equal(loaded.occupancy, empty_stack.occupancy)
    assert np.allclose(loaded.thermal, empty_stack.thermal)
