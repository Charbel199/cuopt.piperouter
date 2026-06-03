import numpy as np
import pytest

from piperouter_solver.grids import GridStack
from piperouter_solver.models import GridFrame


@pytest.fixture
def empty_stack():
    """A 10x10x3 open grid, 0.1 m cells, no obstacles/heat/em."""
    frame = GridFrame(
        bounds_min=np.zeros(3),
        cell_size=0.1,
        res_xyz=(10, 10, 3),
    )
    shape = frame.res_xyz
    return GridStack(
        frame=frame,
        occupancy=np.zeros(shape, dtype=np.uint8),
        surface_dist=np.full(shape, 5.0, dtype=np.float32),  # far from any surface
        thermal=np.full(shape, 20.0, dtype=np.float32),       # ambient
        em=np.zeros(shape, dtype=np.float32),
    )


@pytest.fixture
def wall_stack(empty_stack):
    """Open grid with a wall at x=5 spanning all y, leaving a gap at y=0."""
    s = empty_stack
    s.occupancy[5, 1:, :] = 1  # wall blocks x=5 except the y=0 row
    return s
