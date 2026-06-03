import numpy as np
import pytest

from piperouter_service.session_store import FilesystemSessionStore
from piperouter_solver.grids import GridStack
from piperouter_solver.models import GridFrame


def _stack():
    frame = GridFrame(bounds_min=np.zeros(3), cell_size=0.1, res_xyz=(4, 4, 2))
    shape = (4, 4, 2)
    return GridStack(
        frame=frame,
        occupancy=np.zeros(shape, dtype=np.uint8),
        surface_dist=np.full(shape, 1.0, dtype=np.float32),
        thermal=np.full(shape, 20.0, dtype=np.float32),
        em=np.zeros(shape, dtype=np.float32),
    )


def test_save_then_load_roundtrip(tmp_path):
    store = FilesystemSessionStore(tmp_path)
    assert not store.exists("sess1")
    store.save_stack("sess1", _stack())
    assert store.exists("sess1")
    loaded = store.load_stack("sess1")
    assert loaded.frame.res_xyz == (4, 4, 2)


def test_path_traversal_session_id_rejected(tmp_path):
    store = FilesystemSessionStore(tmp_path)
    for bad in ("../evil", "a/b", "..", ""):
        with pytest.raises(ValueError):
            store.grid_path(bad)
