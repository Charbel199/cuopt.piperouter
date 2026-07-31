# Fixtures shared by the extension tests.
#
# The suite covers the extension's Kit-independent modules (USD scene ops, voxelizer,
# grid IO, solver client) and imports nothing from omni.kit or omni.ui, so it runs
# without a Kit runtime. Dependencies are the real thing rather than mocks: an
# in-memory Usd.Stage and the actual solver service over HTTP, so a schema or wire
# format change on either side shows up here as a failure.

import socket
import threading
import time
import urllib.request
from pathlib import Path

import pytest
from pxr import Gf, Usd, UsdGeom

REPO_ROOT = Path(__file__).resolve().parents[3]


def make_cube_mesh_stage(center=(0.5, 0.5, 0.5), half=0.2):
    """In-memory stage with a single box Mesh (outward-wound quads)."""
    stage = Usd.Stage.CreateInMemory()
    # Pin meters so the units-aware pipeline (geometry ×mpu) is an identity in tests.
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Obstacle")
    cx, cy, cz = center
    h = half
    pts = [
        (cx - h, cy - h, cz - h), (cx + h, cy - h, cz - h),
        (cx + h, cy + h, cz - h), (cx - h, cy + h, cz - h),
        (cx - h, cy - h, cz + h), (cx + h, cy - h, cz + h),
        (cx + h, cy + h, cz + h), (cx - h, cy + h, cz + h),
    ]
    mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in pts])
    mesh.GetFaceVertexCountsAttr().Set([4, 4, 4, 4, 4, 4])
    mesh.GetFaceVertexIndicesAttr().Set([
        0, 3, 2, 1,   # -z
        4, 5, 6, 7,   # +z
        0, 1, 5, 4,   # -y
        2, 3, 7, 6,   # +y
        1, 2, 6, 5,   # +x
        0, 4, 7, 3,   # -x
    ])
    return stage, mesh


@pytest.fixture
def cube_stage():
    stage, _ = make_cube_mesh_stage()
    return stage


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def solver_server(tmp_path):
    """Run the solver service on a background thread; yield (base_url, grid_dir).

    The extension writes grid .npz files into grid_dir and the service reads them
    back from the same path, so both sides must be given one shared directory.
    """
    import uvicorn
    from piperouter_service.app import app_factory
    from piperouter_service.session_store import FilesystemSessionStore

    grid_dir = tmp_path / "grids"
    grid_dir.mkdir()
    wt = REPO_ROOT / "services" / "solver" / "wire_types.json"
    app = app_factory(FilesystemSessionStore(grid_dir), wt)
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port,
                                           log_level="warning"))
    th = threading.Thread(target=server.run, daemon=True)
    th.start()
    base = f"http://127.0.0.1:{port}"
    for _ in range(100):   # wait up to ~10 s for the port to start answering
        try:
            urllib.request.urlopen(base + "/health", timeout=1)
            break
        except Exception:
            time.sleep(0.1)
    yield base, str(grid_dir)
    server.should_exit = True
    th.join(timeout=5)
