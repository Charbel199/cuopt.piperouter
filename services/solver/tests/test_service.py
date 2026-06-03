import numpy as np
import pytest
from fastapi.testclient import TestClient

from piperouter_service.app import app_factory
from piperouter_service.session_store import FilesystemSessionStore
from piperouter_solver.grids import GridStack
from piperouter_solver.models import GridFrame

WIRE = {
    "id": "w1", "label": "w1", "kind": "wire", "outer_diameter_mm": 10.0,
    "min_bend_radius_mm": 50.0, "cost_per_m": 1.0, "mass_per_m_kg": 0.1,
    "max_temp_c": 200.0, "em_sensitivity": 0.0, "color": [1.0, 0.0, 0.0],
}


def _stack(blocked_wall=False):
    frame = GridFrame(bounds_min=np.zeros(3), cell_size=0.1, res_xyz=(10, 10, 3))
    shape = (10, 10, 3)
    occ = np.zeros(shape, dtype=np.uint8)
    if blocked_wall:
        occ[5, :, :] = 1
    return GridStack(
        frame=frame, occupancy=occ,
        surface_dist=np.full(shape, 5.0, dtype=np.float32),
        thermal=np.full(shape, 20.0, dtype=np.float32),
        em=np.zeros(shape, dtype=np.float32),
    )


@pytest.fixture
def client(tmp_path):
    from pathlib import Path
    store = FilesystemSessionStore(tmp_path)
    wt = Path(__file__).resolve().parents[1] / "wire_types.json"
    return TestClient(app_factory(store, wt)), store


def _world(frame, ijk):
    return [float(x) for x in frame.grid_to_world(ijk)]


def test_health_ok(client):
    c, _ = client
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["backend"] in ("scipy", "gpu")


def test_wire_types_lists_defaults(client):
    c, _ = client
    ids = [t["id"] for t in c.get("/wire_types").json()["types"]]
    assert "pwr_4awg" in ids and "ac_pipe_12" in ids


def test_solve_open_grid_routes(client):
    c, store = client
    s = _stack()
    store.save_stack("sess", s)
    body = {
        "session_id": "sess",
        "route": {"wire": WIRE, "start": _world(s.frame, (0, 5, 1)),
                  "end": _world(s.frame, (9, 5, 1)), "connectivity": 6},
    }
    r = c.post("/solve", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["status"] == "routed"
    assert len(out["polyline"]) >= 2
    assert out["length_m"] > 0


def test_solve_walled_grid_no_path(client):
    c, store = client
    s = _stack(blocked_wall=True)
    store.save_stack("sess", s)
    body = {
        "session_id": "sess",
        "route": {"wire": WIRE, "start": _world(s.frame, (0, 5, 1)),
                  "end": _world(s.frame, (9, 5, 1)), "connectivity": 26},
    }
    out = c.post("/solve", json=body).json()
    assert out["status"] == "no_path"


def test_solve_missing_session_404(client):
    c, _ = client
    body = {
        "session_id": "nope",
        "route": {"wire": WIRE, "start": [0, 0, 0], "end": [1, 1, 1]},
    }
    assert c.post("/solve", json=body).status_code == 404


def test_solve_with_locked_route_forces_detour(client):
    c, store = client
    s = _stack()  # open 10x10x3
    store.save_stack("sess", s)
    # lock a wall of tube across x=5 spanning y=1..9 (leave the y=0 row open)
    locked_poly = [_world(s.frame, (5, j, 1)) for j in range(1, 10)]
    body = {
        "session_id": "sess",
        "route": {"wire": WIRE, "start": _world(s.frame, (0, 5, 1)),
                  "end": _world(s.frame, (9, 5, 1)), "connectivity": 26},
        "locked_routes": [{"polyline": locked_poly, "outer_diameter_mm": 20.0}],
    }
    out = c.post("/solve", json=body).json()
    assert out["status"] == "routed"
    # the routed polyline must avoid every locked cell (it may detour in y or z)
    locked_cells = {(5, j, 1) for j in range(1, 10)}
    route_cells = {s.frame.world_to_grid(p) for p in out["polyline"]}
    assert route_cells.isdisjoint(locked_cells)
    # sanity: the unlocked straight route WOULD have used (5,5,1)
    assert (5, 5, 1) in locked_cells


def test_solve_all_two_offset_wires(client):
    c, store = client
    s = _stack()
    store.save_stack("sess", s)
    routes = [
        {"wire": WIRE, "start": _world(s.frame, (0, 5, 1)),
         "end": _world(s.frame, (9, 5, 1)), "connectivity": 26, "priority": 0},
        {"wire": WIRE, "start": _world(s.frame, (0, 5, 2)),
         "end": _world(s.frame, (9, 5, 2)), "connectivity": 26, "priority": 1},
    ]
    out = c.post("/solve_all", json={"session_id": "sess", "routes": routes}).json()
    assert out["routed"] == 2
    assert out["no_path"] == 0
    assert len(out["results"]) == 2
