import numpy as np

from omni.piperouter.solver_client import SolverClient

WIRE = {
    "id": "w1", "label": "w1", "kind": "wire", "outer_diameter_mm": 10.0,
    "min_bend_radius_mm": 50.0, "cost_per_m": 1.0, "mass_per_m_kg": 0.1,
    "max_temp_c": 200.0, "em_sensitivity": 0.0, "color": [1.0, 0.0, 0.0],
}


def _write_open_grid(grid_dir, session_id):
    from pathlib import Path

    from piperouter_solver.grids import GridStack
    from piperouter_solver.models import GridFrame
    frame = GridFrame(bounds_min=np.zeros(3), cell_size=0.1, res_xyz=(10, 10, 3))
    shape = (10, 10, 3)
    stack = GridStack(
        frame=frame, occupancy=np.zeros(shape, np.uint8),
        surface_dist=np.full(shape, 5.0, np.float32),
        thermal=np.full(shape, 20.0, np.float32), em=np.zeros(shape, np.float32))
    p = Path(grid_dir) / session_id / "stack.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    stack.save(p)
    return frame


def test_health_reports_backend(solver_server):
    base, _ = solver_server
    h = SolverClient(base).health()
    assert h["status"] == "ok"
    assert h["backend"] in ("scipy", "gpu")


def test_solve_all_over_real_http(solver_server):
    base, grid_dir = solver_server
    frame = _write_open_grid(grid_dir, "sess")
    client = SolverClient(base)
    routes = [{
        "wire": WIRE,
        "start": [float(x) for x in frame.grid_to_world((0, 5, 1))],
        "end": [float(x) for x in frame.grid_to_world((9, 5, 1))],
        "connectivity": 6, "priority": 0,
    }]
    out = client.solve_all("sess", routes)
    assert out["routed"] == 1
    assert len(out["results"][0]["polyline"]) >= 2
