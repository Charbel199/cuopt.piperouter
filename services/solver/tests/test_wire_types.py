import json
from pathlib import Path

from piperouter_solver.wire_types import load_wire_types

REPO_DEFAULT = Path(__file__).resolve().parents[1] / "wire_types.json"


def test_loads_default_library_keyed_by_id():
    types = load_wire_types(REPO_DEFAULT)
    assert "pwr_4awg" in types
    wt = types["pwr_4awg"]
    assert wt.kind == "wire"
    assert wt.outer_diameter_mm == 11.0
    assert wt.max_temp_c == 105
    assert len(wt.color) == 3


def test_radius_helper_is_half_diameter_in_meters():
    types = load_wire_types(REPO_DEFAULT)
    wt = types["ac_pipe_12"]
    # 12 mm outer diameter -> 0.006 m radius
    assert abs(wt.radius_m - 0.006) < 1e-9


def test_loads_inner_diameter_for_tubes(tmp_path):
    p = tmp_path / "wt.json"
    p.write_text(json.dumps({"types": [
        {"id": "cable", "label": "Cable", "kind": "wire",
         "outer_diameter_mm": 10.0, "min_bend_radius_mm": 50.0,
         "cost_per_m": 1.0, "mass_per_m_kg": 0.1, "max_temp_c": 100.0,
         "em_sensitivity": 0.2, "color": [1, 0, 0]},
        {"id": "hose", "label": "Hose", "kind": "pipe",
         "outer_diameter_mm": 12.0, "inner_diameter_mm": 8.0,
         "min_bend_radius_mm": 90.0, "cost_per_m": 2.0, "mass_per_m_kg": 0.3,
         "max_temp_c": 120.0, "em_sensitivity": 0.0, "color": [0, 0, 1]},
    ]}))
    lib = load_wire_types(p)
    assert lib["cable"].inner_diameter_mm == 0.0   # default for solid wire
    assert lib["hose"].inner_diameter_mm == 8.0
