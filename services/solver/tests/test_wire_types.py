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
