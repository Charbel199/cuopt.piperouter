import numpy as np
from omni.piperouter import bundles


# ---------- bundle_diameter ----------

def test_bundle_diameter_single():
    # single wire OD 10 mm -> sqrt(10^2) = 10
    assert abs(bundles.bundle_diameter([10.0]) - 10.0) < 1e-6


def test_bundle_diameter_two():
    # two wires 10 mm -> sqrt(10^2 + 10^2) = 14.14...
    expected = (10.0**2 + 10.0**2) ** 0.5
    assert abs(bundles.bundle_diameter([10.0, 10.0]) - expected) < 1e-6


def test_bundle_diameter_empty_raises():
    try:
        bundles.bundle_diameter([])
        assert False, "should raise"
    except ValueError:
        pass


# ---------- stitch_polylines ----------

def test_stitch_connects_branch_trunk_branch():
    start_branch = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    trunk        = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    end_branch   = [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    result = bundles.stitch_polylines(start_branch, trunk, end_branch)
    assert result[0] == [0.0, 0.0, 0.0]
    assert result[-1] == [4.0, 0.0, 0.0]
    assert len(result) == 5   # 2+3+2 = 7 minus 2 deduped junctions = 5


def test_stitch_deduplicates_junction_points():
    sb = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    tr = [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    eb = [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    r = bundles.stitch_polylines(sb, tr, eb)
    for a, b in zip(r[:-1], r[1:]):
        assert a != b, f"duplicate point {a}"


# ---------- validate_members ----------

def test_validate_accepts_same_kind():
    wires = [{"name": "a", "bundle_id": "B"}, {"name": "b", "bundle_id": "B"}]
    type_map = {"a": {"kind": "wire"}, "b": {"kind": "wire"}}
    ok, err = bundles.validate_members(wires, type_map)
    assert ok and err == ""


def test_validate_rejects_mixed_kind():
    wires = [{"name": "a", "bundle_id": "B"}, {"name": "c", "bundle_id": "B"}]
    type_map = {"a": {"kind": "wire"}, "c": {"kind": "pipe"}}
    ok, err = bundles.validate_members(wires, type_map)
    assert not ok and "kind" in err


def test_validate_rejects_fewer_than_two():
    wires = [{"name": "a", "bundle_id": "B"}]
    type_map = {"a": {"kind": "wire"}}
    ok, err = bundles.validate_members(wires, type_map)
    assert not ok and "2" in err


# ---------- trunk_spec ----------

def test_trunk_spec_uses_thickest_member():
    specs = [
        {"outer_diameter_mm": 4.0, "min_bend_radius_mm": 20.0, "cost_per_m": 2.0,
         "mass_per_m_kg": 0.04, "max_temp_c": 90.0, "em_sensitivity": 0.9,
         "id": "sig", "label": "sig", "kind": "wire", "inner_diameter_mm": 0.0,
         "color": [1, 1, 0]},
        {"outer_diameter_mm": 11.0, "min_bend_radius_mm": 55.0, "cost_per_m": 6.4,
         "mass_per_m_kg": 0.21, "max_temp_c": 105.0, "em_sensitivity": 0.2,
         "id": "pwr", "label": "pwr", "kind": "wire", "inner_diameter_mm": 0.0,
         "color": [1, 0, 0]},
    ]
    ts = bundles.trunk_spec(specs, bundle_id="B1")
    assert ts["outer_diameter_mm"] == bundles.bundle_diameter(
        [s["outer_diameter_mm"] for s in specs])
    assert ts["min_bend_radius_mm"] == 55.0   # max of members
    assert ts["max_temp_c"] == 90.0           # min of members (strictest)
    assert ts["id"] == "bundle_B1_trunk"
