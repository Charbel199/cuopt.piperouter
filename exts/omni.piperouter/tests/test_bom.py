from omni.piperouter import bom


def _rows():
    return [
        {"wire_id": "power_0", "status": "routed", "length_m": 2.0, "cost": 12.8, "mass": 0.42},
        {"wire_id": "signal_1", "status": "no_path", "length_m": 0.0, "cost": 0.0, "mass": 0.0},
        {"wire_id": "ac_2", "status": "routed", "length_m": 1.5, "cost": 13.65, "mass": 0.51},
    ]


def test_totals_only_count_routed():
    s = bom.summarize(_rows())
    assert s["n_routed"] == 2
    assert s["n_no_path"] == 1
    assert round(s["total_length"], 2) == 3.5
    assert round(s["total_cost"], 2) == 26.45
    assert round(s["total_mass"], 2) == 0.93


def test_type_labels_join_by_wire_id():
    s = bom.summarize(_rows(), {"power_0": "Power cable 4 AWG", "ac_2": "AC line 12 mm"})
    by_id = {r["wire_id"]: r for r in s["rows"]}
    assert by_id["power_0"]["type"] == "Power cable 4 AWG"
    assert by_id["signal_1"]["type"] == ""   # missing label -> blank


def test_no_path_row_present_but_zeroed():
    s = bom.summarize(_rows())
    sig = next(r for r in s["rows"] if r["wire_id"] == "signal_1")
    assert sig["status"] == "no_path"
    assert sig["cost"] == 0.0 and sig["length_m"] == 0.0


def test_reason_is_carried_through():
    rows = [{"wire_id": "x", "status": "no_path", "reason": "Start is too hot"}]
    s = bom.summarize(rows)
    assert s["rows"][0]["reason"] == "Start is too hot"


def test_empty():
    s = bom.summarize([])
    assert s["rows"] == [] and s["total_cost"] == 0.0 and s["n_routed"] == 0
