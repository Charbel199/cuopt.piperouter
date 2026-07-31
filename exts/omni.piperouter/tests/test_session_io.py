from pxr import Usd, UsdGeom

from omni.piperouter import scene_ops, session_io


def test_serialize_drops_ui_handles_and_keeps_logic():
    wires = [{
        "key": "wire_0", "name": "pwr", "type_index": 2, "type_id": "pwr_4awg",
        "weights": {"bend": 3.0}, "waypoints": ["/m/w0_wp0"], "wp_slots": [1],
        "wp_counter": 1, "start_head_idx": 0, "end_head_idx": 0, "locked": True,
        "status": "routed", "length_m": 1.2, "cost": 3.4,
        # UI handles, which must not be serialized
        "combo": object(), "name_model": object(), "_swatch": object(),
    }]
    data = session_io.serialize(wires, [], {"resolution": 48}, {"key_counter": 1})
    w = data["wires"][0]
    assert w["name"] == "pwr" and w["type_id"] == "pwr_4awg" and w["wp_slots"] == [1]
    assert "combo" not in w and "name_model" not in w and "_swatch" not in w
    assert data["settings"]["resolution"] == 48 and data["version"] == session_io.SCHEMA_VERSION

    ws, bs, settings, counters = session_io.deserialize(data)
    assert ws[0]["locked"] is True and settings["resolution"] == 48
    assert counters["key_counter"] == 1


def test_session_round_trips_through_usd_stage():
    s = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(s, 0.01)
    payload = session_io.serialize(
        [{"name": "w0", "type_id": "sig_can", "weights": {"em": 2.0}}],
        [{"id": "b0", "name": "b0", "members": ["w0"]}],
        {"resolution": 56}, {"bundle_counter": 1})
    scene_ops.write_session(s, payload)
    assert scene_ops.read_session(s) == payload


def test_read_session_none_when_absent():
    s = Usd.Stage.CreateInMemory()
    UsdGeom.Scope.Define(s, scene_ops.PIPEROUTER_ROOT)
    assert scene_ops.read_session(s) is None
