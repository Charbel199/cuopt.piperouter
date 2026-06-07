"""Procedural mini engine-bay sample scene + pre-placed wires, so the expert can
hit Route All immediately. pxr-only (headless-testable)."""
from __future__ import annotations

from pxr import UsdGeom

from . import scene_ops

SAMPLE_ROOT = "/World/Sample"

# Author in Omniverse-native CENTIMETRES (metersPerUnit = 0.01) so imported cm assets
# compose 1:1 in the viewport. SCALE / COMPLEX_SCALE are physical sizes in METRES; _M2U
# converts metres -> stage units (cm). The routing pipeline reads metersPerUnit and
# converts back to metres for the solver, so nothing downstream changes.
METERS_PER_UNIT = 0.01
_M2U = 1.0 / METERS_PER_UNIT  # 100 stage units per metre

# Overall size of the sample bay (physical metres). Bump this to make it bigger.
SCALE = 4.0
MARKER_RADIUS = 0.02 * SCALE * _M2U  # draggable markers, sized to the scene (small)

# Base (unit) layout, scaled by SCALE on build. Each descriptor: name (unique route
# id), wire-type id, and world endpoints.
SAMPLE_WIRES = [
    {"name": "power_0", "type_id": "pwr_4awg",  "start": (0.1, 0.2, 0.35), "end": (1.9, 0.2, 0.30)},
    {"name": "signal_1", "type_id": "sig_can",  "start": (0.1, 0.8, 0.45), "end": (1.9, 0.5, 0.20)},
    {"name": "ac_2",     "type_id": "ac_pipe_12", "start": (0.1, 0.5, 0.50), "end": (1.9, 0.85, 0.40)},
]


def _s(t):
    return tuple(float(v) * SCALE * _M2U for v in t)


def _clear(stage, path):
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        stage.RemovePrim(prim.GetPath())


def build_sample_scene(stage):
    """Build /World/Sample + start/end markers; return the wire descriptors (with
    scaled world coords). Idempotent: clears any prior sample, markers, routes first."""
    for p in (SAMPLE_ROOT, scene_ops.MARKERS_SCOPE, scene_ops.ROUTES_SCOPE,
              scene_ops.DEBUG_SCOPE):
        _clear(stage, p)

    UsdGeom.Xform.Define(stage, "/World")
    # Centimetres — Omniverse's native unit — so imported cm assets line up 1:1.
    # Geometry is authored at SCALE metres * 100 units/m; the routing pipeline reads
    # this metersPerUnit and converts back to metres for the solver.
    UsdGeom.SetStageMetersPerUnit(stage, METERS_PER_UNIT)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.Scope.Define(stage, SAMPLE_ROOT)

    box = scene_ops.author_box_mesh
    # ground plane
    box(stage, SAMPLE_ROOT + "/ground", _s((1.0, 0.5, -0.05)), _s((2.2, 1.2, 0.05)), (0.30, 0.30, 0.32))
    # firewall: two boxes leaving a routing gap at mid-y
    box(stage, SAMPLE_ROOT + "/firewall_a", _s((1.0, 0.2, 0.30)), _s((0.12, 0.40, 0.60)), (0.50, 0.50, 0.55))
    box(stage, SAMPLE_ROOT + "/firewall_b", _s((1.0, 0.8, 0.30)), _s((0.12, 0.40, 0.60)), (0.50, 0.50, 0.55))
    # hot engine block
    eb = box(stage, SAMPLE_ROOT + "/engine_block", _s((0.55, 0.50, 0.25)), _s((0.50, 0.50, 0.50)), (0.60, 0.30, 0.20))
    scene_ops.write_tags(eb.GetPrim(), temp_c=120.0)
    # components — one is an EM source
    ca = box(stage, SAMPLE_ROOT + "/comp_a", _s((1.50, 0.80, 0.20)), _s((0.30, 0.25, 0.40)), (0.25, 0.40, 0.60))
    scene_ops.write_tags(ca.GetPrim(), em=0.8)
    box(stage, SAMPLE_ROOT + "/comp_b", _s((1.60, 0.25, 0.20)), _s((0.30, 0.30, 0.40)), (0.30, 0.50, 0.40))

    out = []
    for w in SAMPLE_WIRES:
        start, end = _s(w["start"]), _s(w["end"])
        scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/{w['name']}_start",
                               start, color=(0.1, 0.9, 0.1), radius=MARKER_RADIUS)
        scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/{w['name']}_end",
                               end, color=(0.9, 0.1, 0.1), radius=MARKER_RADIUS)
        out.append({"name": w["name"], "type_id": w["type_id"], "start": start, "end": end})
    return out


# ----------------------------------------------------------------- complex scene
# A much fuller engine-bay + chassis: ~15 obstacles (engine, exhaust, transmission,
# radiator, battery, ECUs, ABS, firewall halves, chassis rails) with thermal/EM
# sources, and 14 wires/pipes spanning the bay using ALL wire + tube types. Built
# under the SAME root as the sample scene, so the two buttons swap cleanly.
COMPLEX_SCALE = 6.0

# (name, center, size, color, temp_c, em) — base units (×COMPLEX_SCALE on build).
COMPLEX_BOXES = [
    ("ground",         (1.50, 1.00, -0.05), (3.40, 2.40, 0.05), (0.30, 0.30, 0.32), None, None),
    ("engine_block",   (0.60, 1.00, 0.30),  (0.60, 0.70, 0.60), (0.55, 0.30, 0.20), 120.0, None),
    ("exhaust",        (0.60, 0.55, 0.18),  (0.55, 0.14, 0.12), (0.45, 0.25, 0.22), 300.0, None),
    ("transmission",   (1.20, 1.00, 0.22),  (0.50, 0.40, 0.40), (0.35, 0.35, 0.40), None, None),
    ("radiator",       (0.15, 1.00, 0.35),  (0.08, 1.20, 0.70), (0.40, 0.45, 0.55), None, None),
    ("coolant_tank",   (0.30, 0.40, 0.40),  (0.20, 0.20, 0.35), (0.20, 0.50, 0.55), None, None),
    ("battery",        (0.30, 1.60, 0.25),  (0.30, 0.35, 0.40), (0.25, 0.45, 0.35), None, None),
    ("alternator",     (0.55, 1.45, 0.45),  (0.22, 0.22, 0.22), (0.30, 0.40, 0.60), None, 0.7),
    ("ecu_main",       (1.95, 1.60, 0.30),  (0.25, 0.35, 0.25), (0.25, 0.40, 0.60), None, 0.9),
    ("ecu_aux",        (2.40, 0.40, 0.30),  (0.20, 0.25, 0.20), (0.25, 0.40, 0.55), None, 0.6),
    ("abs_module",     (1.80, 0.60, 0.20),  (0.20, 0.20, 0.20), (0.40, 0.40, 0.45), None, 0.4),
    ("firewall_a",     (1.55, 0.50, 0.35),  (0.10, 0.90, 0.70), (0.50, 0.50, 0.55), None, None),
    ("firewall_b",     (1.55, 1.50, 0.35),  (0.10, 0.90, 0.70), (0.50, 0.50, 0.55), None, None),
    ("chassis_rail_l", (1.50, 0.25, 0.10),  (3.00, 0.12, 0.12), (0.40, 0.40, 0.42), None, None),
    ("chassis_rail_r", (1.50, 1.75, 0.10),  (3.00, 0.12, 0.12), (0.40, 0.40, 0.42), None, None),
]

# (name, wire-type id, start, end) — base units.
COMPLEX_WIRES = [
    ("batt_pwr_0",     "pwr_4awg",        (0.30, 1.60, 0.45), (1.95, 1.58, 0.45)),
    ("starter_pwr_1",  "pwr_4awg",        (0.40, 1.25, 0.65), (1.00, 1.05, 0.65)),
    ("alt_pwr_2",      "pwr_4awg",        (0.55, 1.45, 0.58), (0.32, 1.60, 0.50)),
    ("ecu_aux_pwr_3",  "pwr_4awg",        (0.32, 1.55, 0.42), (2.40, 0.42, 0.35)),
    ("can_eng_4",      "sig_can",         (0.62, 1.05, 0.55), (1.92, 1.55, 0.42)),
    ("can_abs_5",      "sig_can",         (1.80, 0.62, 0.34), (1.95, 1.58, 0.42)),
    ("can_exh_6",      "sig_can",         (0.65, 0.40, 0.40), (2.05, 1.45, 0.55)),
    ("can_cabin_7",    "sig_can",         (1.62, 1.00, 0.50), (2.60, 1.00, 0.40)),
    ("ac_8",           "ac_pipe_12",      (0.20, 0.42, 0.55), (2.00, 0.50, 0.40)),
    ("coolant_in_9",   "coolant_hose_16", (0.30, 1.00, 0.50), (0.55, 1.00, 0.70)),
    ("coolant_out_10", "coolant_hose_16", (0.60, 1.20, 0.70), (0.30, 1.20, 0.55)),
    ("heater_11",      "coolant_hose_8",  (0.62, 1.00, 0.56), (2.00, 1.00, 0.45)),
    ("brake_fl_12",    "brake_line_6",    (1.80, 0.60, 0.32), (0.20, 0.30, 0.18)),
    ("brake_rl_13",    "brake_line_6",    (1.82, 0.62, 0.32), (2.60, 1.70, 0.18)),
]


def build_complex_scene(stage):
    """Build a large engine-bay + chassis under SAMPLE_ROOT with many obstacles,
    thermal/EM sources, and 14 wires/pipes (all types). Returns wire descriptors
    (scaled world coords). Idempotent and replaces any prior sample/complex scene."""
    for p in (SAMPLE_ROOT, scene_ops.MARKERS_SCOPE, scene_ops.ROUTES_SCOPE,
              scene_ops.DEBUG_SCOPE):
        _clear(stage, p)

    UsdGeom.Xform.Define(stage, "/World")
    # Centimetres (see build_sample_scene) — native Omniverse unit.
    UsdGeom.SetStageMetersPerUnit(stage, METERS_PER_UNIT)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.Scope.Define(stage, SAMPLE_ROOT)

    s = COMPLEX_SCALE

    def sc(t):
        return tuple(float(v) * s * _M2U for v in t)

    for name, center, size, color, temp_c, em in COMPLEX_BOXES:
        mesh = scene_ops.author_box_mesh(stage, f"{SAMPLE_ROOT}/{name}", sc(center), sc(size), color)
        if temp_c is not None or em is not None:
            scene_ops.write_tags(mesh.GetPrim(), temp_c=temp_c, em=em)

    radius = 0.02 * s * _M2U
    out = []
    for name, type_id, start, end in COMPLEX_WIRES:
        ws, we = sc(start), sc(end)
        scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/{name}_start",
                               ws, color=(0.1, 0.9, 0.1), radius=radius)
        scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/{name}_end",
                               we, color=(0.9, 0.1, 0.1), radius=radius)
        out.append({"name": name, "type_id": type_id, "start": ws, "end": we})
    return out
