"""Procedural mini engine-bay sample scene + pre-placed wires, so the expert can
hit Route All immediately. pxr-only (headless-testable)."""
from __future__ import annotations

from pxr import UsdGeom

from . import scene_ops

SAMPLE_ROOT = "/World/Sample"

# Overall size of the sample bay (world units). Bump this to make it bigger.
SCALE = 4.0
MARKER_RADIUS = 0.04 * SCALE  # draggable markers, sized to the scene

# Base (unit) layout, scaled by SCALE on build. Each descriptor: name (unique route
# id), wire-type id, and world endpoints.
SAMPLE_WIRES = [
    {"name": "power_0", "type_id": "pwr_4awg",  "start": (0.1, 0.2, 0.35), "end": (1.9, 0.2, 0.30)},
    {"name": "signal_1", "type_id": "sig_can",  "start": (0.1, 0.8, 0.45), "end": (1.9, 0.5, 0.20)},
    {"name": "ac_2",     "type_id": "ac_pipe_12", "start": (0.1, 0.5, 0.50), "end": (1.9, 0.85, 0.40)},
]


def _s(t):
    return tuple(float(v) * SCALE for v in t)


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
