"""Generate USD test scenes, each built to exercise one constraint in isolation.

Each scene ships its own wires and session data, so the workflow is: open the .usd in
Kit, open the PipeRouter panel, press ROUTE ALL. The geometry is deliberately schematic:
plain boxes with known dimensions, so a wrong route is obvious by eye and the numbers can
be checked by hand.

    python3 exts/omni.piperouter/scenes/make_test_scenes.py [outdir]

Scenes:
  test_bend.usd       U-channels of decreasing width, plus one corner routed with three
                      stiffnesses. Exposes whether the bend rating is respected.
  test_clearance.usd  A wall of slots from generous to narrower than the cable.
  test_thermal_em.usd A hot zone and an emitter, each with a clean detour available.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "omni"))

from pxr import Usd, UsdGeom                                            # noqa: E402
from piperouter import scene_ops, session_io, wire_library              # noqa: E402

M2U = 100.0                      # stage authored in centimetres, Omniverse's default
ROOT = "/World/TestScene"
WEIGHTS = ("surface", "bend", "thermal", "em", "smoothing")


def _lib_index():
    """wire-type id -> its index in the library, which the panel stores per wire."""
    types = wire_library.load_wire_library()
    return {t["id"]: i for i, t in enumerate(types)}, types


def _u(v):
    """Metres to stage units."""
    return tuple(float(x) * M2U for x in v)


def _box(stage, name, center, size, color, temp_c=None, em=None):
    m = scene_ops.author_box_mesh(stage, f"{ROOT}/{name}", _u(center), _u(size), color)
    if temp_c is not None or em is not None:
        scene_ops.write_tags(m.GetPrim(), temp_c=temp_c, em=em)
    return m


def _new_stage(path):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0 / M2U)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Scope.Define(stage, ROOT)
    return stage


def _safe(name):
    """USD prim names allow only alphanumerics and underscore."""
    out = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    return out if out[0].isalpha() or out[0] == "_" else "w_" + out


def _finish(stage, wires, path, notes, resolution=250):
    """Author markers, write the session, and save."""
    idx, _types = _lib_index()
    out = []
    for w in wires:
        key = _safe(w["name"])
        scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/{key}_start",
                               _u(w["start"]), (0.1, 0.9, 0.1), 0.9)
        scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/{key}_end",
                               _u(w["end"]), (0.9, 0.1, 0.1), 0.9)
        out.append({
            "key": key, "name": key, "type_id": w["type_id"],
            "type_index": idx.get(w["type_id"], 0),
            "weights": {k: float(w.get("weights", {}).get(k, 1.0)) for k in WEIGHTS},
            "waypoints": [], "wp_slots": [], "wp_counter": 0,
            "start_head_idx": 0, "end_head_idx": 0, "locked": False,
        })
    scene_ops.write_session(stage, session_io.serialize(
        out, [], {"resolution": resolution, "clearance_mm": 0.0,
                  "global_algo": "octree_lattice", "local_algo": "fibre"},
        {"wire": len(out)}))
    stage.GetRootLayer().Save()
    print(f"  {path.name:<22} {len(out):>2} wires   {notes}")


# ---------------------------------------------------------------- bend gauntlet
# Channel inner width W with a central divider of thickness d: each lane is (W-d)/2
# wide, so the U-turn around the divider tip has a centreline radius of
#     R = d/2 + (W-d)/4
# Those radii are the point of the scene, so they go in the wire names.
BEND_WIDTHS = [0.60, 0.40, 0.28, 0.16, 0.09]
BEND_DIVIDER = 0.010


def _uturn_radius_mm(width, divider=BEND_DIVIDER):
    return 1000.0 * (divider / 2.0 + (width - divider) / 4.0)


def build_bend(outdir):
    """U-channels whose available turn radius shrinks below the pipe's rating.

    Each channel is sealed at both ends. A central divider runs from the near cap and
    stops short of the far one, leaving a pocket. A cable starting in one lane can only
    reach the other lane by running the length of the channel and turning 180 degrees
    around the divider tip, so the turn is unavoidable and its radius is set purely by
    geometry:  R = divider/2 + lane/2, printed in each wire name. The pipe is rated for
    120 mm, so only the widest channel can hold a legal turn.
    """
    stage = _new_stage(outdir / "test_bend.usd")
    _box(stage, "floor", (2.10, 1.05, -0.03), (4.40, 2.30, 0.06), (0.30, 0.30, 0.32))

    t, h, zc = 0.05, 0.34, 0.17
    wires, x = [], 0.45
    for wdt in BEND_WIDTHS:
        r = _uturn_radius_mm(wdt)
        depth = max(0.55, 0.9 * wdt + 0.45)          # pocket deep enough for the turn
        y0, y1 = 0.45, 0.45 + depth
        yc = (y0 + y1) / 2
        _box(stage, f"ch{int(wdt*1000)}_l", (x - wdt/2 - t/2, yc, zc), (t, depth, h),
             (0.48, 0.48, 0.54))
        _box(stage, f"ch{int(wdt*1000)}_r", (x + wdt/2 + t/2, yc, zc), (t, depth, h),
             (0.48, 0.48, 0.54))
        _box(stage, f"ch{int(wdt*1000)}_far", (x, y1 + t/2, zc), (wdt + 2*t, t, h),
             (0.48, 0.48, 0.54))
        # near cap seals the mouth, so going around the outside is not an option
        _box(stage, f"ch{int(wdt*1000)}_near", (x, y0 - t/2, zc), (wdt + 2*t, t, h),
             (0.48, 0.48, 0.54))
        div_len = depth - max(0.60 * wdt, 0.12)      # pocket left at the far end
        _box(stage, f"ch{int(wdt*1000)}_div", (x, y0 + div_len/2, zc),
             (BEND_DIVIDER, div_len, h), (0.62, 0.42, 0.30))
        lane = (wdt - BEND_DIVIDER) / 2.0
        off = BEND_DIVIDER/2 + lane/2
        wires.append({"name": f"uturn_R{int(round(r))}mm", "type_id": "cooling_main_24",
                      "start": (x - off, y0 + 0.06, zc),
                      "end":   (x + off, y0 + 0.06, zc)})
        x += wdt + 0.34

    # Does the bend slider help? Same geometry, same pipe, weight 1 against weight 8.
    wdt = BEND_WIDTHS[0]
    depth = max(0.55, 0.9 * wdt + 0.45)
    y0 = 0.45
    lane = (wdt - BEND_DIVIDER) / 2.0
    off = BEND_DIVIDER / 2 + lane / 2
    wires[0]["name"] = f"uturn_R{int(round(_uturn_radius_mm(wdt)))}mm_bend1"
    wires.append({"name": f"uturn_R{int(round(_uturn_radius_mm(wdt)))}mm_bend8",
                  "type_id": "cooling_main_24", "weights": {"bend": 8.0},
                  "start": (0.45 - off, y0 + 0.14, zc), "end": (0.45 + off, y0 + 0.14, zc)})

    # One right-angle corner, three stiffnesses. A stiffer pipe should cut it wider.
    _box(stage, "corner_block", (3.55, 1.55, 0.17), (0.60, 0.60, 0.34), (0.40, 0.42, 0.48))
    for n, tid in enumerate(("sig_can", "brake_line_6", "cooling_main_24")):
        z = 0.09 + n * 0.10
        wires.append({"name": f"corner_{tid}", "type_id": tid,
                      "start": (3.10, 1.10, z), "end": (4.00, 2.00, z)})

    _finish(stage, wires, outdir / "test_bend.usd",
            "sealed U-turns of radius "
            + "/".join(str(int(round(_uturn_radius_mm(w)))) for w in BEND_WIDTHS)
            + " mm vs pipe Rmin 120 mm", resolution=400)


# ---------------------------------------------------------------- clearance slots
def build_clearance(outdir):
    """One wall, slots from generous down to narrower than the cable itself."""
    stage = _new_stage(outdir / "test_clearance.usd")
    _box(stage, "floor", (1.50, 0.60, -0.03), (3.20, 1.40, 0.06), (0.30, 0.30, 0.32))

    gaps = [0.150, 0.080, 0.040, 0.024, 0.014]
    wall_x, wall_t, wall_h, zc = 1.50, 0.10, 0.40, 0.20
    y = 0.15
    wires, edges = [], []
    for g in gaps:
        edges.append((y, y + g))
        y += g + 0.16
    span_end = y
    # wall segments between the slots
    prev = -0.10
    for n, (a, b) in enumerate(edges):
        seg_c = (prev + a) / 2
        _box(stage, f"wall{n}", (wall_x, seg_c, zc), (wall_t, max(a - prev, 0.01), wall_h),
             (0.50, 0.50, 0.56))
        prev = b
    _box(stage, "wall_last", (wall_x, (prev + span_end + 0.10) / 2, zc),
         (wall_t, span_end + 0.10 - prev, wall_h), (0.50, 0.50, 0.56))

    for (a, b), g in zip(edges, gaps):
        mid = (a + b) / 2
        wires.append({"name": f"slot_{int(g*1000)}mm", "type_id": "sig_can",
                      "start": (0.80, mid, zc), "end": (2.25, mid, zc)})
    _finish(stage, wires, outdir / "test_clearance.usd",
            "slots 150/80/40/24/14mm; sig_can OD 4mm. raise clearance to close them in turn",
            resolution=400)


# ---------------------------------------------------------------- thermal and EM
def build_thermal_em(outdir):
    """A hot manifold and an emitter, far apart in a bay big enough to route around them.

    Every tagged prim radiates over its own size plus a one metre margin, so the bay has
    to be large or the field swamps it and nothing routes. Here the two sources sit 2.6 m
    apart in a 6 x 5 m bay with no walls at all: the only thing bending a route is the
    field. Wires come in pairs sharing endpoints but differing in temperature rating or
    EM sensitivity, so the gap between their lengths is the constraint being measured.
    """
    stage = _new_stage(outdir / "test_thermal_em.usd")
    _box(stage, "floor", (3.00, 2.50, -0.04), (6.20, 5.20, 0.08), (0.30, 0.30, 0.32))
    _box(stage, "exhaust", (3.00, 1.20, 0.22), (0.34, 0.20, 0.30), (0.45, 0.24, 0.20),
         temp_c=300.0)
    _box(stage, "inverter", (3.00, 3.80, 0.24), (0.40, 0.40, 0.44), (0.28, 0.40, 0.62),
         em=1.0)

    wires = [
        # straight across the manifold, three temperature ratings
        # thermal weight 0 on purpose: with the soft cost switched off, the only thing
        # separating these three is the hard melt cutoff at their own rating, so the
        # detour lengths should order themselves 90C > 135C > 160C.
        {"name": "hot_sig_can_90C", "type_id": "sig_can", "weights": {"thermal": 0.0},
         "start": (0.40, 1.20, 0.24), "end": (5.60, 1.20, 0.24)},
        {"name": "hot_ac_pipe_135C", "type_id": "ac_pipe_12", "weights": {"thermal": 0.0},
         "start": (0.40, 1.26, 0.30), "end": (5.60, 1.26, 0.30)},
        {"name": "hot_brake_160C", "type_id": "brake_line_6", "weights": {"thermal": 0.0},
         "start": (0.40, 1.14, 0.18), "end": (5.60, 1.14, 0.18)},
        # straight past the emitter, sensitive wire against an indifferent pipe
        {"name": "em_sig_can_sens09", "type_id": "sig_can",
         "start": (0.40, 3.80, 0.26), "end": (5.60, 3.80, 0.26)},
        {"name": "em_brake_sens00", "type_id": "brake_line_6",
         "start": (0.40, 3.74, 0.18), "end": (5.60, 3.74, 0.18)},
    ]
    _finish(stage, wires, outdir / "test_thermal_em.usd",
            "6x5 m bay, exhaust 300C and emitter 2.6 m apart: detour length is the measurement",
            resolution=200)


def main():
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)
    for f in ("test_bend.usd", "test_clearance.usd", "test_thermal_em.usd"):
        p = outdir / f
        if p.exists():
            p.unlink()
    print(f"writing to {outdir}")
    build_bend(outdir)
    build_clearance(outdir)
    build_thermal_em(outdir)
    print("\nopen a scene in Kit, then press ROUTE ALL.")


if __name__ == "__main__":
    main()
