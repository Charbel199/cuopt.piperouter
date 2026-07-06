"""Pure bundle helpers (omni-free, headless-testable).

A bundle groups same-kind wires that share a routed trunk corridor (merge->split)
then fan out as individual branches. This module contains only math and validation;
all USD / routing calls live in router_session and panel.
"""
from __future__ import annotations

import math


def bundle_diameter(outer_diameters_mm: list[float]) -> float:
    """Circular-packing estimate: sqrt(sum of d^2).

    Gives the diameter of a circle whose area equals the sum of member areas -
    a practical harness-sizing shortcut used in automotive design.
    """
    if not outer_diameters_mm:
        raise ValueError("bundle_diameter requires at least one diameter")
    return math.sqrt(sum(d * d for d in outer_diameters_mm))


def stitch_polylines(start_branch: list, trunk: list, end_branch: list) -> list:
    """Concatenate start_branch + trunk + end_branch into one continuous polyline.

    Deduplicates the junction points (last of start_branch == first of trunk, etc.)
    so the result has no consecutive equal points. Each input is a list of [x,y,z].
    """
    def _as_list(p):
        return [float(p[0]), float(p[1]), float(p[2])]

    pts = [_as_list(p) for p in start_branch]
    for p in trunk:
        lp = _as_list(p)
        if lp != pts[-1]:
            pts.append(lp)
    for p in end_branch:
        lp = _as_list(p)
        if lp != pts[-1]:
            pts.append(lp)
    return pts


def validate_members(wires: list[dict], type_map: dict[str, dict]) -> tuple[bool, str]:
    """Check that bundle members are valid.

    wires: wire dicts that share a bundle_id (at least 2 required).
    type_map: {wire_name -> spec_dict with 'kind'}.
    Returns (ok, error_message).
    """
    if len(wires) < 2:
        return False, "A bundle requires at least 2 member wires."
    kinds = {type_map[w["name"]]["kind"] for w in wires if w["name"] in type_map}
    if len(kinds) > 1:
        return False, (f"All bundle members must be the same kind; "
                       f"found: {', '.join(sorted(kinds))}")
    return True, ""


def trunk_spec(member_specs: list[dict], bundle_id: str) -> dict:
    """Build a synthetic wire spec for the trunk route.

    Uses the maximum min_bend_radius (stiffest member), minimum max_temp_c
    (strictest thermal limit), and bundle_diameter for the outer diameter.
    Cost/mass are set to 0 - the trunk BOM row is computed separately by summing
    member costs over the trunk length.
    """
    od = bundle_diameter([s["outer_diameter_mm"] for s in member_specs])
    return {
        "id": f"bundle_{bundle_id}_trunk",
        "label": f"Bundle {bundle_id} trunk",
        "kind": member_specs[0]["kind"],
        "outer_diameter_mm": od,
        "inner_diameter_mm": 0.0,
        "min_bend_radius_mm": max(s["min_bend_radius_mm"] for s in member_specs),
        "max_temp_c": min(s["max_temp_c"] for s in member_specs),
        "em_sensitivity": max(s["em_sensitivity"] for s in member_specs),
        "cost_per_m": 0.0,   # trunk BOM computed from members
        "mass_per_m_kg": sum(s["mass_per_m_kg"] for s in member_specs),
        "color": [0.75, 0.75, 0.75],   # neutral gray-white for bundle trunks
    }
