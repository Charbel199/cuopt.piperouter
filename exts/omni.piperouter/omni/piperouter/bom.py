"""Pure BOM (bill-of-materials) summarization (omni-free, headless-testable).

Turns the solver's per-route BOM rows into table rows + totals for the panel's
Output / BOM table. Only successfully-routed wires contribute to the totals.
"""
from __future__ import annotations


def summarize(bom, type_labels=None):
    """Aggregate BOM rows into table rows + totals.

    bom: list of dicts {wire_id, status, length_m, cost, mass}.
    type_labels: optional {wire_id: type_label} for the Type column.

    Returns {rows, total_cost, total_mass, total_length, n_routed, n_no_path}.
    Each row: {wire_id, type, length_m, mass, cost, status}. No-path wires appear
    as rows (status != "routed") but contribute 0 to the totals.
    """
    type_labels = type_labels or {}
    rows = []
    total_cost = total_mass = total_length = 0.0
    n_routed = n_no_path = 0
    for b in bom:
        routed = b.get("status") == "routed"
        length = float(b.get("length_m", 0.0)) if routed else 0.0
        mass = float(b.get("mass", 0.0)) if routed else 0.0
        cost = float(b.get("cost", 0.0)) if routed else 0.0
        rows.append({
            "wire_id": b.get("wire_id", "?"),
            "type": type_labels.get(b.get("wire_id"), ""),
            "length_m": length,
            "mass": mass,
            "cost": cost,
            "status": b.get("status", "?"),
        })
        if routed:
            n_routed += 1
            total_cost += cost
            total_mass += mass
            total_length += length
        else:
            n_no_path += 1
    return {
        "rows": rows,
        "total_cost": total_cost,
        "total_mass": total_mass,
        "total_length": total_length,
        "n_routed": n_routed,
        "n_no_path": n_no_path,
    }
