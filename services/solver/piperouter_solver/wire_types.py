from __future__ import annotations

import json
from pathlib import Path

from .models import WireType


def load_wire_types(path: str | Path) -> dict[str, WireType]:
    data = json.loads(Path(path).read_text())
    out: dict[str, WireType] = {}
    for entry in data["types"]:
        wt = WireType(
            id=entry["id"],
            label=entry["label"],
            kind=entry["kind"],
            outer_diameter_mm=float(entry["outer_diameter_mm"]),
            min_bend_radius_mm=float(entry["min_bend_radius_mm"]),
            cost_per_m=float(entry["cost_per_m"]),
            mass_per_m_kg=float(entry["mass_per_m_kg"]),
            max_temp_c=float(entry["max_temp_c"]),
            em_sensitivity=float(entry["em_sensitivity"]),
            color=tuple(float(c) for c in entry["color"]),
            inner_diameter_mm=float(entry.get("inner_diameter_mm", 0.0)),
        )
        out[wt.id] = wt
    return out
