"""Load the bundled wire/pipe library. The chosen type's full spec is sent in each
route request, so the solver never needs its own copy to agree with the extension.
"""
from __future__ import annotations

import json
from pathlib import Path

_DEFAULT = Path(__file__).resolve().parents[2] / "data" / "wire_types.json"

_SPEC_KEYS = (
    "id", "label", "kind", "outer_diameter_mm", "inner_diameter_mm",
    "min_bend_radius_mm", "cost_per_m", "mass_per_m_kg", "max_temp_c",
    "em_sensitivity", "color",
)


def load_wire_library(path=None):
    p = Path(path) if path else _DEFAULT
    return json.loads(p.read_text())["types"]


def as_spec(entry: dict) -> dict:
    return {k: entry[k] for k in _SPEC_KEYS}


def by_id(types, type_id):
    for t in types:
        if t["id"] == type_id:
            return t
    raise KeyError(type_id)
