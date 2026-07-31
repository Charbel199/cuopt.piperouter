"""Serialize and deserialize a panel session as a JSON-safe dict.

The dict is embedded in the USD stage (see scene_ops.write_session) so that one Save
carries the whole session. Geometry, markers and route tubes already live in the
stage; this captures the panel's logical state: wire/bundle definitions, weights,
waypoint order, headings and settings.

Only JSON-safe fields survive. The omni.ui widget handles the live wire dicts carry
(combo, name_model, _swatch) are dropped here and rebuilt when the panel redraws.
"""
from __future__ import annotations

# A v1 session recording "lattice" as its global planner may just have been taking the
# default of the day rather than choosing it, so the panel migrates those on load.
SCHEMA_VERSION = 2

_WIRE_KEYS = (
    "key", "name", "type_index", "type_id", "weights", "waypoints", "wp_slots",
    "wp_counter", "start_head_idx", "end_head_idx", "locked",
    "status", "reason", "note", "length_m", "cost", "polyline", "cells", "raw_polyline",
)
_BUNDLE_KEYS = (
    "id", "name", "kind", "type_index", "type_id", "members", "merge_marker",
    "split_marker", "waypoints", "wp_counter", "weights",
    "status", "reason", "trunk_length_m", "trunk_polyline",
)


def _pick(d, keys):
    return {k: d[k] for k in keys if k in d}


def serialize(wires, bundles, settings, counters):
    """Build the JSON-safe session dict from the panel's lists and scalar state."""
    return {
        "version": SCHEMA_VERSION,
        "settings": dict(settings),
        "counters": dict(counters),
        "wires": [_pick(w, _WIRE_KEYS) for w in wires],
        "bundles": [_pick(b, _BUNDLE_KEYS) for b in bundles],
    }


def deserialize(data):
    """Return (wires, bundles, settings, counters) from a session dict.

    Missing keys are tolerated so older or partial files still load.
    """
    return (
        list(data.get("wires", [])),
        list(data.get("bundles", [])),
        dict(data.get("settings", {})),
        dict(data.get("counters", {})),
    )
