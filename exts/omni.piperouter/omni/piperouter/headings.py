"""Axis-label <-> unit-vector mapping for pinned departure/arrival headings.
Kept omni-free so it can be unit-tested headless and imported by the panel."""
from __future__ import annotations

HEADING_OPTIONS = ("None", "+X", "-X", "+Y", "-Y", "+Z", "-Z")

_VECTORS = {
    "+X": (1.0, 0.0, 0.0), "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0), "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0), "-Z": (0.0, 0.0, -1.0),
}


def axis_to_vector(label):
    """Return the unit vector tuple for an axis label, or None for 'None'."""
    return _VECTORS.get(label)
