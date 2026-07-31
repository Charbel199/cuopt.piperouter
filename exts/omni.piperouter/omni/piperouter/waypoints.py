"""Waypoint-order helpers, kept omni-free so they can be tested without Kit.

The per-wire `waypoints` list doubles as the route order (the solver routes
start -> wp[0] -> wp[1] -> ... -> end), so reordering the list reorders the legs.
"""
from __future__ import annotations


def reorder(seq, src, dst):
    """Return a new list with the item at index `src` moved to index `dst`.

    Indices are clamped into range, and degenerate indices return a plain copy so a
    no-op drag cannot corrupt the list.
    """
    items = list(seq)
    n = len(items)
    if n == 0:
        return items
    src = max(0, min(int(src), n - 1))
    dst = max(0, min(int(dst), n - 1))
    if src == dst:
        return items
    items.insert(dst, items.pop(src))
    return items
