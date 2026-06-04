"""Pure waypoint-order helpers (omni-free, headless-testable).

The per-wire `waypoints` list IS the route order (the solver routes
start -> wp[0] -> wp[1] -> ... -> end), so reordering this list reorders the
legs. Kept omni-free so the reorder math can be unit-tested without Kit.
"""
from __future__ import annotations


def reorder(seq, src, dst):
    """Return a NEW list with the item at index `src` moved to index `dst`.

    Indices are clamped into range; equal/degenerate indices return a plain copy
    (so a no-op drag never corrupts the list).
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
