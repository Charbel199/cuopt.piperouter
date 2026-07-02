"""Axis-label <-> unit-vector mapping for pinned departure/arrival headings, plus
azimuth/elevation <-> vector conversions for the CUSTOM (rotatable-gizmo) heading mode.
Kept omni-free so it can be unit-tested headless and imported by the panel."""
from __future__ import annotations

import math

CUSTOM = "Custom"
HEADING_OPTIONS = ("None", "+X", "-X", "+Y", "-Y", "+Z", "-Z", CUSTOM)

_VECTORS = {
    "+X": (1.0, 0.0, 0.0), "-X": (-1.0, 0.0, 0.0),
    "+Y": (0.0, 1.0, 0.0), "-Y": (0.0, -1.0, 0.0),
    "+Z": (0.0, 0.0, 1.0), "-Z": (0.0, 0.0, -1.0),
}


def axis_to_vector(label):
    """Unit vector tuple for an axis label; None for 'None' AND for 'Custom' (the
    custom direction is read from the marker's rotation, not from the label)."""
    return _VECTORS.get(label)


def _ground_axes(up_axis):
    """(forward, side, up) world unit vectors for azimuth/elevation, given the stage
    up-axis letter. Azimuth 0 = forward, measured toward side; elevation measured
    toward up. Y-up: forward=+X, side=+Z. Z-up: forward=+X, side=+Y."""
    if str(up_axis).upper() == "Y":
        return (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)
    return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def angles_to_vector(azimuth_deg, elevation_deg, up_axis="Z"):
    """Unit world vector from azimuth (degrees around the up-axis, 0 = +X) and
    elevation (degrees toward the up-axis, +90 = straight up)."""
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    f, s, u = _ground_axes(up_axis)
    ch = math.cos(el)
    cf, cs, cu = ch * math.cos(az), ch * math.sin(az), math.sin(el)
    return tuple(cf * f[i] + cs * s[i] + cu * u[i] for i in range(3))


def vector_to_angles(v, up_axis="Z"):
    """(azimuth_deg, elevation_deg) of a world vector — inverse of angles_to_vector.
    A (near-)zero vector maps to (0, 0)."""
    f, s, u = _ground_axes(up_axis)
    x = sum(float(v[i]) * f[i] for i in range(3))
    y = sum(float(v[i]) * s[i] for i in range(3))
    z = sum(float(v[i]) * u[i] for i in range(3))
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-9:
        return 0.0, 0.0
    el = math.degrees(math.asin(max(-1.0, min(1.0, z / n))))
    az = math.degrees(math.atan2(y, x)) if (abs(x) > 1e-12 or abs(y) > 1e-12) else 0.0
    return az, el
