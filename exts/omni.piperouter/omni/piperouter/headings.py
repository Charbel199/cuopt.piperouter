"""Heading conversions for pinned departure/arrival directions.

Maps axis labels to unit vectors, and azimuth/elevation to vectors for the Custom
(rotatable-gizmo) heading mode. Omni-free so it can be tested headless.
"""
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
    """Return the unit vector for an axis label, or None.

    'Custom' also yields None: its direction comes from the marker's rotation rather
    than from the label.
    """
    return _VECTORS.get(label)


def _ground_axes(up_axis):
    """Return (forward, side, up) world unit vectors for the stage up-axis letter.

    Azimuth 0 points along forward and is measured toward side; elevation is measured
    toward up. Y-up gives forward=+X, side=+Z; Z-up gives forward=+X, side=+Y.
    """
    if str(up_axis).upper() == "Y":
        return (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)
    return (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)


def angles_to_vector(azimuth_deg, elevation_deg, up_axis="Z"):
    """Return the unit world vector for an azimuth/elevation pair.

    Azimuth is degrees around the up-axis with 0 = +X; elevation is degrees toward the
    up-axis, so +90 points straight up.
    """
    az = math.radians(float(azimuth_deg))
    el = math.radians(float(elevation_deg))
    f, s, u = _ground_axes(up_axis)
    ch = math.cos(el)
    cf, cs, cu = ch * math.cos(az), ch * math.sin(az), math.sin(el)
    return tuple(cf * f[i] + cs * s[i] + cu * u[i] for i in range(3))


def vector_to_angles(v, up_axis="Z"):
    """Return (azimuth_deg, elevation_deg) of a world vector.

    Inverse of angles_to_vector. A near-zero vector maps to (0, 0).
    """
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
