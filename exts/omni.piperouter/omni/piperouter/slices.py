"""Render flattened 2D cross-section views (XY / XZ / YZ projections) of the voxel
space, for the extension's debug panel. Pure numpy (headless-testable); the omni.ui
side just pushes the returned RGBA buffers into image providers.

Each view collapses the perpendicular axis so a single image shows ALL obstacles and
the WHOLE route at once:
  XY = top view (collapse Z), XZ = front view (collapse Y), YZ = side view (collapse X).

The obstacle/thermal/clearance background is the voxel grid upscaled to a high pixel
resolution; the routes are then drawn as continuous lines (not per-voxel dots) so they
stay crisp regardless of grid coarseness. Layers, back to front: light background ->
thermal tint (warm = red) -> clearance halo (light grey) -> occupancy (dark grey) ->
routes (wire color, start green / end red).
"""
from __future__ import annotations

import numpy as np

# plane name -> (collapsed_axis, horizontal_axis, vertical_axis)
_PLANES = {"xy": (2, 0, 1), "xz": (1, 0, 2), "yz": (0, 1, 2)}

_BG = (0.93, 0.93, 0.95)
_OCC = (0.20, 0.20, 0.24)
_HALO = (0.62, 0.62, 0.68)
_GREEN = (0.10, 0.85, 0.15)
_RED = (0.95, 0.10, 0.10)


def _dilate2d(mask, n):
    out = mask.copy()
    for _ in range(int(n)):
        d = out.copy()
        d[1:, :] |= out[:-1, :]
        d[:-1, :] |= out[1:, :]
        d[:, 1:] |= out[:, :-1]
        d[:, :-1] |= out[:, 1:]
        out = d
    return out


def _disk(img, r, c, color, rad):
    h, w = img.shape[0], img.shape[1]
    r0, r1 = max(0, r - rad), min(h, r + rad + 1)
    c0, c1 = max(0, c - rad), min(w, c + rad + 1)
    img[r0:r1, c0:c1] = color


def _line(img, r0, c0, r1, c1, color, thick):
    """Bresenham line with a square brush of half-width `thick`."""
    r0, c0, r1, c1 = int(r0), int(c0), int(r1), int(c1)
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    while True:
        _disk(img, r0, c0, color, thick)
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc


def render_views(bounds_min, cell_size, res_xyz, occ, thermal, routes,
                 clearance_cells=0, ambient=20.0, target_px=720):
    """Return {"xy","xz","yz"} -> RGBA uint8 array (H, W, 4) at ~target_px."""
    bounds_min = np.asarray(bounds_min, float)
    occ = occ.astype(bool)
    res = np.asarray(res_xyz, int)
    thermal = np.asarray(thermal, float)
    t_lo, t_hi = ambient, max(float(thermal.max()), ambient + 1e-6)
    cell_size = float(cell_size)

    out = {}
    for name, (ca, ha, va) in _PLANES.items():
        occ2d = occ.any(axis=ca)                         # (res[ha], res[va])
        therm2d = thermal.max(axis=ca)
        halo2d = (_dilate2d(occ2d, clearance_cells) & ~occ2d) if clearance_cells > 0 \
            else np.zeros_like(occ2d)

        hd, vd = occ2d.shape
        base = np.empty((hd, vd, 3), dtype=np.float32)
        base[:] = _BG
        t01 = np.clip((therm2d - t_lo) / (t_hi - t_lo), 0.0, 1.0)
        warm = t01 > 0.02
        base[warm] = (1.0 - t01[warm, None]) * np.array(_BG) + t01[warm, None] * np.array([1.0, 0.35, 0.2])
        base[halo2d] = _HALO
        base[occ2d] = _OCC

        # orient to image space: rows = vertical axis (flipped so + is up), cols = horiz
        img0 = np.transpose(base, (1, 0, 2))[::-1]       # (vd, hd, 3)
        scale = max(1, int(round(target_px / max(vd, hd))))
        img = np.repeat(np.repeat(img0, scale, axis=0), scale, axis=1)
        ih, iw = img.shape[0], img.shape[1]

        # draw routes as continuous lines at the high pixel resolution
        thick = max(1, scale // 3)
        for route in routes:
            color = np.asarray(route.get("color", (0.9, 0.1, 0.1)), float)
            pts = route.get("points", [])
            px = []
            for p in pts:
                p = np.asarray(p, float)
                fc_h = (p[ha] - bounds_min[ha]) / cell_size   # fractional cell coords
                fc_v = (p[va] - bounds_min[va]) / cell_size
                col = int(np.clip(fc_h * scale, 0, iw - 1))
                row = int(np.clip((vd - fc_v) * scale, 0, ih - 1))
                px.append((row, col))
            for (r0, c0), (r1, c1) in zip(px[:-1], px[1:]):
                _line(img, r0, c0, r1, c1, color, thick)
            if px:
                _disk(img, *px[0], _GREEN, thick + 2)        # start
                _disk(img, *px[-1], _RED, thick + 2)         # end

        rgba = np.empty((ih, iw, 4), dtype=np.uint8)
        rgba[..., :3] = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        rgba[..., 3] = 255
        out[name] = rgba
    return out
