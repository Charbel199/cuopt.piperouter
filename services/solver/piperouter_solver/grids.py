from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .models import GridFrame


def _xp():
    """cupy when PIPEROUTER_GPU_BUILD=1 and importable, else numpy (mirrors lattice.xp)."""
    import os
    if os.environ.get("PIPEROUTER_GPU_BUILD") == "1":
        try:
            import cupy
            return cupy
        except Exception:
            pass
    return np


def dilate6(mask, k):
    """6-connected binary dilation by k iterations, via shifted ORs.

    Equivalent to scipy.ndimage.binary_dilation(structure=generate_binary_structure(3,1),
    iterations=k), but as vectorized slab ops that run on either numpy or cupy (when
    PIPEROUTER_GPU_BUILD=1), which is much faster than scipy's C loop on big grids.
    """
    m = np.asarray(mask, dtype=bool)
    if k <= 0 or not m.any() or m.all():
        return m.copy()
    xp = _xp()
    out = xp.asarray(m)
    for _ in range(int(k)):
        nxt = out.copy()
        nxt[1:, :, :] |= out[:-1, :, :]
        nxt[:-1, :, :] |= out[1:, :, :]
        nxt[:, 1:, :] |= out[:, :-1, :]
        nxt[:, :-1, :] |= out[:, 1:, :]
        nxt[:, :, 1:] |= out[:, :, :-1]
        nxt[:, :, :-1] |= out[:, :, 1:]
        out = nxt
    if xp is not np:
        out = xp.asnumpy(out)
    return out


@dataclass
class GridStack:
    """The four aligned voxel fields produced by the extension's Warp pass."""

    frame: GridFrame
    occupancy: np.ndarray     # uint8 (nx, ny, nz), 1 = blocked
    surface_dist: np.ndarray  # float32, distance (m) to nearest surface
    thermal: np.ndarray       # float32, temperature in deg C
    em: np.ndarray            # float32, EM strength in [0, 1]
    # Optional per-object clearance: class grid (uint8; 0 = untagged geometry) and the
    # clearance in metres for class id i+1 at clearance_values[i].
    clearance_class: np.ndarray | None = None
    clearance_values: tuple = ()

    def dilate_occupancy(self, radius_m: float) -> np.ndarray:
        """Grow blocked cells by a wire radius so the tube body clears meshes.

        Uses round-half-up on radius/cell: a wire whose radius is under half a cell fits
        in the free space around the centerline and does not dilate (it may legitimately
        run flush against a surface); larger radii grow the blocked region by the number
        of cells they actually span.

        Cached per dilation-cell count, since occupancy is immutable for a stack's
        lifetime and route_one/planners call this several times per wire with the same
        radii. Treat the returned array as read-only; callers copy via .astype() or
        combine with `|`/`&` into a new array.
        """
        # The +1e-9 keeps the round-half-up boundary robust against float error, e.g.
        # 0.15/0.1 evaluates to 1.4999..., which must still round to 2.
        cells = int(radius_m / self.frame.cell_size + 0.5 + 1e-9)
        cache = self.__dict__.setdefault("_dilate_cache", {})
        hit = cache.get(cells)
        if hit is not None:
            return hit
        if cells <= 0:
            out = self.occupancy.copy()
        else:
            out = dilate6(self.occupancy.astype(bool), cells).astype(np.uint8)
        cache[cells] = out
        return out

    def dilate_class(self, class_id: int, dist_m: float) -> np.ndarray:
        """Dilate a single clearance class's voxels by dist_m (class 0 = untagged occupancy).

        Cached per (class_id, cells); read-only, same convention as dilate_occupancy.
        """
        cells = int(dist_m / self.frame.cell_size + 0.5 + 1e-9)
        cache = self.__dict__.setdefault("_cls_dilate_cache", {})
        key = (int(class_id), cells)
        hit = cache.get(key)
        if hit is not None:
            return hit
        occ_b = self.occupancy.astype(bool)
        if self.clearance_class is None:
            mask = occ_b if class_id == 0 else np.zeros_like(occ_b)
        elif class_id == 0:
            mask = occ_b & (self.clearance_class == 0)
        else:
            mask = self.clearance_class == class_id
        if cells > 0 and mask.any():
            mask = dilate6(mask, cells)
        cache[key] = mask
        return mask

    def save(self, path: str | Path) -> None:
        # Uncompressed: the handoff file lives on tmpfs (/dev/shm), so compression would
        # only cost CPU.
        np.savez(
            path,
            bounds_min=self.frame.bounds_min,
            cell_size=np.float64(self.frame.cell_size),
            res_xyz=np.asarray(self.frame.res_xyz, dtype=np.int64),
            occupancy=self.occupancy,
            surface_dist=self.surface_dist,
            thermal=self.thermal,
            em=self.em,
            **({"clearance_class": self.clearance_class,
                "clearance_values": np.asarray(self.clearance_values, dtype=np.float64)}
               if self.clearance_class is not None and len(self.clearance_values) else {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "GridStack":
        d = np.load(path)
        frame = GridFrame(
            bounds_min=d["bounds_min"],
            cell_size=float(d["cell_size"]),
            res_xyz=tuple(int(v) for v in d["res_xyz"]),
        )
        return cls(
            frame=frame,
            occupancy=d["occupancy"],
            surface_dist=d["surface_dist"],
            thermal=d["thermal"],
            em=d["em"],
            clearance_class=d["clearance_class"] if "clearance_class" in d.files else None,
            clearance_values=(tuple(float(v) for v in d["clearance_values"])
                              if "clearance_values" in d.files else ()),
        )
