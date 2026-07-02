from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from .models import GridFrame


@dataclass
class GridStack:
    """The four aligned voxel fields produced by the extension's Warp pass."""

    frame: GridFrame
    occupancy: np.ndarray     # uint8 (nx, ny, nz), 1 = blocked
    surface_dist: np.ndarray  # float32, distance (m) to nearest surface
    thermal: np.ndarray       # float32, temperature in deg C
    em: np.ndarray            # float32, EM strength in [0, 1]
    # per-object clearance (optional): class grid (uint8; 0 = untagged geometry) and the
    # clearance in METRES for class id i+1 at clearance_values[i].
    clearance_class: np.ndarray | None = None
    clearance_values: tuple = ()

    def dilate_occupancy(self, radius_m: float) -> np.ndarray:
        """Grow blocked cells by a wire radius so the tube body clears meshes.

        Uses round-half-up on radius/cell: a wire whose radius is under half a
        cell fits in the free space around the centerline and does not dilate
        (it may legitimately run flush against a surface); larger radii grow the
        blocked region by the number of cells they actually span.

        CACHED per dilation-cell count (occupancy is immutable for a stack's
        lifetime, and route_one/planners call this several times per wire with the
        same radii). Treat the returned array as READ-ONLY — every current caller
        copies via .astype() or combines with `|`/`&` into a new array.
        """
        # +1e-9 keeps the round-half-up boundary robust against float error
        # (e.g. 0.15/0.1 evaluates to 1.4999..., which must round to 2).
        cells = int(radius_m / self.frame.cell_size + 0.5 + 1e-9)
        cache = self.__dict__.setdefault("_dilate_cache", {})
        hit = cache.get(cells)
        if hit is not None:
            return hit
        if cells <= 0:
            out = self.occupancy.copy()
        else:
            structure = ndimage.generate_binary_structure(3, 1)
            out = ndimage.binary_dilation(
                self.occupancy.astype(bool), structure=structure, iterations=cells
            ).astype(np.uint8)
        cache[cells] = out
        return out

    def dilate_class(self, class_id: int, dist_m: float) -> np.ndarray:
        """Dilate ONE clearance class's voxels by dist_m (class 0 = untagged occupancy).
        Cached per (class_id, cells) — read-only, same convention as dilate_occupancy."""
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
            structure = ndimage.generate_binary_structure(3, 1)
            mask = ndimage.binary_dilation(mask, structure=structure, iterations=cells)
        cache[key] = mask
        return mask

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
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
