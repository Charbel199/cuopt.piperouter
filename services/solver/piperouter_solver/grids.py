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
        )
