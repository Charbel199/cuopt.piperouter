from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class GridFrame:
    """Axis-aligned uniform-cubic-cell voxel frame."""

    bounds_min: np.ndarray  # (3,) float64 world-space origin of the padded grid
    cell_size: float
    res_xyz: tuple[int, int, int]

    @classmethod
    def from_bounds(
        cls,
        bounds_min: Sequence[float],
        bounds_max: Sequence[float],
        resolution: int,
    ) -> "GridFrame":
        bmin = np.asarray(bounds_min, dtype=np.float64)
        bmax = np.asarray(bounds_max, dtype=np.float64)
        extent = bmax - bmin
        longest = float(np.max(extent))
        cell = longest / resolution
        res = np.maximum(1, np.ceil(extent / cell).astype(int))
        grid_extent = res * cell
        padding = (grid_extent - extent) * 0.5
        bmin_padded = bmin - padding
        return cls(
            bounds_min=bmin_padded,
            cell_size=float(cell),
            res_xyz=(int(res[0]), int(res[1]), int(res[2])),
        )

    def world_to_grid(self, point: Sequence[float]) -> tuple[int, int, int]:
        p = np.asarray(point, dtype=np.float64)
        idx = np.floor((p - self.bounds_min) / self.cell_size).astype(int)
        hi = np.array(self.res_xyz) - 1
        idx = np.clip(idx, 0, hi)
        return (int(idx[0]), int(idx[1]), int(idx[2]))

    def grid_to_world(self, idx: Sequence[int]) -> np.ndarray:
        a = np.asarray(idx, dtype=np.float64)
        return self.bounds_min + (a + 0.5) * self.cell_size


@dataclass(frozen=True)
class WireType:
    id: str
    label: str
    kind: str  # "wire" | "pipe"
    outer_diameter_mm: float
    min_bend_radius_mm: float
    cost_per_m: float
    mass_per_m_kg: float
    max_temp_c: float
    em_sensitivity: float
    color: tuple[float, float, float]

    @property
    def radius_m(self) -> float:
        return (self.outer_diameter_mm / 1000.0) / 2.0


@dataclass
class RouteRequest:
    wire: "WireType"
    start: tuple[float, float, float]            # world space
    end: tuple[float, float, float]              # world space
    waypoints: list[tuple[float, float, float]] = field(default_factory=list)
    weights: dict = field(default_factory=dict)  # {"surface","thermal","em","bend"}
    connectivity: int = 26
    priority: int = 0                            # lower routes first in route_all
    # Extra safety margin (m) kept from meshes, ON TOP of the wire's own radius. 0 =
    # the route only needs to avoid the mesh itself (may run flush against surfaces).
    clearance_m: float = 0.0


@dataclass
class RouteResult:
    wire_id: str
    status: str                  # "routed" | "no_path"
    polyline: list = field(default_factory=list)  # list of (3,) world points
    length_m: float = 0.0
    cells: list = field(default_factory=list)      # occupied (i,j,k) for obstacle reuse


@dataclass
class SolveReport:
    results: list  # list[RouteResult]

    @property
    def routed(self) -> int:
        return sum(1 for r in self.results if r.status == "routed")

    @property
    def no_path(self) -> int:
        return sum(1 for r in self.results if r.status == "no_path")
