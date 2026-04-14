import math
import numpy as np

class OccupancyGrid3D:

    def __init__(self, bounds_min, bounds_max, resolution):
        self.bounds_min = np.array(bounds_min, dtype=np.float64)
        self.bounds_max = np.array(bounds_max, dtype=np.float64)

        # uniform cubic cells sized from the longest axis
        extent = self.bounds_max - self.bounds_min
        longest = float(np.max(extent))
        self.cell_size = np.full(3, longest / resolution)

        # per-axis resolution (may differ if scene is not cubic)
        self.res_xyz = np.maximum(1, np.ceil(extent / self.cell_size[0]).astype(int))
        self.resolution = int(np.max(self.res_xyz))

        # expand bounds so the grid fits evenly
        grid_extent = self.res_xyz * self.cell_size[0]
        padding = (grid_extent - extent) * 0.5
        self.bounds_min = self.bounds_min - padding
        self.bounds_max = self.bounds_min + grid_extent

        self.occupied = np.zeros(
            (self.res_xyz[0], self.res_xyz[1], self.res_xyz[2]), dtype=bool
        )

    def world_to_grid(self, point):
        p = np.asarray(point, dtype=np.float64)
        idx = ((p - self.bounds_min) / self.cell_size[0]).astype(int)
        return tuple(np.clip(idx, 0, self.res_xyz - 1))

    def grid_to_world(self, idx):
        return self.bounds_min + (np.array(idx, dtype=np.float64) + 0.5) * self.cell_size[0]

    def mark_box(self, box_min, box_max, clearance=0.0):
        expanded_min = np.asarray(box_min) - clearance
        expanded_max = np.asarray(box_max) + clearance
        i0, j0, k0 = self.world_to_grid(expanded_min)
        i1, j1, k1 = self.world_to_grid(expanded_max)
        self.occupied[i0 : i1 + 1, j0 : j1 + 1, k0 : k1 + 1] = True


# pre-compute 26-connected neighborhood offsets and distances
_NEIGHBORS_26 = []
for _di in (-1, 0, 1):
    for _dj in (-1, 0, 1):
        for _dk in (-1, 0, 1):
            if _di == 0 and _dj == 0 and _dk == 0:
                continue
            _NEIGHBORS_26.append(
                (_di, _dj, _dk, math.sqrt(_di * _di + _dj * _dj + _dk * _dk))
            )


def smooth_path(points, subdivisions=8):
    # catmull-rom spline to smooth out cuopt output
    if len(points) < 2:
        return list(points)
    if len(points) == 2:
        return [
            points[0] * (1.0 - t) + points[1] * t
            for t in np.linspace(0.0, 1.0, max(subdivisions, 2))
        ]

    ext = [points[0]] + list(points) + [points[-1]]
    out = []
    for i in range(1, len(ext) - 2):
        p0, p1, p2, p3 = ext[i - 1], ext[i], ext[i + 1], ext[i + 2]
        for j in range(subdivisions):
            t = j / subdivisions
            t2, t3 = t * t, t * t * t
            pt = 0.5 * (
                2.0 * p1
                + (-p0 + p2) * t
                + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3
            )
            out.append(pt)
    out.append(points[-1])
    return out
