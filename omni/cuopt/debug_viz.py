"""
Debug visualizations for the occupancy grid and CSR waypoint graph.

Creates USD geometry in /World/Debug so we can inspect what the solver
sees: which cells are blocked, which are free, and how the waypoint
graph connects them.
"""

import numpy as np
from pxr import UsdGeom, Gf, Vt, Sdf, Usd


def show_occupancy_grid(stage, grid, show_free=False):
    # render blocked cells as red cubes
    _clear_debug(stage)
    _ensure_xform(stage, "/World")
    _ensure_xform(stage, "/World/Debug")
    _ensure_xform(stage, "/World/Debug/Grid")

    res = grid.resolution
    cell_scale = grid.cell_size * 0.95

    # blocked cells
    bi, bj, bk = np.where(grid.occupied)
    blocked_positions = [grid.grid_to_world((bi[n], bj[n], bk[n]))
                         for n in range(len(bi))]

    if blocked_positions:
        _create_instancer(
            stage, "/World/Debug/Grid/Blocked",
            blocked_positions, cell_scale,
            color=Gf.Vec3f(0.9, 0.15, 0.15), opacity=0.35,
        )



    return len(blocked_positions)


def show_waypoint_graph(stage, grid):
    # render free cells near obstacles as green cubes.
    # only shows cells within 3 cells of a blocked cell to keep the visualization readable. These are the nodes the router can use.
    _ensure_xform(stage, "/World/Debug")
    _ensure_xform(stage, "/World/Debug/Graph")

    dilated = _dilate(grid.occupied, iterations=3)
    near_surface = dilated & ~grid.occupied

    fi, fj, fk = np.where(near_surface)
    node_positions = [grid.grid_to_world((fi[n], fj[n], fk[n]))
                      for n in range(len(fi))]

    if node_positions:
        node_scale = grid.cell_size * 0.2
        _create_instancer(
            stage, "/World/Debug/Graph/Nodes",
            node_positions, node_scale,
            color=Gf.Vec3f(0.2, 0.9, 0.3), opacity=0.6,
        )

    return len(node_positions)


def clear_debug(stage):
    _clear_debug(stage)


def _dilate(volume, iterations=1):
    # binary dilation using 6-connected shifts (no scipy dependency)
    out = volume.copy()
    for _ in range(iterations):
        out = (
            out
            | np.roll(out, 1, axis=0) | np.roll(out, -1, axis=0)
            | np.roll(out, 1, axis=1) | np.roll(out, -1, axis=1)
            | np.roll(out, 1, axis=2) | np.roll(out, -1, axis=2)
        )
    return out


def _clear_debug(stage):
    prim = stage.GetPrimAtPath("/World/Debug")
    if prim and prim.IsValid():
        stage.RemovePrim("/World/Debug")


def _ensure_xform(stage, path):
    if not stage.GetPrimAtPath(path).IsValid():
        UsdGeom.Xform.Define(stage, path)


def _create_instancer(stage, path, positions, scale, color, opacity=1.0):
    # create a PointInstancer with a cube prototype at the given positions

    instancer = UsdGeom.PointInstancer.Define(stage, path)

    proto_path = f"{path}/Protos/Cell"
    cube = UsdGeom.Cube.Define(stage, proto_path)
    cube.GetSizeAttr().Set(1.0)
    cube.GetDisplayColorAttr().Set(Vt.Vec3fArray([color]))
    if opacity < 1.0:
        pv_api = UsdGeom.PrimvarsAPI(cube.GetPrim())
        pv_api.CreatePrimvar(
            "displayOpacity", Sdf.ValueTypeNames.FloatArray
        ).Set(Vt.FloatArray([opacity]))

    n = len(positions)
    instancer.GetPrototypesRel().AddTarget(proto_path)
    instancer.GetProtoIndicesAttr().Set(Vt.IntArray([0] * n))

    pos_array = Vt.Vec3fArray([
        Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in positions
    ])
    instancer.GetPositionsAttr().Set(pos_array)

    scale_vec = Gf.Vec3f(float(scale[0]), float(scale[1]), float(scale[2]))
    instancer.GetScalesAttr().Set(Vt.Vec3fArray([scale_vec] * n))
