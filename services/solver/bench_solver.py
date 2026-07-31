"""Time every global planner on a pre-saved voxel grid, and report which backend is live.

Solver-only: no extension, Warp or pxr import, so it runs inside the solver container
where cupy and cuGraph are installed. Takes the GridStack the extension hands off on
disk plus a list of endpoint cells.

Usage:
    PIPEROUTER_GPU_BUILD=1 python3 bench_solver.py <grid.npz> <endpoints.json>
"""
from __future__ import annotations

import json
import os
import sys
import time

from piperouter_solver.grids import GridStack
from piperouter_solver.models import RouteRequest, WireType
from piperouter_solver.solver import Solver

ALGOS = ("lattice", "octree_lattice", "astar", "fmm", "medial")


def _backend():
    import importlib.util
    has_cupy = importlib.util.find_spec("cupy") is not None
    has_cugraph = importlib.util.find_spec("cugraph") is not None
    # Importable is not enough; check cuGraph and cudf actually load.
    cugraph_live = False
    if has_cugraph:
        try:
            import cudf  # noqa: F401
            import cugraph  # noqa: F401
            cugraph_live = True
        except Exception:
            cugraph_live = False
    print(f"backend: cupy={'YES' if has_cupy else 'no'} | "
          f"cuGraph(SSSP)={'YES' if cugraph_live else 'no'} | "
          f"PIPEROUTER_GPU_BUILD={os.environ.get('PIPEROUTER_GPU_BUILD', '0')}")


def main():
    grid_path, ep_path = sys.argv[1], sys.argv[2]
    stack = GridStack.load(grid_path)
    eps = json.load(open(ep_path))
    _backend()
    print(f"grid res {tuple(stack.frame.res_xyz)}, {len(eps)} wires\n")
    solver = Solver()
    print(f"{'global':>16} {'conn':>5} {'routed':>8} {'time_ms':>9} {'ms/wire':>8}")
    for algo in ALGOS:
        for conn in (18, 26):
            routed = 0
            t0 = time.perf_counter()
            for e in eps:
                w = WireType(**e["wire"])
                req = RouteRequest(
                    wire=w,
                    start=tuple(stack.frame.grid_to_world(e["start"])),
                    end=tuple(stack.frame.grid_to_world(e["end"])),
                    connectivity=conn,
                    weights={"surface": 1.0, "bend": 1.0, "thermal": 1.0,
                             "em": 1.0, "smoothing": 1.0},
                    global_planner=algo, local_optimizer="fibre")
                if solver.route_one(stack, req).status == "routed":
                    routed += 1
            dt = (time.perf_counter() - t0) * 1e3
            print(f"{algo:>16} {conn:>5} {routed:>5}/{len(eps):<2} {dt:>9.0f} "
                  f"{dt / max(len(eps), 1):>8.1f}")


if __name__ == "__main__":
    main()
