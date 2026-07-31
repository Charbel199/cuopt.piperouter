"""Core-pipeline benchmark: where the time goes, and how it scales.

Times each stage (voxelize, lattice build, SSSP, smoothing) for one representative wire
on the complex demo scene across grid resolutions x connectivities, and reports the
expanded-graph size in nodes and edges alongside it.

Run:  PYTHONPATH=services/solver python3 services/solver/bench_core.py
"""
from __future__ import annotations

import time

import numpy as np
from pxr import Usd

from omni.piperouter import sample_scene, wire_library
from omni.piperouter.router_session import RouterSession
from piperouter_solver import smoothing
from piperouter_solver.backend import shortest_path
from piperouter_solver.fields import melt_mask
from piperouter_solver.grids import GridFrame, GridStack
from piperouter_solver.lattice import ExpandedLatticeBuilder
from piperouter_solver.models import WireType


def _wire_from_spec(spec):
    return WireType(
        id="bench", label="bench", kind="wire",
        outer_diameter_mm=float(spec["outer_diameter_mm"]),
        min_bend_radius_mm=float(spec["min_bend_radius_mm"]),
        cost_per_m=float(spec.get("cost_per_m", 1.0)),
        mass_per_m_kg=float(spec.get("mass_per_m_kg", 0.1)),
        max_temp_c=float(spec.get("max_temp_c", 200.0)),
        em_sensitivity=float(spec.get("em_sensitivity", 1.0)),
        color=tuple(spec.get("color", (1.0, 0.0, 0.0))),
    )


def main():
    stage = Usd.Stage.CreateInMemory()
    descr = sample_scene.build_complex_scene(stage)   # cm scene, ~15 obstacles
    sess = RouterSession()
    types = wire_library.load_wire_library()
    # A long diagonal wire, one that has to thread the bay.
    d = descr[0]
    spec = wire_library.as_spec(wire_library.by_id(types, d["type_id"]))
    wire = _wire_from_spec(spec)
    builder = ExpandedLatticeBuilder()

    print(f"{'res':>4} {'conn':>4} {'free_cells':>11} {'nodes':>10} {'edges':>11} "
          f"{'voxel_ms':>9} {'build_ms':>9} {'sssp_ms':>9} {'smooth_ms':>10} {'total_ms':>9}")
    for res in (40, 56, 72):
        t0 = time.perf_counter()
        gb, cell, rxyz, occ, sd, thermal, em = sess.compute_grids(stage, resolution=res)
        voxel_ms = (time.perf_counter() - t0) * 1e3
        mpu = sess.mpu
        stack = GridStack(
            frame=GridFrame(bounds_min=np.asarray(gb, dtype=float), cell_size=float(cell),
                            res_xyz=tuple(int(v) for v in rxyz)),
            occupancy=occ, surface_dist=sd, thermal=thermal, em=em,
        )
        start = np.asarray(d["start"], dtype=float) * mpu
        end = np.asarray(d["end"], dtype=float) * mpu
        a = stack.frame.world_to_grid(tuple(start))
        b = stack.frame.world_to_grid(tuple(end))
        free_cells = int((occ == 0).sum())

        for conn in (6, 18, 26):
            weights = {"surface": 1.0, "bend": 1.0, "thermal": 1.0, "em": 1.0}
            t1 = time.perf_counter()
            g = builder.build(stack, wire, weights, conn, a, b, None, clearance_m=0.0)
            build_ms = (time.perf_counter() - t1) * 1e3

            t2 = time.perf_counter()
            path, _cost = shortest_path(g.src, g.dst, g.weight, g.n_nodes,
                                        g.source_id, g.sink_id)
            sssp_ms = (time.perf_counter() - t2) * 1e3

            smooth_ms = float("nan")
            if path is not None:
                cells = []
                for node in path:
                    if node in (g.source_id, g.sink_id):
                        continue
                    c = g.cell_of(node)
                    if not cells or cells[-1] != c:
                        cells.append(c)
                poly = [np.asarray(stack.frame.grid_to_world(c), dtype=float) for c in cells]
                if len(poly) >= 3:
                    blocked = (stack.dilate_occupancy(wire.radius_m).astype(bool)
                               | melt_mask(stack, wire))
                    t3 = time.perf_counter()
                    smoothing.smooth_path(poly, stack.frame, blocked, wire,
                                          None, None, 1.0, fixed_idx=[0, len(poly) - 1])
                    smooth_ms = (time.perf_counter() - t3) * 1e3

            total = build_ms + sssp_ms + (0.0 if smooth_ms != smooth_ms else smooth_ms)
            print(f"{res:>4} {conn:>4} {free_cells:>11} {g.n_nodes:>10} {g.src.size:>11} "
                  f"{voxel_ms:>9.1f} {build_ms:>9.1f} {sssp_ms:>9.1f} {smooth_ms:>10.1f} "
                  f"{total:>9.1f}")


if __name__ == "__main__":
    main()
