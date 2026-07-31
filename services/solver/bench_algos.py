"""Compare routing algorithms: run every global x local combination over the same fixture
wires on the complex demo scene and print a scored table.

Per combo: how many wires routed, then over the wires that every combo routed, the mean
cost, bend count, surface-hug distance and wall-clock time.

Run:  PYTHONPATH=services/solver:exts/omni.piperouter python3 services/solver/bench_algos.py
"""
from __future__ import annotations

import time

import numpy as np
from pxr import Usd

from omni.piperouter import sample_scene, wire_library
from omni.piperouter.router_session import RouterSession
from piperouter_solver import optimizers, planners
from piperouter_solver.grids import GridFrame, GridStack
from piperouter_solver.models import RouteRequest, WireType
from piperouter_solver.solver import Solver

RES = 56
N_WIRES = 14


def _wire(spec):
    return WireType(
        id="w", label="w", kind="wire",
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
    descr = sample_scene.build_complex_scene(stage)[:N_WIRES]
    sess = RouterSession()
    gb, cell, rxyz, occ, sd, thermal, em = sess.compute_grids(stage, resolution=RES)
    mpu = sess.mpu
    stack = GridStack(
        frame=GridFrame(bounds_min=np.asarray(gb, float), cell_size=float(cell),
                        res_xyz=tuple(int(v) for v in rxyz)),
        occupancy=occ, surface_dist=sd, thermal=thermal, em=em)
    types = wire_library.load_wire_library()

    jobs = []
    for d in descr:
        spec = wire_library.as_spec(wire_library.by_id(types, d["type_id"]))
        jobs.append((_wire(spec),
                     tuple(np.asarray(d["start"], float) * mpu),
                     tuple(np.asarray(d["end"], float) * mpu)))

    solver = Solver()
    sd = stack.surface_dist
    snx, sny, snz = sd.shape
    n = len(jobs)

    def turn_count(poly):
        """Count vertices turning by more than 20 deg, i.e. the real bends."""
        p = [np.asarray(x, dtype=float) for x in poly]
        c = 0
        for i in range(1, len(p) - 1):
            a, b = p[i] - p[i - 1], p[i + 1] - p[i]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                continue
            if np.degrees(np.arccos(np.clip(np.dot(a / na, b / nb), -1, 1))) > 20.0:
                c += 1
        return c

    def hug_mm(poly):
        """Mean distance to the nearest surface along the route, in mm.

        Low values mean the route hugs surfaces it could be clipped to; high values
        mean it floats in open air.
        """
        vals = []
        for p in poly:
            i, j, k = stack.frame.world_to_grid((float(p[0]), float(p[1]), float(p[2])))
            if 0 <= i < snx and 0 <= j < sny and 0 <= k < snz:
                vals.append(float(sd[i, j, k]))
        return (sum(vals) / len(vals) * 1000.0) if vals else 0.0

    # Pass 1: route everything, keeping the per-combo, per-wire result.
    res_by = {}
    time_by = {}
    for gname in planners.GLOBAL_PLANNERS:
        for lname in optimizers.LOCAL_OPTIMIZERS:
            per = {}
            t0 = time.perf_counter()
            for wi, (wire, a, b) in enumerate(jobs):
                req = RouteRequest(wire=wire, start=a, end=b, connectivity=18,
                                   weights={"surface": 1.0, "bend": 1.0, "thermal": 1.0,
                                            "em": 1.0, "smoothing": 1.0},
                                   global_planner=gname, local_optimizer=lname)
                r = solver.route_one(stack, req)
                per[wi] = r if r.status == "routed" else None
            res_by[(gname, lname)] = per
            time_by[(gname, lname)] = (time.perf_counter() - t0) * 1e3

    # Common set: wires every combo routed, so cost/turns/hug compare like for like.
    common = [wi for wi in range(n) if all(res_by[c][wi] for c in res_by)]

    rows = []
    for c, per in res_by.items():
        routed = sum(1 for v in per.values() if v)
        costs, turns, hugs = [], [], []
        for wi in common:
            r = per[wi]
            costs.append(r.length_m * jobs[wi][0].cost_per_m)
            turns.append(turn_count(r.polyline))
            hugs.append(hug_mm(r.polyline))
        rows.append({
            "g": c[0], "l": c[1], "routed": routed,
            "cost": sum(costs) / len(costs) if costs else float("inf"),
            "turns": sum(turns) / len(turns) if turns else 0.0,
            "hug": sum(hugs) / len(hugs) if hugs else 0.0,
            "time": time_by[c],
        })

    # Rank: among the combos routing the most wires, prefer low cost, few turns and a
    # low hug distance. Combos that route fewer wires are gated down rather than
    # rewarded for the easier average.
    rmax = max(r["routed"] for r in rows)
    cmin = min(r["cost"] for r in rows)
    tmin = min(r["turns"] for r in rows) or 1.0
    hmin = min(r["hug"] for r in rows) or 1.0
    for r in rows:
        gate = 1.0 if r["routed"] == rmax else 0.6 * r["routed"] / rmax
        r["score"] = gate / (1.0 + (r["cost"] / cmin - 1.0)
                             + 0.5 * (r["turns"] / tmin - 1.0)
                             + 0.5 * (r["hug"] / hmin - 1.0))
    rows.sort(key=lambda r: r["score"], reverse=True)

    print(f"complex scene, {n} wires, res {RES}, conn 18 | metrics over the "
          f"{len(common)} wires ALL combos routed\n")
    print(f"{'rank':>4} {'global':>8} {'local':>12} {'routed':>8} {'cost($)':>9} "
          f"{'turns':>7} {'hug_mm':>8} {'time_ms':>9} {'score':>7}")
    for rk, r in enumerate(rows, 1):
        print(f"{rk:>4} {r['g']:>8} {r['l']:>12} {r['routed']:>5}/{n:<2} {r['cost']:>9.2f} "
              f"{r['turns']:>7.1f} {r['hug']:>8.0f} {r['time']:>9.0f} {r['score']:>7.3f}")
    print("\n(routed = honest count, endpoints-in-objects now correctly rejected; "
          "cost/turns/hug over the common set; hug LOWER = better surface-hug)")


if __name__ == "__main__":
    main()
