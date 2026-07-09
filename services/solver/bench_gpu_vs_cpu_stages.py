"""Per-stage GPU-vs-CPU benchmark of the routing solve. Runs INSIDE the container.
Usage: python3 bench_stages.py <stack.npz> <bench.json> <gpu|cpu>
Wraps the production code path (octree_lattice + fibre) with timers; CPU mode forces
numpy edge-build, scipy Dijkstra SSSP and scipy smoothing on the same machine."""
import json, os, sys, time

import numpy as np

mode = sys.argv[3]
PLANNER = sys.argv[4] if len(sys.argv) > 4 else "octree_lattice"
os.environ["PIPEROUTER_GPU_BUILD"] = "1" if mode == "gpu" else "0"
os.environ["PIPEROUTER_GPU_SMOOTH"] = "1" if mode == "gpu" else "0"

from piperouter_solver import backend, planners, smoothing
from piperouter_solver.grids import GridStack
from piperouter_solver.lattice import ExpandedLatticeBuilder
from piperouter_solver.models import RouteRequest, WireType
from piperouter_solver.solver import Solver

T = {"octree": 0.0, "graph_build": 0.0, "sssp": 0.0, "smoothing": 0.0}
def _wrap(fn, key, forced=None):
    def w(*a, **k):
        t0 = time.perf_counter()
        r = (forced or fn)(*a, **k)
        T[key] += time.perf_counter() - t0
        return r
    return w

planners.octree_leaves = _wrap(planners.octree_leaves, "octree")
planners.leaf_adjacency = _wrap(planners.leaf_adjacency, "octree")
planners.octree_corridor = _wrap(planners.octree_corridor, "octree")
ExpandedLatticeBuilder.build = _wrap(ExpandedLatticeBuilder.build, "graph_build")
planners.shortest_path = _wrap(planners.shortest_path, "sssp",
                               forced=backend._scipy_sssp if mode == "cpu" else None)
smoothing.smooth_path = _wrap(smoothing.smooth_path, "smoothing")

stack = GridStack.load(sys.argv[1])
bench = json.load(open(sys.argv[2]))
reqs = []
for w in bench["wires"]:
    spec = w["spec"]
    wt = WireType(id=w["name"], label=w["name"], kind=spec["kind"],
                  outer_diameter_mm=spec["outer_diameter_mm"],
                  min_bend_radius_mm=spec["min_bend_radius_mm"],
                  cost_per_m=spec.get("cost_per_m", 0.0),
                  mass_per_m_kg=spec.get("mass_per_m_kg", 0.0),
                  max_temp_c=spec.get("max_temp_c", 1e9),
                  em_sensitivity=spec.get("em_sensitivity", 0.0),
                  inner_diameter_mm=spec.get("inner_diameter_mm", 0.0),
                  color=(0.8, 0.2, 0.2))
    reqs.append(RouteRequest(
        wire=wt, start=tuple(w["start"]), end=tuple(w["end"]),
        weights=dict(w.get("weights", {})), connectivity=26,
        start_heading=tuple(w["sh"]) if w.get("sh") else None,
        end_heading=tuple(w["eh"]) if w.get("eh") else None,
        global_planner=PLANNER,
        clearance_m=float(bench.get("clearance_m", 0.0))))

# warm-up (JIT, cuda context, caches) on the first wire, then reset and measure
Solver().route_one(stack, reqs[0])
for c in ("_octree_cache", "_dilate_cache", "_cls_dilate_cache",
          "_norm_fields", "_soft_cache", "_melt_cache"):
    stack.__dict__.pop(c, None)               # measure every build/cache honestly once
for k in T: T[k] = 0.0

t0 = time.perf_counter()
report = Solver().route_all(stack, reqs)
total = time.perf_counter() - t0
routed = sum(1 for r in report.results if r.status == "routed")
out = {"mode": mode, "planner": PLANNER, "routed": routed, "n": len(reqs), "total_s": total}
out.update({k: v for k, v in T.items()})
out["other_s"] = total - sum(T.values())
print("BENCH_JSON " + json.dumps(out))
