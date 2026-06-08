"""Profile the lattice's GPU stages — edge-build (cupy) vs SSSP (cuGraph) — per wire, so
we can see where a GPU win would have to come from. Solver-only (runs in the container).

    PIPEROUTER_GPU_BUILD=1 python3 bench_gpu_stages.py <grid.npz> <endpoints.json>
"""
from __future__ import annotations

import json
import os
import sys
import time

from piperouter_solver.backend import shortest_path
from piperouter_solver.grids import GridStack
from piperouter_solver.lattice import ExpandedLatticeBuilder
from piperouter_solver.models import WireType

WEIGHTS = {"surface": 1.0, "bend": 1.0, "thermal": 1.0, "em": 1.0}
CONN = 26


def _backend():
    import importlib.util as u
    print(f"cupy={'YES' if u.find_spec('cupy') else 'no'} | "
          f"cuGraph={'YES' if u.find_spec('cugraph') else 'no'} | "
          f"GPU_BUILD={os.environ.get('PIPEROUTER_GPU_BUILD', '0')}\n")


def main():
    stack = GridStack.load(sys.argv[1])
    eps = json.load(open(sys.argv[2]))
    _backend()
    b = ExpandedLatticeBuilder()

    # warmup so cupy/cuGraph JIT + first-alloc don't pollute the timings
    e = eps[0]
    w = WireType(**e["wire"])
    g = b.build(stack, w, WEIGHTS, CONN, tuple(e["start"]), tuple(e["end"]), None, clearance_m=0.0)
    shortest_path(g.src, g.dst, g.weight, g.n_nodes, g.source_id, g.sink_id)

    tot_build = tot_sssp = 0.0
    n = edges = nodes = 0
    for e in eps:
        w = WireType(**e["wire"])
        t0 = time.perf_counter()
        g = b.build(stack, w, WEIGHTS, CONN, tuple(e["start"]), tuple(e["end"]),
                    None, clearance_m=0.0)
        t1 = time.perf_counter()
        shortest_path(g.src, g.dst, g.weight, g.n_nodes, g.source_id, g.sink_id)
        t2 = time.perf_counter()
        tot_build += (t1 - t0)
        tot_sssp += (t2 - t1)
        edges = int(g.src.size)
        nodes = int(g.n_nodes)
        n += 1

    print(f"per wire (avg over {n}, conn {CONN}):  nodes={nodes}  edges={edges}")
    print(f"  edge-build (cupy):  {tot_build / n * 1e3:8.1f} ms")
    print(f"  SSSP (cuGraph):     {tot_sssp / n * 1e3:8.1f} ms")
    print(f"  build+sssp total:   {(tot_build + tot_sssp) / n * 1e3:8.1f} ms")


if __name__ == "__main__":
    main()
