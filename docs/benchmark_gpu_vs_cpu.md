# GPU vs CPU benchmark — accelerated-3d-routing

**Scene:** `engine_demo.usd` (BEV engine bay, 8 meshes / 816 triangles, 8 cooling-circuit
wires with their session weights + one pinned heading). **Hardware:** same workstation for
both modes (RTX PRO 6000 vs its host CPUs) — identical code path, stage-instrumented; CPU
mode forces numpy edge-build, SciPy Dijkstra SSSP and SciPy smoothing.
Reproduce: `python3 services/solver/bench_gpu_vs_cpu_host.py` (host: voxelize + grid prep),
then `docker exec piperouter-solver python3 .../bench_gpu_vs_cpu_stages.py <stack.npz>
<bench.json> gpu|cpu [lattice]`.

## 1. Voxelization (NVIDIA Warp) — runs before EVERY solve

| resolution | grid | cell | GPU (CUDA) | CPU | speed-up |
|---|---|---|---|---|---|
| 120 | 0.1M cells | 26.7 mm | 0.5 ms | 45 ms | ~90× |
| 180 | 0.5M cells | 17.8 mm | 1.0 ms | 116 ms | ~115× |
| 250 | 1.2M cells | 12.8 mm | 1.3 ms | 247 ms | ~190× |

## 2. Exact search (dense heading-expanded lattice, 8 wires end-to-end)

The mode used when soft costs must be honoured exactly; ~26 direction-states per free
cell (≈30M states at res 250).

| resolution | GPU total | CPU total | speed-up | GPU SSSP | CPU SSSP | GPU build | CPU build |
|---|---|---|---|---|---|---|---|
| 120 | 3.9 s | 21.5 s | **5.4×** | 3.4 s | 16.8 s | 0.48 s | 4.6 s |
| 160 | 7.5 s | 54.8 s | **7.3×** | 6.4 s | 44.4 s | 1.0 s | 10.2 s |
| 180 | 9.7 s | 81.2 s | **8.3×** | 8.5 s | 66.5 s | 1.2 s | 14.6 s |
| 250 | 22.2 s | **235 s (3.9 min)** | **10.6×** | 19.9 s | 197.8 s | 1.9 s | 36.8 s |

The gap **widens with resolution** (5.4× → 10.6×): cuGraph SSSP is ~10× and the cupy
edge-build ~19× at res 250. On CPU, the exact solve is a coffee break (29 s/wire);
on GPU it stays inside a design conversation (2.8 s/wire).

## 3. Honest finding: the octree corridor makes THIS small scene CPU-sized

With the default `octree_lattice` (corridor prune, ~10× smaller graphs), the 8-wire
demo solves in ~1.4-1.7 s in BOTH modes — at corridor-sized graphs (≪1M edges),
cuGraph's launch/transfer overhead cancels its compute win. Implications:

* the octree is doing its job — it is a CPU-side algorithmic optimization;
* **actionable**: dispatch SSSP by graph size (SciPy below ~1M edges, cuGraph above)
  for the best of both;
* the GPU necessity claim rests on **scale**: this demo bay is 816 triangles at
  ≤1.2M cells. Production automotive CAD is millions of triangles, full-vehicle
  volumes, resolutions beyond 250, exactness requirements, and joint optimization
  (negotiation = N wires × rounds of re-solves) — every one of those multiplies the
  workload toward and beyond the dense-mode numbers above, where the GPU is already
  10× and pulling away.

**E2E at demo settings, exact mode, full pipeline (voxelize + route 8 wires):
GPU ≈ 22 s vs CPU ≈ 235 s — 10.6×, growing with problem size.**

## 4. High-resolution sweep (down to 5 mm cells, res 640)

Voxelization (Warp): 300 → 9.9 ms vs 383 ms (39×) · 400 → 21 ms vs 785 ms (38×) ·
640 (5 mm) → 119 ms vs 2516 ms (21×).

**Dense exact mode hits a wall for EVERYONE above ~13 mm cells:** at res 300 the CPU
needed **7.2 minutes** (431 s; SSSP 367 s) while the GPU path ran **out of memory** —
cuGraph's edge-list graph at ~1.35 B edges needs >90 GB transient device memory on this
card. At res 400+ (3.2 B edges) and 640 (13 B edges) neither side can even represent
the problem. Exact search at millimetre resolution is not a "faster hardware" question
— it requires the hierarchical pruning (octree) that the product already does.

**Production (octree) mode at 5 mm, 8 wires: ~13 s end-to-end either way** (GPU 13.3 s
incl. a 2.0 s one-off GPU-smoothing anomaly — production runs smoothing on CPU;
CPU 13.5 s). Per-stage at 640: cupy edge-build 2.2 s vs numpy 4.4 s (2×), SSSP parity
(corridor graphs still under cuGraph's efficiency threshold), and the new bottlenecks
are CPU-side stages: python octree subdivision (4.3 s) + dilations/fields/np-IO
("other", 4.1 s) — i.e. the **optimization roadmap**: Warp-ify dilation & octree,
size-dispatch SSSP.

## 5. Reconciling with the GUI stopwatch

The tables above time the SOLVE stages inside the container (grids preloaded, one
warm-up route). What ROUTE ALL in Kit measures on top: Warp module load + octree cache
on the first click, mesh collection + fields + grid save (~0.1-0.2 s), HTTP + npz load,
and USD tube authoring (~20 ms). Kit's own log at res 250 reads "voxelize 104 ms +
route 1653 ms = 1757 ms" — consistent with the 1.69 s solve-only figure here; a
hand-timed ~2.5 s includes UI refresh and warm-up.

## 6. Production mode (octree), full sweep — end-to-end (voxelize + 8-wire solve)

| resolution | cell | GPU e2e | CPU e2e | factor |
|---|---|---|---|---|
| 120 | 26.7 mm | 0.66 s | 0.52 s | CPU ×1.26 |
| 180 | 17.8 mm | 1.10 s | 1.03 s | CPU ×1.07 |
| 250 | 12.8 mm | 1.69 s | 1.67 s | ≈ tie |
| 300 | 10.7 mm | 2.06 s | 2.40 s | GPU ×1.17 |
| 400 | 8.0 mm | 4.00 s | 4.97 s | GPU ×1.24 |
| 640 | 5.0 mm | 11.46 s | 16.00 s | GPU ×1.40 |

**The crossover is at ~11 mm cells** — below it the pruned graphs are so small the CPU
is fine; above it the GPU pulls ahead and keeps widening. Per-stage GPU advantage in
octree mode rises monotonically with resolution: graph build ×0.83 → ×1.98 (break-even
at res ~250), cuGraph SSSP ×0.49 → ×1.07 (break-even at res ~640), voxelize ×21–×170
throughout. Every trend line points further in the GPU's favour as scenes grow.
(GPU totals exclude the 2.0 s one-off anomaly of the non-default GPU-smoothing path.)

Interactive visualization of all of the above: `docs/benchmark_gpu_vs_cpu.html`.

## 7. Extreme resolution (res 800 / 1000 — 4.0 / 3.2 mm) and Amdahl's law

| resolution | cell | GPU e2e | CPU e2e | factor | vox GPU/CPU |
|---|---|---|---|---|---|
| 800 | 4.0 mm | 23.2 s | 33.1 s | ×1.43 | 264 ms / 4.5 s (×17) |
| 1000 | 3.2 mm | 41.5 s | 57.2 s | ×1.38 | 487 ms / 7.6 s (×16) |

(The user-observed 44 s ROUTE ALL at res 1000 matches the 41–43 s GPU solve + handoff.)

**Why isn't the GPU winning more? Measured Amdahl.** At res 1000 the GPU-mode wall time
splits: octree build 16.6 s (Python, CPU) + dilations/fields/IO 16.2 s (SciPy/NumPy,
CPU) = **76% never touches the GPU**; graph build 7.1 s (cuPy, ×2.2 vs CPU 15.8 s) and
SSSP 1.0 s (cuGraph, ×1.2) are already GPU-won. Search fell from 20% of wall time at
res 250 to 2% at res 1000 — the bottleneck moved because the GPU did its job. The grey
(CPU) stages are all embarrassingly parallel → the roadmap: Warp-ify occupancy dilation
and cost fields, GPU octree build, keep SSSP size-dispatch. If those stages get even a
×10 (conservative for Warp on 78M-cell grids), res-1000 total drops to ~11-12 s.

## 8. Round 2 (2026-07-08) — the 76% attacked, remeasured

The section-7 roadmap is implemented, entirely inside the solver (no new dependencies,
equivalence-tested against the old code — same octree leaves, same dilations, same
cost fields; 125 solver + 88 extension tests green):

* **Octree build**: the pure-Python recursion became a **3D integral image** (padded
  prefix-sum, so any box's any/all test is 8 array lookups) with **level-vectorized
  subdivision** — Python cost is per level (~log n), not per box (millions). cuPy runs
  the prefix-sum on GPU when `PIPEROUTER_GPU_BUILD=1`. 16.6 s → 1.5 s at res 1000.
* **Leaf adjacency**: slab compares + edge dedup + CSR grouping, array ops end-to-end
  (cuPy on GPU) — no per-edge/per-node Python loop. → 0.2 s on GPU at res 1000.
* **Corridor A\***: leaf centres and per-edge lengths precomputed per scene octree;
  per-wire edge weights one vectorized pass. Same search, same costs. 8.4 s → 1.1 s.
* **Dilation**: SciPy `binary_dilation` → shifted-OR slab kernel, identical output,
  runs on NumPy or cuPy (`grids.dilate6`), used by occupancy/class dilation and prior-
  route marking.
* **Cost fields / melt masks**: normalized fields and combined soft cost cached per
  stack + weights (was recomputed per wire); melt mask cached per temperature rating.
* **Grid handoff**: `/dev/shm` npz now uncompressed (it's tmpfs — zlib only burned CPU
  on both sides), and the service caches the loaded `GridStack` per session (mtime-
  keyed), so repeat solves of the same scene also reuse the scene octree and fields.

| stage (8-wire solve) | res 250 GPU / CPU | res 640 GPU / CPU | res 1000 GPU / CPU |
|---|---|---|---|
| octree (corridor build + adjacency + A*) | 0.14 / 0.09 s | 0.67 / 0.97 s | 2.32 / 5.40 s |
| graph build (heading lattice) | 0.46 / 0.50 s | 0.85 / 3.54 s | 2.11 / 12.34 s |
| SSSP (cuGraph / SciPy Dijkstra) | 0.38 / 0.24 s | 0.71 / 0.77 s | 1.05 / 1.31 s |
| dilations · fields · IO · glue | 0.04 / 0.08 s | 0.41 / 1.08 s | 1.78 / 5.41 s |
| **solve total** | **1.05 / 0.92 s** | **2.66 / 6.37 s** | **7.28 / 24.50 s** |
| voxelize (Warp) | 0.001 / 0.247 s | 0.119 / 2.516 s | 0.487 / 7.648 s |
| **end-to-end** | **1.05 / 1.17 s (×1.11)** | **2.78 / 8.89 s (×3.20)** | **7.77 / 32.15 s (×4.14)** |

Smoothing runs its production CPU path in both modes (~0.01-0.02 s, folded into glue);
GPU totals exclude the known one-off of the non-default GPU-smoothing path. At 12.8 mm
the pruned graphs are small enough that the CPU solve is at parity and the GPU case
rests on voxelize; at 5 mm and below every stage favours the GPU. A repeat ROUTE ALL
on an unchanged scene over HTTP: 0.9 s at res 250, 5.0 s at res 1000 (the service
caches the loaded `GridStack` per session, so scene octree + fields carry over).

## 9. Round 3 (2026-07-08) — the residual host work, moved to the device

* **Octree subdivision fully on xp**: the integral image already ran on cuPy; now the
  per-level box classification, child generation and the leaf list stay on the device
  too (leaves return as one `(n,6)` array instead of ~1M Python tuples), and `leaf_of`
  went int64 → int32. 1.5 s → ~0.6 s at res 1000.
* **Per-leaf soft means**: was a 78M-cell gather + two `np.bincount` per wire; now one
  xp bincount cached across wires sharing weights, with exact denominators from box
  volumes. ~2.8 s → ~0.1 s.
* **Corridor band**: the per-point box-painting loop became a scatter + separable box
  dilation on xp (also fixed a latent slice-wraparound bug for heading rays leaving
  the grid more than r+1 cells below an edge).
* **Lattice build on-device ordinals**: the free-cell ordinal grid (was a grid-sized
  int64 host array uploaded per wire, ~624 MB at res 1000) and `argwhere` are built
  directly on cuPy; the soft field's device copy is cached per weights. Host↔device
  traffic per wire dropped to the free mask (~78 MB).
* **Corridor A\* kept on CPU deliberately**: measured cuGraph SSSP on the leaf graph
  is roughly cost-neutral vs the float-only A* (~0.13 s/call either way) and changes
  tie-breaking — no win, extra risk.

Everything remains equivalence-tested (127 solver + 88 extension tests green).

## 10. Fleet scaling — 8 → 50 wires (2026-07-08)

Fleets built from the 8 real connectors plus jittered copies (endpoints validated
against the occupancy grid), routed sequentially with earlier wires as obstacles.
One-time work (octree, cost fields) is cached across the fleet, so scaling is governed
by the per-wire search — the GPU's stage (~0.9 s/wire GPU vs ~3.8 s/wire CPU at 3.2 mm).

| fleet | res 640 (5 mm) GPU / CPU | res 1000 (3.2 mm) GPU / CPU |
|---|---|---|
| 8 wires | 2.7 / 6.4 s | 7.3 / 24.5 s |
| 20 wires | 6.8 / 19.2 s | 16.6 / 63.6 s |
| 30 wires | 10.9 / 28.5 s | 23.0 / 87.0 s |
| 50 wires | 17.2 / 50.1 s (×2.9) | 44.9 / 185.3 s (×4.1) |

Routed counts are identical between modes at every point except 50 wires / 3.2 mm
(GPU 45, CPU 46 of 50 — one equal-cost tie-break lands differently). Wires a fleet
genuinely cannot fit report no_path honestly (see below).

**Solver change shipped with this experiment** (found because 30-wire fleets broke it):
under congestion the ±4-cell corridor band can be saturated by prior routes; the old
code then fell back to the FULL lattice — infeasible at high resolution (13 B edges at
res 640: GPU OOM, CPU effectively hangs). Now the planner (1) escalates the band 4 → 12
cells before considering full fallback, and (2) skips the full-lattice fallback when its
graph would exceed ~1 B edges (measured infeasibility threshold), reporting no_path with
a logged reason instead of taking the whole solve down. Small grids keep the old
never-worse fallback. 130 solver + 88 extension tests green.

## 11. The editing loop — warm single-wire re-route (2026-07-08)

Drag-to-edit measured as the app experiences it: scene solved once (caches warm), then
each wire re-solved with its endpoint nudged 3 cells. Median of 24 re-routes:

| cell size | GPU | CPU | |
|---|---|---|---|
| 5 mm (res 640) | **264 ms** (p90 359) | 660 ms (p90 776) | ×2.5 |
| 3.2 mm (res 1000) | **533 ms** (p90 758) | 2 082 ms (p90 2 350) | ×3.9 |

Both modes reuse the same cached scene octree and cost fields, so the difference is
purely the per-wire search. Below ~500 ms a re-route reads as live feedback under the
cursor; at 2 s every nudge is a stall — at millimetre resolution the live-editing
workflow exists only on the GPU. Derived from the same data: at a fixed ~3 s ROUTE ALL
budget the CPU affords 8 mm cells and the GPU 5 mm — 4× more cells for the same wait.
