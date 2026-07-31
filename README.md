# Accelerated 3D Routing

Constraint-aware 3D routing of cables and pipes through automotive USD scenes, inside
NVIDIA Omniverse. An Omniverse Kit extension (window name: PipeRouter) voxelizes the
stage with NVIDIA Warp and hands the grids to a containerized solver, which finds the
routes with NVIDIA cuGraph shortest paths on the GPU. Electric wires, CAN buses, AC
lines and cooling circuits are all routed the same way, with different constraints.

![Engine bay with routed pipes](docs/pipes_output.png)

## Architecture

| Piece | Where | What |
|---|---|---|
| `services/solver/piperouter_solver` | pure Python, GPU optional | grids → direction-aware weighted lattice (default: adaptive `octree_lattice`) → SSSP → fibre-neutre smoothing; all constraint math |
| `services/solver/piperouter_service` | Docker, cuGraph/GPU | FastAPI `/solve` and `/solve_all`; grids handed over via `/dev/shm/piperouter` |
| `exts/omni.piperouter` | Omniverse Kit | omni.ui panel, Warp voxelization, thermal/EM fields, USD tube authoring |

Built on NVIDIA Warp for GPU voxelization in-process in Kit, NVIDIA cuGraph for
single-source shortest paths in the solver container, numpy and scipy for the solver
core, FastAPI for the service, and OpenUSD with omni.ui for the extension. There is a
scipy Dijkstra fallback throughout, so everything also runs without a GPU.

Hard constraints are collision and safety clearance (a global default or per-object
clearance tags), a thermal melt cutoff, waypoints, and pinned start/end headings from
either axis presets or a rotatable arrow gizmo. Soft constraints enter as weighted edge
cost: surface-hug, thermal, EM scaled by wire sensitivity, a bend penalty, and smoothing
strength. Wire types (cost, mass, diameter, bend radius, temperature rating) live in
`wire_types.json`.

## Quick start

```bash
# 1. Start the GPU solver (workstation with an NVIDIA GPU + Container Toolkit)
docker compose up --build -d
curl http://localhost:8000/health        # {"status":"ok","backend":"gpu"}

# 2. Enable the extension in USD Composer / Isaac Sim:
#    Window > Extensions > add this repo's exts/ to the search paths > enable omni.piperouter
```

`exts/omni.piperouter/docs/README.md` has the full workflow: Route All, refining a single
wire with waypoints, locking it, then exporting a BOM. Service wiring is in
`docker-compose.yml`.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e services/solver
.venv/bin/pip install numpy scipy pytest fastapi "uvicorn[standard]" httpx usd-core warp-lang

# solver core + service
cd services/solver && ../../.venv/bin/pytest -p no:pqm -q          # 130 tests
# extension headless logic (Warp voxelize, USD authoring, real-HTTP solve)
cd exts/omni.piperouter && ../../.venv/bin/pytest -p no:pqm -q     # 88 tests
```

The GPU paths (cuGraph in the container, Warp in the extension) run on real hardware
when it is present. The solver core falls back to scipy Dijkstra, so the test suites
pass on a machine without a GPU.

## Status

Working today: the solver core, the GPU service, and a Kit extension covering Route All,
per-wire refinement with waypoints (double-click a tube to drop one), bundles with shared
trunks, the start/end heading gizmo, thermal/EM/clearance tagging including instanced
CAD, buried-endpoint rescue, occupancy/thermal/EM overlays with per-wire debug views,
session save/load into a single USD, and BOM export.

The default planner is the adaptive `octree_lattice`, which confines the search to a
corridor and is roughly an order of magnitude faster than the dense lattice at high
resolution. `docs/benchmark_gpu_vs_cpu.md` has the measured GPU-versus-CPU numbers per
stage and per resolution.

Possible next steps: design-space keep-in zones, drag-to-edit local re-solve, raceway
corridors, a bundle-diameter formula, and pressure-drop checks for cooling circuits.
