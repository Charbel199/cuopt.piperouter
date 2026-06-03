# PipeRouter

Constraint-aware routing of electric wires and AC/fluid pipes through a car (USD),
inside NVIDIA Omniverse. An Omniverse Kit extension voxelizes the scene with **Warp**
and an in-its-own-container **cuGraph** service solves the routes on the GPU. This is
the v2 redesign that replaces the original cuOpt POC (see `omni/cuopt/`).

![Engine bay with routed pipes](docs/pipes_output.png)

## Architecture

| Piece | Where | What |
|---|---|---|
| `services/solver/piperouter_solver` | pure Python (GPU-optional) | grids → direction-aware weighted lattice → SSSP → polyline; all constraint math |
| `services/solver/piperouter_service` | **Docker, cuGraph/GPU** | FastAPI `/solve` `/solve_all`; grids handed over via `/dev/shm/piperouter` |
| `exts/omni.piperouter` | **Omniverse Kit** | omni.ui panel, Warp voxelization + thermal/EM fields, USD tube authoring, the expert workflow |

Constraints: **hard** = collision + per-wire clearance, thermal melt cutoff, waypoints;
**soft** (weighted edge cost) = surface-hug, thermal, EM (× wire sensitivity), bend/turn
penalty. Wire types (cost/mass/diameter/bend/temp) live in `wire_types.json`.

Design + plans: `docs/superpowers/specs/2026-06-03-piperouter-cugraph-extension-design.md`
and `docs/superpowers/plans/2026-06-03-piperouter-m{1,2,3,4}-*.md`.

## Quick start

```bash
# 1. Start the GPU solver (workstation with an NVIDIA GPU + Container Toolkit)
docker compose up --build -d
curl http://localhost:8000/health        # {"status":"ok","backend":"gpu"}

# 2. Enable the extension in USD Composer / Isaac Sim:
#    Window > Extensions > add this repo's exts/ to the search paths > enable omni.piperouter
```

See `exts/omni.piperouter/docs/README.md` for the full workflow (Route All → refine one
wire with waypoints → lock → BOM export) and `docker-compose.yml` for the service.

## Development

```bash
python3 -m venv .venv && .venv/bin/pip install -e services/solver
.venv/bin/pip install numpy scipy pytest fastapi "uvicorn[standard]" httpx usd-core warp-lang

# solver core + service (51 tests)
cd services/solver && ../../.venv/bin/pytest -p no:pqm -q
# extension headless logic (16 tests: Warp voxelize, USD authoring, real-HTTP solve)
cd exts/omni.piperouter && ../../.venv/bin/pytest -p no:pqm -q
```

GPU paths (cuGraph in the container, Warp in the extension) run on the hardware; the
solver core falls back to scipy dijkstra so tests pass without a GPU.

## Status

M1 (solver core), M2 (GPU service), M3 (extension fields+routing), M4 (refine loop +
tagging + overlay + BOM export) are built and tested. Phase 2 (not yet built): cuSolver
curvature smoothing, cuOpt multi-wire bundling, hierarchical-corridor lattice, real
thermal/EM field solves.
