# omni.piperouter

Constraint-aware wire / pipe routing for automotive USD, inside Omniverse Kit
(USD Composer, Isaac Sim, or any kit-app-template app).

The extension reads the live USD stage, **voxelizes it with Warp in-process** (occupancy
+ surface-distance + thermal + EM fields), hands the grids to a **cuGraph solver running
in a Docker container**, and authors the returned routes as tubes — with a two-phase
expert workflow: *Route All*, then *refine one wire at a time* with waypoints.

## Architecture

```
omni.piperouter (Kit, this extension)            piperouter-solver (Docker, cuGraph/GPU)
  panel.py / extension.py  ── omni.ui UI            app.py        FastAPI /solve /solve_all
  voxelizer.py  ── Warp (GPU) occupancy+dist        solver.py     direction-aware lattice + SSSP
  fields.py     ── thermal / EM splat               obstacles.py  locked-route rasterization
  scene_ops.py  ── markers, tubes, tags, overlay
  router_session.py ── orchestration                grids handed over via /dev/shm/piperouter
  solver_client.py  ── HTTP (stdlib urllib) ───────► http://localhost:8000
```

Only `extension.py` / `panel.py` import `omni.*`. Everything else is pure
`pxr`/`warp`/`numpy`/stdlib and is covered by headless pytest under `tests/`.

## Running

1. **Start the solver** (GPU workstation, from the repo root):
   ```bash
   docker compose up --build -d        # builds piperouter-solver, serves :8000
   curl http://localhost:8000/health   # {"status":"ok","backend":"gpu"}
   ```
   The container mounts `/dev/shm/piperouter` so it reads the grids the extension writes.

2. **Enable the extension** in your Kit app (USD Composer / Isaac Sim):
   add this repo's `exts/` folder to *Window ▸ Extensions ▸ search paths*, then enable
   **omni.piperouter**. A "PipeRouter" window appears.

## Workflow

The panel is organized into collapsible sections: **Scene & Setup · Wires · Selected
wire · Tagging · Output**, with a colored connection status line on top.

- **Quick start:** click **Create sample scene** (Scene & Setup) — it builds a procedural
  mini engine-bay (ground, a firewall with a gap, a hot engine block, an EM component)
  and pre-places three wires (power / CAN / AC pipe). Then click **ROUTE ALL**.
- **Phase A — Route All:** or click *+ Add wire* (spawns green start / red end markers),
  drag them in the viewport, pick a wire type, set the grid resolution, click
  **ROUTE ALL**. Routes appear as colored tubes; each wire row shows a status dot
  (green routed / red no-path / blue locked), its color swatch, and inline length/$; the
  BOM fills in under Output. Wires can be renamed and deleted per row.
- **Phase B — Refine one wire:** click a wire to select it. Tune the soft-constraint
  sliders (surface-hug / bend / thermal / EM), **+ Add waypoint** (drops a blue marker
  the route must pass through), then **Re-route this wire** — only that wire re-solves,
  with every *Locked* wire treated as an obstacle. **Lock** it when happy.
- **Tagging:** select a mesh/xform, enter a temperature °C and/or EM strength, click
  *Tag* — writes `piperouter:temp_c` / `piperouter:em_strength` onto the prim (persisted
  in the USD). These drive the thermal/EM cost fields on the next solve.
- **Overlay:** toggle *Show occupancy overlay* to see the voxelized obstacles.
- **Export BOM:** writes `<path>.json` and `<path>.csv`.

## Constraints (see the design spec)

Hard: collision + per-wire clearance, thermal melt cutoff (`max_temp_c`), waypoints.
Soft (weighted edge cost): surface-hug, thermal, EM (× wire `em_sensitivity`), and the
bend/turn penalty (scaled by `min_bend_radius_mm`). Wire types live in
`data/wire_types.json`.

## Tests
```bash
cd exts/omni.piperouter && <repo>/.venv/bin/pytest -p no:pqm -q   # 16 headless tests
```
The `omni.ui` panel itself is verified by loading the extension in a real Kit runtime.

## Notes / phase 2
- USD writes happen on the main thread (Route All briefly blocks the UI); moving the
  voxelize+solve off-thread with progress is a follow-up.
- Phase 2 hooks: cuSolver curvature smoothing, cuOpt multi-wire bundling, hierarchical
  -corridor lattice (Option 2), real thermal/EM field solves.
