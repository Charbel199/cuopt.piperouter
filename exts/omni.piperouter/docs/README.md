# omni.piperouter

Constraint-aware 3D routing of cables and pipes in automotive USD, inside Omniverse Kit
(USD Composer, Isaac Sim, or any kit-app-template app). Part of the
[accelerated-3d-routing](https://github.com/Charbel199/accelerated-3d-routing) project.

The extension reads the live USD stage and voxelizes it with NVIDIA Warp in-process,
producing occupancy, surface-distance, thermal and EM fields. It hands those grids to an
NVIDIA cuGraph solver running in a Docker container (with a scipy Dijkstra CPU fallback)
and authors the returned routes as tubes. The workflow has two phases: route everything,
then refine one wire at a time with waypoints.

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

Only `extension.py` and `panel.py` import `omni.*`. Everything else is pure
`pxr`/`warp`/`numpy`/stdlib and is covered by the headless pytest suite in `tests/`.

## Running

1. Start the solver on a GPU workstation, from the repo root:
   ```bash
   docker compose up --build -d        # builds piperouter-solver, serves :8000
   curl http://localhost:8000/health   # {"status":"ok","backend":"gpu"}
   ```
   The container mounts `/dev/shm/piperouter` so it reads the grids the extension writes.

2. Enable the extension in your Kit app: add this repo's `exts/` folder to
   *Window ▸ Extensions ▸ search paths*, then enable **omni.piperouter**. A PipeRouter
   window appears.

## Workflow

The panel is organized into collapsible sections (Scene & Setup, Wires, Selected wire,
Tagging, Output) with a colored connection status line on top.

To see it working straight away, click **Create sample scene** under Scene & Setup. It
builds a procedural mini engine bay (ground, a firewall with a gap, a hot engine block,
an EM component) and pre-places three wires: power, CAN, and an AC pipe. Then click
**ROUTE ALL**.

Routing everything: click **+ Add wire** to spawn a green start marker and a red end
marker, drag them in the viewport, pick a wire type, set the grid resolution, and click
**ROUTE ALL**. Routes appear as colored tubes. Each wire row shows a status dot (green
for routed, red for no path, blue for locked), its color swatch, and inline length and
cost. The BOM fills in under Output. Wires can be renamed and deleted per row.

Refining one wire: click a wire to select it. Tune the soft-constraint sliders
(surface-hug, bend, thermal, EM), use **+ Add waypoint** to drop a blue marker the route
must pass through, then click **Re-route this wire**. Only that wire re-solves, and every
locked wire is treated as an obstacle. Click **Lock** when you are happy with it.

Tagging: select a mesh or xform, enter a temperature in °C and/or an EM strength, then
click **Tag**. This writes `piperouter:temp_c` and `piperouter:em_strength` onto the prim,
persisted in the USD, and those drive the thermal and EM cost fields on the next solve.

The *Show occupancy overlay* toggle displays the voxelized obstacles. **Export BOM**
writes both `<path>.json` and `<path>.csv`.

## Constraints

Hard constraints are collision, per-wire clearance, the thermal melt cutoff
(`max_temp_c`), and waypoints. Soft constraints enter the edge cost as weights:
surface-hug, thermal, EM scaled by the wire's `em_sensitivity`, and a bend penalty scaled
by `min_bend_radius_mm`. Wire types live in `data/wire_types.json`.

## Tests

```bash
cd exts/omni.piperouter && <repo>/.venv/bin/pytest -p no:pqm -q   # 88 headless tests
```

The `omni.ui` panel itself is verified by loading the extension in a real Kit runtime.

## Known limitations

USD writes happen on the main thread, so Route All briefly blocks the UI; moving
voxelize and solve off-thread with a progress indicator is still to do. Candidates for
future work: design-space keep-in zones, drag-to-edit local re-solve, raceway corridors,
a bundle-diameter formula, and real thermal/EM field solves rather than splats.
