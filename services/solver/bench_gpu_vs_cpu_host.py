"""Host half of the GPU-vs-CPU benchmark: voxelization timings and bench-data prep.

Times Warp voxelization on CUDA against CPU at each resolution, then writes the grids and
the bench.json that bench_gpu_vs_cpu_stages.py consumes inside the container.

The scene must be a saved routing session (markers and wire settings authored by the
extension), since the benchmark reuses its wires and their weights.

Run from the repo root:  python3 services/solver/bench_gpu_vs_cpu_host.py <scene.usd>
"""
import json, sys, time
import numpy as np
from pxr import Usd, UsdGeom
sys.path.insert(0, "exts/omni.piperouter/omni")
sys.path.insert(0, "services/solver")
from piperouter import grid_io, headings, scene_ops, session_io, voxelizer, wire_library
from piperouter.router_session import RouterSession

if len(sys.argv) < 2:
    sys.exit(__doc__.strip().splitlines()[-1])
USD = sys.argv[1]
RESOLUTIONS = [int(r) for r in sys.argv[2:]] or [120, 180, 250]

stage = Usd.Stage.Open(USD)
mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
data = scene_ops.read_session(stage)
wires_d = session_io.deserialize(data)[0]
lib = wire_library.load_wire_library()

# ---- endpoints + specs (meters)
wires = []
for w in wires_d:
    s = scene_ops.get_world_pos(stage, f"{scene_ops.MARKERS_SCOPE}/{w['key']}_start")
    e = scene_ops.get_world_pos(stage, f"{scene_ops.MARKERS_SCOPE}/{w['key']}_end")
    if s is None or e is None: continue
    def hv(idx):
        lab = headings.HEADING_OPTIONS[idx] if 0 <= idx < len(headings.HEADING_OPTIONS) else "None"
        v = headings.axis_to_vector(lab)
        return list(v) if v else None
    wires.append({
        "name": w["name"],
        "spec": wire_library.as_spec(wire_library.by_id(lib, w["type_id"])),
        "start": [float(x) * mpu for x in s], "end": [float(x) * mpu for x in e],
        "weights": dict(w.get("weights", {})),
        "sh": hv(int(w.get("start_head_idx", 0))), "eh": hv(int(w.get("end_head_idx", 0)))})
json.dump({"wires": wires, "clearance_m": 0.0},
          open("/dev/shm/piperouter/bench.json", "w"))
print(f"bench.json written: {len(wires)} wires")

# ---- mesh soup, collected once
prims = scene_ops.list_collidable_meshes(stage)
bounds = scene_ops.compute_bounds(stage, prims)
bmin, bmax = np.asarray(bounds[0]) * mpu, np.asarray(bounds[1]) * mpu
mk = scene_ops.marker_positions(stage)
if mk:
    mp = np.asarray(mk, float) * mpu
    bmin, bmax = np.minimum(bmin, mp.min(0)), np.maximum(bmax, mp.max(0))
pad = (bmax - bmin) * 0.05 + 1e-3
bmin, bmax = bmin - pad, bmax + pad
pts, idx = voxelizer.collect_meshes(stage, prims)
pts = np.asarray(pts, dtype=np.float32) * mpu
print(f"meshes: {len(prims)}, tris: {len(idx)//3}")

vox = {}
for res in RESOLUTIONS:
    gbmin, cell, rxyz = grid_io.frame_from_bounds(bmin, bmax, res)
    row = {}
    for dev in ("cuda:0", "cpu"):
        voxelizer.voxelize(pts, idx, gbmin, cell, rxyz, device=dev)   # warm-up/compile
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            occ, sd = voxelizer.voxelize(pts, idx, gbmin, cell, rxyz, device=dev)
            ts.append(time.perf_counter() - t0)
        row[dev] = min(ts)
    ncells = int(np.prod(rxyz))
    vox[res] = {"cuda": row["cuda:0"], "cpu": row["cpu"],
                "cells": ncells, "occ": int(occ.astype(bool).sum()),
                "cell_mm": cell * 1000}
    print(f"res {res}: grid {rxyz} ({ncells/1e6:.1f}M cells, {cell*1000:.1f} mm) "
          f"voxelize cuda {row['cuda:0']*1e3:.0f} ms vs cpu {row['cpu']*1e3:.0f} ms")
    # Full grids, cost fields included, for the container-side stage.
    sess = RouterSession(grid_dir="/dev/shm/piperouter", solver_url="http://localhost:8000")
    sess.voxelize_scene(stage, f"bench_r{res}", resolution=res)
json.dump(vox, open("/dev/shm/piperouter/bench_vox.json", "w"))
print("grids + voxel timings saved")
