"""Routing orchestration, omni-free so it is fully testable headlessly.

voxelize_scene: read stage -> bounds -> Warp voxelize + thermal/EM fields -> write
grids to the shared dir. route_all: build route payloads from the wire list, call the
solver, author tubes, compute the BOM.

A `wire` dict: {name (unique route id), spec (wire-type props), start, end,
waypoints, weights, connectivity, priority}.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from . import fields, grid_io, scene_ops, voxelizer
from .solver_client import SolverClient

# Logs surface in the Kit console (and stderr headless) tagged [piperouter].
log = logging.getLogger("piperouter")


# Authored tubes render at least this thick so routes are visible at scene scale.
# Purely cosmetic — BOM and clearance use the physical wire spec, not this.
MIN_DISPLAY_DIAMETER_M = 0.04

# Thermal/EM fields radiate over (object characteristic size + this margin), with a
# floor so even small tagged prims have a usable reach. Tune to taste.
FIELD_MARGIN_M = 1.0
MIN_FALLOFF_M = 0.75


class RouterSession:
    def __init__(self, grid_dir="/dev/shm/piperouter", solver_url="http://localhost:8000"):
        self.grid_dir = Path(grid_dir)
        self.client = SolverClient(solver_url)
        self.frame = None         # (bounds_min, cell_size, res_xyz) from the last voxelize
        self.last_stats = {}      # stats from the last compute_grids (for logging/UI)
        self.last_grids = None    # (bounds_min, cell, res, occ, thermal, em) for views/overlay

    def compute_grids(self, stage, resolution=64, pad_frac=0.05, clearance_m=0.0):
        """Read the stage and build the four voxel grids (occupancy, surface distance,
        thermal, EM) IN MEMORY. The safety clearance is BAKED INTO the occupancy here
        (occupied cells dilated by round(clearance/cell)), so the single saved grid is
        the prohibited-voxel set used by the router AND shown by the overlay/2D views —
        one source of truth. Shared by voxelize_scene and the debug overlay.

        Returns: (bounds_min, cell_size, res_xyz, occupancy, surface_dist, thermal, em).
        """
        t0 = time.perf_counter()
        prims = scene_ops.list_collidable_meshes(stage)
        bounds = scene_ops.compute_bounds(stage, prims)
        if bounds is None:
            raise ValueError("no collidable geometry in stage")
        bmin, bmax = bounds
        # Grow bounds to include all markers (endpoints + waypoints) so routing to a
        # marker dragged beyond the geometry isn't clamped to the grid edge.
        markers = scene_ops.marker_positions(stage)
        if markers:
            mp = np.asarray(markers, dtype=float)
            bmin = np.minimum(bmin, mp.min(axis=0))
            bmax = np.maximum(bmax, mp.max(axis=0))
        pad = (bmax - bmin) * pad_frac + 1e-3
        bmin, bmax = bmin - pad, bmax + pad

        gbmin, cell, res = grid_io.frame_from_bounds(bmin, bmax, resolution)
        pts, idx = voxelizer.collect_meshes(stage, prims)
        occ, sd = voxelizer.voxelize(pts, idx, gbmin, cell, res)

        # BAKE the safety clearance into the occupancy: grow prohibited voxels by
        # round(clearance/cell). This IS the keep-out the router avoids and the views
        # show — change clearance -> more prohibited voxels -> re-solve.
        clearance_cells = int(float(clearance_m) / cell + 0.5 + 1e-9)
        if clearance_cells > 0:
            occ = grid_io.dilate_mask(occ.astype(bool), clearance_cells).astype(np.uint8)

        # Thermal / EM fields: each tagged prim becomes a source at its bbox centre.
        # The field radiates over `char_size + FIELD_MARGIN_M` so a big hot block heats
        # a region proportional to its size (a fixed falloff vanished once the scene
        # was scaled up). MIN_FALLOFF keeps tiny tagged prims from having ~zero reach.
        tags = scene_ops.read_thermal_em_tags(stage)
        thermal_sources = [(c, t, max(MIN_FALLOFF_M, char + FIELD_MARGIN_M))
                           for (c, t, e, char) in tags if t is not None]
        em_sources = [(c, e, max(MIN_FALLOFF_M, char + FIELD_MARGIN_M))
                      for (c, t, e, char) in tags if e is not None]
        thermal = fields.thermal_field(gbmin, cell, res, thermal_sources)
        em = fields.em_field(gbmin, cell, res, em_sources)

        self.last_stats = {
            "res": tuple(int(v) for v in res),
            "cells": int(res[0] * res[1] * res[2]),
            "occupied": int(occ.sum()),
            "n_thermal_sources": len(thermal_sources),
            "n_em_sources": len(em_sources),
            "thermal_max_c": float(thermal.max()),
            "seconds": round(time.perf_counter() - t0, 2),
        }
        log.info("[piperouter] voxelized %s cells (%d occupied), %d thermal / %d EM "
                 "sources, max %.0f°C, %.2fs",
                 self.last_stats["res"], self.last_stats["occupied"],
                 len(thermal_sources), len(em_sources),
                 self.last_stats["thermal_max_c"], self.last_stats["seconds"])
        return gbmin, cell, res, occ, sd, thermal, em

    def voxelize_scene(self, stage, session_id, resolution=64, pad_frac=0.05, clearance_m=0.0):
        gbmin, cell, res, occ, sd, thermal, em = self.compute_grids(
            stage, resolution, pad_frac, clearance_m=clearance_m)
        path = self.grid_dir / session_id / "stack.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        grid_io.save_grids(path, gbmin, cell, res, occ, sd, thermal, em)
        self.frame = (gbmin, cell, res)
        self.last_grids = (gbmin, cell, res, occ, thermal, em)  # cached for views/overlay
        return session_id

    def route_all(self, stage, session_id, wires):
        """Returns (results, bom). Results are matched back to wires by unique name."""
        routes = []
        for w in wires:
            spec = dict(w["spec"])
            spec["id"] = w["name"]  # unique per-route id (echoed back as wire_id)
            routes.append({
                "wire": spec,
                "start": [float(x) for x in w["start"]],
                "end": [float(x) for x in w["end"]],
                "waypoints": [[float(x) for x in wp] for wp in w.get("waypoints", [])],
                "weights": dict(w.get("weights", {})),
                "connectivity": int(w.get("connectivity", 26)),
                "priority": int(w.get("priority", 0)),
                "clearance_m": 0.0,  # already baked into the voxel grid (compute_grids)
                "start_heading": w.get("start_heading"),
                "end_heading": w.get("end_heading"),
            })

        resp = self.client.solve_all(session_id, routes)
        by_name = {w["name"]: w for w in wires}

        scene_ops.clear_routes(stage)
        bom = []
        for res in resp["results"]:
            w = by_name[res["wire_id"]]
            spec = w["spec"]
            if res["status"] != "routed":
                bom.append({"wire_id": res["wire_id"], "status": res["status"],
                            "length_m": 0.0, "cost": 0.0, "mass": 0.0})
                continue
            diameter = max(float(spec["outer_diameter_mm"]) / 1000.0, MIN_DISPLAY_DIAMETER_M)
            color = spec.get("color", (0.8, 0.1, 0.1))
            scene_ops.author_tube(
                stage, f"{scene_ops.ROUTES_SCOPE}/{res['wire_id']}",
                res["polyline"], diameter, color)
            length = float(res["length_m"])
            bom.append({
                "wire_id": res["wire_id"], "status": "routed", "length_m": length,
                "cost": length * float(spec.get("cost_per_m", 0.0)),
                "mass": length * float(spec.get("mass_per_m_kg", 0.0)),
            })
        return resp["results"], bom

    def refine_wire(self, stage, session_id, wire, locked_wires):
        """Re-route a single wire (honouring its waypoints + weights) while every
        other routed wire in `locked_wires` acts as an obstacle, so the re-routed
        wire never overlaps the rest. Authors/replaces only this wire's tube.
        Returns (result_dict, bom_row)."""
        locked_routes = []
        for lw in locked_wires:
            poly = lw.get("polyline")
            if not poly:
                continue
            locked_routes.append({
                "polyline": [[float(x) for x in p] for p in poly],
                "outer_diameter_mm": float(lw["spec"]["outer_diameter_mm"]),
            })

        spec = dict(wire["spec"])
        spec["id"] = wire["name"]
        route = {
            "wire": spec,
            "start": [float(x) for x in wire["start"]],
            "end": [float(x) for x in wire["end"]],
            "waypoints": [[float(x) for x in wp] for wp in wire.get("waypoints", [])],
            "weights": dict(wire.get("weights", {})),
            "connectivity": int(wire.get("connectivity", 26)),
            "clearance_m": 0.0,  # already baked into the voxel grid (compute_grids)
            "start_heading": wire.get("start_heading"),
            "end_heading": wire.get("end_heading"),
        }
        res = self.client.solve(session_id, route, locked_routes=locked_routes)

        path = f"{scene_ops.ROUTES_SCOPE}/{res['wire_id']}"
        existing = stage.GetPrimAtPath(path)
        if existing and existing.IsValid():
            stage.RemovePrim(existing.GetPath())
        if res["status"] == "routed":
            diameter = max(float(spec["outer_diameter_mm"]) / 1000.0, MIN_DISPLAY_DIAMETER_M)
            scene_ops.author_tube(stage, path, res["polyline"], diameter,
                                  spec.get("color", (0.8, 0.1, 0.1)))
            length = float(res["length_m"])
            bom_row = {"wire_id": res["wire_id"], "status": "routed", "length_m": length,
                       "cost": length * float(spec.get("cost_per_m", 0.0)),
                       "mass": length * float(spec.get("mass_per_m_kg", 0.0))}
        else:
            bom_row = {"wire_id": res["wire_id"], "status": res["status"],
                       "length_m": 0.0, "cost": 0.0, "mass": 0.0}
        return res, bom_row
