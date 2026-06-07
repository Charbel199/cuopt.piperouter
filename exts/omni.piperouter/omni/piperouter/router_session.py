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

from . import bundles as bundle_lib
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
        self.mpu = 1.0            # meters per stage unit (set in compute_grids)

    @staticmethod
    def _mpu(stage):
        """Meters per stage unit from the stage's metersPerUnit metadata (Omniverse
        default 0.01 = cm). The solver always works in meters; this is the conversion
        factor between stage coordinates and meters."""
        try:
            from pxr import UsdGeom
            v = float(UsdGeom.GetStageMetersPerUnit(stage))
            return v if v > 1e-9 else 1.0
        except Exception:
            return 1.0

    @staticmethod
    def _scaled(pts, f):
        """Scale a list of [x,y,z] points by factor f."""
        return [[float(p[0]) * f, float(p[1]) * f, float(p[2]) * f] for p in pts]

    def compute_grids(self, stage, resolution=64, pad_frac=0.05, clearance_m=0.0):
        """Read the stage and build the four voxel grids (occupancy, surface distance,
        thermal, EM) IN MEMORY. The safety clearance is BAKED INTO the occupancy here
        (occupied cells dilated by round(clearance/cell)), so the single saved grid is
        the prohibited-voxel set used by the router AND shown by the overlay/2D views —
        one source of truth. Shared by voxelize_scene and the debug overlay.

        UNITS: the solver works in METERS. The stage may use any metersPerUnit (cm by
        default in Omniverse, mm for many CAD imports), so all geometry read from the
        stage is multiplied by `mpu` (meters per stage unit) here — the resulting grid
        (gbmin, cell) is in METERS. The route methods convert endpoints/polylines at the
        same boundary. self.mpu is cached for the authoring side.

        Returns: (bounds_min, cell_size, res_xyz, occupancy, surface_dist, thermal, em).
        """
        t0 = time.perf_counter()
        mpu = self._mpu(stage)
        self.mpu = mpu
        prims = scene_ops.list_collidable_meshes(stage)
        bounds = scene_ops.compute_bounds(stage, prims)
        if bounds is None:
            raise ValueError("no collidable geometry in stage")
        bmin, bmax = np.asarray(bounds[0]) * mpu, np.asarray(bounds[1]) * mpu  # -> meters
        # Grow bounds to include all markers (endpoints + waypoints) so routing to a
        # marker dragged beyond the geometry isn't clamped to the grid edge.
        markers = scene_ops.marker_positions(stage)
        if markers:
            mp = np.asarray(markers, dtype=float) * mpu   # -> meters
            bmin = np.minimum(bmin, mp.min(axis=0))
            bmax = np.maximum(bmax, mp.max(axis=0))
        pad = (bmax - bmin) * pad_frac + 1e-3
        bmin, bmax = bmin - pad, bmax + pad

        gbmin, cell, res = grid_io.frame_from_bounds(bmin, bmax, resolution)
        pts, idx = voxelizer.collect_meshes(stage, prims)
        pts = np.asarray(pts, dtype=np.float32) * mpu     # stage units -> meters
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
        # source centres + characteristic size are in stage units -> scale to meters
        thermal_sources = [(np.asarray(c) * mpu, t,
                            max(MIN_FALLOFF_M, char * mpu + FIELD_MARGIN_M))
                           for (c, t, e, char) in tags if t is not None]
        em_sources = [(np.asarray(c) * mpu, e,
                       max(MIN_FALLOFF_M, char * mpu + FIELD_MARGIN_M))
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
        self.last_grids = (gbmin, cell, res, occ, sd, thermal, em)  # + sd for debug views
        return session_id

    def route_all(self, stage, session_id, wires):
        """Returns (results, bom). Results are matched back to wires by unique name."""
        mpu = self._mpu(stage)
        inv = 1.0 / mpu
        routes = []
        for w in wires:
            spec = dict(w["spec"])
            spec["id"] = w["name"]  # unique per-route id (echoed back as wire_id)
            routes.append({
                "wire": spec,
                # stage units -> meters for the solver
                "start": self._scaled([w["start"]], mpu)[0],
                "end": self._scaled([w["end"]], mpu)[0],
                "waypoints": self._scaled(w.get("waypoints", []), mpu),
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
                            "length_m": 0.0, "cost": 0.0, "mass": 0.0,
                            "reason": res.get("reason", "")})
                continue
            diameter = max(float(spec["outer_diameter_mm"]) / 1000.0, MIN_DISPLAY_DIAMETER_M)
            color = spec.get("color", (0.8, 0.1, 0.1))
            # solver polyline is meters -> back to stage units for authoring
            scene_ops.author_tube(
                stage, f"{scene_ops.ROUTES_SCOPE}/{res['wire_id']}",
                self._scaled(res["polyline"], inv), diameter * inv, color)
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

        mpu = self._mpu(stage)
        inv = 1.0 / mpu
        spec = dict(wire["spec"])
        spec["id"] = wire["name"]
        route = {
            "wire": spec,
            "start": self._scaled([wire["start"]], mpu)[0],
            "end": self._scaled([wire["end"]], mpu)[0],
            "waypoints": self._scaled(wire.get("waypoints", []), mpu),
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
            scene_ops.author_tube(stage, path, self._scaled(res["polyline"], inv),
                                  diameter * inv, spec.get("color", (0.8, 0.1, 0.1)))
            length = float(res["length_m"])
            bom_row = {"wire_id": res["wire_id"], "status": "routed", "length_m": length,
                       "cost": length * float(spec.get("cost_per_m", 0.0)),
                       "mass": length * float(spec.get("mass_per_m_kg", 0.0))}
        else:
            bom_row = {"wire_id": res["wire_id"], "status": res["status"],
                       "length_m": 0.0, "cost": 0.0, "mass": 0.0,
                       "reason": res.get("reason", "")}
        return res, bom_row

    def route_all_with_bundles(self, stage, session_id, wires, bundles):
        """Bundle-aware Route All.

        Two-phase algorithm supporting wires that are in MULTIPLE bundles:

        Phase 1 — route ALL trunks in bundle order. Each trunk's cells become
        obstacles for subsequent bundles so they don't collide.

        Phase 2 — for each member wire, determine its complete segment sequence
        across ALL bundles it belongs to (in order), then route every individual
        (non-trunk) segment. Example for a wire in B1 then B2:
            wire_start -> B1_merge  [individual]
            B1_merge   -> B1_split  [trunk B1, already routed]
            B1_split   -> B2_merge  [individual — between-bundle segment]
            B2_merge   -> B2_split  [trunk B2, already routed]
            B2_split   -> wire_end  [individual]
        Then stitch all segments + trunks into one continuous polyline and author
        one per-wire colored tube (trunks are visually covered by their thick tubes).
        Returns (results, bom).
        """
        mpu = self._mpu(stage)
        inv = 1.0 / mpu
        member_names = {name for b in bundles for name in b["members"]}
        wire_by_name = {w["name"]: w for w in wires}

        # --- Phase 0: route non-bundle wires first ---
        non_bundle = [w for w in wires if w["name"] not in member_names]
        if non_bundle:
            nb_results, nb_bom = self.route_all(stage, session_id, non_bundle)
        else:
            scene_ops.clear_routes(stage)
            nb_results, nb_bom = [], []

        all_results = list(nb_results)
        all_bom = list(nb_bom)

        # --- Phase 1: route every trunk in bundle order ---
        trunk_data = {}   # bid -> {poly, len, ts, member_specs, conn}
        failed_bundles = set()

        for b in bundles:
            bid = b["id"]
            member_wires = [wire_by_name[n] for n in b["members"]
                            if n in wire_by_name]
            if not member_wires:
                continue
            member_specs = [w["spec"] for w in member_wires]

            type_map = {w["name"]: w["spec"] for w in member_wires}
            ok, err = bundle_lib.validate_members(member_wires, type_map)
            if not ok:
                for w in member_wires:
                    reason = f"Bundle validation failed: {err}"
                    all_results.append({"wire_id": w["name"], "status": "no_path",
                                        "reason": reason, "polyline": [],
                                        "length_m": 0.0})
                    all_bom.append({"wire_id": w["name"], "status": "no_path",
                                    "length_m": 0.0, "cost": 0.0, "mass": 0.0,
                                    "reason": reason})
                failed_bundles.add(bid)
                continue

            merge_pos = scene_ops.get_world_pos(stage, b["merge_marker"])
            split_pos = scene_ops.get_world_pos(stage, b["split_marker"])
            if merge_pos is not None:
                merge_pos = np.asarray(merge_pos) * mpu   # stage units -> meters
            if split_pos is not None:
                split_pos = np.asarray(split_pos) * mpu
            if merge_pos is None or split_pos is None:
                reason = "Bundle merge/split markers not found in stage."
                for w in member_wires:
                    all_results.append({"wire_id": w["name"], "status": "no_path",
                                        "reason": reason, "polyline": [],
                                        "length_m": 0.0})
                    all_bom.append({"wire_id": w["name"], "status": "no_path",
                                    "length_m": 0.0, "cost": 0.0, "mass": 0.0,
                                    "reason": reason})
                failed_bundles.add(bid)
                continue

            ts = bundle_lib.trunk_spec(member_specs, bid)
            conn = member_wires[0].get("connectivity", 18)
            trunk_weights = dict(b.get("weights", {}))
            trunk_route = {
                "wire": ts,
                "start": [float(x) for x in merge_pos],
                "end": [float(x) for x in split_pos],
                "waypoints": [], "weights": trunk_weights,
                "connectivity": conn, "priority": 0, "clearance_m": 0.0,
            }
            trunk_res = self.client.solve(session_id, trunk_route)
            if trunk_res["status"] != "routed":
                reason = ("Bundle trunk failed: "
                          + trunk_res.get("reason", "could not be routed."))
                for w in member_wires:
                    all_results.append({"wire_id": w["name"], "status": "no_path",
                                        "reason": reason, "polyline": [],
                                        "length_m": 0.0})
                    all_bom.append({"wire_id": w["name"], "status": "no_path",
                                    "length_m": 0.0, "cost": 0.0, "mass": 0.0,
                                    "reason": reason})
                failed_bundles.add(bid)
                continue

            trunk_poly = trunk_res["polyline"]
            trunk_len = float(trunk_res["length_m"])
            trunk_od_m = max(float(ts["outer_diameter_mm"]) / 1000.0,
                             MIN_DISPLAY_DIAMETER_M)
            scene_ops.author_tube(
                stage, f"{scene_ops.ROUTES_SCOPE}/bundle_{bid}_trunk",
                self._scaled(trunk_poly, inv), trunk_od_m * inv, tuple(ts["color"]))

            # Use the bundle's harness type cost_per_m when the panel has set it;
            # fall back to summing individual member wire costs.
            bundle_type_cost_pm = float(b.get("bundle_type_cost_pm", 0.0))
            combined_cost_pm = (bundle_type_cost_pm if bundle_type_cost_pm > 0
                                else sum(float(s.get("cost_per_m", 0.0))
                                         for s in member_specs))
            trunk_id = f"bundle_{bid}_trunk"
            all_bom.append({"wire_id": trunk_id, "status": "routed",
                            "length_m": trunk_len,
                            "cost": combined_cost_pm * trunk_len,
                            "mass": float(ts["mass_per_m_kg"]) * trunk_len,
                            "reason": ""})
            all_results.append({"wire_id": trunk_id, "status": "routed",
                                 "polyline": trunk_poly, "length_m": trunk_len})
            trunk_data[bid] = {
                "poly": trunk_poly, "len": trunk_len, "ts": ts,
                "merge_pos": merge_pos, "split_pos": split_pos,
                "member_specs": member_specs, "conn": conn,
            }

        # --- Phase 2: for each member wire, route all individual segments ---
        # Collect each unique member wire and the ordered bundles it belongs to.
        wire_bundles: dict[str, list] = {}
        for b in bundles:
            if b["id"] in failed_bundles:
                continue
            for name in b["members"]:
                if name not in wire_by_name or b["id"] not in trunk_data:
                    continue
                wire_bundles.setdefault(name, []).append(b)

        already_added = set()   # wire names already in all_results

        for wire_name, wire_bundle_list in wire_bundles.items():
            w = wire_by_name[wire_name]
            spec = dict(w["spec"])
            weights = dict(w.get("weights", {}))
            wconn = w.get("connectivity", 18)
            od_m = max(float(spec["outer_diameter_mm"]) / 1000.0,
                       MIN_DISPLAY_DIAMETER_M)
            color = spec.get("color", (0.8, 0.1, 0.1))

            # wire endpoints stage units -> meters (trunk merge/split already meters)
            w_start_m = self._scaled([w["start"]], mpu)[0]
            w_end_m = self._scaled([w["end"]], mpu)[0]

            # Build the ordered list of (from_pos, to_pos, is_trunk, bundle_id)
            # for this wire across all its bundles. All positions in METERS.
            segments = []          # (from, to, is_trunk, bid)
            prev_split = None
            for b in wire_bundle_list:
                td = trunk_data[b["id"]]
                seg_start = prev_split if prev_split is not None else w_start_m
                segments.append((seg_start, td["merge_pos"], False, b["id"]))
                segments.append((td["merge_pos"], td["split_pos"], True, b["id"]))
                prev_split = td["split_pos"]
            # Final segment: last split -> wire end
            last_start = prev_split if prev_split is not None else w_start_m
            segments.append((last_start, w_end_m, False, None))

            # Route each non-trunk segment, collect results.
            full_poly = []
            branch_len = 0.0
            failed = False

            for seg_idx, (seg_from, seg_to, is_trunk, bid) in enumerate(segments):
                if is_trunk:
                    td = trunk_data[bid]
                    # Stitch the shared trunk into the full polyline
                    full_poly = bundle_lib.stitch_polylines(
                        full_poly, td["poly"], []) if full_poly else list(td["poly"])
                else:
                    seg_route = {
                        "wire": spec,
                        "start": [float(x) for x in seg_from],
                        "end": [float(x) for x in seg_to],
                        "waypoints": [], "weights": weights,
                        "connectivity": wconn, "priority": 0, "clearance_m": 0.0,
                    }
                    seg_res = self.client.solve(session_id, seg_route)
                    if seg_res["status"] != "routed":
                        reason = seg_res.get("reason", "Branch segment could not be routed.")
                        all_results.append({"wire_id": wire_name, "status": "no_path",
                                            "reason": reason, "polyline": [],
                                            "length_m": 0.0})
                        all_bom.append({"wire_id": wire_name, "status": "no_path",
                                        "length_m": 0.0, "cost": 0.0, "mass": 0.0,
                                        "reason": reason})
                        failed = True
                        break
                    poly = seg_res["polyline"]
                    branch_len += float(seg_res["length_m"])
                    full_poly = (bundle_lib.stitch_polylines(full_poly, poly, [])
                                 if full_poly else list(poly))
                    # Author individual segment tube (colored; overlapping trunk
                    # sections are visually hidden under the thicker trunk tube)
                    if len(poly) >= 2:
                        scene_ops.author_tube(
                            stage,
                            f"{scene_ops.ROUTES_SCOPE}/{wire_name}_seg{seg_idx}",
                            self._scaled(poly, inv), od_m * inv, color)

            if not failed and wire_name not in already_added:
                total_trunk_len = sum(trunk_data[b["id"]]["len"]
                                      for b in wire_bundle_list)
                all_results.append({
                    "wire_id": wire_name, "status": "routed",
                    "polyline": full_poly,
                    "length_m": branch_len + total_trunk_len,
                })
                all_bom.append({
                    "wire_id": wire_name, "status": "routed",
                    "length_m": branch_len,   # BOM: branch-only length
                    "cost": branch_len * float(spec.get("cost_per_m", 0.0)),
                    "mass": branch_len * float(spec.get("mass_per_m_kg", 0.0)),
                    "reason": "",
                })
                already_added.add(wire_name)

        return all_results, all_bom
