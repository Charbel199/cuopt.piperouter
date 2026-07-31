"""Routing orchestration, omni-free so it is fully testable headlessly.

voxelize_scene reads the stage, takes its bounds, runs the Warp voxelizer and the
thermal/EM fields, then writes the grids to the shared directory. route_all builds route
payloads from the wire list, calls the solver, authors tubes and computes the BOM.

A `wire` dict holds: name (which is the unique route id), spec (the wire-type
properties), start, end, waypoints, weights, connectivity and priority.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from . import bundles as bundle_lib
from . import fields, grid_io, scene_ops, voxelizer
from .solver_client import SolverClient

# Logs surface in the Kit console, and on stderr headless, tagged [piperouter].
log = logging.getLogger("piperouter")


# Thermal and EM fields radiate over the object's characteristic size plus this margin,
# with a floor so even small tagged prims have a usable reach.
FIELD_MARGIN_M = 1.0
MIN_FALLOFF_M = 0.75


class RouterSession:
    def __init__(self, grid_dir="/dev/shm/piperouter", solver_url="http://localhost:8000"):
        self.grid_dir = Path(grid_dir)
        self.client = SolverClient(solver_url)
        self.frame = None         # (bounds_min, cell_size, res_xyz) from the last voxelize
        self.last_stats = {}      # stats from the last compute_grids, for logging and UI
        self.last_grids = None    # (bounds_min, cell, res, occ, thermal, em) for views/overlay
        self.mpu = 1.0            # meters per stage unit, set in compute_grids
        self.last_clearance_m = 0.0   # safety clearance in metres, sent with each route
        # Routing algorithms, sent to the solver in every route request.
        self.global_planner = "octree_lattice"
        self.local_optimizer = "fibre"

    @staticmethod
    def _mpu(stage):
        """Return metres per stage unit, from the stage's metersPerUnit metadata.

        Omniverse defaults to 0.01, i.e. centimetres. The solver always works in metres,
        so this is the conversion factor between stage coordinates and metres.
        """
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

    @staticmethod
    def _hdg(p, q):
        """Return the unit direction from p to q, or None for a zero-length step.

        World axes align with grid axes, so the result also serves as a lattice heading.
        """
        d = np.asarray(q, dtype=float) - np.asarray(p, dtype=float)
        n = float(np.linalg.norm(d))
        return [float(x) for x in (d / n)] if n > 1e-9 else None

    def _display_diameter_m(self, real_diameter_m):
        """Return the tube display diameter in metres for a wire's physical gauge.

        The scene is at real scale, so tubes are drawn at true thickness: a 1.3 mm wire
        is 1.3 mm and a 16 mm hose is 16 mm, which keeps types visually distinct. The
        floor exists only to avoid a degenerate zero-radius tube, as with bundle types
        whose thickness comes from the bundle-diameter formula rather than a fixed gauge.
        """
        return max(float(real_diameter_m), 0.0005)   # 0.5 mm, a guard against zero

    def compute_grids(self, stage, resolution=64, pad_frac=0.05, clearance_m=0.0):
        """Build the four voxel grids in memory from the stage.

        Returns (bounds_min, cell_size, res_xyz, occupancy, surface_dist, thermal, em),
        and is shared by voxelize_scene and the debug overlay.

        Occupancy is the raw mesh: clearance is deliberately not baked in, so the solver
        can still distinguish real mesh from clearance halo. It applies clearance itself
        as a relaxable band, waived near endpoints. `clearance_m` is remembered here for
        the route requests and for re-dilating the overlay and 2D views at display time.

        Units: the solver works in metres, while the stage may use any metersPerUnit,
        centimetres by default in Omniverse and millimetres for many CAD imports. All
        geometry read from the stage is multiplied by `mpu` here, so the resulting grid
        (gbmin, cell) is metric. The route methods convert endpoints and polylines at the
        same boundary, and self.mpu is cached for the authoring side.
        """
        t0 = time.perf_counter()
        mpu = self._mpu(stage)
        self.mpu = mpu
        prims = scene_ops.list_collidable_meshes(stage)
        bounds = scene_ops.compute_bounds(stage, prims)
        if bounds is None:
            raise ValueError("no collidable geometry in stage")
        bmin, bmax = np.asarray(bounds[0]) * mpu, np.asarray(bounds[1]) * mpu  # -> metres
        # Bounds must cover every marker, endpoints and waypoints alike, or a marker
        # dragged past the geometry gets clamped to the grid edge.
        markers = scene_ops.marker_positions(stage)
        if markers:
            mp = np.asarray(markers, dtype=float) * mpu   # -> metres
            bmin = np.minimum(bmin, mp.min(axis=0))
            bmax = np.maximum(bmax, mp.max(axis=0))
        pad = (bmax - bmin) * pad_frac + 1e-3
        bmin, bmax = bmin - pad, bmax + pad

        gbmin, cell, res = grid_io.frame_from_bounds(bmin, bmax, resolution)
        t_setup = time.perf_counter()
        pts, idx = voxelizer.collect_meshes(stage, prims)
        pts = np.asarray(pts, dtype=np.float32) * mpu     # stage units -> metres
        occ, sd = voxelizer.voxelize(pts, idx, gbmin, cell, res)
        t_vox = time.perf_counter()

        # Clearance travels to the solver in each route request rather than being baked
        # into occ, which keeps occ the raw mesh. That distinction matters downstream:
        # endpoints are relocated out of real mesh only, clearance is a relaxable band
        # waived around endpoints, and the overlay re-dilates by this value for display.
        self.last_clearance_m = float(clearance_m)

        # Per-object clearance, a minimum distance per component category. Prims tagged
        # with CLEARANCE_ATTR are voxelized per distinct value into a class grid and the
        # solver honours each class's own distance; untagged geometry falls back to the
        # global default. Values are processed in ascending order so that the largest
        # clearance wins where classes overlap.
        cls_grid = np.zeros(tuple(int(r) for r in res), dtype=np.uint8)
        cls_values: list[float] = []
        groups: dict[float, list] = {}
        proxy_tags = scene_ops.read_proxy_tags(stage)
        for prim in prims:
            c = scene_ops.clearance_for_prim(prim, proxy_tags=proxy_tags)
            if c is not None and c > 0.0:
                groups.setdefault(round(float(c), 6), []).append(prim)
        for val in sorted(groups):
            gpts, gidx = voxelizer.collect_meshes(stage, groups[val])
            if len(gpts) == 0:
                continue
            gocc, _gsd = voxelizer.voxelize(
                np.asarray(gpts, dtype=np.float32) * mpu, gidx, gbmin, cell, res)
            cls_values.append(float(val))
            cls_grid[gocc.astype(bool)] = len(cls_values)   # class ids are 1-based
        self.last_clearance_classes = (cls_grid, cls_values) if cls_values else None

        # Thermal and EM fields: each tagged prim becomes a source at its bbox centre.
        # The field radiates over char_size + FIELD_MARGIN_M, so a large hot block heats
        # a region proportional to its size, and MIN_FALLOFF_M keeps tiny tagged prims
        # from having near-zero reach.
        tags = scene_ops.read_thermal_em_tags(stage)
        # Source centres and characteristic sizes are in stage units, so scale to metres.
        thermal_sources = [(np.asarray(c) * mpu, t,
                            max(MIN_FALLOFF_M, char * mpu + FIELD_MARGIN_M))
                           for (c, t, e, char) in tags if t is not None]
        em_sources = [(np.asarray(c) * mpu, e,
                       max(MIN_FALLOFF_M, char * mpu + FIELD_MARGIN_M))
                      for (c, t, e, char) in tags if e is not None]
        thermal = fields.thermal_field(gbmin, cell, res, thermal_sources)
        em = fields.em_field(gbmin, cell, res, em_sources)
        t_fields = time.perf_counter()

        self.last_stats = {
            "res": tuple(int(v) for v in res),
            "cells": int(res[0] * res[1] * res[2]),
            "occupied": int(occ.sum()),
            "n_thermal_sources": len(thermal_sources),
            "n_em_sources": len(em_sources),
            "thermal_max_c": float(thermal.max()),
            "seconds": round(t_fields - t0, 2),
        }
        # Per-step breakdown, so a slow voxelization can be attributed to a stage.
        log.info("[piperouter] voxelize %s = %d occ | %d meshes/%d tris | "
                 "TIMING setup %.0fms + voxelize(GPU) %.0fms + fields %.0fms = %.0fms total",
                 self.last_stats["res"], self.last_stats["occupied"],
                 len(prims), len(idx) // 3,
                 (t_setup - t0) * 1e3, (t_vox - t_setup) * 1e3,
                 (t_fields - t_vox) * 1e3, (t_fields - t0) * 1e3)
        return gbmin, cell, res, occ, sd, thermal, em

    def voxelize_scene(self, stage, session_id, resolution=64, pad_frac=0.05, clearance_m=0.0):
        gbmin, cell, res, occ, sd, thermal, em = self.compute_grids(
            stage, resolution, pad_frac, clearance_m=clearance_m)
        path = self.grid_dir / session_id / "stack.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        t_save = time.perf_counter()
        cc = getattr(self, "last_clearance_classes", None)
        grid_io.save_grids(path, gbmin, cell, res, occ, sd, thermal, em,
                           clearance_class=cc[0] if cc else None,
                           clearance_values=cc[1] if cc else None)
        log.info("[piperouter] grid handoff saved in %.0fms", (time.perf_counter() - t_save) * 1e3)
        self.frame = (gbmin, cell, res)
        self.last_grids = (gbmin, cell, res, occ, sd, thermal, em)  # sd is for debug views
        return session_id

    def route_all(self, stage, session_id, wires):
        """Route every wire and return (results, bom).

        Results are matched back to wires by their unique name.
        """
        mpu = self._mpu(stage)
        inv = 1.0 / mpu
        routes = []
        for w in wires:
            spec = dict(w["spec"])
            spec["id"] = w["name"]  # unique per-route id, echoed back as wire_id
            routes.append({
                "wire": spec,
                # stage units -> metres for the solver
                "start": self._scaled([w["start"]], mpu)[0],
                "end": self._scaled([w["end"]], mpu)[0],
                "waypoints": self._scaled(w.get("waypoints", []), mpu),
                "weights": dict(w.get("weights", {})),
                "connectivity": int(w.get("connectivity", 26)),
                "priority": int(w.get("priority", 0)),
                "clearance_m": float(self.last_clearance_m),  # applied solver-side
                "start_heading": w.get("start_heading"),
                "end_heading": w.get("end_heading"),
                "global_planner": self.global_planner,
                "local_optimizer": self.local_optimizer,
            })

        t_solve = time.perf_counter()
        resp = self.client.solve_all(session_id, routes)
        t_done = time.perf_counter()
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
            diameter = self._display_diameter_m(float(spec["outer_diameter_mm"]) / 1000.0)
            color = spec.get("color", (0.8, 0.1, 0.1))
            # The solver polyline is in metres; author in stage units.
            scene_ops.author_tube(
                stage, f"{scene_ops.ROUTES_SCOPE}/{res['wire_id']}",
                self._scaled(res["polyline"], inv), diameter * inv, color)
            length = float(res["length_m"])
            bom.append({
                "wire_id": res["wire_id"], "status": "routed", "length_m": length,
                "cost": length * float(spec.get("cost_per_m", 0.0)),
                "mass": length * float(spec.get("mass_per_m_kg", 0.0)),
            })
        n_routed = sum(1 for r in resp["results"] if r["status"] == "routed")
        log.info("[piperouter] route_all (%s/%s): solve(GPU/HTTP) %.0fms + author %.0fms "
                 "= %.0fms for %d wires (%d routed)",
                 self.global_planner, self.local_optimizer,
                 (t_done - t_solve) * 1e3, (time.perf_counter() - t_done) * 1e3,
                 (time.perf_counter() - t_solve) * 1e3, len(wires), n_routed)
        return resp["results"], bom

    def refine_wire(self, stage, session_id, wire, locked_wires):
        """Re-route one wire, honouring its waypoints and weights.

        Every routed wire in `locked_wires` acts as an obstacle, so the re-routed wire
        cannot overlap the rest. Only this wire's tube is authored or replaced. Returns
        (result_dict, bom_row).
        """
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
            "clearance_m": float(self.last_clearance_m),  # applied solver-side
            "start_heading": wire.get("start_heading"),
            "end_heading": wire.get("end_heading"),
            "global_planner": self.global_planner,
            "local_optimizer": self.local_optimizer,
        }
        res = self.client.solve(session_id, route, locked_routes=locked_routes)

        path = f"{scene_ops.ROUTES_SCOPE}/{res['wire_id']}"
        existing = stage.GetPrimAtPath(path)
        if existing and existing.IsValid():
            stage.RemovePrim(existing.GetPath())
        if res["status"] == "routed":
            diameter = self._display_diameter_m(float(spec["outer_diameter_mm"]) / 1000.0)
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
        """Bundle-aware Route All, returning (results, bom).

        Runs in two phases, and a wire may belong to several bundles.

        Phase 1 routes the trunks in bundle order. Each trunk's cells become obstacles
        for later bundles, so trunks cannot collide.

        Phase 2 works out each member wire's full segment sequence across the bundles it
        belongs to, in order, and routes every non-trunk segment. For a wire in B1 then
        B2 that is:

            wire_start -> B1_merge  individual
            B1_merge   -> B1_split  trunk B1, already routed
            B1_split   -> B2_merge  individual, between bundles
            B2_merge   -> B2_split  trunk B2, already routed
            B2_split   -> wire_end  individual

        The segments and trunks are then stitched into one continuous polyline and
        authored as a single coloured tube per wire; the thicker trunk tubes visually
        cover the shared stretches.
        """
        mpu = self._mpu(stage)
        inv = 1.0 / mpu
        member_names = {name for b in bundles for name in b["members"]}
        wire_by_name = {w["name"]: w for w in wires}

        # --- Phase 0: non-bundle wires ---
        non_bundle = [w for w in wires if w["name"] not in member_names]
        if non_bundle:
            nb_results, nb_bom = self.route_all(stage, session_id, non_bundle)
        else:
            scene_ops.clear_routes(stage)
            nb_results, nb_bom = [], []

        all_results = list(nb_results)
        all_bom = list(nb_bom)

        # --- Phase 1: trunks, in bundle order ---
        trunk_data = {}   # bundle id -> {poly, len, ts, member_specs, conn}
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
                merge_pos = np.asarray(merge_pos) * mpu   # stage units -> metres
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
            # Trunk waypoints are stored as stage marker paths; resolve them to world
            # positions in metres so the shared trunk passes through them.
            trunk_wps = []
            for wp_path in b.get("waypoints", []):
                wp = scene_ops.get_world_pos(stage, wp_path)
                if wp is not None:
                    trunk_wps.append([float(x) for x in (np.asarray(wp) * mpu)])
            trunk_route = {
                "wire": ts,
                "start": [float(x) for x in merge_pos],
                "end": [float(x) for x in split_pos],
                "waypoints": trunk_wps, "weights": trunk_weights,
                "connectivity": conn, "priority": 0, "clearance_m": float(self.last_clearance_m),
                "global_planner": self.global_planner,
                "local_optimizer": self.local_optimizer,
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
            trunk_od_m = self._display_diameter_m(float(ts["outer_diameter_mm"]) / 1000.0)
            scene_ops.author_tube(
                stage, f"{scene_ops.ROUTES_SCOPE}/bundle_{bid}_trunk",
                self._scaled(trunk_poly, inv), trunk_od_m * inv, tuple(ts["color"]))

            # Prefer the bundle's harness-type cost_per_m when the panel has set one,
            # otherwise sum the individual member costs.
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

        # --- Phase 2: the individual segments of each member wire ---
        # Collect each unique member wire and the ordered bundles it belongs to.
        wire_bundles: dict[str, list] = {}
        for b in bundles:
            if b["id"] in failed_bundles:
                continue
            for name in b["members"]:
                if name not in wire_by_name or b["id"] not in trunk_data:
                    continue
                wire_bundles.setdefault(name, []).append(b)

        already_added = set()   # wire names already present in all_results

        for wire_name, wire_bundle_list in wire_bundles.items():
            w = wire_by_name[wire_name]
            spec = dict(w["spec"])
            weights = dict(w.get("weights", {}))
            wconn = w.get("connectivity", 18)
            od_m = self._display_diameter_m(float(spec["outer_diameter_mm"]) / 1000.0)
            color = spec.get("color", (0.8, 0.1, 0.1))

            # Wire endpoints go from stage units to metres; trunk merge/split are metric
            # already.
            w_start_m = self._scaled([w["start"]], mpu)[0]
            w_end_m = self._scaled([w["end"]], mpu)[0]

            # This wire's own waypoints, bucketed by slot, meaning how many of its
            # bundles precede them. gaps[s] holds the waypoints visited in the s-th gap
            # and gaps[K] the final stretch to the wire end. All positions in metres.
            K = len(wire_bundle_list)
            wp_m = self._scaled(w.get("waypoints", []), mpu)
            wp_slots = w.get("waypoint_slots", [])
            gaps = [[] for _ in range(K + 1)]
            for wi_, pos in enumerate(wp_m):
                s = wp_slots[wi_] if wi_ < len(wp_slots) else 0
                gaps[min(max(int(s), 0), K)].append([float(x) for x in pos])

            # Ordered (from, to, is_trunk, bid, waypoints) across all this wire's
            # bundles, with its waypoints interleaved. All positions in metres.
            segments = []
            prev_split = None
            for s, b in enumerate(wire_bundle_list):
                td = trunk_data[b["id"]]
                seg_start = prev_split if prev_split is not None else w_start_m
                segments.append((seg_start, td["merge_pos"], False, b["id"], gaps[s]))
                segments.append((td["merge_pos"], td["split_pos"], True, b["id"], []))
                prev_split = td["split_pos"]
            # Final segment runs from the last split to the wire end, carrying any
            # trailing waypoints.
            last_start = prev_split if prev_split is not None else w_start_m
            segments.append((last_start, w_end_m, False, None, gaps[K]))

            # Route each non-trunk segment. Heading continuity is threaded across
            # segments so a branch joins and leaves the shared trunk smoothly rather than
            # kinking at the merge or split.
            full_poly = []
            branch_len = 0.0
            failed = False
            prev_heading = None   # arrival heading carried into the next segment

            for seg_idx, (seg_from, seg_to, is_trunk, bid, seg_wps) in enumerate(segments):
                if is_trunk:
                    td = trunk_data[bid]
                    full_poly = bundle_lib.stitch_polylines(
                        full_poly, td["poly"], []) if full_poly else list(td["poly"])
                    tp = td["poly"]
                    if len(tp) >= 2:   # next branch leaves the split along the trunk exit
                        prev_heading = self._hdg(tp[-2], tp[-1])
                else:
                    # When a trunk comes next, arrive at its merge aligned with the
                    # trunk's entry direction so the join is smooth.
                    goal_h = None
                    nxt = segments[seg_idx + 1] if seg_idx + 1 < len(segments) else None
                    if nxt is not None and nxt[2]:
                        ntp = trunk_data[nxt[3]]["poly"]
                        if len(ntp) >= 2:
                            goal_h = self._hdg(ntp[0], ntp[1])
                    seg_route = {
                        "wire": spec,
                        "start": [float(x) for x in seg_from],
                        "end": [float(x) for x in seg_to],
                        "waypoints": [list(x) for x in seg_wps], "weights": weights,
                        "connectivity": wconn, "priority": 0, "clearance_m": float(self.last_clearance_m),
                        "start_heading": prev_heading, "end_heading": goal_h,
                        "global_planner": self.global_planner,
                        "local_optimizer": self.local_optimizer,
                    }
                    seg_res = self.client.solve(session_id, seg_route)
                    if seg_res["status"] != "routed" and (prev_heading or goal_h):
                        # The continuity headings over-constrained this segment; retry
                        # without them.
                        seg_route["start_heading"] = None
                        seg_route["end_heading"] = None
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
                    if len(poly) >= 2:   # carry this segment's exit heading to the next
                        prev_heading = self._hdg(poly[-2], poly[-1])
                    branch_len += float(seg_res["length_m"])
                    full_poly = (bundle_lib.stitch_polylines(full_poly, poly, [])
                                 if full_poly else list(poly))
                    # Author the segment's own coloured tube; where it overlaps a trunk it
                    # is hidden under the thicker trunk tube.
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
                    "length_m": branch_len,   # branch only; the trunk gets its own row
                    "cost": branch_len * float(spec.get("cost_per_m", 0.0)),
                    "mass": branch_len * float(spec.get("mass_per_m_kg", 0.0)),
                    "reason": "",
                })
                already_added.add(wire_name)

        return all_results, all_bom
