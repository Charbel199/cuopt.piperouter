"""Kit entry point. Owns the RouterSession + grid session; exposes route/refine/
overlay/tag operations to the panel. Thin: imports omni.* so it loads only in Kit.
"""
from __future__ import annotations

import carb
import omni.ext
import omni.usd

from . import scene_ops
from .panel import PipeRouterPanel
from .router_session import RouterSession


class PipeRouterExtension(omni.ext.IExt):
    def on_startup(self, ext_id):
        carb.log_info("[piperouter] extension startup")
        self._counter = 0
        self._sid = None
        self._session = None
        self._url = None
        self._panel = PipeRouterPanel(get_stage=self._get_stage, api=self)

    # --- helpers -----------------------------------------------------------
    def _get_stage(self):
        return omni.usd.get_context().get_stage()

    def _ensure_session(self, url):
        if self._session is None or self._url != url:
            self._session = RouterSession(solver_url=url)
            self._url = url
        return self._session

    def _voxelize(self, resolution, url, clearance_m=0.0):
        stage = self._get_stage()
        if stage is None:
            raise RuntimeError("no USD stage is open")
        s = self._ensure_session(url)
        self._counter += 1
        self._sid = f"sess_{self._counter}"
        s.voxelize_scene(stage, self._sid, resolution=resolution, clearance_m=clearance_m)
        return self._sid

    # --- operations called by the panel ------------------------------------
    def health(self, url):
        """Quick reachability probe for the panel's connection indicator."""
        from .solver_client import SolverClient
        try:
            return SolverClient(url, timeout=3.0).health(), None
        except Exception as exc:
            return None, str(exc)

    def select_prim(self, path):
        """Select a prim (e.g. a waypoint marker) so the user can find/drag it."""
        return self.select_prims([path])

    def select_prims(self, paths):
        """Select several prims at once (e.g. a wire's start/end/waypoints + its tube)
        so the whole wire highlights in the viewport when picked in the panel."""
        try:
            stage = self._get_stage()
            valid = [p for p in paths
                     if stage and stage.GetPrimAtPath(p) and stage.GetPrimAtPath(p).IsValid()]
            omni.usd.get_context().get_selection().set_selected_prim_paths(valid, True)
            return None
        except Exception as exc:
            return str(exc)

    def create_sample_scene(self):
        from . import sample_scene
        stage = self._get_stage()
        if stage is None:
            return None, "no USD stage is open"
        try:
            wires = sample_scene.build_sample_scene(stage)
            carb.log_info(f"[piperouter] sample scene created with {len(wires)} wires")
            self._frame_scene(stage)
            return wires, None
        except Exception as exc:
            carb.log_error(f"[piperouter] sample scene failed: {exc}")
            return None, str(exc)

    def create_complex_scene(self):
        from . import sample_scene
        stage = self._get_stage()
        if stage is None:
            return None, "no USD stage is open"
        try:
            wires = sample_scene.build_complex_scene(stage)
            carb.log_info(f"[piperouter] complex scene created with {len(wires)} wires")
            self._frame_scene(stage)
            return wires, None
        except Exception as exc:
            carb.log_error(f"[piperouter] complex scene failed: {exc}")
            return None, str(exc)

    def _frame_scene(self, stage):
        """Fit the active viewport camera to show the whole scene.

        Tries the Kit command API first (most reliable), then falls back to
        manually positioning the perspective camera based on the scene bounding box.
        Fully guarded — a failure here is non-critical.
        """
        try:
            import numpy as np
            from pxr import Gf, Usd, UsdGeom

            # Compute world bounding box of /World
            world = stage.GetPrimAtPath("/World")
            if not world or not world.IsValid():
                return
            bbox_cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), ["default", "render"])
            bbox = bbox_cache.ComputeWorldBound(world)
            rng = bbox.GetRange()
            if rng.IsEmpty():
                return

            center = rng.GetMidpoint()
            size   = rng.GetSize()
            diag   = float(Gf.Vec3d(*size).GetLength())

            # --- Try Kit command approach first ---
            try:
                import omni.usd
                import omni.kit.commands
                omni.usd.get_context().get_selection().set_selected_prim_paths(
                    ["/World"], True)
                # Try several command names used in different Kit versions. Check the
                # command is REGISTERED first — calling execute() on an unknown command
                # makes the command subsystem log a noisy [Error] even when we catch it.
                for cmd in ("FrameViewportSelection",
                            "FocusViewport",
                            "ViewportFrameSelection"):
                    try:
                        if omni.kit.commands.get_command_class(cmd) is None:
                            continue   # not in this Kit build -> skip silently
                        omni.kit.commands.execute(cmd)
                        carb.log_info(f"[piperouter] framed scene via command '{cmd}'")
                        omni.usd.get_context().get_selection().clear_selected_prim_paths()
                        return
                    except Exception:
                        continue
                omni.usd.get_context().get_selection().clear_selected_prim_paths()
            except Exception:
                pass

            # --- Fallback: move the perspective camera via USD ---
            # Place it at ~1.5× the diagonal from the center, slightly above and back.
            cam_prim = stage.GetPrimAtPath("/OmniverseKit_Persp")
            if not cam_prim or not cam_prim.IsValid():
                cam_prim = stage.GetPrimAtPath("/World/Camera")
            if not cam_prim or not cam_prim.IsValid():
                carb.log_info("[piperouter] _frame_scene: no perspective camera found, "
                              "press F in the viewport to frame the scene")
                return

            dist = diag * 1.5
            cam_pos = Gf.Vec3d(
                float(center[0]),
                float(center[1]) + dist * 0.5,
                float(center[2]) + dist,
            )
            xf = UsdGeom.Xformable(cam_prim)
            ops = [o for o in xf.GetOrderedXformOps()
                   if o.GetOpType() == UsdGeom.XformOp.TypeTranslate]
            op = ops[0] if ops else xf.AddTranslateOp()
            op.Set(cam_pos)
            carb.log_info(f"[piperouter] framed scene: camera moved to {cam_pos}")
        except Exception as exc:
            carb.log_warn(f"[piperouter] _frame_scene failed (non-critical): {exc}")

    def route_all(self, wires, resolution, url, global_planner="lattice",
                  local_optimizer="fibre"):
        try:
            clr = wires[0].get("clearance_m", 0.0) if wires else 0.0
            carb.log_info(f"[piperouter] Route All: {len(wires)} wire(s) at resolution "
                          f"{resolution}, safety clearance {clr} m, "
                          f"algo {global_planner}/{local_optimizer}")
            import time as _t
            t0 = _t.perf_counter()
            s = self._ensure_session(url)
            s.global_planner, s.local_optimizer = global_planner, local_optimizer
            clr = wires[0].get("clearance_m", 0.0) if wires else 0.0
            self._voxelize(resolution, url, clearance_m=clr)   # bakes clearance into the grid
            t_vox = _t.perf_counter()
            cell = s.last_grids[1] if getattr(s, "last_grids", None) else None
            self._clearance_note = None
            if cell:
                keepout = int(clr / cell + 0.5 + 1e-9)
                carb.log_info(f"[piperouter] clearance {clr * 1000:.0f}mm = {keepout} prohibited "
                              f"voxel-layer(s) (grid cell {cell * 1000:.0f}mm)")
                if clr > 0 and clr < 0.5 * cell:
                    self._clearance_note = (f"clearance {clr * 1000:.0f}mm < grid cell "
                                            f"{cell * 1000:.0f}mm -> ignored; raise resolution")
                    carb.log_warn(f"[piperouter] {self._clearance_note}")
            results, bom = s.route_all(self._get_stage(), self._sid, wires)
            t_end = _t.perf_counter()
            routed = sum(1 for r in results if r["status"] == "routed")
            carb.log_info(
                f"[piperouter] Route All done: {routed}/{len(results)} routed, "
                f"{len(results) - routed} no-path | TIMING voxelize {(t_vox - t0) * 1e3:.0f}ms "
                f"+ route {(t_end - t_vox) * 1e3:.0f}ms = {(t_end - t0) * 1e3:.0f}ms total")
            return results, bom, None
        except Exception as exc:
            carb.log_error(f"[piperouter] route_all failed: {exc}")
            return None, None, str(exc)

    def route_all_bundles(self, wires, bundles, resolution, url,
                          global_planner="lattice", local_optimizer="fibre"):
        try:
            clr = wires[0].get("clearance_m", 0.0) if wires else 0.0
            carb.log_info(f"[piperouter] Route All (bundles): {len(wires)} wire(s), "
                          f"{len(bundles)} bundle(s), resolution {resolution}")
            s = self._ensure_session(url)
            s.global_planner, s.local_optimizer = global_planner, local_optimizer
            self._voxelize(resolution, url, clearance_m=clr)
            results, bom = s.route_all_with_bundles(
                self._get_stage(), self._sid, wires, bundles)
            routed = sum(1 for r in results if r["status"] == "routed")
            carb.log_info(f"[piperouter] Route All (bundles) done: "
                          f"{routed}/{len(results)} routed")
            return results, bom, None
        except Exception as exc:
            carb.log_error(f"[piperouter] route_all_bundles failed: {exc}")
            return None, None, str(exc)

    def refine(self, wire, locked_wires, resolution, url, global_planner="lattice",
               local_optimizer="fibre"):
        try:
            s = self._ensure_session(url)
            s.global_planner, s.local_optimizer = global_planner, local_optimizer
            # always re-voxelize: the grid is framed to include all current markers,
            # so a freshly-dragged waypoint (possibly beyond the geometry) is covered
            clr = float(wire.get("clearance_m", 0.0))
            self._voxelize(resolution, url, clearance_m=clr)   # bakes clearance into the grid
            cell = s.last_grids[1] if getattr(s, "last_grids", None) else None
            keepout = int(clr / cell + 0.5 + 1e-9) if cell else 0
            carb.log_info(f"[piperouter] re-route '{wire.get('name')}': "
                          f"{len(wire.get('waypoints', []))} waypoint(s), "
                          f"avoiding {len(locked_wires)} other routed wire(s), clearance {clr * 1000:.0f}mm "
                          f"= {keepout} prohibited voxel-layer(s)")
            res, bom_row = s.refine_wire(self._get_stage(), self._sid, wire, locked_wires)
            carb.log_info(f"[piperouter] re-route '{wire.get('name')}' -> {res['status']}")
            return res, bom_row, None
        except Exception as exc:
            carb.log_error(f"[piperouter] refine failed: {exc}")
            return None, None, str(exc)

    def show_overlay(self, resolution, url, mode, clearance_m=0.0):
        """Author a debug point cloud under /World/PipeRouter/debug for the chosen
        field. mode in {"none","occupancy","thermal","em"}. The occupancy cloud is
        grown by `clearance_m` so it shows the actual keep-out volume (matching the
        safety clearance the solver routes against)."""
        try:
            import numpy as np
            stage = self._get_stage()
            if stage is None:
                return "no USD stage is open"
            scene_ops.clear_debug(stage)
            mode = (mode or "none").lower()
            if mode == "none":
                carb.log_info("[piperouter] overlay cleared")
                return None

            s = self._ensure_session(url)
            # reuse the grids from the last route (matches what the router saw, and
            # avoids re-voxelizing); fall back to a fresh voxelize if not routed yet
            grids = getattr(s, "last_grids", None)
            if grids is not None:
                # occ already has the safety clearance baked in (compute_grids)
                gbmin, cell, res, occ, _sd, thermal, em = grids
            else:
                gbmin, cell, res, occ, _sd, thermal, em = s.compute_grids(
                    stage, resolution, clearance_m=clearance_m)
            ambient = 20.0

            if mode == "occupancy":
                mask, vals, lo = occ > 0, None, 0.0   # prohibited voxels (mesh + clearance)
            elif mode == "thermal":
                mask, vals, lo = thermal > ambient + 0.5, thermal, ambient
            elif mode == "em":
                mask, vals, lo = em > 1e-3, em, 0.0
            else:
                return f"unknown overlay mode: {mode}"

            ijk = np.argwhere(mask)
            if len(ijk) == 0:
                hint = {"thermal": " — tag a prim with a °C value",
                        "em": " — tag a prim with an EM strength"}.get(mode, "")
                carb.log_warn(f"[piperouter] overlay '{mode}': nothing to show{hint}")
                return f"overlay '{mode}': nothing to show{hint}"

            # grid is in METERS; convert centres + point size back to stage units
            inv = 1.0 / float(getattr(s, "mpu", 1.0) or 1.0)
            centers = (gbmin + (ijk + 0.5) * cell) * inv
            cell_stage = cell * inv
            cap = 200000  # subsample so a fine grid doesn't author millions of points
            step = (len(centers) // cap + 1) if len(centers) > cap else 1

            if mode == "occupancy":
                # ~0.85*cell so beads nearly fill their voxel and the overlay reads as a
                # continuous occupied shell instead of sparse floating dots.
                scene_ops.author_points(stage, scene_ops.DEBUG_SCOPE + "/occ",
                                        centers[::step], size=cell_stage * 0.85,
                                        color=(0.2, 0.6, 1.0))
            else:
                v = vals[mask]
                hi = max(float(v.max()), lo + 1e-6)
                t01 = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
                if mode == "thermal":   # cold blue -> hot red
                    colors = np.stack([t01, np.full_like(t01, 0.1), 1.0 - t01], axis=1)
                else:                   # em: low teal -> high magenta
                    colors = np.stack([t01, 1.0 - t01, np.full_like(t01, 0.7)], axis=1)
                scene_ops.author_colored_points(
                    stage, scene_ops.DEBUG_SCOPE + f"/{mode}",
                    centers[::step], colors[::step], size=cell_stage * 0.8)

            carb.log_info(f"[piperouter] overlay '{mode}': {len(ijk)} cells "
                          f"(showing {len(centers[::step])})")
            return None
        except Exception as exc:
            carb.log_error(f"[piperouter] overlay failed: {exc}")
            return str(exc)

    def show_wire_debug(self, wire, mode):
        """Author per-wire debug geometry under /World/PipeRouter/debug.
        mode: "cells" | "raw_path" | "cost_terrain" | "clearance" | "bend_radius" | "none"
        """
        try:
            import numpy as np
            stage = self._get_stage()
            if stage is None:
                return "no USD stage is open"
            scene_ops.clear_debug(stage)
            # Always reset visibility first, so switching wires/modes never leaves a
            # previously-hidden cable hidden.
            scene_ops.set_all_routes_visible(stage)
            if mode == "none" or not wire:
                return None

            wire_name = wire.get("name", "?")
            spec = wire.get("spec", {})
            color = tuple(float(c) for c in spec.get("color", (0.8, 0.1, 0.1)))
            # Hide this wire's final tube so the debug geometry below is clearly visible.
            scene_ops.hide_route(stage, wire_name)

            s = getattr(self, "_session", None)
            grids = getattr(s, "last_grids", None) if s else None
            # The grids/polylines are in METERS (solver space); convert lengths back
            # to stage units when authoring debug geometry so it lines up with the USD.
            inv = 1.0 / float(getattr(s, "mpu", 1.0) or 1.0)
            # the cable-representing debug curves use the wire's REAL diameter (same as the
            # routed tube), so the debug view is to-scale rather than a fixed fat line.
            _real_d = float(spec.get("outer_diameter_mm", 1.5)) / 1000.0
            dia_stage = (s._display_diameter_m(_real_d) if s else max(_real_d, 5e-4)) * inv

            if mode == "cells":
                cells = wire.get("cells", [])
                if not cells or grids is None:
                    return "[piperouter] cells: no cell data (route first)"
                gbmin, cell, _res, _occ, _sd, _th, _em = grids
                scene_ops.author_wire_cells(stage, wire_name, cells,
                                            np.asarray(gbmin) * inv, float(cell) * inv,
                                            color=color)
                carb.log_info(f"[piperouter] wire debug 'cells': {len(cells)} voxels for {wire_name}")

            elif mode == "raw_path":
                raw = wire.get("raw_polyline")
                if not raw:
                    return "[piperouter] raw_path: no raw polyline (route first)"
                poly = wire.get("polyline", [])
                raw_s = (np.asarray(raw, dtype=np.float64) * inv).tolist()
                # raw stair-step a bit thinner than the cable so the smooth tube reads over it
                scene_ops.author_raw_path(stage, wire_name + "_raw", raw_s,
                                          color=(0.9, 0.9, 0.1), width=dia_stage * 0.5)
                if poly and len(poly) >= 2:
                    poly_s = (np.asarray(poly, dtype=np.float64) * inv).tolist()
                    scene_ops.author_tube(stage,
                                          f"{scene_ops.DEBUG_SCOPE}/smooth_{wire_name}",
                                          poly_s, dia_stage, color)
                carb.log_info(f"[piperouter] wire debug 'raw_path': {len(raw)} raw pts, {len(poly)} smooth pts")

            elif mode == "cost_terrain":
                if grids is None:
                    return "[piperouter] cost_terrain: route first"
                cells = wire.get("cells", [])
                if not cells:
                    return "[piperouter] cost_terrain: route first (no path cells)"
                from . import fields as ext_fields
                gbmin, cell, res, occ, sd, thermal, em = grids
                nx, ny, nz = int(res[0]), int(res[1]), int(res[2])
                weights = wire.get("weights", {})
                cost = ext_fields.soft_cost_field(sd, thermal, em, spec, weights)

                # Build a mask of cells within CORRIDOR_R cells of the wire's path
                CORRIDOR_R = 6
                path_mask = np.zeros((nx, ny, nz), dtype=bool)
                for ci, cj, ck in cells:
                    i0, i1 = max(0, ci - CORRIDOR_R), min(nx, ci + CORRIDOR_R + 1)
                    j0, j1 = max(0, cj - CORRIDOR_R), min(ny, cj + CORRIDOR_R + 1)
                    k0, k1 = max(0, ck - CORRIDOR_R), min(nz, ck + CORRIDOR_R + 1)
                    path_mask[i0:i1, j0:j1, k0:k1] = True

                mask = path_mask & (cost > 1e-3)
                ijk = np.argwhere(mask).astype(np.float64)
                if len(ijk) == 0:
                    return "[piperouter] cost_terrain: costs are ~zero near this path (check sliders)"
                centres = (np.asarray(gbmin, dtype=np.float64) + (ijk + 0.5) * float(cell)) * inv
                cv = cost[mask]
                hi = float(cv.max()) + 1e-6
                t01 = np.clip(cv / hi, 0.0, 1.0)
                # blue (cheap near path) -> red (expensive near path)
                cols = np.column_stack([t01, 1.0 - t01, np.zeros_like(t01)])
                scene_ops.author_colored_points(stage,
                    f"{scene_ops.DEBUG_SCOPE}/cost_{wire_name}", centres, cols,
                    size=float(cell) * inv * 0.55)
                carb.log_info(f"[piperouter] wire debug 'cost_terrain': {len(centres)} corridor cells")

            elif mode == "bend_radius":
                poly = wire.get("polyline")
                if not poly or len(poly) < 3:
                    return "[piperouter] bend_radius: route first"
                min_bend = float(spec.get("min_bend_radius_mm", 50.0))
                scene_ops.author_bend_heatmap(stage, wire_name, poly, min_bend,
                                              pos_scale=inv, width=dia_stage)
                carb.log_info(f"[piperouter] wire debug 'bend_radius': min={min_bend}mm")

            return None
        except Exception as exc:
            carb.log_error(f"[piperouter] wire debug failed: {exc}")
            return str(exc)

    def slice_views(self, routes, target_px=1024):
        """Render the XY/XZ/YZ projection images from the EXACT routing grid (last
        voxelize), so the obstacles + clearance halo shown are precisely what the
        router removed — the route can never appear to cross them. (Obstacles are at
        the routing resolution; raise it for finer views + finer routing together.)
        `routes` = [{"points": [[x,y,z],...], "color": (r,g,b)}]."""
        try:
            from . import slices
            s = self._ensure_session(self._url or "http://localhost:8000")
            grids = getattr(s, "last_grids", None)
            if grids is None:
                return None, "route first (no voxel grids yet)"
            gbmin, cell, res, occ, _sd, thermal, em = grids
            # occ already includes the clearance keep-out (baked in compute_grids),
            # so there's no separate halo to draw — show the prohibited voxels as-is
            imgs = slices.render_views(gbmin, cell, res, occ, thermal, routes,
                                       target_px=target_px)
            return imgs, None
        except Exception as exc:
            carb.log_error(f"[piperouter] slice views failed: {exc}")
            return None, str(exc)

    def create_view_camera(self, plane):
        """Create/position a camera looking along the given axis and make it the active
        viewport camera. plane in {"xy"(top), "xz"(front), "yz"(side)}."""
        try:
            import numpy as np
            from pxr import Gf, UsdGeom
            stage = self._get_stage()
            if stage is None:
                return "no USD stage is open"
            prims = scene_ops.list_collidable_meshes(stage)
            bounds = scene_ops.compute_bounds(stage, prims)
            if bounds is None:
                return "no geometry to frame"
            bmin, bmax = bounds
            center = 0.5 * (bmin + bmax)
            dist = float(np.linalg.norm(bmax - bmin)) * 2.4 + 1e-3
            offset, up = {
                "xy": ((0, 0, dist), (0, 1, 0)),    # top, looking down -Z
                "xz": ((0, -dist, 0), (0, 0, 1)),   # front, looking +Y
                "yz": ((dist, 0, 0), (0, 0, 1)),    # side, looking -X
            }[plane]
            eye = center + np.asarray(offset, float)
            path = f"{scene_ops.PIPEROUTER_ROOT}/cameras/{plane}"
            cam = UsdGeom.Camera.Define(stage, path)
            # hide the camera gizmo in the viewport (still usable as a viewport camera)
            UsdGeom.Imageable(cam).CreateVisibilityAttr().Set(UsdGeom.Tokens.invisible)
            view = Gf.Matrix4d().SetLookAt(
                Gf.Vec3d(*[float(x) for x in eye]),
                Gf.Vec3d(*[float(x) for x in center]),
                Gf.Vec3d(*[float(x) for x in up]))
            xf = UsdGeom.Xformable(cam)
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(view.GetInverse())
            try:
                from omni.kit.viewport.utility import get_active_viewport
                get_active_viewport().camera_path = path
            except Exception as exc:
                carb.log_warn(f"[piperouter] camera created at {path} but could not set "
                              f"active viewport: {exc}")
                return None
            carb.log_info(f"[piperouter] view camera '{plane}' active ({path})")
            return None
        except Exception as exc:
            carb.log_error(f"[piperouter] create_view_camera failed: {exc}")
            return str(exc)

    def list_tags(self):
        stage = self._get_stage()
        return scene_ops.list_tagged_prims(stage) if stage else []

    def clear_tag(self, path):
        stage = self._get_stage()
        if stage is None:
            return "no stage"
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            scene_ops.clear_tags(prim)
            carb.log_info(f"[piperouter] cleared tag on {path}")
        return None

    def write_tag(self, temp_c, em):
        stage = self._get_stage()
        sel = omni.usd.get_context().get_selection().get_selected_prim_paths()
        if not sel:
            return "select a prim in the stage first"
        for p in sel:
            prim = stage.GetPrimAtPath(p)
            if prim and prim.IsValid():
                scene_ops.write_tags(prim, temp_c=temp_c, em=em)
        carb.log_info(f"[piperouter] tagged {len(sel)} prim(s): temp_c={temp_c}, em={em}")
        return None

    def on_shutdown(self):
        carb.log_info("[piperouter] extension shutdown")
        if getattr(self, "_panel", None):
            self._panel.destroy()
            self._panel = None
        self._session = None  # drop the RouterSession reference too
