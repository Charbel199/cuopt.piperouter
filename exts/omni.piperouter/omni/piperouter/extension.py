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
            return wires, None
        except Exception as exc:
            carb.log_error(f"[piperouter] complex scene failed: {exc}")
            return None, str(exc)

    def route_all(self, wires, resolution, url):
        try:
            clr = wires[0].get("clearance_m", 0.0) if wires else 0.0
            carb.log_info(f"[piperouter] Route All: {len(wires)} wire(s) at resolution "
                          f"{resolution}, safety clearance {clr} m")
            s = self._ensure_session(url)
            clr = wires[0].get("clearance_m", 0.0) if wires else 0.0
            self._voxelize(resolution, url, clearance_m=clr)   # bakes clearance into the grid
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
            routed = sum(1 for r in results if r["status"] == "routed")
            carb.log_info(f"[piperouter] Route All done: {routed}/{len(results)} routed, "
                          f"{len(results) - routed} no-path")
            return results, bom, None
        except Exception as exc:
            carb.log_error(f"[piperouter] route_all failed: {exc}")
            return None, None, str(exc)

    def refine(self, wire, locked_wires, resolution, url):
        try:
            s = self._ensure_session(url)
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
                gbmin, cell, res, occ, thermal, em = grids
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

            centers = gbmin + (ijk + 0.5) * cell
            cap = 200000  # subsample so a fine grid doesn't author millions of points
            step = (len(centers) // cap + 1) if len(centers) > cap else 1

            if mode == "occupancy":
                scene_ops.author_points(stage, scene_ops.DEBUG_SCOPE + "/occ",
                                        centers[::step], size=cell * 0.4,
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
                    centers[::step], colors[::step], size=cell * 0.5)

            carb.log_info(f"[piperouter] overlay '{mode}': {len(ijk)} cells "
                          f"(showing {len(centers[::step])})")
            return None
        except Exception as exc:
            carb.log_error(f"[piperouter] overlay failed: {exc}")
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
            gbmin, cell, res, occ, thermal, em = grids
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
