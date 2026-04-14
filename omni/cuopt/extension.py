"""cuOpt Pipe Router - Omniverse Extension entry point."""

import time
import traceback

import carb
import numpy as np
import omni.ext
import omni.usd

from .pathfinding import OccupancyGrid3D, smooth_path
from .cuopt_solver import solve as cuopt_solve
from .scene_builder import (
    create_sample_scene,
    get_obstacle_bounds,
    get_pipe_markers,
    voxelize_obstacles,
    fill_interior,
    create_tube_mesh,
    mark_tube_obstacle,
    count_bends,
    path_length,
    clear_pipes,
    PIPE_PALETTE,
)
from .debug_viz import (
    show_occupancy_grid,
    show_waypoint_graph,
    clear_debug,
)

try:
    from .warp_voxelizer import voxelize_obstacles_gpu
    _HAS_WARP = True
except Exception:
    _HAS_WARP = False

from .ui.panel import CuoptPanel

LOG = "[omni.cuopt]"


class CuoptPipeRouterExtension(omni.ext.IExt):

    def on_startup(self, _ext_id: str) -> None:
        try:
            carb.log_warn(f"{LOG} starting up (warp: {_HAS_WARP})")
            self._panel = CuoptPanel(
                on_create_scene=self._create_scene,
                on_route_all=self._route_all,
                on_clear=self._clear,
            )
        except Exception as e:
            carb.log_error(f"{LOG} startup failed: {e}\n{traceback.format_exc()}")

    def on_shutdown(self) -> None:
        if getattr(self, "_panel", None):
            self._panel.destroy()

    def _create_scene(self) -> None:
        try:
            stage = omni.usd.get_context().get_stage()
            create_sample_scene(stage)
            pipes = get_pipe_markers(stage)
            self._panel.set_status(
                f"Scene created: {len(pipes)} pipes. "
                "Drag the spheres to set endpoints."
            )
        except Exception as e:
            carb.log_error(f"{LOG} scene creation failed: {e}")
            self._panel.set_status(f"Error: {e}")

    def _route_all(self, resolution, clearance, tube_radius,
                   server_url, bend_penalty,
                   show_grid, show_graph) -> None:
        try:
            t_total = time.time()
            timings = {}
            stage = omni.usd.get_context().get_stage()
            clear_debug(stage)

            # read scene
            t = time.time()
            pipes = get_pipe_markers(stage)
            if not pipes:
                self._panel.set_status("No pipe markers found. Create a scene first.")
                return
            obstacle_boxes = get_obstacle_bounds(stage)
            if not obstacle_boxes:
                self._panel.set_status("No obstacles found. Create a scene first.")
                return
            timings["scene"] = time.time() - t

            # compute bounds + build grid
            t = time.time()
            all_pts = []
            for p in pipes:
                all_pts.extend(p["start"])
                all_pts.extend(p["end"])
            for lo, hi in obstacle_boxes:
                all_pts.extend(lo)
                all_pts.extend(hi)
            arr = np.array(all_pts).reshape(-1, 3)
            scene_size = arr.max(axis=0) - arr.min(axis=0)
            margin = max(5.0, float(np.max(scene_size)) * 0.1)
            bounds_min = tuple(arr.min(axis=0) - margin)
            bounds_max = tuple(arr.max(axis=0) + margin)

            grid = OccupancyGrid3D(bounds_min, bounds_max, resolution)
            ri, rj, rk = grid.res_xyz
            total_cells = int(ri * rj * rk)
            timings["grid"] = time.time() - t

            # voxelize
            t = time.time()
            voxelized = False
            vox_method = "bbox"
            if _HAS_WARP:
                try:
                    self._panel.set_status("Voxelizing with Warp (GPU)...")
                    n = voxelize_obstacles_gpu(stage, grid, clearance)
                    voxelized = n > 0
                    if voxelized:
                        vox_method = "warp"
                except Exception as exc:
                    carb.log_warn(f"{LOG} warp failed: {exc}")

            if not voxelized:
                self._panel.set_status("Voxelizing obstacles (CPU)...")
                n = voxelize_obstacles(stage, grid, clearance)
                if n:
                    vox_method = "cpu+fill"
                    fill_interior(grid)
                else:
                    for obs_min, obs_max in obstacle_boxes:
                        grid.mark_box(obs_min, obs_max, clearance)
            timings["voxelize"] = time.time() - t

            occupied = int(grid.occupied.sum())
            carb.log_warn(
                f"{LOG} grid {ri}x{rj}x{rk}={total_cells} cells, "
                f"{occupied} blocked ({100.0 * occupied / total_cells:.1f}%), "
                f"method={vox_method}"
            )

            # debug viz
            if show_grid:
                t = time.time()
                show_occupancy_grid(stage, grid)
                timings["debug_grid"] = time.time() - t

            if show_graph:
                t = time.time()
                show_waypoint_graph(stage, grid)
                timings["debug_graph"] = time.time() - t

            # route pipes
            clear_pipes(stage)
            results = []

            for pipe in pipes:
                name = pipe["name"]
                start = pipe["start"]
                end = pipe["end"]
                pal = PIPE_PALETTE[pipe["index"] % len(PIPE_PALETTE)]

                self._panel.set_status(f"Routing {name}...")

                t = time.time()
                raw, msg = cuopt_solve(
                    grid, start, end,
                    server_url=server_url, time_limit=5,
                    bend_penalty=bend_penalty,
                )
                carb.log_warn(f"{LOG} cuopt: {msg}")

                if raw is None:
                    timings[name] = time.time() - t
                    results.append(f"{name}: no path")
                    continue

                # smooth and deduplicate
                raw[0] = np.array(start, dtype=np.float64)
                raw[-1] = np.array(end, dtype=np.float64)
                path_pts = smooth_path(raw)
                filtered = [path_pts[0]]
                for pt in path_pts[1:]:
                    if np.linalg.norm(pt - filtered[-1]) > 0.01:
                        filtered.append(pt)
                path_pts = filtered

                create_tube_mesh(
                    stage, path_pts, pipe_name=name,
                    radius=tube_radius, color=pal["tube"],
                )
                mark_tube_obstacle(grid, path_pts, tube_radius * 2.5)

                timings[name] = time.time() - t
                length = path_length(path_pts)
                bends = count_bends(path_pts)
                results.append(f"{name}: L={length:.0f} bends={bends}")

            timings["TOTAL"] = time.time() - t_total

            parts = [f"{k}={v:.3f}s" for k, v in timings.items()]
            carb.log_warn(f"{LOG} {' | '.join(parts)}")

            summary = "[CUOPT] " + "  |  ".join(results)
            self._panel.set_status(summary)

        except Exception as e:
            carb.log_error(f"{LOG} routing failed: {e}\n{traceback.format_exc()}")
            self._panel.set_status(f"Error: {e}")

    def _clear(self) -> None:
        try:
            stage = omni.usd.get_context().get_stage()
            clear_pipes(stage)
            clear_debug(stage)
            self._panel.set_status("Cleared.")
        except Exception as e:
            self._panel.set_status(f"Error: {e}")
