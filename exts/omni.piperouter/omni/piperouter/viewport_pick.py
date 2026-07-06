"""Double-click-in-viewport -> world point under the cursor, via the active viewport's
pick query. Used to drop a waypoint exactly on the wire the user double-clicks.

GUI / viewport-API dependent, so EVERY Kit call is guarded: if the viewport API is absent
or differs, the picker silently does nothing (a [piperouter] warning is logged once).
Mirrors the defensive style of viewport_labels.py. UNVERIFIED in a live GUI - the exact
pick-query / gesture-payload API can vary across Kit builds; logging is verbose on purpose
so a failure points at the offending step.
"""
from __future__ import annotations

import carb

_FRAME_ID = "omni.piperouter.pick"


class ViewportPicker:
    """on_pick(world_xyz: list[float], hit_prim_path: str) fires on a viewport double-click,
    with the world-space surface point and the prim under the cursor."""

    def __init__(self, on_pick):
        self._on_pick = on_pick
        self._scene_view = None
        self._frame = None
        self._vp_api = None
        self._screen = None
        self._warned = False

    # ------------------------------------------------------------------
    def enable(self):
        """Lazily attach a double-click gesture to the active viewport. Safe to call
        repeatedly (no-op once attached)."""
        if self._screen is not None:
            return True
        try:
            from omni.kit.viewport.utility import get_active_viewport_window
            from omni.ui import scene as sc

            vpw = get_active_viewport_window()
            if vpw is None:
                return False
            self._vp_api = vpw.viewport_api
            self._frame = vpw.get_frame(_FRAME_ID)
            with self._frame:
                self._scene_view = sc.SceneView()
            self._vp_api.add_scene_view(self._scene_view)
            with self._scene_view.scene:
                # a full-screen invisible quad that reports double-clicks
                self._screen = sc.Screen(
                    gesture=sc.DoubleClickGesture(self._on_double_click))
            carb.log_info("[piperouter] viewport double-click picker armed")
            return True
        except Exception as exc:  # noqa: BLE001
            if not self._warned:
                self._warned = True
                carb.log_warn(f"[piperouter] viewport picker unavailable: {exc}")
            return False

    # ------------------------------------------------------------------
    def _on_double_click(self, *args):
        try:
            gesture = args[0] if args else None
            # NDC mouse position in [-1, 1] from the gesture payload (field name varies)
            payload = (getattr(gesture, "gesture_payload", None)
                       or getattr(getattr(gesture, "sender", None), "gesture_payload", None))
            ndc = getattr(payload, "mouse", None)
            if ndc is None:
                carb.log_warn("[piperouter] pick: no mouse NDC in gesture payload")
                return
            res = self._vp_api.resolution  # (w, h) of the render target
            px = (int((float(ndc[0]) * 0.5 + 0.5) * float(res[0])),
                  int((1.0 - (float(ndc[1]) * 0.5 + 0.5)) * float(res[1])))
            # ask the renderer what's under that pixel; callback gets (path, world_pos)
            self._vp_api.request_query(px, self._on_query, query_name="piperouter.pick")
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[piperouter] pick double-click failed: {exc}")

    def _on_query(self, path, world_pos, *args):
        try:
            if not path or world_pos is None:
                return
            xyz = [float(world_pos[0]), float(world_pos[1]), float(world_pos[2])]
            self._on_pick(xyz, str(path))
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[piperouter] pick query callback failed: {exc}")

    # ------------------------------------------------------------------
    def destroy(self):
        try:
            if self._vp_api is not None and self._scene_view is not None:
                self._vp_api.remove_scene_view(self._scene_view)
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._scene_view is not None:
                self._scene_view.destroy()
        except Exception:  # noqa: BLE001
            pass
        self._scene_view = self._frame = self._vp_api = self._screen = None
