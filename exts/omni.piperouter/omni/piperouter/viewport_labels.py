"""Screen-space order labels in the viewport, via omni.ui.scene.

Draws a small text label (waypoint order number 1..N, plus S/E for the wire's
endpoints) anchored at given world positions, following the active viewport
camera. This is GUI-only and viewport-API dependent, so EVERY Kit call is
guarded: if the viewport API is absent or differs, the extension still loads and
routes — it just shows no labels (a [piperouter] warning is logged). Mirrors the
defensive style of the camera helper. UNVERIFIED in a live GUI.

Usage:
    labels = ViewportOrderLabels()
    labels.update([(world_pos, "1", 0xFFFFFFFF), ...])  # rebuilds the overlay
    labels.clear()                                       # remove all labels
    labels.destroy()                                     # on panel teardown
"""
from __future__ import annotations

import carb

_FRAME_ID = "omni.piperouter.order_labels"
_LABEL_SIZE = 22


class ViewportOrderLabels:
    def __init__(self):
        self._scene_view = None
        self._frame = None
        self._vp_api = None
        self._items = []  # list of (world_pos (3,), text, color_abgr)

    def _ensure_overlay(self):
        """Lazily create the SceneView bound to the active viewport. Returns True
        if an overlay is available to draw into."""
        if self._scene_view is not None:
            return True
        try:
            from omni.kit.viewport.utility import get_active_viewport_window
            from omni.ui import scene as sc

            vpw = get_active_viewport_window()
            if vpw is None:
                return False
            self._frame = vpw.get_frame(_FRAME_ID)
            with self._frame:
                self._scene_view = sc.SceneView()
            # Bind to the viewport so the scene gets the camera's view/projection
            # each frame (labels then track orbit/pan/zoom).
            self._vp_api = vpw.viewport_api
            self._vp_api.add_scene_view(self._scene_view)
            return True
        except Exception as exc:  # noqa: BLE001 — viewport API varies across Kit
            carb.log_warn(f"[piperouter] viewport order-labels unavailable: {exc}")
            self._scene_view = None
            self._frame = None
            self._vp_api = None
            return False

    def update(self, items):
        """Rebuild the overlay. items: list of (world_pos, text, color_abgr)."""
        self._items = list(items)
        if not self._ensure_overlay():
            return
        try:
            import omni.ui as ui
            from omni.ui import scene as sc

            self._scene_view.scene.clear()
            with self._scene_view.scene:
                for pos, text, color in self._items:
                    mtx = sc.Matrix44.get_translation_matrix(
                        float(pos[0]), float(pos[1]), float(pos[2]))
                    with sc.Transform(transform=mtx):
                        # sc.Label takes an omni.ui.Alignment (NOT omni.ui.scene.*)
                        sc.Label(str(text), alignment=ui.Alignment.CENTER,
                                 color=int(color), size=_LABEL_SIZE)
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[piperouter] viewport order-label draw failed: {exc}")

    def clear(self):
        self.update([])

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
        try:
            if self._frame is not None:
                self._frame.clear()
        except Exception:  # noqa: BLE001
            pass
        self._scene_view = None
        self._frame = None
        self._vp_api = None
        self._items = []
