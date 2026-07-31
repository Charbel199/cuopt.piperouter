"""Viewport heads-up display: a semi-transparent stats box over the active viewport.

Styled like the Kit FPS counter. Shows total cost, mass, length and wires routed,
refreshed after every Route All. All viewport calls are guarded, so if the viewport API
is unavailable the HUD quietly does nothing.
"""
from __future__ import annotations

import carb

_FRAME_ID = "omni.piperouter.hud"

# Colors are ABGR.
_BG       = 0xCC0A0A0A   # semi-transparent near-black
_GREEN    = 0xFF76B900   # NVIDIA green
_LABEL    = 0xFF999999   # dim label
_VALUE    = 0xFFEEEEEE   # bright value
_WARN     = 0xFFFF6B35   # orange for no-path
_TITLE    = 0xFFFFFFFF


class ViewportHUD:
    """Persistent viewport overlay showing route stats and selected-wire weights."""

    def __init__(self):
        self._frame = None
        self._visible = True
        self._warned_draw = False  # a draw failure recurs every frame, so warn once

    # ------------------------------------------------------------------
    def _ensure_frame(self):
        if self._frame is not None:
            return True
        try:
            from omni.kit.viewport.utility import get_active_viewport_window
            vpw = get_active_viewport_window()
            if vpw is None:
                return False
            self._frame = vpw.get_frame(_FRAME_ID)
            return True
        except Exception as exc:
            carb.log_warn(f"[piperouter] HUD unavailable: {exc}")
            return False

    # ------------------------------------------------------------------
    def update(self, stats: dict, selected_wire: dict | None):
        """Rebuild the HUD.

        Every `stats` key is optional and renders as '-' when missing: total_cost,
        total_mass, total_length, n_routed, n_total, n_no_path. `selected_wire` is the
        panel wire dict for the current selection, or None.
        """
        if not self._visible:
            return
        if not self._ensure_frame():
            return
        try:
            import omni.ui as ui
            self._frame.clear()
            if not self._visible:
                return
            with self._frame:
                self._build(ui, stats, selected_wire)
        except Exception as exc:
            if not self._warned_draw:
                self._warned_draw = True
                carb.log_warn(f"[piperouter] HUD draw failed (silencing further): {exc}")

    def _build(self, ui, stats, wire):
        # Spacers push the box into the bottom-right corner.
        with ui.VStack():
            ui.Spacer()
            with ui.HStack(height=0):
                ui.Spacer()
                self._build_stats(ui, stats)

    def _build_stats(self, ui, s):
        import omni.ui as ui2
        n_r  = s.get("n_routed", 0)
        n_t  = s.get("n_total", 0)
        n_np = s.get("n_no_path", 0)
        cost   = s.get("total_cost")
        mass   = s.get("total_mass")
        length = s.get("total_length")

        with ui2.ZStack(width=260):
            ui2.Rectangle(style={"background_color": _BG, "border_radius": 8,
                                  "border_width": 1, "border_color": 0xFF333333})
            with ui2.VStack(spacing=0):
                # Title bar.
                with ui2.HStack(height=32):
                    ui2.Spacer(width=10)
                    ui2.Rectangle(width=4, height=18,
                                   style={"background_color": _GREEN,
                                          "border_radius": 2})
                    ui2.Spacer(width=8)
                    ui2.Label("ROUTE STATS", style={"color": _TITLE,
                                                     "font_size": 14})
                    ui2.Spacer()
                ui2.Rectangle(height=1, style={"background_color": 0xFF333333})

                rows = [
                    ("WIRES",  f"{n_r}/{n_t}"
                               + (f"  ({n_np} no-path)" if n_np else " routed"),
                     _WARN if n_np else _VALUE),
                    ("COST",   f"${cost:.2f}"     if cost   is not None else "-", _VALUE),
                    ("MASS",   f"{mass:.3f} kg"   if mass   is not None else "-", _VALUE),
                    ("LENGTH", f"{length:.2f} m"  if length is not None else "-", _VALUE),
                ]
                for lbl, val, col in rows:
                    with ui2.HStack(height=28):
                        ui2.Spacer(width=14)
                        ui2.Label(lbl, width=70,
                                   style={"color": _LABEL, "font_size": 13})
                        ui2.Label(val, style={"color": col, "font_size": 14})
                        ui2.Spacer(width=10)
                ui2.Spacer(height=8)

    # ------------------------------------------------------------------
    def set_visible(self, visible: bool):
        self._visible = visible
        if not visible and self._frame is not None:
            try:
                self._frame.clear()
            except Exception:
                pass

    def clear(self):
        self.update({}, None)

    def destroy(self):
        try:
            if self._frame is not None:
                self._frame.clear()
        except Exception:
            pass
        self._frame = None
