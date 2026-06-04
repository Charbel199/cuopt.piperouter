"""omni.ui panel for the two-phase expert workflow, organized into collapsible
sections with status feedback, color swatches, and per-wire controls.

Phase A: Create sample scene (or add wires) -> Route All. Phase B: select a wire ->
tune sliders, add waypoints, Re-route just it (others locked as obstacles), Lock.
Plus thermal/EM tagging, an occupancy overlay, a node-count readout, and BOM export.

Each wire keeps a stable `key` (used for marker prim paths so rename is safe) and an
editable `name` (the display + route id).
"""
from __future__ import annotations

import asyncio
import csv
import json

import omni.kit.app
import omni.ui as ui
import omni.usd
from pxr import Tf, Usd

from . import bom as bom_lib
from . import headings, scene_ops, viewport_labels, waypoints, wire_library

_WEIGHTS = ("surface", "bend", "thermal", "em", "smoothing")

# Slider help — shown as tooltips so the soft constraints are self-explanatory.
_WEIGHT_HELP = {
    "surface": ("Surface hug", "Pull the route toward nearby surfaces so it can be "
                "clipped down (higher = hug closer; 0 = ignore)."),
    "bend": ("Bend (gentle)", "Penalize sharp turns for gentler curves; respects the "
             "wire's min bend radius (higher = straighter; 0 = allow sharp turns)."),
    "thermal": ("Thermal avoid", "Steer away from hot regions, from prims tagged with "
                "°C (higher = avoid more; 0 = ignore heat)."),
    "em": ("EM avoid", "Steer away from EM sources, scaled by the wire type's EM "
           "sensitivity (higher = avoid more; 0 = ignore EM)."),
    "smoothing": ("Smoothing", "Fibre-neutre curve smoothing strength (cuSolver). "
                  "0 = off (raw grid path); higher = smoother, gentler curves. "
                  "Always stays clear of obstacles."),
}

_DOT = {  # status -> ABGR color for the ● status dot
    "routed": 0xFF33CC33,
    "no_path": 0xFF3333CC,
    "locked": 0xFFCC7A33,
    "unrouted": 0xFF888888,
}
_PRIMARY = 0xFF2A7DBE
_OK = 0xFF33CC33
_BAD = 0xFF3333CC


def _abgr(rgb):
    r, g, b = (max(0, min(255, int(c * 255))) for c in rgb)
    return 0xFF000000 | (b << 16) | (g << 8) | r


class PipeRouterPanel:
    def __init__(self, get_stage, api, default_url="http://localhost:8000"):
        self._get_stage = get_stage
        self._api = api
        self._default_url = default_url
        self._types = wire_library.load_wire_library()
        self._type_labels = [t["label"] for t in self._types]
        self._type_ids = [t["id"] for t in self._types]
        self._wires = []
        self._selected = None
        self._key_counter = 0
        self._last_bom = []
        self._building = False
        self._refresh_pending = False
        self._need_wires = False
        self._need_inspector = False
        self._need_tags = False
        self._need_bom = False
        self._vp_labels = viewport_labels.ViewportOrderLabels()
        self._window = ui.Window("PipeRouter", width=520, height=780)
        self._build()

    # ---------------------------------------------------------------- build
    def _build(self):
        with self._window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, height=0):
                    ui.Label("PipeRouter", style={"font_size": 20})
                    with ui.HStack(height=0, spacing=6):
                        ui.Button("Reconnect", width=100, clicked_fn=self._check_connection)
                        self._status_dot = ui.Rectangle(
                            width=12, height=12,
                            style={"background_color": 0xFF888888, "border_radius": 6})
                        self._status = ui.Label("checking...")
                    self._progress = ui.Label("", style={"color": 0xFFBBBBBB})

                    self._section_setup()
                    self._section_wires()
                    self._section_inspector()
                    self._section_views()
                    self._section_tagging()
                    self._section_output()
        self._check_connection()
        self._obj_listener = None
        self._stage_sub = None
        self._register_stage_listener()

    def _register_stage_listener(self):
        """Watch the stage so the tagged-prims list updates when the scene changes
        (e.g. an object is deleted) — not just when we tag from the panel."""
        try:
            if self._obj_listener is not None:
                self._obj_listener.Revoke()
                self._obj_listener = None
            stage = self._get_stage()
            if stage is not None:
                self._obj_listener = Tf.Notice.Register(
                    Usd.Notice.ObjectsChanged, self._on_objects_changed, stage)
            if self._stage_sub is None:
                self._stage_sub = omni.usd.get_context().get_stage_event_stream() \
                    .create_subscription_to_pop(self._on_stage_event, name="piperouter.stage")
        except Exception:
            pass

    def _on_objects_changed(self, notice, sender):
        # coalesced (one rebuild next frame); rebuilding only reads the stage, so it
        # cannot re-trigger this notice
        self._schedule(tags=True)

    def _on_stage_event(self, e):
        if e.type == int(omni.usd.StageEventType.OPENED):
            self._register_stage_listener()   # re-bind to the new stage
            self._schedule(tags=True)

    def _section_views(self):
        self._views_frame = ui.CollapsableFrame("Cross-sections + cameras", collapsed=True)
        with self._views_frame:
            with ui.VStack(spacing=6, height=0):
                with ui.HStack(height=0, spacing=4):
                    ui.Label("Jump camera:", width=90)
                    ui.Button("Top", clicked_fn=lambda: self._api.create_view_camera("xy"))
                    ui.Button("Front", clicked_fn=lambda: self._api.create_view_camera("xz"))
                    ui.Button("Side", clicked_fn=lambda: self._api.create_view_camera("yz"))
                ui.Button("Refresh 2D views", clicked_fn=lambda: self._refresh_views(force=True))
                # full-width images stacked vertically (much larger than side-by-side)
                self._providers = {}
                for key, label in (("xy", "XY (top)"), ("xz", "XZ (front)"), ("yz", "YZ (side)")):
                    ui.Label(label, height=0, style={"color": 0xFF999999})
                    prov = ui.ByteImageProvider()
                    self._providers[key] = prov
                    ui.ImageWithProvider(prov, height=300)

    def _section_setup(self):
        with ui.CollapsableFrame("Scene & Setup", collapsed=False):
            with ui.VStack(spacing=4, height=0):
                ui.Button("Create sample scene", clicked_fn=self._create_sample, height=28)
                with ui.HStack(height=0):
                    ui.Label("Grid resolution", width=110)
                    self._res = ui.IntField()
                    self._res.model.set_value(64)
                    self._res.model.add_value_changed_fn(lambda m: self._update_readout())
                self._readout = ui.Label("", style={"color": 0xFF999999})
                with ui.HStack(height=0):
                    ui.Label("Safety clearance (m)", width=130,
                             tooltip="Extra gap kept from meshes ON TOP of the wire's "
                                     "radius. 0 = the route may run flush against a "
                                     "surface (just won't intersect it).")
                    self._clearance = ui.FloatField()
                    self._clearance.model.set_value(0.0)
                with ui.HStack(height=0):
                    ui.Label("Debug overlay", width=110)
                    # None / Occupancy / Thermal / EM — voxelizes + authors a colored
                    # point cloud under /World/PipeRouter/debug.
                    self._overlay_combo = ui.ComboBox(0, "None", "Occupancy", "Thermal", "EM")
                    self._overlay_combo.model.add_item_changed_fn(self._on_overlay)
        self._update_readout()

    def _section_wires(self):
        with ui.CollapsableFrame("Wires", collapsed=False):
            with ui.VStack(spacing=4, height=0):
                self._wire_stack = ui.VStack(spacing=3, height=0)
                self._rebuild_wires()
                with ui.HStack(height=0):
                    ui.Button("+ Add wire", clicked_fn=self._add_wire)
                    ui.Button("ROUTE ALL", clicked_fn=self._on_route_all, height=34,
                              style={"background_color": _PRIMARY, "color": 0xFFFFFFFF})

    def _section_inspector(self):
        self._inspector_frame = ui.CollapsableFrame("Selected wire", collapsed=False)
        with self._inspector_frame:
            self._inspector = ui.VStack(spacing=4, height=0)
            self._rebuild_inspector()

    def _section_tagging(self):
        with ui.CollapsableFrame("Tagging (thermal / EM)", collapsed=True):
            with ui.VStack(spacing=4, height=0):
                ui.Label("Select a prim in the stage, set values, then Tag:",
                         style={"color": 0xFF999999})
                with ui.HStack(height=0):
                    ui.Label("Temp °C", width=70)
                    self._temp = ui.FloatField()
                    ui.Label("EM", width=30)
                    self._em = ui.FloatField()
                    ui.Button("Tag", width=60, clicked_fn=self._on_tag)
                ui.Separator()
                ui.Label("Tagged prims:", style={"color": 0xFF999999})
                self._tag_stack = ui.VStack(spacing=2, height=0)
                self._rebuild_tags()

    _TEMP_COLOR = 0xFF4466EE   # warm red-orange (thermal)
    _EM_COLOR = 0xFFCCAA33     # teal-gold (EM)

    def _rebuild_tags(self):
        if self._window is None:
            return
        self._tag_stack.clear()
        tags = self._api.list_tags()
        with self._tag_stack:
            if not tags:
                ui.Label("  (none — select a prim above and Tag it)",
                         style={"color": 0xFF888888})
                return
            for t in tags:
                has_t = t["temp_c"] is not None
                has_e = t["em"] is not None
                name = t["path"].rsplit("/", 1)[-1]
                # presence dot: warm if thermal, teal if EM-only
                dot = self._TEMP_COLOR if has_t else self._EM_COLOR
                with ui.HStack(height=0, spacing=6):
                    ui.Rectangle(width=10, height=10, tooltip=t["path"],
                                 style={"background_color": dot, "border_radius": 5})
                    ui.Label(name, width=130, tooltip=t["path"])
                    ui.Label(f"{t['temp_c']:.0f}°C" if has_t else "", width=52,
                             style={"color": self._TEMP_COLOR})
                    ui.Label(f"EM {t['em']:.2f}" if has_e else "", width=64,
                             style={"color": self._EM_COLOR})
                    ui.Spacer()
                    ui.Button("Locate", width=58,
                              clicked_fn=lambda p=t["path"]: self._api.select_prim(p))
                    ui.Button("Remove", width=58,
                              clicked_fn=lambda p=t["path"]: self._remove_tag(p))

    def _remove_tag(self, path):
        self._api.clear_tag(path)
        self._schedule(tags=True)

    # BOM table column widths (px); shared by header + rows for alignment.
    _BOM_COLS = (("Wire", 120), ("Type", 130), ("Length", 70), ("Mass", 64),
                 ("Cost", 64), ("", 16))

    def _section_output(self):
        with ui.CollapsableFrame("Output / BOM", collapsed=False):
            with ui.VStack(spacing=4, height=0):
                with ui.HStack(height=0):
                    ui.Label("Export path", width=90)
                    self._bom_path = ui.StringField()
                    self._bom_path.model.set_value("/tmp/piperouter_bom")
                    ui.Button("Export", width=70, clicked_fn=self._on_export)
                # the table is rebuilt into this container by _rebuild_bom
                self._bom_table = ui.VStack(spacing=2, height=0)
        self._rebuild_bom()

    def _rebuild_bom(self):
        if self._window is None or getattr(self, "_bom_table", None) is None:
            return
        self._bom_table.clear()
        s = bom_lib.summarize(self._last_bom, self._bom_type_labels())
        with self._bom_table:
            if not s["rows"]:
                ui.Label("(no routes yet — Route All to populate)",
                         style={"color": 0xFF888888})
                return
            # header row
            with ui.HStack(height=0, spacing=4):
                for title, wpx in self._BOM_COLS:
                    ui.Label(title, width=wpx, style={"color": 0xFF999999, "font_size": 13})
            ui.Rectangle(height=1, style={"background_color": 0xFF444444})
            for r in s["rows"]:
                routed = r["status"] == "routed"
                with ui.HStack(height=0, spacing=4):
                    ui.Label(r["wire_id"], width=120)
                    ui.Label(r["type"], width=130, style={"color": 0xFFAAAAAA})
                    ui.Label(f"{r['length_m']:.2f} m" if routed else "—", width=70)
                    ui.Label(f"{r['mass']:.2f} kg" if routed else "—", width=64)
                    ui.Label(f"${r['cost']:.2f}" if routed else "—", width=64)
                    ui.Rectangle(width=12, height=12,
                                 tooltip="routed" if routed else "no path",
                                 style={"background_color": _DOT.get(r["status"], 0xFF888888),
                                        "border_radius": 6})
            ui.Rectangle(height=1, style={"background_color": 0xFF444444})
            # totals row
            with ui.HStack(height=0, spacing=4):
                ui.Label(f"TOTAL ({s['n_routed']} routed"
                         + (f", {s['n_no_path']} no-path" if s["n_no_path"] else "") + ")",
                         width=250, style={"font_size": 14})
                ui.Label(f"{s['total_length']:.2f} m", width=70, style={"font_size": 14})
                ui.Label(f"{s['total_mass']:.2f} kg", width=64, style={"font_size": 14})
                ui.Label(f"${s['total_cost']:.2f}", width=64,
                         style={"font_size": 14, "color": _OK})

    def _bom_type_labels(self):
        """{wire_id -> wire-type label} so the BOM Type column is filled."""
        return {w["name"]: self._type_labels[w["type_index"]] for w in self._wires}

    # ----------------------------------------------------------- connection
    def _check_connection(self):
        url = self._url_value()
        info, err = self._api.health(url)
        if err:
            self._status.text = f"Not connected  ({err.splitlines()[0][:40]})"
            self._status_dot.set_style({"background_color": _BAD, "border_radius": 6})
        else:
            self._status.text = f"Connected  (backend: {info.get('backend', '?')})"
            self._status_dot.set_style({"background_color": _OK, "border_radius": 6})

    def _url_value(self):
        return self._default_url  # URL is fixed for now; Reconnect re-probes it

    def _update_readout(self):
        res = self._res.model.get_value_as_int()
        # Cells are uniform cubes: cell size = (longest scene axis) / resolution, and
        # the other axes get however many of those cubes fit. So `res` is the voxel
        # count along the LONGEST axis; spacing is identical on all three axes.
        self._readout.text = (f"≈{res} voxels across the longest axis · uniform cubic "
                              f"cells. Higher = finer but slower.")

    # ------------------------------------------------- deferred UI refresh
    # omni.ui forbids clearing/rebuilding a container from inside an event/draw
    # callback ("Container::clear was called during an event or draw"). So event
    # handlers request a refresh and we rebuild on the next frame instead.
    def _schedule(self, wires=False, inspector=False, tags=False, bom=False):
        self._need_wires = self._need_wires or wires
        self._need_inspector = self._need_inspector or inspector
        self._need_tags = self._need_tags or tags
        self._need_bom = self._need_bom or bom
        if self._refresh_pending:
            return
        self._refresh_pending = True
        asyncio.ensure_future(self._deferred_refresh())

    async def _deferred_refresh(self):
        try:
            await omni.kit.app.get_app().next_update_async()
        except Exception:
            pass
        self._refresh_pending = False
        if self._need_wires:
            self._need_wires = False
            self._rebuild_wires()
        if self._need_inspector:
            self._need_inspector = False
            self._rebuild_inspector()
        if self._need_tags:
            self._need_tags = False
            self._rebuild_tags()
        if self._need_bom:
            self._need_bom = False
            self._rebuild_bom()
        # viewport order labels track marker drags / selection, refreshed every
        # coalesced frame (cheap; reads stage positions for the selected wire)
        self._refresh_vp_labels()

    # ----------------------------------------------------------------- wires
    def _new_wire(self, key, name, type_index=0):
        return {"key": key, "name": name, "type_index": type_index,
                "weights": {k: 1.0 for k in _WEIGHTS}, "waypoints": [], "wp_counter": 0,
                "locked": False, "polyline": None, "status": "unrouted",
                "length_m": 0.0, "cost": 0.0, "combo": None, "name_model": None,
                "_swatch": None, "start_head_idx": 0, "end_head_idx": 0}

    def _add_wire(self):
        stage = self._get_stage()
        if stage is None:
            self._progress.text = "open a USD stage first"
            return
        key = f"wire_{self._key_counter}"
        self._key_counter += 1
        scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/{key}_start",
                               (0.0, 0.0, 0.0), color=(0.1, 0.9, 0.1), radius=0.12)
        scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/{key}_end",
                               (0.5, 0.0, 0.0), color=(0.9, 0.1, 0.1), radius=0.12)
        self._wires.append(self._new_wire(key, key))
        self._schedule(wires=True)

    def _create_sample(self):
        descriptors, err = self._api.create_sample_scene()
        if err:
            self._progress.text = f"sample scene: {err}"
            return
        self._wires = []
        self._selected = None
        self._key_counter = 0
        for d in descriptors:
            ti = self._type_ids.index(d["type_id"]) if d["type_id"] in self._type_ids else 0
            # markers were authored by build_sample_scene under the descriptor name
            self._wires.append(self._new_wire(d["name"], d["name"], ti))
            self._key_counter += 1
        self._schedule(wires=True, inspector=True, tags=True)
        self._progress.text = f"sample scene ready — {len(descriptors)} wires. Click ROUTE ALL."

    def _delete_wire(self, idx):
        stage = self._get_stage()
        w = self._wires[idx]
        if stage is not None:
            for suffix in ("_start", "_end"):
                p = stage.GetPrimAtPath(f"{scene_ops.MARKERS_SCOPE}/{w['key']}{suffix}")
                if p and p.IsValid():
                    stage.RemovePrim(p.GetPath())
            for wp in w["waypoints"]:
                p = stage.GetPrimAtPath(wp)
                if p and p.IsValid():
                    stage.RemovePrim(p.GetPath())
        del self._wires[idx]
        if self._selected == idx:
            self._selected = None
        elif self._selected is not None and self._selected > idx:
            self._selected -= 1
        self._schedule(wires=True, inspector=True)

    def _rebuild_wires(self):
        if self._window is None:
            return
        self._building = True  # suppress callbacks fired while widgets are constructed
        try:
            self._wire_stack.clear()
            with self._wire_stack:
                if not self._wires:
                    ui.Label("(no wires — Create sample scene or + Add wire)",
                             style={"color": 0xFF888888})
                for idx, w in enumerate(self._wires):
                    color = self._types[w["type_index"]]["color"]
                    status = "locked" if w["locked"] else w["status"]
                    is_sel = idx == self._selected
                    with ui.HStack(height=0, spacing=4):
                        # select button — ASCII ">" + highlight (the old glyph showed
                        # as "?" in the Kit font); click selects + highlights the wire
                        ui.Button(">" if is_sel else " ", width=26,
                                  clicked_fn=lambda i=idx: self._select(i),
                                  style={"background_color": _PRIMARY} if is_sel else {})
                        # type color swatch
                        w["_swatch"] = ui.Rectangle(width=12, height=12,
                                                    style={"background_color": _abgr(color)})
                        # status chip (green routed / red no-path / blue locked / grey)
                        ui.Rectangle(width=12, height=12, tooltip=status,
                                     style={"background_color": _DOT.get(status, _DOT["unrouted"]),
                                            "border_radius": 6})
                        nm = ui.StringField(width=90)
                        nm.model.set_value(w["name"])
                        nm.model.add_value_changed_fn(lambda m, i=idx: self._rename(i, m))
                        w["name_model"] = nm
                        combo = ui.ComboBox(w["type_index"], *self._type_labels)
                        combo.model.add_item_changed_fn(lambda m, e, i=idx: self._set_type(i, m))
                        w["combo"] = combo
                        figs = (f"{w['length_m']:.2f}m ${w['cost']:.0f}"
                                if w["status"] == "routed" else "")
                        ui.Label(figs, width=80, style={"color": 0xFFAAAAAA})
                        ui.Button("X", width=22, clicked_fn=lambda i=idx: self._delete_wire(i))
        finally:
            self._building = False

    def _rename(self, idx, model):
        if not self._building and idx < len(self._wires):
            self._wires[idx]["name"] = model.get_value_as_string()

    def _set_type(self, idx, model):
        # NB: do NOT rebuild the list here — rebuilding constructs new ComboBoxes whose
        # item_changed callbacks re-enter this handler, which previously cleared the
        # list mid-build and made other wires vanish. Update in place instead.
        if self._building or idx >= len(self._wires):
            return
        try:
            i = model.get_item_value_model().as_int
        except Exception:
            return
        self._wires[idx]["type_index"] = i
        sw = self._wires[idx].get("_swatch")
        if sw is not None:
            sw.set_style({"background_color": _abgr(self._types[i]["color"])})

    def _select(self, idx):
        self._selected = idx
        # highlight the whole wire in the viewport: start/end + waypoints + its tube
        w = self._wires[idx]
        paths = [f"{scene_ops.MARKERS_SCOPE}/{w['key']}_start",
                 f"{scene_ops.MARKERS_SCOPE}/{w['key']}_end",
                 *w["waypoints"],
                 f"{scene_ops.ROUTES_SCOPE}/{w['name']}"]
        self._api.select_prims(paths)
        self._schedule(wires=True, inspector=True)

    def _toggle_lock(self):
        w = self._wires[self._selected]
        w["locked"] = not w["locked"]
        self._schedule(wires=True, inspector=True)

    # ------------------------------------------------------------- inspector
    def _rebuild_inspector(self):
        if self._window is None:
            return
        self._inspector.clear()
        with self._inspector:
            if self._selected is None or self._selected >= len(self._wires):
                ui.Label("(select a wire above)", style={"color": 0xFF888888})
                return
            w = self._wires[self._selected]
            status = "locked" if w["locked"] else w["status"]
            with ui.HStack(height=0, spacing=6):
                ui.Rectangle(width=14, height=14,
                             style={"background_color": _abgr(self._types[w["type_index"]]["color"])})
                ui.Rectangle(width=14, height=14, tooltip=status,
                             style={"background_color": _DOT.get(status, _DOT["unrouted"]),
                                    "border_radius": 7})
                ui.Label(f"{w['name']}", style={"font_size": 16})
                ui.Label(f"· {self._type_labels[w['type_index']]}",
                         style={"color": 0xFFAAAAAA})
            ui.Label("Soft constraints (0 = ignore, 10 = strong). Hover for details.",
                     style={"color": 0xFF999999})
            self._sliders = {}
            for k in _WEIGHTS:
                label, help_text = _WEIGHT_HELP[k]
                with ui.HStack(height=0):
                    ui.Label(label, width=100, tooltip=help_text)
                    s = ui.FloatSlider(min=0.0, max=10.0, tooltip=help_text)
                    s.model.set_value(w["weights"][k])
                    # Persist the value into the wire AS IT CHANGES (not just on
                    # Re-route) so switching to another wire and back keeps it.
                    s.model.add_value_changed_fn(lambda m, kk=k: self._set_weight(kk, m))
                    self._sliders[k] = s
            # Optional pinned departure/arrival headings (None = free).
            ui.Label("Pinned headings (optional; None = free direction)",
                     style={"color": 0xFF999999})
            with ui.HStack(height=0):
                ui.Label("Start heading", width=100,
                         tooltip="Force the wire to LEAVE the start in this direction.")
                start_combo = ui.ComboBox(w["start_head_idx"], *headings.HEADING_OPTIONS)
                start_combo.model.add_item_changed_fn(
                    lambda m, *_: self._set_heading("start_head_idx", m))
            with ui.HStack(height=0):
                ui.Label("End heading", width=100,
                         tooltip="Force the wire to ARRIVE at the end in this direction.")
                end_combo = ui.ComboBox(w["end_head_idx"], *headings.HEADING_OPTIONS)
                end_combo.model.add_item_changed_fn(
                    lambda m, *_: self._set_heading("end_head_idx", m))
            ui.Button("+ Add waypoint (route must pass through)",
                      clicked_fn=self._add_waypoint)
            if w["waypoints"]:
                ui.Label("  drag the :: handle to reorder (order = route sequence)",
                         style={"color": 0xFF888888})
                for i, wp_path in enumerate(w["waypoints"]):
                    with ui.HStack(height=0, spacing=4) as row:
                        # drag handle: starts a drag carrying this row's index
                        handle = ui.Label("::", width=18, style={"color": 0xFFAAAAAA},
                                          tooltip="Drag to reorder")
                        handle.set_drag_fn(lambda j=i: str(j))
                        ui.Label(f"#{i + 1}", width=34,
                                 style={"color": 0xFFDDDDDD, "font_size": 15})
                        ui.Button("Locate", width=64,
                                  clicked_fn=lambda p=wp_path: self._api.select_prim(p))
                        ui.Button("Delete", width=64,
                                  clicked_fn=lambda j=i: self._delete_waypoint(j))
                    # whole row is a drop target -> move dragged index to here
                    row.set_accept_drop_fn(lambda *_: True)
                    row.set_drop_fn(lambda e, dst=i: self._reorder_waypoint(
                        int(e.mime_data), dst))
            else:
                ui.Label("  (no waypoints)", style={"color": 0xFF888888})
            with ui.HStack(height=0):
                ui.Button("Re-route this wire", clicked_fn=self._on_refine,
                          style={"background_color": _PRIMARY, "color": 0xFFFFFFFF})
                ui.Button("Unlock" if w["locked"] else "Lock", clicked_fn=self._toggle_lock)

    def _set_weight(self, k, model):
        # Live-write a slider value into the currently selected wire's weights so it
        # survives selecting another wire and coming back.
        if self._selected is None or self._selected >= len(self._wires):
            return
        self._wires[self._selected]["weights"][k] = float(model.get_value_as_float())

    def _set_heading(self, key, model):
        # Persist the chosen heading axis index into the selected wire so it
        # survives switching wires (mirrors _set_weight for the sliders).
        if self._selected is None or self._selected >= len(self._wires):
            return
        idx = int(model.get_item_value_model().get_value_as_int())
        self._wires[self._selected][key] = idx

    def _add_waypoint(self):
        stage = self._get_stage()
        if stage is None or self._selected is None:
            return
        w = self._wires[self._selected]
        n = w["wp_counter"]
        w["wp_counter"] += 1
        path = f"{scene_ops.MARKERS_SCOPE}/{w['key']}_wp{n}"
        # spawn near the wire's start so it's easy to find, then the user drags it
        start = scene_ops.get_world_pos(stage, f"{scene_ops.MARKERS_SCOPE}/{w['key']}_start")
        pos = (float(start[0]) + 0.3, float(start[1]), float(start[2])) if start is not None \
            else (0.25, 0.0, 0.0)
        scene_ops.spawn_marker(stage, path, pos, color=(0.1, 0.5, 0.9), radius=0.1)
        w["waypoints"].append(path)
        self._api.select_prim(path)  # auto-select so it can be dragged immediately
        self._schedule(inspector=True)

    def _delete_waypoint(self, idx):
        if self._selected is None:
            return
        w = self._wires[self._selected]
        if idx >= len(w["waypoints"]):
            return
        path = w["waypoints"][idx]
        stage = self._get_stage()
        if stage is not None:
            p = stage.GetPrimAtPath(path)
            if p and p.IsValid():
                stage.RemovePrim(p.GetPath())
        del w["waypoints"][idx]
        self._schedule(inspector=True)

    def _reorder_waypoint(self, src, dst):
        # drag-drop reorder of the selected wire's waypoints (order = route legs).
        if self._selected is None or self._selected >= len(self._wires):
            return
        w = self._wires[self._selected]
        w["waypoints"] = waypoints.reorder(w["waypoints"], src, dst)
        self._schedule(inspector=True)  # re-numbers rows + refreshes viewport labels

    def _refresh_vp_labels(self):
        """Update the viewport order-number overlay for the SELECTED wire: S at the
        start marker, 1..N above each waypoint (in route order), E at the end."""
        vpl = getattr(self, "_vp_labels", None)
        if vpl is None:
            return
        stage = self._get_stage()
        if stage is None or self._selected is None or self._selected >= len(self._wires):
            vpl.clear()
            return
        w = self._wires[self._selected]
        up = self._stage_up_offset(stage)
        items = []

        def _add(path, text, color):
            p = scene_ops.get_world_pos(stage, path)
            if p is not None:
                items.append(((p[0] + up[0], p[1] + up[1], p[2] + up[2]), text, color))

        _add(f"{scene_ops.MARKERS_SCOPE}/{w['key']}_start", "S", 0xFF33CC33)  # green
        for i, wp_path in enumerate(w["waypoints"]):
            _add(wp_path, str(i + 1), 0xFFFFFFFF)
        _add(f"{scene_ops.MARKERS_SCOPE}/{w['key']}_end", "E", 0xFF3333CC)    # red
        vpl.update(items)

    @staticmethod
    def _stage_up_offset(stage):
        """A small world-space offset along the stage up-axis, so labels float just
        ABOVE their markers rather than sitting on them."""
        try:
            from pxr import UsdGeom
            axis = UsdGeom.GetStageUpAxis(stage)
            d = 0.25
            return (0.0, d, 0.0) if axis == UsdGeom.Tokens.y else (0.0, 0.0, d)
        except Exception:
            return (0.0, 0.0, 0.25)

    # --------------------------------------------------------------- solving
    def _gather_wire(self, w, priority=0):
        stage = self._get_stage()
        start = scene_ops.get_world_pos(stage, f"{scene_ops.MARKERS_SCOPE}/{w['key']}_start")
        end = scene_ops.get_world_pos(stage, f"{scene_ops.MARKERS_SCOPE}/{w['key']}_end")
        if start is None or end is None:
            return None
        wps = []
        for wp_path in w["waypoints"]:
            p = scene_ops.get_world_pos(stage, wp_path)
            if p is not None:
                wps.append([float(x) for x in p])
        sh = headings.axis_to_vector(headings.HEADING_OPTIONS[w.get("start_head_idx", 0)])
        eh = headings.axis_to_vector(headings.HEADING_OPTIONS[w.get("end_head_idx", 0)])
        return {"name": w["name"], "spec": wire_library.as_spec(self._types[w["type_index"]]),
                "start": [float(x) for x in start], "end": [float(x) for x in end],
                "waypoints": wps, "weights": dict(w["weights"]),
                "connectivity": 26, "priority": priority,
                "clearance_m": float(self._clearance.model.get_value_as_float()),
                "start_heading": list(sh) if sh else None,
                "end_heading": list(eh) if eh else None}

    def _on_route_all(self):
        pairs = [(w, self._gather_wire(w, i)) for i, w in enumerate(self._wires)]
        wires = [g for (_w, g) in pairs if g]
        if not wires:
            self._progress.text = "add a wire first"
            return
        self._progress.text = "voxelizing + routing all..."
        results, bom, err = self._api.route_all(wires, self._res.model.get_value_as_int(),
                                                self._url_value())
        if err:
            self._progress.text = f"error: {err}"
            return
        by_name = {r["wire_id"]: r for r in (results or [])}
        for w in self._wires:
            r = by_name.get(w["name"])
            if r:
                w["status"] = r["status"]
                w["polyline"] = r.get("polyline") if r["status"] == "routed" else None
        for b in (bom or []):
            for w in self._wires:
                if w["name"] == b["wire_id"]:
                    w["length_m"] = b["length_m"]
                    w["cost"] = b["cost"]
        self._last_bom = bom or []
        note = getattr(self._api, "_clearance_note", None)
        self._progress.text = f"done — note: {note}" if note else "done"
        self._schedule(wires=True, bom=True)
        self._refresh_views()
        self._refresh_overlay()

    def _on_refine(self):
        if self._selected is None:
            return
        w = self._wires[self._selected]
        for k, s in getattr(self, "_sliders", {}).items():
            w["weights"][k] = s.model.get_value_as_float()
        wire = self._gather_wire(w)
        if wire is None:
            self._progress.text = "wire has no endpoints"
            return
        # Every OTHER already-routed wire is an obstacle, so re-routing this one
        # never overlaps the rest (wires must not overlap, even on single re-route).
        # Locking is no longer required to be avoided — it now just freezes a wire.
        obstacles = [{"spec": wire_library.as_spec(self._types[ow["type_index"]]),
                      "polyline": ow["polyline"]}
                     for ow in self._wires if ow is not w and ow.get("polyline")]
        self._progress.text = f"re-routing {w['name']}..."
        result, bom_row, err = self._api.refine(wire, obstacles,
                                                self._res.model.get_value_as_int(),
                                                self._url_value())
        if err:
            self._progress.text = f"error: {err}"
            return
        w["status"] = result["status"] if result else "no_path"
        w["polyline"] = result.get("polyline") if w["status"] == "routed" else None
        if bom_row:
            w["length_m"] = bom_row["length_m"]
            w["cost"] = bom_row["cost"]
            # replace this wire's row in the BOM (or append if absent) so the table
            # reflects the re-route without needing a full Route All
            self._last_bom = [b for b in self._last_bom if b["wire_id"] != bom_row["wire_id"]]
            self._last_bom.append(bom_row)
        self._progress.text = f"{w['name']}: {w['status']}"
        self._schedule(wires=True, bom=True)
        self._refresh_views()
        self._refresh_overlay()

    # ------------------------------------------------------ tag / overlay / io
    def _refresh_views(self, force=False):
        """Rebuild the XY/XZ/YZ projection images. The render voxelizes at a high
        resolution, so auto-calls (after routing) are skipped unless the section is
        open; the 'Refresh 2D views' button forces it."""
        frame = getattr(self, "_views_frame", None)
        if not force and frame is not None and frame.collapsed:
            return
        routes = [{"points": w["polyline"], "color": self._types[w["type_index"]]["color"]}
                  for w in self._wires if w.get("polyline") and w["status"] == "routed"]
        imgs, err = self._api.slice_views(routes)
        if err or not imgs:
            if err:
                self._progress.text = f"views: {err}"
            return
        for key, prov in self._providers.items():
            img = imgs.get(key)
            if img is None:
                continue
            try:
                h, w = int(img.shape[0]), int(img.shape[1])
                prov.set_bytes_data(img.reshape(-1).tolist(), [w, h])
            except Exception as exc:  # omni.ui image API differences — non-fatal
                self._progress.text = f"views: {exc}"
                return

    def _on_tag(self):
        temp = self._temp.model.get_value_as_float()
        em = self._em.model.get_value_as_float()
        err = self._api.write_tag(temp if temp != 0.0 else None, em if em != 0.0 else None)
        self._progress.text = err if err else "tagged selection"
        if not err:
            self._schedule(tags=True)

    def _refresh_overlay(self):
        """Re-render the active debug overlay (occupancy/thermal/em) with the latest
        grids; no-op if the overlay is set to None."""
        combo = getattr(self, "_overlay_combo", None)
        if combo is None:
            return
        idx = combo.model.get_item_value_model().as_int
        if idx == 0:                      # "None"
            return
        mode = ("none", "occupancy", "thermal", "em")[idx]
        clearance = float(self._clearance.model.get_value_as_float())
        err = self._api.show_overlay(self._res.model.get_value_as_int(),
                                     self._url_value(), mode, clearance_m=clearance)
        if err:
            self._progress.text = f"overlay: {err}"

    def _on_overlay(self, model, item=None):
        mode = ("none", "occupancy", "thermal", "em")[model.get_item_value_model().as_int]
        self._progress.text = f"overlay: {mode}..."
        clearance = float(self._clearance.model.get_value_as_float())
        err = self._api.show_overlay(self._res.model.get_value_as_int(),
                                     self._url_value(), mode, clearance_m=clearance)
        self._progress.text = f"overlay: {err}" if err else f"overlay: {mode}"

    def _on_export(self):
        if not self._last_bom:
            self._progress.text = "nothing to export yet"
            return
        base = self._bom_path.model.get_value_as_string()
        try:
            with open(base + ".json", "w") as f:
                json.dump(self._last_bom, f, indent=2)
            with open(base + ".csv", "w", newline="") as f:
                wr = csv.writer(f)
                wr.writerow(["wire_id", "status", "length_m", "cost", "mass"])
                for b in self._last_bom:
                    wr.writerow([b["wire_id"], b["status"], b["length_m"], b["cost"], b["mass"]])
            self._progress.text = f"exported {base}.json / .csv"
        except Exception as exc:
            self._progress.text = f"export error: {exc}"

    def destroy(self):
        if getattr(self, "_obj_listener", None) is not None:
            self._obj_listener.Revoke()
            self._obj_listener = None
        self._stage_sub = None
        if getattr(self, "_vp_labels", None) is not None:
            self._vp_labels.destroy()
            self._vp_labels = None
        if self._window:
            self._window.destroy()
            self._window = None
        # break reference cycles so the extension object is released on reload
        self._api = None
        self._get_stage = None
        self._wires = []
        self._sliders = {}
