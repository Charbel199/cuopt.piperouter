"""omni.ui panel for the two-phase routing workflow.

Phase A: create a scene (or add wires), then Route All. Phase B: select a wire, tune
its sliders, add waypoints, re-route just that wire with the others locked as
obstacles, then lock it.

Each wire keeps a stable `key`, used for marker prim paths so renaming is safe, and an
editable `name` used as the display label and the route id.
"""
from __future__ import annotations

import asyncio
import csv
import json
import traceback

import carb
import omni.kit.app
import omni.ui as ui
import omni.usd
from pxr import Tf, Usd

from . import bom as bom_lib
from . import bundles as bundle_lib
from . import headings, help_window, hud as hud_mod, scene_ops, session_io, viewport_labels, viewport_pick, waypoints, wire_library

_WEIGHTS = ("surface", "bend", "thermal", "em", "smoothing")

# Selectable routing algorithms. Index order must match the ComboBox order; the _ALGOS
# tuples hold the values sent to the solver, the _LABELS hold what the combo shows.
# octree_lattice is the default: the bend-aware lattice restricted to a corridor band,
# falling back to the full lattice when the band is too tight. Plain lattice is the
# exhaustive mode, for when soft costs must be followed exactly.
_GLOBAL_ALGOS = ("octree_lattice", "dense", "lattice", "astar", "fmm", "rrt", "octree",
                 "medial")
_LOCAL_ALGOS = ("fibre", "none", "trajopt", "elastic_rod")
_GLOBAL_SPECIAL = {"octree_lattice": "octree_lattice (fast - default)",
                   "dense": "dense (best quality, needs GPU)",
                   "lattice": "lattice (exhaustive)"}
_GLOBAL_LABELS = tuple(_GLOBAL_SPECIAL.get(n, f"{n} (experimental)") for n in _GLOBAL_ALGOS)
_LOCAL_LABELS = tuple(n if n in ("fibre", "none") else f"{n} (experimental)" for n in _LOCAL_ALGOS)

# Slider help, shown as tooltips.
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
_WARN = 0xFF1E96E6         # amber, for non-fatal warnings such as a relocated endpoint
_ACCENT = 0xFF76B900       # NVIDIA green, for captions and selection
_HINT_BG = 0xFF2B2B2B      # empty-state card background
_DIVIDER = 0xFF333333

# Uniform control heights so buttons line up across the whole panel.
_CTA_H = 30                # primary call-to-action (Route All, Re-route)
_BTN_H = 26                # standard button
_MINI_H = 22               # compact in-row buttons (S / E / X / Up / Dn)

_DEFAULT_WEIGHT = 1.0      # neutral soft-constraint weight (used by "reset")

# Kit authors these viewport cameras into the stage; we drop them from exported sessions
# so re-opening the file doesn't spam "Updated /OmniverseKit_* to Sdf.SpecifierOver".
_VIEWPORT_CAMERAS = ("OmniverseKit_Persp", "OmniverseKit_Front",
                     "OmniverseKit_Top", "OmniverseKit_Right")


def _abgr(rgb):
    r, g, b = (max(0, min(255, int(c * 255))) for c in rgb)
    return 0xFF000000 | (b << 16) | (g << 8) | r


class PipeRouterPanel:
    def __init__(self, get_stage, api, default_url="http://localhost:8000"):
        self._get_stage = get_stage
        self._api = api
        self._default_url = default_url
        all_types = wire_library.load_wire_library()
        # Routable types shown in the Wires section.
        self._types = [t for t in all_types if t["kind"] in ("wire", "pipe")]
        self._type_labels = [t["label"] for t in self._types]
        self._type_ids = [t["id"] for t in self._types]
        # Harness types, used for the cost and label of trunk BOM rows.
        self._bundle_types = [t for t in all_types if t["kind"] == "bundle"]
        self._bundle_type_labels = [t["label"] for t in self._bundle_types]
        self._bundle_type_ids = [t["id"] for t in self._bundle_types]
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
        self._need_bundles = False
        self._need_resync = False   # re-sync panel to stage after marker deletion or scene swap
        self._bundles = []
        self._bundle_counter = 0
        self._selected_bundle = None   # index into self._bundles, or None
        self._active_debug = None      # (wire_name, mode) of the live per-wire debug view
        self._checklist_collapsed = {}  # bid -> collapsed?, driven by the user
        self._vp_labels = viewport_labels.ViewportOrderLabels()
        self._hud = hud_mod.ViewportHUD()
        # Double-click on the selected wire's tube drops a waypoint there.
        self._picker = viewport_pick.ViewportPicker(on_pick=self._on_viewport_pick)
        self._hud_visible = True
        self._help = help_window.HelpWindow()
        self._window = ui.Window("PipeRouter", width=520, height=780)
        self._build()
        # The tutorial opens once on enable, unless the user unticked "show on startup".
        if self._help.show_on_startup():
            self._help.show()

    # ---------------------------------------------------------------- build
    def _build(self):
        with self._window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6, height=0):
                    with ui.HStack(height=0, spacing=8):
                        ui.Rectangle(width=4, height=24,
                                     style={"background_color": 0xFF76B900, "border_radius": 2})
                        ui.Label("PipeRouter", style={"font_size": 20})
                        ui.Spacer()
                        ui.Button("HUD", width=40, tooltip="Toggle viewport stats overlay",
                                  clicked_fn=self._toggle_hud)
                        ui.Button("?", width=28, tooltip="Help / tutorial",
                                  clicked_fn=self._open_help)
                    with ui.HStack(height=0, spacing=6):
                        self._status_dot = ui.Rectangle(
                            width=10, height=10,
                            style={"background_color": 0xFF888888, "border_radius": 5})
                        self._status = ui.Label("checking...", style={"color": 0xFF999999})
                        ui.Spacer()
                        ui.Button("Reconnect", width=90, height=22,
                                  clicked_fn=self._check_connection)
                    self._progress = ui.Label("", style={"color": 0xFFBBBBBB})

                    self._section_setup()
                    self._section_wires()
                    self._section_bundles()
                    self._section_inspector()
                    self._section_views()
                    self._section_tagging()
                    self._section_output()
        self._check_connection()
        self._obj_listener = None
        self._stage_sub = None
        self._register_stage_listener()

    def _register_stage_listener(self):
        """Watch the stage so the tagged-prims list tracks changes made outside the panel,
        such as an object being deleted in the viewport."""
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
        # Coalesced into one rebuild next frame. The rebuild only reads the stage, so it
        # cannot re-trigger this notice. resync drops wires and bundles whose markers the
        # user just deleted in the viewport.
        self._schedule(tags=True, resync=True)

    def _on_stage_event(self, e):
        if e.type == int(omni.usd.StageEventType.OPENED):
            self._register_stage_listener()   # re-bind to the new stage
            # A different scene invalidates wires tied to the old stage's markers, so
            # resync empties the panel to match. Load re-populates it with its own markers.
            self._schedule(tags=True, resync=True)

    def _section_views(self):
        self._views_frame = ui.CollapsableFrame("Cross-sections + cameras", collapsed=True)
        with self._views_frame:
            with ui.VStack(spacing=6, height=0):
                with ui.HStack(height=0, spacing=4):
                    ui.Label("Jump camera:", width=90)
                    ui.Button("Top", height=_BTN_H,
                              clicked_fn=lambda: self._api.create_view_camera("xy"))
                    ui.Button("Front", height=_BTN_H,
                              clicked_fn=lambda: self._api.create_view_camera("xz"))
                    ui.Button("Side", height=_BTN_H,
                              clicked_fn=lambda: self._api.create_view_camera("yz"))
                ui.Button("Refresh 2D views", height=_BTN_H,
                          clicked_fn=lambda: self._refresh_views(force=True))
                # Full-width images stacked vertically.
                self._providers = {}
                for key, label in (("xy", "XY (top)"), ("xz", "XZ (front)"), ("yz", "YZ (side)")):
                    ui.Label(label, height=0, style={"color": 0xFF999999})
                    prov = ui.ByteImageProvider()
                    self._providers[key] = prov
                    ui.ImageWithProvider(prov, height=300)

    # Uniform left-column width so every labelled control lines up.
    _LBL = 130

    def _caption(self, text):
        ui.Spacer(height=2)
        ui.Label(text, style={"color": _ACCENT, "font_size": 11})

    def _hint(self, text):
        """Draw a rounded empty-state card, styled to match the help window."""
        with ui.ZStack(height=0):
            ui.Rectangle(style={"background_color": _HINT_BG, "border_radius": 6})
            with ui.HStack(height=0):
                ui.Spacer(width=10)
                with ui.VStack(height=0):
                    ui.Spacer(height=8)
                    ui.Label(text, word_wrap=True,
                             style={"color": 0xFF9A9A9A, "font_size": 12})
                    ui.Spacer(height=8)
                ui.Spacer(width=10)

    def _select_bar(self, selected, on_click, tooltip):
        """Draw a row's selection indicator: a slim vertical bar, solid green when the row
        is selected and a faint outline otherwise."""
        style = ({"background_color": _ACCENT, "border_radius": 2} if selected
                 else {"background_color": 0x00000000, "border_radius": 2,
                       "border_width": 1, "border_color": 0xFF555555})
        ui.Button(" ", width=8, height=_MINI_H, clicked_fn=on_click,
                  tooltip=tooltip, style=style)

    def _section_setup(self):
        with ui.CollapsableFrame("Scene & Setup", collapsed=False):
            with ui.VStack(spacing=6, height=0):
                # Scene actions.
                self._caption("SCENE")
                with ui.HStack(height=0, spacing=4):
                    ui.Button("Create sample scene", clicked_fn=self._create_sample,
                              height=28, tooltip="Small engine-bay demo: 3 wires.")
                    ui.Button("Create complex scene", clicked_fn=self._create_complex,
                              height=28,
                              tooltip="Full engine-bay + chassis: 15 obstacles, 14 "
                                      "wires/pipes across all types.")
                with ui.HStack(height=0, spacing=4):
                    ui.Button("Save...", clicked_fn=self._on_save, height=26,
                              tooltip="Save the whole session (geometry, markers, routes, "
                                      "wires/bundles/settings) to one editable .usd file.")
                    ui.Button("Load...", clicked_fn=self._on_load, height=26,
                              tooltip="Open a saved .usd session and restore everything.")
                    ui.Button("Export .usdz...", clicked_fn=self._on_export_usdz, height=26,
                              tooltip="Export the session to a single compressed .usdz "
                                      "archive for sharing / handoff.")
                    ui.Button("Reset", clicked_fn=self._on_reset, height=26,
                              style={"background_color": 0xFF223344},
                              tooltip="Remove all wires, bundles, markers and routed tubes "
                                      "and start fresh. Obstacle meshes are kept. The panel "
                                      "also auto-clears when you open a new scene or delete "
                                      "markers.")
                ui.Rectangle(height=1, style={"background_color": 0xFF333333})
                # Solver settings.
                self._caption("SOLVER")
                with ui.HStack(height=0):
                    ui.Label("Grid resolution", width=self._LBL)
                    self._res = ui.IntField()
                    self._res.model.set_value(64)
                    self._res.model.add_value_changed_fn(lambda m: self._update_readout())
                self._readout = ui.Label("", style={"color": 0xFF999999, "font_size": 12})
                with ui.HStack(height=0):
                    ui.Label("Connectivity", width=self._LBL,
                             tooltip="Moves allowed per step = graph size = SPEED. "
                                     "Fast(6)=axis-only (blocky, ~4x faster); "
                                     "Balanced(18)=+2D diagonals; Smooth(26)=+3D corners "
                                     "(densest, slowest). Smoothing rounds all of them.")
                    # Index 0/1/2 maps to 6/18/26 connectivity; default is Smooth (26).
                    self._conn_combo = ui.ComboBox(2, "Fast (6)", "Balanced (18)",
                                                   "Smooth (26)")
                with ui.HStack(height=0):
                    ui.Label("Safety clearance (m)", width=self._LBL,
                             tooltip="DEFAULT extra gap kept from meshes ON TOP of the "
                                     "wire's radius, for objects without a clearance tag. "
                                     "Tag individual objects (Tagging section) for "
                                     "per-object distances. 0 = flush routing allowed.")
                    self._clearance = ui.FloatField()
                    self._clearance.model.set_value(0.0)
                with ui.HStack(height=0):
                    ui.Label("Debug overlay", width=self._LBL,
                             tooltip="Visualize the voxel field as a colored point cloud "
                                     "under /World/PipeRouter/debug.")
                    self._overlay_combo = ui.ComboBox(0, "None", "Occupancy", "Thermal", "EM")
                    self._overlay_combo.model.add_item_changed_fn(self._on_overlay)
                # Experimental algorithm pickers, tucked away by default.
                with ui.CollapsableFrame("Advanced - routing algorithms", collapsed=True):
                    with ui.VStack(spacing=4, height=0):
                        ui.Label("Defaults (octree_lattice + fibre) are recommended. Pick "
                                 "'lattice (exhaustive)' when soft costs must be followed "
                                 "exactly; the rest are experimental / for benchmarking.",
                                 word_wrap=True,
                                 style={"color": 0xFF888888, "font_size": 12})
                        with ui.HStack(height=0):
                            ui.Label("Global algorithm", width=self._LBL,
                                     tooltip="Path-finding method. octree_lattice = default "
                                             "(bend-aware lattice confined to an octree "
                                             "corridor, ~10x faster at high res, falls back "
                                             "to the full lattice when needed); lattice = "
                                             "exhaustive full-grid search; others for "
                                             "comparison: astar, fmm (Eikonal), rrt, octree, "
                                             "medial.")
                            self._global_combo = ui.ComboBox(0, *_GLOBAL_LABELS)
                        with ui.HStack(height=0):
                            ui.Label("Local optimizer", width=self._LBL,
                                     tooltip="Smoothing/shaping method. fibre = production "
                                             "least-squares default; none = raw; trajopt = "
                                             "SDF trajectory optimization; elastic_rod = "
                                             "physics rod.")
                            self._local_combo = ui.ComboBox(0, *_LOCAL_LABELS)
        self._update_readout()

    def _section_wires(self):
        with ui.CollapsableFrame("Wires", collapsed=False):
            with ui.VStack(spacing=4, height=0):
                self._wire_stack = ui.VStack(spacing=3, height=0)
                self._rebuild_wires()
                with ui.HStack(height=0, spacing=4):
                    ui.Button("+ Add wire", clicked_fn=self._add_wire, height=_CTA_H, width=110)
                    ui.Button("ROUTE ALL", clicked_fn=self._on_route_all, height=_CTA_H,
                              style={"background_color": _PRIMARY, "color": 0xFFFFFFFF,
                                     "font_size": 15})

    def _section_inspector(self):
        # Collapsed until something is selected; _select and _select_bundle expand it.
        self._inspector_frame = ui.CollapsableFrame("Selected wire / bundle", collapsed=True)
        with self._inspector_frame:
            self._inspector = ui.VStack(spacing=4, height=0)
            self._rebuild_inspector()

    # -------------------------------------------------------------- bundles
    _BUNDLE_START_COLOR = (0.9, 0.7, 0.1)   # amber, bundle start
    _BUNDLE_END_COLOR   = (0.9, 0.4, 0.0)   # orange, bundle end

    def _section_bundles(self):
        self._bundles_frame = ui.CollapsableFrame("Bundles", collapsed=True)
        with self._bundles_frame:
            with ui.VStack(spacing=4, height=0):
                self._bundle_stack = ui.VStack(spacing=3, height=0)
                self._rebuild_bundles()
                ui.Button("+ New bundle", clicked_fn=self._new_bundle, height=_BTN_H)

    def _new_bundle(self):
        stage = self._get_stage()
        if stage is None:
            self._progress.text = "open a USD stage first"
            return
        bid = f"b{self._bundle_counter}"
        self._bundle_counter += 1
        inv = self._stage_inv(stage)  # meters -> stage units, whatever the stage uses
        merge_path = f"{scene_ops.MARKERS_SCOPE}/{bid}_merge"
        split_path = f"{scene_ops.MARKERS_SCOPE}/{bid}_split"
        r = self._marker_radius(stage)   # scene-relative marker size
        scene_ops.spawn_waypoint_marker(stage, merge_path, (0.3 * inv, 0.3 * inv, 0.3 * inv),
                                         color=self._BUNDLE_START_COLOR, radius=r)
        scene_ops.spawn_waypoint_marker(stage, split_path, (0.8 * inv, 0.3 * inv, 0.3 * inv),
                                         color=self._BUNDLE_END_COLOR, radius=r)
        default_type_id = self._bundle_type_ids[0] if self._bundle_type_ids else ""
        self._bundles.append({
            "id": bid, "name": bid, "kind": "wire",
            "type_id": default_type_id,
            "type_index": 0,
            "members": [],
            "merge_marker": merge_path,
            "split_marker": split_path,
            "waypoints": [],     # marker paths the shared trunk must pass through
            "wp_counter": 0,
            "trunk_polyline": None,
            "trunk_length_m": 0.0,
            "status": "unrouted",
            "reason": "",
            "weights": {k: 1.0 for k in _WEIGHTS},
        })
        self._schedule(bundles=True)
        self._progress.text = (f"Bundle {bid}: drag the amber marker (Bundle Start) "
                               f"and orange marker (Bundle End) to position the "
                               f"shared trunk, then tick wires in the checklist below.")

    def _rebuild_bundles(self):
        if self._window is None or not hasattr(self, "_bundle_stack"):
            return
        self._bundle_stack.clear()
        with self._bundle_stack:
            if not self._bundles:
                self._hint("No bundles yet. Click '+ New bundle', drag the amber & orange "
                           "trunk markers, then tick the wires that share the trunk.")
                return
            for bi, b in enumerate(self._bundles):
                status_color = _DOT.get(b["status"], _DOT["unrouted"])
                is_sel = bi == self._selected_bundle
                # Bundle header row, styled to match the wire rows.
                with ui.HStack(height=0, spacing=4):
                    self._select_bar(is_sel, lambda i=bi: self._select_bundle(i),
                                     "Select this bundle")
                    # Gray swatch; bundles have no per-bundle color.
                    ui.Rectangle(width=12, height=12,
                                 style={"background_color": 0xFFBBBBBB, "border_radius": 2,
                                        "border_width": 1, "border_color": 0xFF222222})
                    chip_tip = b["status"]
                    if b["status"] == "no_path" and b.get("reason"):
                        chip_tip = f"no path: {b['reason']}"
                    ui.Rectangle(width=12, height=12, tooltip=chip_tip,
                                 style={"background_color": status_color,
                                        "border_radius": 6})
                    nm = ui.StringField(width=90)
                    nm.model.set_value(b["name"])
                    nm.model.add_value_changed_fn(
                        lambda m, i=bi: self._rename_bundle(i, m))
                    # Harness type; drives the cost and label of the trunk BOM row.
                    if self._bundle_type_labels:
                        tidx = b.get("type_index", 0)
                        btc = ui.ComboBox(tidx, *self._bundle_type_labels)
                        btc.model.add_item_changed_fn(
                            lambda m, e, i=bi: self._set_bundle_type(i, m))
                    tl = b.get("trunk_length_m", 0.0)
                    if b["status"] == "routed" and tl:
                        bt = (self._bundle_types[b.get("type_index", 0)]
                              if self._bundle_types else None)
                        cost = tl * float(bt["cost_per_m"]) if bt else 0.0
                        fig = f"{tl:.1f}m ${cost:.0f}"
                    else:
                        fig = ""
                    ui.Label(fig, width=72, style={"color": 0xFFAAAAAA})
                    ui.Button("S", width=22, height=_MINI_H,
                              tooltip="Select the bundle START (amber) marker",
                              clicked_fn=lambda bb=b: self._api.select_prim(
                                  bb["merge_marker"]))
                    ui.Button("E", width=22, height=_MINI_H,
                              tooltip="Select the bundle END (orange) marker",
                              clicked_fn=lambda bb=b: self._api.select_prim(
                                  bb["split_marker"]))
                    ui.Button("Up", width=28, height=_MINI_H,
                              tooltip="Move bundle up (routes earlier)",
                              clicked_fn=lambda i=bi: self._reorder_bundle(i, i - 1))
                    ui.Button("Dn", width=28, height=_MINI_H,
                              tooltip="Move bundle down (routes later)",
                              clicked_fn=lambda i=bi: self._reorder_bundle(i, i + 1))
                    ui.Button("X", width=22, height=_MINI_H, tooltip="Delete this bundle",
                              clicked_fn=lambda i=bi: self._delete_bundle(i),
                              style={"color": 0xFFCC6666})
                # Collapsible wire checklist.
                mem_count = len(b["members"])
                checklist_label = (f"Wires ({mem_count} selected)"
                                   if mem_count else "Wires (none selected)")
                # Honour the user's last collapsed state; default open when there are
                # no members yet.
                default_collapsed = self._checklist_collapsed.get(b["id"], mem_count > 0)
                cf = ui.CollapsableFrame(checklist_label, collapsed=default_collapsed)
                cf.set_collapsed_changed_fn(
                    lambda v, bid=b["id"]: self._checklist_collapsed.__setitem__(bid, v))
                with cf:
                    with ui.VStack(spacing=2, height=0):
                        compatible = [w for w in self._wires
                                      if self._types[w["type_index"]].get("kind") == "wire"]
                        if compatible:
                            for w in compatible:
                                in_bundle = w["name"] in b["members"]
                                color = self._types[w["type_index"]]["color"]
                                with ui.HStack(height=0, spacing=6):
                                    tick_col = _abgr(color) if in_bundle else 0xFF444444
                                    ui.Rectangle(
                                        width=14, height=14,
                                        tooltip="Click to add/remove from bundle",
                                        style={"background_color": tick_col,
                                               "border_radius": 3},
                                    ).set_mouse_pressed_fn(
                                        lambda _x, _y, _b, _m, wn=w["name"], i=bi:
                                            self._toggle_bundle_member(i, wn))
                                    ui.Label(w["name"], width=160,
                                             style={"color": 0xFFDDDDDD
                                                    if in_bundle else 0xFF888888})
                                    if in_bundle:
                                        ui.Label("in bundle",
                                                 style={"color": 0xFF33CC33,
                                                        "font_size": 12})
                        else:
                            ui.Label("  (no wire-type wires in the scene yet)",
                                     style={"color": 0xFF888888})
                if b["status"] == "no_path" and b.get("reason"):
                    ui.Label(f"   -> {b['reason']}", word_wrap=True,
                             style={"color": _BAD, "font_size": 12})
                ui.Rectangle(height=1, style={"background_color": 0xFF333333})

    def _rename_bundle(self, idx, model):
        if idx < len(self._bundles):
            self._bundles[idx]["name"] = model.get_value_as_string()

    def _bundles_with_cost(self):
        """Return bundles list with bundle_type_cost_pm injected from the harness type."""
        result = []
        for b in self._bundles:
            entry = dict(b)
            tid = b.get("type_id", "")
            bt = next((t for t in self._bundle_types if t["id"] == tid), None)
            entry["bundle_type_cost_pm"] = float(bt["cost_per_m"]) if bt else 0.0
            result.append(entry)
        return result

    def _set_bundle_type(self, idx, model):
        if idx < len(self._bundles) and self._bundle_types:
            i = int(model.get_item_value_model().get_value_as_int())
            i = max(0, min(i, len(self._bundle_types) - 1))
            self._bundles[idx]["type_index"] = i
            self._bundles[idx]["type_id"] = self._bundle_type_ids[i]

    def _delete_bundle(self, idx):
        if idx >= len(self._bundles):
            return
        b = self._bundles[idx]
        stage = self._get_stage()
        if stage:
            for path in (b["merge_marker"], b["split_marker"], *b.get("waypoints", [])):
                p = stage.GetPrimAtPath(path)
                if p and p.IsValid():
                    stage.RemovePrim(p.GetPath())
            trunk_prim = stage.GetPrimAtPath(
                f"{scene_ops.ROUTES_SCOPE}/bundle_{b['id']}_trunk")
            if trunk_prim and trunk_prim.IsValid():
                stage.RemovePrim(trunk_prim.GetPath())
        if self._selected_bundle == idx:
            self._selected_bundle = None
        elif self._selected_bundle is not None and self._selected_bundle > idx:
            self._selected_bundle -= 1
        del self._bundles[idx]
        self._schedule(bundles=True, wires=True, inspector=True)

    def _toggle_bundle_member(self, bundle_idx, wire_name):
        """Toggle a single wire in/out of a bundle. A wire can be in multiple bundles."""
        if bundle_idx >= len(self._bundles):
            return
        b = self._bundles[bundle_idx]
        # Pin the checklist open before the rebuild. Without an entry in the state dict
        # the default is "collapsed once mem_count > 0", so the very first tick would
        # fold the checklist shut under the user's cursor.
        self._checklist_collapsed[b["id"]] = False
        if wire_name in b["members"]:
            b["members"].remove(wire_name)
        else:
            b["members"].append(wire_name)
        # Re-aggregate trunk weights from the members as a starting point.
        members = [w for w in self._wires if w["name"] in b["members"]]
        if members:
            for k in _WEIGHTS:
                b["weights"][k] = max(w["weights"].get(k, 1.0) for w in members)
        self._schedule(bundles=True, inspector=True)

    def _reorder_bundle(self, src, dst):
        """Move the bundle at index `src` to index `dst`. Order is routing order."""
        n = len(self._bundles)
        if not (0 <= src < n and 0 <= dst < n) or src == dst:
            return
        self._bundles.insert(dst, self._bundles.pop(src))
        if self._selected_bundle == src:
            self._selected_bundle = dst
        self._schedule(bundles=True)

    def _select_bundle(self, idx):
        """Select a bundle for editing in the inspector. Clears any wire selection."""
        self._selected_bundle = idx
        self._selected = None
        b = self._bundles[idx]
        self._api.select_prims([b["merge_marker"], b["split_marker"]])
        if getattr(self, "_inspector_frame", None) is not None:
            self._inspector_frame.collapsed = False   # reveal the inspector on select
        self._schedule(bundles=True, wires=True, inspector=True)

    def _section_tagging(self):
        with ui.CollapsableFrame("Tagging (thermal / EM / clearance)", collapsed=True):
            with ui.VStack(spacing=4, height=0):
                ui.Label("Select a prim in the stage, set values, then Tag:",
                         style={"color": 0xFF999999})
                with ui.HStack(height=0, spacing=4):
                    ui.Label("Temp °C", width=70)
                    self._temp = ui.FloatField()
                    ui.Label("EM", width=30)
                    self._em = ui.FloatField()
                    ui.Label("Clr mm", width=50,
                             tooltip="Per-object safety clearance (mm): routes keep this "
                                     "distance from THIS object. Untagged objects use the "
                                     "global Safety clearance. 0 = untagged.")
                    self._clr_tag = ui.FloatField()
                    ui.Button("Tag", width=60, height=_BTN_H, clicked_fn=self._on_tag)
                ui.Separator()
                ui.Label("Tagged prims:", style={"color": 0xFF999999})
                self._tag_stack = ui.VStack(spacing=2, height=0)
                self._rebuild_tags()

    _TEMP_COLOR = 0xFF4466EE   # warm red-orange
    _EM_COLOR = 0xFFCCAA33     # teal-gold
    _CLR_COLOR = 0xFF55CC55    # green

    def _rebuild_tags(self):
        if self._window is None:
            return
        self._tag_stack.clear()
        tags = self._api.list_tags()
        with self._tag_stack:
            if not tags:
                self._hint("Nothing tagged yet. Select a prim in the stage, set a "
                           "temperature or EM value above, then click Tag.")
                return
            for t in tags:
                has_t = t["temp_c"] is not None
                has_e = t["em"] is not None
                has_c = t.get("clearance_m") is not None
                name = t["path"].rsplit("/", 1)[-1]
                # Presence dot: warm if thermal, teal if EM, green if clearance only.
                dot = (self._TEMP_COLOR if has_t
                       else self._EM_COLOR if has_e else self._CLR_COLOR)
                with ui.HStack(height=0, spacing=6):
                    ui.Rectangle(width=10, height=10, tooltip=t["path"],
                                 style={"background_color": dot, "border_radius": 5})
                    ui.Label(name, width=130, tooltip=t["path"])
                    ui.Label(f"{t['temp_c']:.0f}°C" if has_t else "", width=52,
                             style={"color": self._TEMP_COLOR})
                    ui.Label(f"EM {t['em']:.2f}" if has_e else "", width=64,
                             style={"color": self._EM_COLOR})
                    ui.Label(f"clr {t['clearance_m'] * 1000:.0f}mm" if has_c else "",
                             width=76, style={"color": self._CLR_COLOR})
                    ui.Spacer()
                    ui.Button("Locate", width=58, height=_MINI_H,
                              clicked_fn=lambda p=t["path"]: self._api.select_prim(p))
                    ui.Button("Remove", width=58, height=_MINI_H, style={"color": 0xFFCC6666},
                              clicked_fn=lambda p=t["path"]: self._remove_tag(p))

    def _remove_tag(self, path):
        self._api.clear_tag(path)
        self._schedule(tags=True)

    # BOM table columns as (title, width_px, numeric). Numeric columns are right-aligned.
    # Header, rows and totals all read these widths so the grid lines up.
    _BOM_COLS = (("Wire", 116, False), ("Type", 124, False), ("Length", 66, True),
                 ("Mass", 60, True), ("Cost", 60, True), ("", 16, False))
    _RIGHT = None  # resolved lazily to ui.Alignment.RIGHT_CENTER; ui may be a stub at import

    def _num_align(self):
        if self._RIGHT is None:
            try:
                type(self)._RIGHT = ui.Alignment.RIGHT_CENTER
            except Exception:
                type(self)._RIGHT = 0
        return self._RIGHT

    def _section_output(self):
        self._output_frame = ui.CollapsableFrame("Output / BOM", collapsed=True)
        with self._output_frame:
            with ui.VStack(spacing=4, height=0):
                with ui.HStack(height=0, spacing=4):
                    ui.Label("Export path", width=90)
                    self._bom_path = ui.StringField()
                    self._bom_path.model.set_value("/tmp/piperouter_bom")
                    ui.Button("Export", width=70, height=_BTN_H, clicked_fn=self._on_export)
                # _rebuild_bom clears and refills this container.
                self._bom_table = ui.VStack(spacing=2, height=0)
        self._rebuild_bom()

    def _rebuild_bom(self):
        if self._window is None or getattr(self, "_bom_table", None) is None:
            return
        self._bom_table.clear()
        s = bom_lib.summarize(self._last_bom, self._bom_type_labels())
        right = self._num_align()
        with self._bom_table:
            if not s["rows"]:
                self._hint("No routes yet. Add wires and click ROUTE ALL to populate the "
                           "bill of materials.")
                return
            # Auto-expand the section the first time results appear.
            if not getattr(self, "_bom_auto_expanded", False):
                self._bom_auto_expanded = True
                if getattr(self, "_output_frame", None) is not None:
                    self._output_frame.collapsed = False
            # Header row.
            with ui.HStack(height=0, spacing=4):
                for title, wpx, numeric in self._BOM_COLS:
                    kw = {"alignment": right} if numeric else {}
                    ui.Label(title, width=wpx,
                             style={"color": 0xFF999999, "font_size": 13}, **kw)
            ui.Rectangle(height=1, style={"background_color": 0xFF444444})
            for r in s["rows"]:
                routed = r["status"] == "routed"
                reason = r.get("reason", "")
                with ui.HStack(height=0, spacing=4):
                    ui.Label(r["wire_id"], width=116)
                    ui.Label(r["type"], width=124, style={"color": 0xFFAAAAAA})
                    ui.Label(f"{r['length_m']:.2f} m" if routed else "-", width=66,
                             alignment=right)
                    ui.Label(f"{r['mass']:.2f} kg" if routed else "-", width=60,
                             alignment=right)
                    ui.Label(f"${r['cost']:.2f}" if routed else "-", width=60,
                             alignment=right)
                    ui.Rectangle(width=12, height=12,
                                 tooltip=reason if (not routed and reason) else
                                         ("routed" if routed else "no path"),
                                 style={"background_color": _DOT.get(r["status"], 0xFF888888),
                                        "border_radius": 6})
                # Spell out why a wire failed, right under its row.
                if not routed and reason:
                    ui.Label(f"   -> {reason}", word_wrap=True,
                             style={"color": _BAD, "font_size": 12})
            ui.Rectangle(height=1, style={"background_color": 0xFF444444})
            # Totals row, on the same column grid as the rows above.
            with ui.HStack(height=0, spacing=4):
                ui.Label(f"TOTAL ({s['n_routed']} routed"
                         + (f", {s['n_no_path']} no-path" if s["n_no_path"] else "") + ")",
                         width=116 + 124 + 4, style={"font_size": 14})
                ui.Label(f"{s['total_length']:.2f} m", width=66, alignment=right,
                         style={"font_size": 14})
                ui.Label(f"{s['total_mass']:.2f} kg", width=60, alignment=right,
                         style={"font_size": 14})
                ui.Label(f"${s['total_cost']:.2f}", width=60, alignment=right,
                         style={"font_size": 14, "color": _OK})

    def _bom_type_labels(self):
        """Return {wire_id: wire-type label}, used to fill the BOM Type column."""
        return {w["name"]: self._type_labels[w["type_index"]] for w in self._wires}

    # ----------------------------------------------------------- connection
    def _toggle_hud(self):
        self._hud_visible = not self._hud_visible
        self._hud.set_visible(self._hud_visible)
        if self._hud_visible:
            self._refresh_hud()

    def _open_help(self):
        if getattr(self, "_help", None) is not None:
            self._help.show()

    def _refresh_hud(self):
        """Push the current BOM totals and selected-wire weights to the viewport HUD."""
        if not self._hud_visible:
            return
        s = bom_lib.summarize(self._last_bom, self._bom_type_labels())
        stats = {
            "total_cost":   s["total_cost"],
            "total_mass":   s["total_mass"],
            "total_length": s["total_length"],
            "n_routed":     s["n_routed"],
            "n_total":      len(self._wires),
            "n_no_path":    s["n_no_path"],
        }
        wire = (self._wires[self._selected]
                if self._selected is not None and self._selected < len(self._wires)
                else None)
        self._hud.update(stats, wire)

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
        return self._default_url  # the URL is fixed; Reconnect just re-probes it

    def _connectivity(self):
        """Return 6, 18 or 26 from the Connectivity combo. Defaults to 26."""
        combo = getattr(self, "_conn_combo", None)
        if combo is None:
            return 26
        idx = int(combo.model.get_item_value_model().get_value_as_int())
        return (6, 18, 26)[idx] if idx in (0, 1, 2) else 26

    def _global_algo(self):
        combo = getattr(self, "_global_combo", None)
        if combo is None:
            return _GLOBAL_ALGOS[0]
        i = int(combo.model.get_item_value_model().get_value_as_int())
        return _GLOBAL_ALGOS[i] if 0 <= i < len(_GLOBAL_ALGOS) else _GLOBAL_ALGOS[0]

    def _local_algo(self):
        combo = getattr(self, "_local_combo", None)
        if combo is None:
            return "fibre"
        i = int(combo.model.get_item_value_model().get_value_as_int())
        return _LOCAL_ALGOS[i] if 0 <= i < len(_LOCAL_ALGOS) else "fibre"

    def _update_readout(self):
        res = max(1, self._res.model.get_value_as_int())
        # Cells are uniform cubes of (longest scene axis) / resolution; the other axes
        # get however many of those cubes fit. So `res` counts voxels along the longest
        # axis and spacing is identical on all three. Prefer the voxel size the last
        # route actually used, which matches the occupancy overlay, and fall back to an
        # estimate from the scene bounds before the first route.
        size_txt = ""
        real = None
        try:
            real = self._api.last_cell_size_m()      # metres, or None
        except Exception:
            real = None
        if real:
            size_txt = f"  ~{self._fmt_len_m(real)} per voxel (routed grid)"
        else:
            stage = self._get_stage() if self._get_stage else None
            longest = self._scene_longest_axis_m(stage) if stage is not None else None
            if longest:
                size_txt = f"  ~{self._fmt_len_m(longest / res)} per voxel (estimate)"
        self._readout.text = (f"~{res} voxels along the longest axis (uniform cubic "
                              f"cells){size_txt}. Higher = finer detail, slower routing.")

    # ------------------------------------------------- deferred UI refresh
    # omni.ui forbids clearing or rebuilding a container from inside an event or draw
    # callback; it raises "Container::clear was called during an event or draw". Event
    # handlers therefore only request a refresh, and the rebuild happens next frame.
    def _schedule(self, wires=False, inspector=False, tags=False, bom=False,
                  bundles=False, resync=False):
        self._need_wires = self._need_wires or wires
        self._need_inspector = self._need_inspector or inspector
        self._need_tags = self._need_tags or tags
        self._need_bom = self._need_bom or bom
        self._need_bundles = self._need_bundles or bundles
        self._need_resync = self._need_resync or resync
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
        if self._need_resync:
            self._need_resync = False
            if self._prune_missing():   # markers vanished, so drop those wires/bundles
                self._need_wires = self._need_bundles = True
                self._need_inspector = self._need_bom = True
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
        if self._need_bundles:
            self._need_bundles = False
            self._rebuild_bundles()
        # Viewport order labels and HUD refresh on every coalesced frame.
        self._refresh_vp_labels()
        self._refresh_hud()

    def _prune_missing(self):
        """Drop any wire or bundle whose markers no longer exist in the stage.

        Covers both a marker deleted in the viewport and a different scene being opened,
        where every old marker is gone and the panel empties to match. Returns True if
        anything was removed. Only edits the panel model, never stage prims.
        """
        stage = self._get_stage()
        if stage is None:
            return False

        def _has(path):
            p = stage.GetPrimAtPath(path)
            return bool(p and p.IsValid())

        m = scene_ops.MARKERS_SCOPE
        kept_w = [w for w in self._wires
                  if _has(f"{m}/{w['key']}_start") and _has(f"{m}/{w['key']}_end")]
        kept_b = [b for b in self._bundles
                  if _has(b.get("merge_marker", "")) and _has(b.get("split_marker", ""))]
        changed = (len(kept_w) != len(self._wires)) or (len(kept_b) != len(self._bundles))
        if not changed:
            return False

        names = {w["name"] for w in kept_w}
        for b in kept_b:                       # forget members that were pruned
            b["members"] = [mm for mm in b.get("members", []) if mm in names]
        self._wires, self._bundles = kept_w, kept_b
        self._selected = None
        self._selected_bundle = None
        self._active_debug = None
        self._last_bom = [r for r in getattr(self, "_last_bom", []) if r.get("wire_id") in names]
        return True

    def _on_reset(self):
        """Clear every wire, bundle and selection, and remove the markers, route tubes and
        debug geometry PipeRouter authored. The user's obstacle meshes are left untouched."""
        stage = self._get_stage()
        self._wires = []
        self._bundles = []
        self._selected = None
        self._selected_bundle = None
        self._active_debug = None
        self._key_counter = 0
        self._bundle_counter = 0
        self._last_bom = []
        if stage is not None:
            for scope in (scene_ops.MARKERS_SCOPE, scene_ops.ROUTES_SCOPE,
                          scene_ops.DEBUG_SCOPE):
                p = stage.GetPrimAtPath(scope)
                if p and p.IsValid():
                    stage.RemovePrim(p.GetPath())
        self._schedule(wires=True, bundles=True, inspector=True, bom=True, tags=True)
        self._refresh_views()
        self._refresh_overlay()
        self._refresh_hud()
        self._refresh_vp_labels()
        self._progress.text = "Reset - cleared wires, bundles, markers and routes (obstacles kept)."

    # ----------------------------------------------------------------- wires
    def _new_wire(self, key, name, type_index=0):
        return {"key": key, "name": name, "type_index": type_index,
                "weights": {k: 1.0 for k in _WEIGHTS}, "waypoints": [], "wp_counter": 0,
                # wp_slots[i] = how many of this wire's bundles precede waypoint i in the
                # route order; 0 means before the first bundle. Parallel to waypoints.
                "wp_slots": [],
                "locked": False, "polyline": None, "status": "unrouted",
                "length_m": 0.0, "cost": 0.0, "combo": None, "name_model": None,
                "_swatch": None, "start_head_idx": 0, "end_head_idx": 0, "reason": "",
                "note": "", "cells": [], "raw_polyline": None}

    @staticmethod
    def _stage_inv(stage):
        """Return 1 / metersPerUnit for the stage.

        Lets defaults be authored at a fixed physical size in meters whatever the stage
        unit is: cm for Omniverse, mm for CAD imports, m for the sample scenes. A value
        in meters times inv lands at the right scale; the routing pipeline converts back
        to meters using metersPerUnit.
        """
        try:
            from pxr import UsdGeom
            mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
            return 1.0 / mpu if mpu > 1e-9 else 1.0
        except Exception:
            return 1.0

    def _scene_diag(self, stage):
        """Return the scene bounding-box diagonal in stage units.

        Uses /World's world bound: one cheap compute, no force-load and no per-mesh loop,
        so it is safe to call on every refresh. Falls back to roughly 1 m worth of stage
        units when the scene is empty.
        """
        try:
            from pxr import Usd, UsdGeom
            bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                   [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            # Prefer /World, then the asset's defaultPrim since imported CAD often has
            # no /World, then the whole stage.
            targets = []
            w = stage.GetPrimAtPath("/World")
            if w and w.IsValid():
                targets.append(w)
            dp = stage.GetDefaultPrim()
            if dp and dp.IsValid():
                targets.append(dp)
            targets.append(stage.GetPseudoRoot())
            for prim in targets:
                rng = bb.ComputeWorldBound(prim).ComputeAlignedRange()
                if not rng.IsEmpty():
                    d = rng.GetSize()
                    diag = float((d[0] ** 2 + d[1] ** 2 + d[2] ** 2) ** 0.5)
                    if diag > 1e-9:
                        return diag
        except Exception:
            pass
        return float(self._stage_inv(stage))   # ~1 m

    def _scene_longest_axis_m(self, stage):
        """Return the longest scene bbox axis in metres, or None when there is no scene.

        Drives the voxel size, since cells are cubes of longest_axis / resolution.
        """
        try:
            from pxr import Usd, UsdGeom
            bb = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                   [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
            mpu = float(UsdGeom.GetStageMetersPerUnit(stage)) or 1.0
            targets = []
            w = stage.GetPrimAtPath("/World")
            if w and w.IsValid():
                targets.append(w)
            dp = stage.GetDefaultPrim()
            if dp and dp.IsValid():
                targets.append(dp)
            targets.append(stage.GetPseudoRoot())
            for prim in targets:
                rng = bb.ComputeWorldBound(prim).ComputeAlignedRange()
                if not rng.IsEmpty():
                    longest = float(max(rng.GetSize()))
                    if longest > 1e-9:
                        return longest * mpu
        except Exception:
            pass
        return None

    @staticmethod
    def _fmt_len_m(m):
        """Format a length in metres for display: mm under 1 cm, cm under 1 m, else m."""
        if m < 0.01:
            return f"{m * 1000.0:.1f} mm"
        if m < 1.0:
            return f"{m * 100.0:.2f} cm"
        return f"{m:.3f} m"

    def _marker_radius(self, stage):
        """Return the draggable-marker radius in stage units.

        Roughly 0.4% of the scene diagonal, clamped to 5-50 mm, so markers scale with the
        asset and neither dwarf a 1 m harness nor vanish in a large engine bay.
        """
        inv = self._stage_inv(stage)
        return float(min(max(self._scene_diag(stage) * 0.004, 0.005 * inv), 0.05 * inv))

    def _add_wire(self):
        stage = self._get_stage()
        if stage is None:
            self._progress.text = "open a USD stage first"
            return
        inv = self._stage_inv(stage)  # meters -> stage units, so the wire is ~0.5 m anywhere
        r = self._marker_radius(stage)
        key = f"wire_{self._key_counter}"
        self._key_counter += 1
        scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/{key}_start",
                               (0.0, 0.0, 0.0), color=(0.1, 0.9, 0.1), radius=r)
        scene_ops.spawn_marker(stage, f"{scene_ops.MARKERS_SCOPE}/{key}_end",
                               (0.5 * inv, 0.0, 0.0), color=(0.9, 0.1, 0.1), radius=r)
        self._wires.append(self._new_wire(key, key))
        self._schedule(wires=True)

    def _create_sample(self):
        self._apply_scene(self._api.create_sample_scene(), "sample")

    def _create_complex(self):
        self._apply_scene(self._api.create_complex_scene(), "complex")

    def _apply_scene(self, result, label):
        """Load the wire descriptors from a freshly built scene into the panel."""
        descriptors, err = result
        if err:
            self._progress.text = f"{label} scene: {err}"
            return
        self._wires = []
        self._selected = None
        self._key_counter = 0
        for d in descriptors:
            ti = self._type_ids.index(d["type_id"]) if d["type_id"] in self._type_ids else 0
            # The scene builder authored the markers under the descriptor name, so the
            # wire key must match it.
            self._wires.append(self._new_wire(d["name"], d["name"], ti))
            self._key_counter += 1
        self._schedule(wires=True, inspector=True, tags=True)
        self._progress.text = f"{label} scene ready - {len(descriptors)} wires. Click ROUTE ALL."

    # ------------------------------------------------------------- save / load
    def _session_state(self):
        """Return the panel session as a JSON-safe dict.

        Geometry, markers and routes live in the stage; this captures only the panel
        state. type_id is stored alongside type_index so wire types survive a change in
        the wire library's order.
        """
        settings = {
            "resolution": self._res.model.get_value_as_int(),
            "connectivity_idx": int(
                self._conn_combo.model.get_item_value_model().get_value_as_int()),
            "clearance": float(self._clearance.model.get_value_as_float()),
            "hud_visible": bool(self._hud_visible),
            "global_algo": self._global_algo(),
            "local_algo": self._local_algo(),
        }
        counters = {"key_counter": self._key_counter,
                    "bundle_counter": self._bundle_counter}
        wires = []
        for w in self._wires:
            wd = dict(w)
            ti = w.get("type_index", 0)
            wd["type_id"] = self._type_ids[ti] if 0 <= ti < len(self._type_ids) else ""
            wires.append(wd)
        return session_io.serialize(wires, self._bundles, settings, counters)

    def _apply_session(self, data):
        """Rebuild the panel from a saved session dict.

        Markers and geometry are already restored by opening the stage.
        """
        wires, bundles, settings, counters = session_io.deserialize(data)

        self._wires = []
        for i, wd in enumerate(wires):
            tid = wd.get("type_id")
            ti = self._type_ids.index(tid) if tid in self._type_ids \
                else int(wd.get("type_index", 0))
            ti = max(0, min(ti, len(self._types) - 1))
            w = self._new_wire(wd.get("key", f"wire_{i}"), wd.get("name", f"wire_{i}"), ti)
            for k in ("weights", "waypoints", "wp_slots", "wp_counter", "start_head_idx",
                      "end_head_idx", "locked", "status", "reason", "note", "length_m",
                      "cost", "polyline", "cells", "raw_polyline"):
                if k in wd:
                    w[k] = wd[k]
            self._wires.append(w)

        self._bundles = []
        for bd in bundles:
            b = dict(bd)
            b.setdefault("waypoints", [])
            b.setdefault("wp_counter", 0)
            b.setdefault("weights", {k: 1.0 for k in _WEIGHTS})
            b.setdefault("trunk_polyline", None)
            b.setdefault("trunk_length_m", 0.0)
            b.setdefault("status", "unrouted")
            b.setdefault("reason", "")
            self._bundles.append(b)

        self._selected = None
        self._selected_bundle = None
        self._active_debug = None
        self._key_counter = int(counters.get("key_counter", len(self._wires)))
        self._bundle_counter = int(counters.get("bundle_counter", len(self._bundles)))

        self._res.model.set_value(int(settings.get("resolution", 64)))
        self._conn_combo.model.get_item_value_model().set_value(
            int(settings.get("connectivity_idx", 2)))   # default Smooth (26)
        self._clearance.model.set_value(float(settings.get("clearance", 0.0)))
        # In v1 sessions "lattice" was the default value rather than a deliberate choice,
        # so migrate those onto octree_lattice. From v2 on, "lattice" was picked by the
        # user and is respected.
        if int(data.get("version", 1)) < 2 and settings.get("global_algo") == "lattice":
            settings["global_algo"] = "octree_lattice"
        if settings.get("global_algo") in _GLOBAL_ALGOS:
            self._global_combo.model.get_item_value_model().set_value(
                _GLOBAL_ALGOS.index(settings["global_algo"]))
        if settings.get("local_algo") in _LOCAL_ALGOS:
            self._local_combo.model.get_item_value_model().set_value(
                _LOCAL_ALGOS.index(settings["local_algo"]))

        # Rebuild the BOM aggregate from the restored routed wires and bundle trunks.
        self._last_bom = []
        for w in self._wires:
            if w.get("status") == "routed":
                ml = self._types[w["type_index"]]
                self._last_bom.append({
                    "wire_id": w["name"], "status": "routed",
                    "length_m": float(w.get("length_m", 0.0)),
                    "cost": float(w.get("cost", 0.0)),
                    "mass": float(w.get("length_m", 0.0)) * float(ml.get("mass_per_m_kg", 0.0)),
                    "reason": "",
                })

        self._schedule(wires=True, bundles=True, inspector=True, bom=True, tags=True)
        # The model has just been set authoritatively from the loaded file, so cancel any
        # resync queued by the stage OPENED event. Otherwise it prunes the fresh wires
        # while the stage is still composing.
        self._need_resync = False
        self._refresh_views()
        self._refresh_overlay()
        self._refresh_hud()
        self._refresh_vp_labels()

    def _pick_file(self, title, apply_label, exts, handler):
        """Open the native Omniverse file dialog and call `handler(full_path)`."""
        try:
            from omni.kit.window.filepicker import FilePickerDialog

            def _apply(filename, dirname):
                name = filename or ""
                path = (dirname.rstrip("/") + "/" + name) if dirname else name
                try:
                    dialog.hide()
                except Exception:
                    pass
                handler(path)

            try:
                dialog = FilePickerDialog(
                    title, apply_button_label=apply_label,
                    click_apply_handler=_apply, file_extension_options=exts,
                )
            except TypeError:
                # Kit builds vary on whether this kwarg exists; retry without it.
                dialog = FilePickerDialog(
                    title, apply_button_label=apply_label, click_apply_handler=_apply,
                )
            dialog.show()
        except Exception as exc:
            self._progress.text = (f"file dialog unavailable ({exc}); "
                                   f"enable omni.kit.window.filepicker")

    def _on_save(self):
        self._pick_file("Save PipeRouter session", "Save",
                        [(".usd", "USD scene")], self._do_save)

    def _on_load(self):
        self._pick_file("Load PipeRouter session", "Load",
                        [(".usd", "USD scene"), (".usda", "USD ascii"),
                         (".usdc", "USD crate"), (".usdz", "USD archive")],
                        self._do_load)

    def _on_export_usdz(self):
        self._pick_file("Export PipeRouter session (.usdz)", "Export",
                        [(".usdz", "USD archive")], self._do_export_usdz)

    def _do_save(self, path):
        stage = self._get_stage()
        if stage is None:
            self._progress.text = "no USD stage open"
            return
        if not path:
            return
        if not path.lower().endswith((".usd", ".usda", ".usdc")):
            path += ".usd"
        try:
            scene_ops.write_session(stage, self._session_state())
            ok = stage.Export(path)
            if ok:
                self._strip_viewport_cameras(path)
            self._progress.text = (f"Saved session -> {path}" if ok
                                   else f"save failed: {path}")
        except Exception as exc:
            self._progress.text = f"save failed: {exc}"

    @staticmethod
    def _strip_viewport_cameras(usd_path):
        """Remove Kit's OmniverseKit_* viewport cameras from an exported .usd file in place.

        Edits the file as a flat Sdf layer rather than opening it as a stage, which leaves
        instancing in the source scene alone and never touches the live stage's cameras.
        """
        try:
            from pxr import Sdf
            layer = Sdf.Layer.FindOrOpen(usd_path)
            if layer is None:
                return
            removed = False
            for name in _VIEWPORT_CAMERAS:
                if layer.GetPrimAtPath(f"/{name}"):
                    del layer.rootPrims[name]
                    removed = True
            if removed:
                layer.Save()
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[piperouter] could not strip viewport cameras "
                          f"from {usd_path}: {exc}")

    def _do_load(self, path):
        if not path:
            return
        try:
            from pxr import Usd as _Usd
            probe = _Usd.Stage.Open(path)
            data = scene_ops.read_session(probe) if probe else None
        except Exception as exc:
            self._progress.text = f"load failed: {exc}"
            return
        if data is None:
            self._progress.text = "no PipeRouter session embedded in that file"
            return
        try:
            omni.usd.get_context().open_stage(path)
        except Exception as exc:
            self._progress.text = f"could not open {path}: {exc}"
            return
        self._apply_session(data)
        self._progress.text = (f"Loaded session <- {path}  "
                               f"({len(self._wires)} wires, {len(self._bundles)} bundles)")

    def _do_export_usdz(self, path):
        stage = self._get_stage()
        if stage is None:
            self._progress.text = "no USD stage open"
            return
        if not path:
            return
        if not path.lower().endswith(".usdz"):
            path += ".usdz"
        try:
            import os
            import tempfile
            from pxr import UsdUtils
            scene_ops.write_session(stage, self._session_state())
            # Package into a local temp file. CreateNewUsdzPackage's zip writer only
            # handles the local filesystem and fails in TfSafeOutputFile::Replace when
            # the destination is an omniverse:// URL or a non-writable folder.
            tmpdir = tempfile.mkdtemp(prefix="piperouter_")
            tmp_src = os.path.join(tmpdir, "session_src.usd")
            tmp_usdz = os.path.join(tmpdir, "session.usdz")
            stage.Export(tmp_src)
            self._strip_viewport_cameras(tmp_src)
            if not UsdUtils.CreateNewUsdzPackage(tmp_src, tmp_usdz):
                self._progress.text = "usdz packaging failed"
                return
            # Then copy the archive to the chosen destination, local or Nucleus.
            if self._copy_file(tmp_usdz, path):
                self._progress.text = f"Exported -> {path}"
            else:
                self._progress.text = f"usdz export: could not write {path}"
        except Exception as exc:
            self._progress.text = f"usdz export failed: {exc}"

    @staticmethod
    def _copy_file(src_local, dst):
        """Copy a local file to `dst`, which may be a local path or a Nucleus URL.

        Falls back to the resolver-aware omni.client when `dst` is not a plain local path.
        """
        import os
        import shutil
        is_url = "://" in dst
        if not is_url:
            try:
                os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                shutil.copyfile(src_local, dst)
                return True
            except Exception:
                pass
        try:
            import omni.client
            with open(src_local, "rb") as f:
                data = f.read()
            return omni.client.write_file(dst, data) == omni.client.Result.OK
        except Exception:
            return False

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
        if self._active_debug and self._active_debug[0] == w["name"]:
            self._active_debug = None
            if stage is not None:
                scene_ops.clear_debug(stage)
                scene_ops.set_all_routes_visible(stage)
        del self._wires[idx]
        if self._selected == idx:
            self._selected = None
        elif self._selected is not None and self._selected > idx:
            self._selected -= 1
        self._schedule(wires=True, inspector=True)

    def _rebuild_wires(self):
        if self._window is None:
            return
        self._building = True  # ignore callbacks fired while the widgets are constructed
        try:
            self._wire_stack.clear()
            with self._wire_stack:
                if not self._wires:
                    self._hint("No wires yet. Click 'Create sample scene' or '+ Add wire', "
                               "then drag the green & red markers into place.")
                for idx, w in enumerate(self._wires):
                    color = self._types[w["type_index"]]["color"]
                    status = "locked" if w["locked"] else w["status"]
                    is_sel = idx == self._selected
                    with ui.HStack(height=0, spacing=4):
                        self._select_bar(is_sel, lambda i=idx: self._select(i),
                                         "Select this wire")
                        w["_swatch"] = ui.Rectangle(
                            width=12, height=12,
                            style={"background_color": _abgr(color), "border_radius": 2,
                                   "border_width": 1, "border_color": 0xFF222222})
                        # Status chip: green routed, red no-path, blue locked, grey
                        # otherwise. Its tooltip carries the failure reason, or amber plus
                        # a note for a routed wire with a warning such as a relocated
                        # buried endpoint.
                        chip_tip = status
                        chip_color = _DOT.get(status, _DOT["unrouted"])
                        if w["status"] == "no_path" and w.get("reason"):
                            chip_tip = f"no path: {w['reason']}"
                        elif status == "routed" and w.get("note"):
                            chip_tip = w["note"]
                            chip_color = _WARN
                        ui.Rectangle(width=12, height=12, tooltip=chip_tip,
                                     style={"background_color": chip_color,
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
                        ui.Label(figs, width=60, style={"color": 0xFFAAAAAA})
                        ui.Button("S", width=22, height=_MINI_H,
                                  tooltip="Select the START marker in the viewport",
                                  clicked_fn=lambda kk=w["key"]: self._api.select_prim(
                                      f"{scene_ops.MARKERS_SCOPE}/{kk}_start"))
                        ui.Button("E", width=22, height=_MINI_H,
                                  tooltip="Select the END marker in the viewport",
                                  clicked_fn=lambda kk=w["key"]: self._api.select_prim(
                                      f"{scene_ops.MARKERS_SCOPE}/{kk}_end"))
                        ui.Button("X", width=22, height=_MINI_H, tooltip="Delete this wire",
                                  clicked_fn=lambda i=idx: self._delete_wire(i),
                                  style={"color": 0xFFCC6666})
        finally:
            self._building = False

    def _rename(self, idx, model):
        if not self._building and idx < len(self._wires):
            self._wires[idx]["name"] = model.get_value_as_string()

    def _set_type(self, idx, model):
        # Never rebuild the wire list from here. A rebuild constructs new ComboBoxes whose
        # item_changed callbacks re-enter this handler, clearing the list mid-build and
        # making the other wires disappear. Update the row in place instead.
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
        self._selected_bundle = None   # wire selection clears bundle selection
        w = self._wires[idx]
        # The schedule below must pass bundles=True so the bundle row highlight clears.
        paths = [f"{scene_ops.MARKERS_SCOPE}/{w['key']}_start",
                 f"{scene_ops.MARKERS_SCOPE}/{w['key']}_end",
                 *w["waypoints"],
                 f"{scene_ops.ROUTES_SCOPE}/{w['name']}"]
        self._api.select_prims(paths)
        self._refresh_hud()
        if getattr(self, "_inspector_frame", None) is not None:
            self._inspector_frame.collapsed = False   # reveal the inspector on select
        self._schedule(wires=True, inspector=True, bundles=True)

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
            # A selected bundle takes over the inspector.
            if (self._selected is None and self._selected_bundle is not None
                    and self._selected_bundle < len(self._bundles)):
                self._rebuild_inspector_bundle(self._selected_bundle)
                return
            if self._selected is None or self._selected >= len(self._wires):
                self._hint("Select a wire or bundle above to tune its constraints, "
                           "headings, waypoints and debug view.")
                return
            w = self._wires[self._selected]
            status = "locked" if w["locked"] else w["status"]
            with ui.HStack(height=0, spacing=6):
                ui.Rectangle(width=14, height=14,
                             style={"background_color": _abgr(self._types[w["type_index"]]["color"]),
                                    "border_radius": 2, "border_width": 1,
                                    "border_color": 0xFF222222})
                ui.Rectangle(width=14, height=14, tooltip=status,
                             style={"background_color": _DOT.get(status, _DOT["unrouted"]),
                                    "border_radius": 7})
                ui.Label(f"{w['name']}", style={"font_size": 16})
                ui.Label(self._type_labels[w['type_index']], style={"color": 0xFFAAAAAA})
                ui.Spacer()
                if w["locked"]:
                    ui.Label("LOCKED", style={"color": 0xFFCC7A33, "font_size": 11})
            if w["status"] == "no_path" and w.get("reason"):
                ui.Label(f"No path: {w['reason']}", word_wrap=True,
                         style={"color": _BAD})
            elif w.get("note"):   # routed, but with a warning such as a moved endpoint
                ui.Label(w["note"], word_wrap=True, style={"color": _WARN})
            # Constraints group.
            ui.Spacer(height=2)
            with ui.HStack(height=0):
                ui.Label("CONSTRAINTS  (0 = ignore, 10 = strong)",
                         style={"color": _ACCENT, "font_size": 11})
                ui.Spacer()
                ui.Button("Reset", width=60, height=_MINI_H,
                          tooltip="Reset all constraints to their default weight",
                          clicked_fn=self._reset_weights)
            self._sliders = {}
            self._slider_vals = {}
            for k in _WEIGHTS:
                label, help_text = _WEIGHT_HELP[k]
                with ui.HStack(height=0, spacing=6):
                    ui.Label(label, width=96, tooltip=help_text)
                    s = ui.FloatSlider(min=0.0, max=10.0, tooltip=help_text)
                    s.model.set_value(w["weights"][k])
                    # Persist the value into the wire as the slider moves, not only on
                    # Re-route, so switching to another wire and back keeps it.
                    s.model.add_value_changed_fn(lambda m, kk=k: self._set_weight(kk, m))
                    self._sliders[k] = s
                    val = ui.Label(f"{w['weights'][k]:.1f}", width=30,
                                   alignment=ui.Alignment.RIGHT_CENTER,
                                   style={"color": 0xFFCCCCCC, "font_size": 12})
                    self._slider_vals[k] = val
            # Pinned headings. None leaves the direction free; Custom is set by rotating
            # the marker's arrow in the viewport or by typing angles below.
            ui.Spacer(height=2)
            ui.Label("HEADINGS  (optional; None = free direction)",
                     style={"color": _ACCENT, "font_size": 11})
            self._heading_row(w, "start",
                              "Force the wire to LEAVE the start in this direction.")
            self._heading_row(w, "end",
                              "Force the wire to ARRIVE at the end in this direction.")
            # Waypoints and route order.
            ui.Spacer(height=2)
            ui.Label("WAYPOINTS & ROUTE ORDER", style={"color": _ACCENT, "font_size": 11})
            ui.Button("+ Add waypoint (route must pass through)", height=_BTN_H,
                      clicked_fn=self._add_waypoint)
            itinerary = self._wire_itinerary(w)
            if itinerary:
                ui.Label("Drag a waypoint by its grip to move it before, between, or after "
                         "the bundle steps:",
                         word_wrap=True, style={"color": 0xFF888888})

                def _dot(color):
                    ui.Rectangle(width=11, height=11,
                                 style={"background_color": color, "border_radius": 6,
                                        "border_width": 1, "border_color": 0x44000000})

                def _anchor(text, color):  # the Start and End rows, which cannot be dragged
                    with ui.HStack(height=24, spacing=8):
                        ui.Spacer(width=6)
                        _dot(color)
                        ui.Label(text, style={"color": 0xFFCCCCCC, "font_size": 14})

                _anchor("Start", 0xFF33CC33)
                for d, tok in enumerate(itinerary):
                    row = ui.HStack(height=26, spacing=8)
                    if tok["kind"] == "bundle":
                        with row:
                            ui.Spacer(width=6)
                            _dot(_abgr(self._BUNDLE_START_COLOR))
                            ui.Label(f"Bundle {tok['name']}  (shared trunk)",
                                     tooltip="Drop a waypoint here to route just before "
                                             "or after this trunk.",
                                     style={"color": _abgr(self._BUNDLE_START_COLOR),
                                            "font_size": 14})
                    else:
                        wpi, wp_path = tok["wpi"], tok["path"]
                        with row:
                            grip = ui.Label(":::", width=16,
                                            style={"color": 0xFF888888, "font_size": 15},
                                            tooltip="Drag to move in the route order")
                            grip.set_drag_fn(lambda dd=d: str(dd))
                            _dot(_abgr((0.1, 0.5, 0.9)))  # matches the blue waypoint marker
                            ui.Label("Waypoint", style={"color": 0xFFDDDDDD,
                                                        "font_size": 14})
                            ui.Spacer()
                            ui.Button("Locate", width=62, height=_MINI_H,
                                      clicked_fn=lambda p=wp_path: self._api.select_prim(p))
                            ui.Button("Delete", width=62, height=_MINI_H,
                                      style={"color": 0xFFCC6666},
                                      clicked_fn=lambda j=wpi: self._delete_waypoint(j))
                    # Every row is a drop target: dropping moves the dragged waypoint here.
                    row.set_accept_drop_fn(lambda *_: True)
                    row.set_drop_fn(lambda e, dst=d: self._move_wire_waypoint(
                        int(e.mime_data), dst))

                end_row = ui.HStack(height=24, spacing=8)
                with end_row:
                    ui.Spacer(width=6)
                    _dot(0xFF3333CC)
                    ui.Label("End", style={"color": 0xFFCCCCCC, "font_size": 14})
                end_row.set_accept_drop_fn(lambda *_: True)
                end_row.set_drop_fn(lambda e, dst=len(itinerary): self._move_wire_waypoint(
                    int(e.mime_data), dst))
            ui.Spacer(height=2)
            with ui.HStack(height=0, spacing=4):
                ui.Button("Re-route this wire", clicked_fn=self._on_refine, height=_CTA_H,
                          style={"background_color": _PRIMARY, "color": 0xFFFFFFFF})
                ui.Button("Unlock" if w["locked"] else "Lock", width=80, height=_CTA_H,
                          clicked_fn=self._toggle_lock)
            # Per-wire debug visualizations.
            ui.Rectangle(height=1, style={"background_color": _DIVIDER})
            ui.Label("DEBUG VIEW  (authors into /World/PipeRouter/debug)",
                     style={"color": _ACCENT, "font_size": 11})
            _WIRE_DEBUG_MODES = (
                "None",
                "Wire cells (claimed voxels)",
                "Grid vs smooth path",
                "Soft-cost terrain",
                "Bend-radius heatmap",
                "Octree lattice (adaptive cells + band)",
            )
            _WIRE_DEBUG_IDS = (
                "none", "cells", "raw_path", "cost_terrain", "bend_radius", "octree",
            )
            # Reflect the live debug view for this wire; it persists across re-routes.
            cur = (self._active_debug[1]
                   if self._active_debug and self._active_debug[0] == w["name"] else "none")
            cur_idx = _WIRE_DEBUG_IDS.index(cur) if cur in _WIRE_DEBUG_IDS else 0
            dd = ui.ComboBox(cur_idx, *_WIRE_DEBUG_MODES)
            dd.model.add_item_changed_fn(
                lambda m, e, ww=w, ids=_WIRE_DEBUG_IDS:
                    self._on_wire_debug(ww, ids[int(m.get_item_value_model().get_value_as_int())]))

    def _debug_payload(self, w):
        """Return the wire dict for show_wire_debug with its spec attached.

        The spec gives the bend heatmap the wire's own min_bend_radius and the cells its
        color.
        """
        d = dict(w)
        d["spec"] = wire_library.as_spec(self._types[w["type_index"]])
        return d

    def _on_wire_debug(self, wire, mode):
        self._active_debug = None if mode == "none" else (wire["name"], mode)
        err = self._api.show_wire_debug(self._debug_payload(wire), mode)
        if err:
            self._progress.text = err

    def _refresh_debug(self):
        """Re-author the live per-wire debug view against the latest grids and polyline.

        Must run after any Route All or Re-route, otherwise the view shows stale geometry.
        """
        if not self._active_debug:
            return
        name, mode = self._active_debug
        w = next((x for x in self._wires if x["name"] == name), None)
        if w is None:
            self._active_debug = None
            return
        self._api.show_wire_debug(self._debug_payload(w), mode)

    def _rebuild_inspector_bundle(self, bidx):
        """Build the bundle inspector, shown inside the Selected wire frame."""
        b = self._bundles[bidx]
        status = b["status"]
        with ui.HStack(height=0, spacing=6):
            ui.Rectangle(width=14, height=14,
                         style={"background_color": _abgr(self._BUNDLE_START_COLOR),
                                "border_radius": 2, "border_width": 1,
                                "border_color": 0xFF222222})
            ui.Rectangle(width=14, height=14, tooltip=status,
                         style={"background_color": _DOT.get(status, _DOT["unrouted"]),
                                "border_radius": 7})
            ui.Label(f"{b['name']}", style={"font_size": 16})
            ui.Label(f"bundle ({b['kind']})", style={"color": 0xFFAAAAAA})
        mem_str = ", ".join(b["members"]) if b["members"] else "(none)"
        ui.Label(f"Members: {mem_str}", word_wrap=True,
                 style={"color": 0xFF999999})
        if status == "no_path" and b.get("reason"):
            ui.Label(f"No path: {b['reason']}", word_wrap=True,
                     style={"color": _BAD})
        # Harness type selector; it drives the trunk BOM cost.
        if self._bundle_type_labels:
            with ui.HStack(height=0):
                ui.Label("Harness type", width=96,
                         tooltip="Sets cost/m for the trunk BOM row.")
                tidx = b.get("type_index", 0)
                btc = ui.ComboBox(tidx, *self._bundle_type_labels)
                btc.model.add_item_changed_fn(
                    lambda m, e, i=bidx: self._set_bundle_type(i, m))
            bt = (self._bundle_types[b.get("type_index", 0)]
                  if self._bundle_types else None)
            if bt:
                ui.Label(f"  ${bt['cost_per_m']:.2f}/m  |  "
                         f"{bt['mass_per_m_kg']*1000:.0f} g/m",
                         style={"color": 0xFF999999, "font_size": 12})
        # Trunk constraints group.
        ui.Spacer(height=2)
        with ui.HStack(height=0):
            ui.Label("TRUNK CONSTRAINTS  (0 = ignore, 10 = strong)",
                     style={"color": _ACCENT, "font_size": 11})
            ui.Spacer()
            ui.Button("Reset", width=60, height=_MINI_H,
                      tooltip="Reset all trunk constraints to their default weight",
                      clicked_fn=lambda i=bidx: self._reset_bundle_weights(i))
        self._bundle_sliders = {}
        self._bundle_slider_vals = {}
        for k in _WEIGHTS:
            label, help_text = _WEIGHT_HELP[k]
            with ui.HStack(height=0, spacing=6):
                ui.Label(label, width=96, tooltip=help_text)
                s = ui.FloatSlider(min=0.0, max=10.0, tooltip=help_text)
                s.model.set_value(b["weights"].get(k, 1.0))
                s.model.add_value_changed_fn(
                    lambda m, kk=k, i=bidx: self._set_bundle_weight(i, kk, m))
                self._bundle_sliders[k] = s
                val = ui.Label(f"{b['weights'].get(k, 1.0):.1f}", width=30,
                               alignment=self._num_align(),
                               style={"color": 0xFFCCCCCC, "font_size": 12})
                self._bundle_slider_vals[k] = val
        # Waypoints group.
        ui.Spacer(height=2)
        ui.Label("WAYPOINTS  (trunk sequence)", style={"color": _ACCENT, "font_size": 11})
        ui.Button("+ Add waypoint (trunk must pass through)", height=_BTN_H,
                  clicked_fn=self._add_bundle_waypoint)
        if b["waypoints"]:
            ui.Label("  drag the :: handle to reorder (order = trunk sequence)",
                     style={"color": 0xFF888888})
            for i, wp_path in enumerate(b["waypoints"]):
                with ui.HStack(height=0, spacing=4) as row:
                    handle = ui.Label("::", width=18, style={"color": 0xFFAAAAAA},
                                      tooltip="Drag to reorder")
                    handle.set_drag_fn(lambda j=i: str(j))
                    ui.Label(f"#{i + 1}", width=34,
                             style={"color": 0xFFDDDDDD, "font_size": 15})
                    ui.Spacer()
                    ui.Button("Locate", width=64, height=_MINI_H,
                              clicked_fn=lambda p=wp_path: self._api.select_prim(p))
                    ui.Button("Delete", width=64, height=_MINI_H,
                              style={"color": 0xFFCC6666},
                              clicked_fn=lambda j=i: self._delete_bundle_waypoint(j))
                row.set_accept_drop_fn(lambda *_: True)
                row.set_drop_fn(lambda e, dst=i: self._reorder_bundle_waypoint(
                    int(e.mime_data), dst))
        else:
            ui.Label("  (no waypoints)", style={"color": 0xFF888888})
        ui.Spacer(height=2)
        ui.Button("Re-route bundle", clicked_fn=self._on_refine_bundle, height=_CTA_H,
                  style={"background_color": _PRIMARY, "color": 0xFFFFFFFF})

    def _set_bundle_weight(self, bidx, k, model):
        if bidx < len(self._bundles):
            v = float(model.get_value_as_float())
            self._bundles[bidx]["weights"][k] = v
            lbl = getattr(self, "_bundle_slider_vals", {}).get(k)
            if lbl is not None:
                lbl.text = f"{v:.1f}"

    def _reset_bundle_weights(self, bidx):
        """Reset every trunk-constraint slider on the bundle to the default weight."""
        if bidx >= len(self._bundles):
            return
        b = self._bundles[bidx]
        for k in _WEIGHTS:
            b["weights"][k] = _DEFAULT_WEIGHT
            s = getattr(self, "_bundle_sliders", {}).get(k)
            if s is not None:
                s.model.set_value(_DEFAULT_WEIGHT)

    def _set_weight(self, k, model):
        # Write the slider value straight into the selected wire's weights so it survives
        # selecting another wire and coming back.
        if self._selected is None or self._selected >= len(self._wires):
            return
        v = float(model.get_value_as_float())
        self._wires[self._selected]["weights"][k] = v
        lbl = getattr(self, "_slider_vals", {}).get(k)
        if lbl is not None:
            lbl.text = f"{v:.1f}"

    def _reset_weights(self):
        """Reset every soft-constraint slider on the selected wire to the default weight."""
        if self._selected is None or self._selected >= len(self._wires):
            return
        w = self._wires[self._selected]
        for k in _WEIGHTS:
            w["weights"][k] = _DEFAULT_WEIGHT
            s = getattr(self, "_sliders", {}).get(k)
            if s is not None:
                s.model.set_value(_DEFAULT_WEIGHT)   # fires _set_weight -> updates readout

    # Start/end heading: axis presets, or a rotatable marker arrow under "Custom".
    _HEAD_COLORS = {"start": (0.2, 0.9, 0.2), "end": (0.9, 0.2, 0.2)}

    @staticmethod
    def _stage_up(stage):
        try:
            from pxr import UsdGeom
            return str(UsdGeom.GetStageUpAxis(stage))
        except Exception:
            return "Z"

    def _marker_path(self, w, which):
        return f"{scene_ops.MARKERS_SCOPE}/{w['key']}_{which}"

    def _heading_row(self, w, which, tip):
        """Build one heading row: a None/axis/Custom combo, plus azimuth and elevation
        fields when Custom is selected.

        The angles and the marker's arrow stay in sync. Typing angles re-aims the marker,
        and rotating the marker in the viewport wins at route time, since the solver reads
        the marker's rotated local +X axis.
        """
        key = f"{which}_head_idx"
        idx = int(w.get(key, 0))
        with ui.HStack(height=0):
            ui.Label(f"{which.capitalize()} heading", width=96, tooltip=tip)
            combo = ui.ComboBox(idx, *headings.HEADING_OPTIONS)
            combo.model.add_item_changed_fn(
                lambda m, *_, k=key: self._set_heading(k, m))
        if 0 <= idx < len(headings.HEADING_OPTIONS) \
                and headings.HEADING_OPTIONS[idx] == headings.CUSTOM:
            stage = self._get_stage()
            az = el = 0.0
            if stage is not None:
                ax = scene_ops.get_world_axis(stage, self._marker_path(w, which))
                if ax is not None:
                    az, el = headings.vector_to_angles(ax, self._stage_up(stage))
            with ui.HStack(height=0, spacing=6):
                ui.Spacer(width=96)
                ui.Label("azim", width=34,
                         tooltip="Degrees around the up-axis (0 = +X). Or just rotate "
                                 "the marker's arrow in the viewport.")
                azf = ui.FloatDrag(min=-180.0, max=180.0, step=1.0)
                azf.model.set_value(az)
                ui.Label("elev", width=30,
                         tooltip="Degrees toward the up-axis (+90 = straight up).")
                elf = ui.FloatDrag(min=-90.0, max=90.0, step=1.0)
                elf.model.set_value(el)
                # Attach the observers after set_value, otherwise building the row fires
                # the callback and writes back a spurious heading.
                azf.model.add_value_changed_fn(
                    lambda m, wh=which, a=azf, e=elf: self._set_heading_angles(wh, a, e))
                elf.model.add_value_changed_fn(
                    lambda m, wh=which, a=azf, e=elf: self._set_heading_angles(wh, a, e))

    def _set_heading(self, key, model):
        # Persist the chosen heading index, then sync the marker's arrow: axis presets aim
        # it, Custom shows it while keeping the current rotation, None hides it. The
        # inspector rebuild makes the angle fields appear or disappear.
        if self._selected is None or self._selected >= len(self._wires):
            return
        idx = int(model.get_item_value_model().get_value_as_int())
        w = self._wires[self._selected]
        if int(w.get(key, 0)) == idx:
            return                      # the combo fired without a real change, e.g. on rebuild
        w[key] = idx
        self._refresh_heading_arrow(w, "start" if key.startswith("start") else "end")
        self._schedule(inspector=True)

    def _refresh_heading_arrow(self, w, which):
        stage = self._get_stage()
        if stage is None:
            return
        idx = int(w.get(f"{which}_head_idx", 0))
        label = (headings.HEADING_OPTIONS[idx]
                 if 0 <= idx < len(headings.HEADING_OPTIONS) else "None")
        path = self._marker_path(w, which)
        color = self._HEAD_COLORS[which]
        inc = which == "end"   # the end arrow is drawn arriving into the marker
        if label == "None":
            scene_ops.set_marker_direction(stage, path, None, show=False)
        elif label == headings.CUSTOM:
            scene_ops.set_marker_direction(stage, path, None, show=True, color=color,
                                           incoming=inc)
        else:
            scene_ops.set_marker_direction(stage, path, headings.axis_to_vector(label),
                                           show=True, color=color, incoming=inc)

    def _set_heading_angles(self, which, az_field, el_field):
        """Re-aim the marker's arrow from the azimuth/elevation fields, in Custom mode."""
        if self._selected is None or self._selected >= len(self._wires):
            return
        stage = self._get_stage()
        if stage is None:
            return
        w = self._wires[self._selected]
        v = headings.angles_to_vector(az_field.model.get_value_as_float(),
                                      el_field.model.get_value_as_float(),
                                      self._stage_up(stage))
        scene_ops.set_marker_direction(stage, self._marker_path(w, which), v,
                                       show=True, color=self._HEAD_COLORS[which],
                                       incoming=(which == "end"))

    def _heading_vector(self, stage, w, which):
        """Return the heading vector to send to the solver.

        None when the direction is free, an axis preset, or for Custom the marker's
        rotated local +X axis read from the stage.
        """
        idx = int(w.get(f"{which}_head_idx", 0))
        label = (headings.HEADING_OPTIONS[idx]
                 if 0 <= idx < len(headings.HEADING_OPTIONS) else "None")
        if label == headings.CUSTOM:
            ax = scene_ops.get_world_axis(stage, self._marker_path(w, which))
            return [float(x) for x in ax] if ax is not None else None
        v = headings.axis_to_vector(label)
        return list(v) if v else None

    def _add_waypoint(self):
        stage = self._get_stage()
        if stage is None or self._selected is None:
            return
        w = self._wires[self._selected]
        n = w["wp_counter"]
        w["wp_counter"] += 1
        path = f"{scene_ops.MARKERS_SCOPE}/{w['key']}_wp{n}"
        # Spawn near the wire's start so it is easy to find, then the user drags it.
        inv = self._stage_inv(stage)  # meters -> stage units, whatever the stage uses
        start = scene_ops.get_world_pos(stage, f"{scene_ops.MARKERS_SCOPE}/{w['key']}_start")
        pos = (float(start[0]) + 0.3 * inv, float(start[1]), float(start[2])) if start is not None \
            else (0.25 * inv, 0.0, 0.0)
        scene_ops.spawn_waypoint_marker(stage, path, pos, color=(0.1, 0.5, 0.9),
                                         radius=self._marker_radius(stage) * 1.2)
        w["waypoints"].append(path)
        # A new waypoint lands in the last gap, after all the wire's bundles and before
        # the end. The user drags it earlier in the itinerary if they want.
        w.setdefault("wp_slots", []).append(len(self._wire_bundles(w)))
        self._api.select_prim(path)  # auto-select so it can be dragged immediately
        self._schedule(inspector=True)

    @staticmethod
    def _snap_to_centerline(world_xyz, polyline_m, inv):
        """Project a surface-hit world point onto the wire's route polyline.

        Keeps a dropped waypoint in the middle of the tube rather than on its skin.
        `polyline_m` is in meters and is scaled by `inv` into stage units to match
        `world_xyz`. Falls back to the raw point when there is no polyline yet.
        """
        import numpy as np
        if not polyline_m or len(polyline_m) < 2:
            return list(world_xyz)
        P = np.asarray(polyline_m, dtype=float) * float(inv)   # meters -> stage units
        q = np.asarray(world_xyz, dtype=float)
        best, best_d = q, 1e18
        for a, b in zip(P[:-1], P[1:]):
            ab = b - a
            L2 = float(ab @ ab)
            t = 0.0 if L2 < 1e-12 else max(0.0, min(1.0, float((q - a) @ ab) / L2))
            proj = a + t * ab
            d = float(((q - proj) ** 2).sum())
            if d < best_d:
                best_d, best = d, proj
        return [float(x) for x in best]

    def _on_viewport_pick(self, world_xyz, hit_path):
        """Handle a viewport double-click by dropping a waypoint on the selected wire.

        Only fires when the click landed on the selected wire's own routed tube or one of
        its bundle branch segments. Waypoints are appended in click order, so clicking
        along the wire in sequence builds the route order naturally.

        Wrapped so that a Kit-version quirk in the pick callback or a stale selection
        after a rebuild surfaces a readable message and a log entry rather than a raw
        traceback in the console.
        """
        try:
            self._do_viewport_pick(world_xyz, hit_path)
        except Exception as exc:  # noqa: BLE001
            carb.log_error("[piperouter] drop-waypoint-on-double-click failed:\n"
                           + traceback.format_exc())
            try:
                self._progress.text = (f"Could not drop a waypoint here: {exc}. "
                                       "Use '+ Add waypoint' in the inspector instead.")
            except Exception:  # noqa: BLE001
                pass

    def _do_viewport_pick(self, world_xyz, hit_path):
        if self._selected is None or self._selected >= len(self._wires):
            return
        w = self._wires[self._selected]
        tube = f"{scene_ops.ROUTES_SCOPE}/{w['name']}"
        hit_path = str(hit_path or "")
        if not (hit_path == tube or hit_path.startswith(tube + "/")
                or hit_path.startswith(tube + "_seg")):
            return   # double-clicked something other than the selected wire's tube
        stage = self._get_stage()
        if stage is None:
            return
        # The pick lands on the tube surface, so snap it onto the route polyline at that
        # spot to put the waypoint on the cable centerline.
        center = self._snap_to_centerline(world_xyz, w.get("polyline"),
                                          self._stage_inv(stage))
        n = w["wp_counter"]
        w["wp_counter"] += 1
        path = f"{scene_ops.MARKERS_SCOPE}/{w['key']}_wp{n}"
        scene_ops.spawn_waypoint_marker(stage, path, tuple(center), color=(0.1, 0.5, 0.9),
                                         radius=self._marker_radius(stage) * 1.2)
        w["waypoints"].append(path)
        w.setdefault("wp_slots", []).append(len(self._wire_bundles(w)))
        self._api.select_prim(path)
        self._schedule(inspector=True, wires=True)
        self._progress.text = f"waypoint dropped on {w['name']} at the double-clicked point"

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
        if idx < len(w.get("wp_slots", [])):
            del w["wp_slots"][idx]
        self._schedule(inspector=True)

    # Per-wire route itinerary: waypoints interleaved with bundle steps.
    def _wire_bundles(self, w):
        """Return the wire's bundles in the global order of `self._bundles`."""
        return [b for b in self._bundles if w["name"] in b["members"]]

    def _wire_itinerary(self, w):
        """Return the ordered route steps between start and end.

        Waypoints are placed by their slot and interleaved with this wire's bundle trunks
        in global order. Each token is {"kind": "wp", "wpi": <waypoint index>, "path": ...}
        or {"kind": "bundle", ...}.
        """
        bundles = self._wire_bundles(w)
        K = len(bundles)
        slots = w.get("wp_slots", [])
        tokens = []
        for s in range(K + 1):
            for i, path in enumerate(w["waypoints"]):
                si = slots[i] if i < len(slots) else K
                if min(max(int(si), 0), K) == s:
                    tokens.append({"kind": "wp", "wpi": i, "path": path})
            if s < K:
                b = bundles[s]
                tokens.append({"kind": "bundle", "id": b["id"], "name": b["name"]})
        return tokens

    def _move_wire_waypoint(self, src_disp, dst_disp):
        """Move the waypoint at itinerary index `src_disp` to index `dst_disp`.

        Every waypoint's slot is recomputed from the new order, a slot being the number of
        bundle steps before it. Bundles keep their global order.
        """
        if self._selected is None or self._selected >= len(self._wires):
            return
        w = self._wires[self._selected]
        tokens = self._wire_itinerary(w)
        if not (0 <= src_disp < len(tokens)) or tokens[src_disp]["kind"] != "wp":
            return
        tokens = waypoints.reorder(tokens, src_disp, dst_disp)
        new_paths, new_slots, bcount = [], [], 0
        for t in tokens:
            if t["kind"] == "bundle":
                bcount += 1
            else:
                new_paths.append(t["path"])
                new_slots.append(bcount)
        w["waypoints"], w["wp_slots"] = new_paths, new_slots
        self._schedule(inspector=True)  # re-renders the itinerary and the viewport labels

    # Bundle trunk waypoints: the shared trunk must pass through these.
    def _add_bundle_waypoint(self):
        stage = self._get_stage()
        if stage is None or self._selected_bundle is None \
                or self._selected_bundle >= len(self._bundles):
            return
        b = self._bundles[self._selected_bundle]
        n = b["wp_counter"]
        b["wp_counter"] += 1
        path = f"{scene_ops.MARKERS_SCOPE}/{b['id']}_wp{n}"
        inv = self._stage_inv(stage)  # meters -> stage units, whatever the stage uses
        merge = scene_ops.get_world_pos(stage, b["merge_marker"])
        pos = (float(merge[0]) + 0.3 * inv, float(merge[1]), float(merge[2])) \
            if merge is not None else (0.25 * inv, 0.0, 0.0)
        # Amber, so trunk waypoints read as belonging to a bundle.
        scene_ops.spawn_waypoint_marker(stage, path, pos, color=(0.95, 0.6, 0.1),
                                         radius=0.045 * inv)
        b["waypoints"].append(path)
        self._api.select_prim(path)  # auto-select so it can be dragged immediately
        self._schedule(inspector=True)

    def _delete_bundle_waypoint(self, idx):
        if self._selected_bundle is None or self._selected_bundle >= len(self._bundles):
            return
        b = self._bundles[self._selected_bundle]
        if idx >= len(b["waypoints"]):
            return
        path = b["waypoints"][idx]
        stage = self._get_stage()
        if stage is not None:
            p = stage.GetPrimAtPath(path)
            if p and p.IsValid():
                stage.RemovePrim(p.GetPath())
        del b["waypoints"][idx]
        self._schedule(inspector=True)

    def _reorder_bundle_waypoint(self, src, dst):
        if self._selected_bundle is None or self._selected_bundle >= len(self._bundles):
            return
        b = self._bundles[self._selected_bundle]
        b["waypoints"] = waypoints.reorder(b["waypoints"], src, dst)
        self._schedule(inspector=True)

    def _refresh_vp_labels(self):
        """Re-author the persistent viewport text for every wire and bundle.

        Wire <n> gets W<n>S at its start, W<n>.<k> above its k-th waypoint and W<n>E at
        its end. Bundle <n> gets B<n>S at its merge marker and B<n>E at its split marker.
        """
        if getattr(self, "_picker", None) is not None:
            self._picker.enable()   # arms the double-click picker once the viewport exists
        vpl = getattr(self, "_vp_labels", None)
        if vpl is None:
            return
        stage = self._get_stage()
        if stage is None:
            vpl.clear()
            return
        up = self._stage_up_offset(stage)
        items = []

        def _add(path, text, color):
            p = scene_ops.get_world_pos(stage, path)
            if p is not None:
                items.append(((p[0] + up[0], p[1] + up[1], p[2] + up[2]), text, color))

        for wi, w in enumerate(self._wires):
            tag = f"W{wi + 1}"
            _add(f"{scene_ops.MARKERS_SCOPE}/{w['key']}_start", f"{tag}S", 0xFF33CC33)  # green
            for k, wp_path in enumerate(w["waypoints"]):
                _add(wp_path, f"{tag}.{k + 1}", 0xFFFFFFFF)                              # waypoint
            _add(f"{scene_ops.MARKERS_SCOPE}/{w['key']}_end", f"{tag}E", 0xFF3333CC)    # red

        for bi, b in enumerate(self._bundles):
            tag = f"B{bi + 1}"
            _add(b["merge_marker"], f"{tag}S", _abgr(self._BUNDLE_START_COLOR))   # amber
            for k, wp_path in enumerate(b.get("waypoints", [])):
                _add(wp_path, f"{tag}.{k + 1}", _abgr(self._BUNDLE_START_COLOR))  # trunk waypoint
            _add(b["split_marker"], f"{tag}E", _abgr(self._BUNDLE_END_COLOR))     # orange

        vpl.update(items)

    def _stage_up_offset(self, stage):
        """Return a world-space offset along the stage up-axis for floating labels.

        Sized at twice the marker radius so labels stay tight to their marker on a 1 m
        harness and on a large engine bay alike.
        """
        try:
            from pxr import UsdGeom
            axis = UsdGeom.GetStageUpAxis(stage)
            d = self._marker_radius(stage) * 2.0   # just above the marker, in stage units
            return (0.0, d, 0.0) if axis == UsdGeom.Tokens.y else (0.0, 0.0, d)
        except Exception:
            return (0.0, 0.0, self._marker_radius(stage) * 2.0)

    # --------------------------------------------------------------- solving
    def _gather_wire(self, w, priority=0):
        stage = self._get_stage()
        start = scene_ops.get_world_pos(stage, f"{scene_ops.MARKERS_SCOPE}/{w['key']}_start")
        end = scene_ops.get_world_pos(stage, f"{scene_ops.MARKERS_SCOPE}/{w['key']}_end")
        if start is None or end is None:
            return None
        wps, wp_slots = [], []
        slots = w.get("wp_slots", [])
        for i, wp_path in enumerate(w["waypoints"]):
            p = scene_ops.get_world_pos(stage, wp_path)
            if p is not None:
                wps.append([float(x) for x in p])
                wp_slots.append(int(slots[i]) if i < len(slots) else 0)
        sh = self._heading_vector(stage, w, "start")
        eh = self._heading_vector(stage, w, "end")
        return {"name": w["name"], "spec": wire_library.as_spec(self._types[w["type_index"]]),
                "start": [float(x) for x in start], "end": [float(x) for x in end],
                "waypoints": wps, "waypoint_slots": wp_slots, "weights": dict(w["weights"]),
                "connectivity": self._connectivity(), "priority": priority,
                "clearance_m": float(self._clearance.model.get_value_as_float()),
                "start_heading": sh, "end_heading": eh}

    def _on_route_all(self):
        pairs = [(w, self._gather_wire(w, i)) for i, w in enumerate(self._wires)]
        wires = [g for (_w, g) in pairs if g]
        if not wires:
            self._progress.text = "add a wire first"
            return
        self._progress.text = "voxelizing + routing all..."

        has_bundles = bool(self._bundles)
        if has_bundles:
            results, bom, err = self._api.route_all_bundles(
                wires, self._bundles_with_cost(),
                self._res.model.get_value_as_int(),
                self._url_value(), self._global_algo(), self._local_algo())
        else:
            results, bom, err = self._api.route_all(
                wires, self._res.model.get_value_as_int(), self._url_value(),
                self._global_algo(), self._local_algo())

        if err:
            self._progress.text = f"error: {err}"
            return
        by_name = {r["wire_id"]: r for r in (results or [])}
        for w in self._wires:
            r = by_name.get(w["name"])
            if r:
                w["status"] = r["status"]
                w["polyline"] = r.get("polyline") if r["status"] == "routed" else None
                w["reason"] = r.get("reason", "") if r["status"] != "routed" else ""
                w["note"] = r.get("note", "")          # e.g. buried-endpoint relocation
                w["cells"] = r.get("cells", [])
                w["raw_polyline"] = r.get("raw_polyline") if r["status"] == "routed" else None
        for b_row in (bom or []):
            for w in self._wires:
                if w["name"] == b_row["wire_id"]:
                    w["length_m"] = b_row["length_m"]
                    w["cost"] = b_row["cost"]
        # Update bundle statuses from the trunk results.
        if has_bundles:
            for b in self._bundles:
                trunk_id = f"bundle_{b['id']}_trunk"
                tr = by_name.get(trunk_id)
                if tr:
                    b["status"] = tr["status"]
                    b["trunk_polyline"] = tr.get("polyline")
                    b["trunk_length_m"] = tr.get("length_m", 0.0)
                    b["reason"] = tr.get("reason", "")
        self._last_bom = bom or []
        note = getattr(self._api, "_clearance_note", None)
        self._progress.text = f"done - note: {note}" if note else "done"
        self._schedule(wires=True, bom=True, bundles=True)
        self._refresh_views()
        self._refresh_overlay()
        self._refresh_hud()
        self._refresh_debug()
        self._update_readout()   # now reports the voxel size the route actually used

    def _on_refine(self):
        if self._selected is None:
            return
        w = self._wires[self._selected]
        for k, s in getattr(self, "_sliders", {}).items():
            w["weights"][k] = s.model.get_value_as_float()

        # If this wire belongs to any bundles, re-routing it must re-route their trunks
        # and branches too, otherwise the wire ignores the shared trunk it is meant to
        # pass through.
        my_bundles = [b for b in self._bundles if w["name"] in b["members"]]
        if my_bundles:
            self._progress.text = (f"re-routing bundle(s) containing {w['name']}...")
            self._run_bundle_reroute(my_bundles, trigger_wire=w["name"])
            return

        wire = self._gather_wire(w)
        if wire is None:
            self._progress.text = "wire has no endpoints"
            return
        # Every other already-routed wire becomes an obstacle, so re-routing this one
        # cannot overlap the rest.
        obstacles = [{"spec": wire_library.as_spec(self._types[ow["type_index"]]),
                      "polyline": ow["polyline"]}
                     for ow in self._wires if ow is not w and ow.get("polyline")]
        self._progress.text = f"re-routing {w['name']}..."
        result, bom_row, err = self._api.refine(wire, obstacles,
                                                self._res.model.get_value_as_int(),
                                                self._url_value(),
                                                self._global_algo(), self._local_algo())
        if err:
            self._progress.text = f"error: {err}"
            return
        w["status"] = result["status"] if result else "no_path"
        w["polyline"] = result.get("polyline") if w["status"] == "routed" else None
        w["reason"] = result.get("reason", "") if result and w["status"] != "routed" else ""
        w["note"] = result.get("note", "") if result else ""
        w["cells"] = result.get("cells", []) if result else []
        w["raw_polyline"] = result.get("raw_polyline") if result and w["status"] == "routed" else None
        if bom_row:
            w["length_m"] = bom_row["length_m"]
            w["cost"] = bom_row["cost"]
            self._last_bom = [b for b in self._last_bom if b["wire_id"] != bom_row["wire_id"]]
            self._last_bom.append(bom_row)
        self._progress.text = (f"{w['name']}: {w['status']}"
                               + (f" - {w['reason']}" if w["reason"] else ""))
        self._schedule(wires=True, bom=True)
        self._refresh_views()
        self._refresh_overlay()
        self._refresh_hud()
        self._refresh_debug()

    def _on_refine_bundle(self):
        """Re-route the currently selected bundle. Called from the bundle inspector."""
        if self._selected_bundle is None or self._selected_bundle >= len(self._bundles):
            return
        b = self._bundles[self._selected_bundle]
        for k, s in getattr(self, "_bundle_sliders", {}).items():
            b["weights"][k] = s.model.get_value_as_float()
        self._progress.text = f"re-routing bundle {b['name']}..."
        self._run_bundle_reroute([b])

    def _run_bundle_reroute(self, bundles_to_reroute, trigger_wire=None):
        """Re-route the given bundles, trunk and member branches alike.

        Every bundle is passed to route_all_bundles, not just the listed ones, so the
        two-phase algorithm can build the full segment sequence for wires shared across
        bundles: a wire in both B1 and B2 needs B1's split position even when only B2 is
        being re-routed.

        `trigger_wire` is the member wire that triggered the re-route, used in the status
        message.
        """
        pairs = [(w, self._gather_wire(w, i)) for i, w in enumerate(self._wires)]
        wires = [g for (_w, g) in pairs if g]
        results, bom, err = self._api.route_all_bundles(
            wires, self._bundles_with_cost(),   # every bundle, with harness type costs
            self._res.model.get_value_as_int(),
            self._url_value(), self._global_algo(), self._local_algo())
        if err:
            self._progress.text = f"bundle re-route error: {err}"
            return
        by_name = {r["wire_id"]: r for r in (results or [])}
        for w in self._wires:
            r = by_name.get(w["name"])
            if r:
                w["status"] = r["status"]
                w["polyline"] = r.get("polyline") if r["status"] == "routed" else None
                w["reason"] = r.get("reason", "") if r["status"] != "routed" else ""
        for b_row in (bom or []):
            for w in self._wires:
                if w["name"] == b_row["wire_id"]:
                    w["length_m"] = b_row["length_m"]
                    w["cost"] = b_row["cost"]
        for b in bundles_to_reroute:
            trunk_id = f"bundle_{b['id']}_trunk"
            tr = by_name.get(trunk_id)
            if tr:
                b["status"] = tr["status"]
                b["trunk_polyline"] = tr.get("polyline")
                b["trunk_length_m"] = tr.get("length_m", 0.0)
                b["reason"] = tr.get("reason", "")
        for b_row in (bom or []):
            self._last_bom = [x for x in self._last_bom if x["wire_id"] != b_row["wire_id"]]
        self._last_bom.extend(bom or [])
        msg = f"bundle re-route done"
        if trigger_wire:
            msg = f"{trigger_wire}: bundle re-routed"
        self._progress.text = msg
        self._schedule(wires=True, bom=True, bundles=True)
        self._refresh_views()
        self._refresh_overlay()
        self._refresh_hud()
        self._refresh_debug()
        self._update_readout()   # now reports the voxel size the route actually used

    # ------------------------------------------------------ tag / overlay / io
    def _refresh_views(self, force=False):
        """Rebuild the XY, XZ and YZ projection images.

        The render voxelizes at a high resolution, so automatic calls after routing are
        skipped while the section is collapsed. The 'Refresh 2D views' button passes
        force=True.
        """
        frame = getattr(self, "_views_frame", None)
        if not force and frame is not None and frame.collapsed:
            return
        routes = [{"points": w["polyline"], "color": self._types[w["type_index"]]["color"]}
                  for w in self._wires if w.get("polyline") and w["status"] == "routed"]
        # Bundle trunks render white so they stand out in the 2D cross-sections.
        for b in getattr(self, "_bundles", []):
            if b.get("trunk_polyline") and b["status"] == "routed":
                routes.append({"points": b["trunk_polyline"], "color": (1.0, 1.0, 1.0)})
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
            except Exception as exc:  # non-fatal; the omni.ui image API varies by build
                self._progress.text = f"views: {exc}"
                return

    def _on_tag(self):
        temp = self._temp.model.get_value_as_float()
        em = self._em.model.get_value_as_float()
        clr_mm = self._clr_tag.model.get_value_as_float()
        err = self._api.write_tag(temp if temp != 0.0 else None, em if em != 0.0 else None,
                                  clearance_m=(clr_mm / 1000.0) if clr_mm > 0.0 else None)
        self._progress.text = err if err else "tagged selection"
        if not err:
            self._schedule(tags=True)

    def _refresh_overlay(self):
        """Re-render the active occupancy, thermal or EM overlay against the latest grids.

        Does nothing when the overlay combo is set to None.
        """
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
        if getattr(self, "_hud", None) is not None:
            self._hud.destroy()
            self._hud = None
        if getattr(self, "_picker", None) is not None:
            self._picker.destroy()
            self._picker = None
        if getattr(self, "_help", None) is not None:
            self._help.destroy()
            self._help = None
        if self._window:
            self._window.destroy()
            self._window = None
        # Break the reference cycles so the extension object is released on reload.
        self._api = None
        self._get_stage = None
        self._wires = []
        self._sliders = {}
