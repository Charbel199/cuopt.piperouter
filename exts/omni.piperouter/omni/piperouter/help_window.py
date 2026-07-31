"""In-app help and tutorial window.

Shown once on extension startup, subject to a persisted setting, and reopened by the '?'
button in the panel header. It is omni.ui only and fully guarded, so a UI hiccup here
never blocks routing.

The layout aims to be scannable rather than a wall of text: a three-step Quick Start
strip on top, then an accordion of short bulleted sections opened on demand.
"""
from __future__ import annotations

import carb

_SETTING = "/persistent/exts/omni.piperouter/showHelpOnStartup"

# Palette, in ABGR.
_GREEN = 0xFF76B900   # NVIDIA green accent
_WHITE = 0xFFEEEEEE
_GREY = 0xFF9A9A9A
_BODY = 0xFFCBCBCB
_CARD = 0xFF2B2B2B
_LINE = 0xFF3A3A3A
_BLUE = 0xFF2A7DBE

# Quick Start rows of (step, title, detail); always visible.
_QUICK = [
    ("1", "Build", "Create a sample scene, or open your own .usd"),
    ("2", "Place", "+ Add wire, drag the green & red markers"),
    ("3", "Route", "Hit ROUTE ALL"),
]

# Accordion sections: (title, [bullets]). Keep each bullet to one short sentence.
_SECTIONS = [
    ("Set up a scene", [
        "'Create sample scene' = 3-wire demo. 'Create complex scene' = full engine bay.",
        "Or just open your own .usd - every mesh becomes a routing obstacle automatically.",
        "Units (cm / mm / m) are detected from the stage, so CAD at any scale just works.",
    ]),
    ("Wires & routing", [
        "'+ Add wire' spawns a green START and red END marker - drag them into place.",
        "Pick a wire type from the dropdown (sets gauge, bend radius and cost).",
        "ROUTE ALL voxelizes the scene and routes every wire around obstacles & hot / EM zones.",
    ]),
    ("Tune a wire", [
        "Select a wire to open its inspector.",
        "Sliders are soft costs - 0 = ignore, higher = avoid more: Surface, Bend, Thermal, EM, Smoothing.",
        "Set Start / End headings to fix the direction the cable leaves and arrives - "
        "pick an axis, or 'Custom' to rotate the marker's arrow / type exact angles.",
        "'Re-route this wire' re-solves only it (others stay as obstacles); 'Lock' freezes it.",
    ]),
    ("Waypoints", [
        "'+ Add waypoint', then drag the blue gizmo where the wire must pass.",
        "Or DOUBLE-CLICK the wire's tube in the viewport to drop one exactly there.",
        "Drag the :: grip in 'Route order' to place it before / between / after the bundles.",
    ]),
    ("Bundles", [
        "'+ New bundle', tick the member wires, then drag the amber & orange trunk markers.",
        "Members fan out from the trunk to their own endpoints.",
        "Bundles can carry their own waypoints and follow the bundle's constraints.",
    ]),
    ("Heat / EM / clearance tags", [
        "In 'Tagging', select a part, set a temperature (°C), EM strength, or a "
        "per-object clearance (mm), then click Tag.",
        "Cables route around heat and EM, and keep each tagged object's own distance.",
        "Untagged objects use the global Safety clearance; cells hotter than a wire's "
        "rating become hard keep-outs.",
    ]),
    ("See what's happening", [
        "Per-wire Debug view: claimed cells, grid-vs-smooth, soft-cost terrain, or bend heatmap.",
        "Scene overlays: occupancy / thermal / EM point clouds.",
        "2D cross-sections show the exact grid; the viewport HUD shows cost / mass / length.",
    ]),
    ("Save, load & share", [
        "'Save session...' writes everything - geometry, markers, routes, settings - to one .usd.",
        "'Load session...' restores it; 'Export .usdz...' makes one shareable archive.",
        "'Reset' clears wires + markers but keeps your obstacle scene.",
    ]),
    ("Algorithms (optional)", [
        "Global algorithm = path finder (default 'octree_lattice' - fast). Local optimizer = shaping (default 'fibre').",
        "Everything tagged '(experimental)' is for benchmarking only.",
        "Pick 'lattice (exhaustive)' when soft costs must be followed exactly; the rest are for benchmarking.",
    ]),
    ("Tips", [
        "Higher Grid resolution = finer routes but slower.",
        "Safety clearance must be >= one grid cell to take effect.",
        "No path found? Raise resolution, or check the endpoint isn't buried inside a part.",
    ]),
]


class HelpWindow:
    def __init__(self):
        self._window = None

    # -------------------------------------------------- persisted "show on startup"
    @staticmethod
    def _settings():
        try:
            import carb.settings
            return carb.settings.get_settings()
        except Exception:
            return None

    def show_on_startup(self) -> bool:
        s = self._settings()
        if s is None:
            return True
        v = s.get(_SETTING)
        return True if v is None else bool(v)

    def _set_show_on_startup(self, model):
        s = self._settings()
        if s is not None:
            try:
                s.set_bool(_SETTING, bool(model.get_value_as_bool()))
            except Exception:
                pass

    # -------------------------------------------------- window
    def show(self):
        try:
            import omni.ui as ui
            if self._window is None:
                self._window = ui.Window("PipeRouter - Help", width=560, height=720)
                with self._window.frame:
                    with ui.ScrollingFrame(
                            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF):
                        with ui.VStack(spacing=0, height=0):
                            self._build(ui)
            self._window.visible = True
        except Exception as exc:  # noqa: BLE001
            carb.log_warn(f"[piperouter] help window failed: {exc}")

    # -------------------------------------------------- pieces
    def _hero(self, ui):
        with ui.HStack(height=0, spacing=10):
            ui.Rectangle(width=5, height=30, style={"background_color": _GREEN, "border_radius": 2})
            with ui.VStack(spacing=1, height=0):
                ui.Label("PipeRouter", style={"font_size": 26, "color": _WHITE})
                ui.Label("Constraint-aware wire & pipe routing, on the GPU.",
                         style={"color": _GREY, "font_size": 12})

    def _quick_start(self, ui):
        ui.Label("QUICK START", style={"font_size": 11, "color": _GREEN})
        ui.Spacer(height=4)
        with ui.ZStack(height=0):
            ui.Rectangle(style={"background_color": _CARD, "border_radius": 6})
            with ui.HStack(height=0, spacing=2):
                ui.Spacer(width=10)
                for i, (num, title, detail) in enumerate(_QUICK):
                    if i:
                        with ui.VStack(width=18):
                            ui.Spacer()
                            ui.Label("->", style={"color": _GREEN, "font_size": 16})
                            ui.Spacer()
                    with ui.VStack(spacing=2, height=0):
                        ui.Spacer(height=10)
                        with ui.HStack(height=0, spacing=6):
                            with ui.ZStack(width=20, height=20):
                                ui.Circle(style={"background_color": _GREEN})
                                ui.Label(num, alignment=ui.Alignment.CENTER,
                                         style={"color": 0xFF000000, "font_size": 13})
                            ui.Label(title, style={"color": _WHITE, "font_size": 15})
                        ui.Label(detail, word_wrap=True,
                                 style={"color": _GREY, "font_size": 12})
                        ui.Spacer(height=10)
                    ui.Spacer(width=10)

    def _section(self, ui, title, bullets, collapsed):
        frame = ui.CollapsableFrame(
            title, collapsed=collapsed,
            style={
                "CollapsableFrame": {"background_color": 0, "secondary_color": 0,
                                     "border_radius": 4, "margin_height": 2},
                "CollapsableFrame:hovered": {"background_color": _CARD},
                "Label::title": {"color": _WHITE, "font_size": 15},
            })
        with frame:
            with ui.VStack(spacing=6, height=0):
                ui.Spacer(height=2)
                for b in bullets:
                    with ui.HStack(spacing=8, height=0):
                        ui.Spacer(width=4)
                        # Font-independent bullet: a small green square top-aligned to
                        # the first text line. A literal bullet character renders as a
                        # '?' glyph in some Kit fonts.
                        with ui.VStack(width=6, height=0):
                            ui.Spacer(height=6)
                            ui.Rectangle(width=5, height=5,
                                         style={"background_color": _GREEN, "border_radius": 1})
                        ui.Label(b, word_wrap=True, style={"color": _BODY, "font_size": 13})
                ui.Spacer(height=2)

    def _footer(self, ui):
        ui.Rectangle(height=1, style={"background_color": _LINE})
        ui.Spacer(height=8)
        with ui.HStack(height=0, spacing=8):
            cb = ui.CheckBox(width=18)
            cb.model.set_value(self.show_on_startup())
            cb.model.add_value_changed_fn(self._set_show_on_startup)
            ui.Label("Show this on startup", style={"color": _GREY, "font_size": 12})
            ui.Spacer()
            ui.Button("Got it", width=90, height=28, clicked_fn=self._close,
                      style={"background_color": _BLUE, "color": _WHITE, "border_radius": 4})

    def _close(self):
        try:
            if self._window is not None:
                self._window.visible = False
        except Exception:  # noqa: BLE001
            pass

    def _build(self, ui):
        with ui.VStack(spacing=10, height=0):
            ui.Spacer(height=4)
            self._hero(ui)
            self._quick_start(ui)
            with ui.VStack(spacing=2, height=0):
                ui.Label("LEARN MORE", style={"font_size": 11, "color": _GREEN})
                ui.Spacer(height=2)
                for i, (title, bullets) in enumerate(_SECTIONS):
                    self._section(ui, title, bullets, collapsed=(i != 0))
            self._footer(ui)
            ui.Spacer(height=6)

    def destroy(self):
        try:
            if self._window is not None:
                self._window.destroy()
        except Exception:  # noqa: BLE001
            pass
        self._window = None
