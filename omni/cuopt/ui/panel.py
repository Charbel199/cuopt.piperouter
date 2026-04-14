import omni.ui as ui


class CuoptPanel:

    WINDOW_TITLE = "cuOpt Pipe Router"

    def __init__(self, on_create_scene, on_route_all, on_clear):
        self._on_create_scene = on_create_scene
        self._on_route_all = on_route_all
        self._on_clear = on_clear

        self._resolution_model = ui.SimpleIntModel(30)
        self._clearance_model = ui.SimpleFloatModel(3.0)
        self._tube_radius_model = ui.SimpleFloatModel(1.0)
        self._bend_penalty_model = ui.SimpleFloatModel(0.0)
        self._server_url_model = ui.SimpleStringModel("http://localhost:5001")
        self._show_grid_model = ui.SimpleBoolModel(False)
        self._show_graph_model = ui.SimpleBoolModel(False)
        self._status_label = None

        self._window = ui.Window(self.WINDOW_TITLE, width=380, height=580)
        self._build()

    def _build(self):
        with self._window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=8, height=0):
                    self._build_scene_section()
                    self._build_server_section()
                    self._build_config_section()
                    self._build_debug_section()
                    self._build_actions_section()
                    self._build_status_section()

    def _build_scene_section(self):
        with ui.CollapsableFrame("Scene Setup", collapsed=False):
            with ui.VStack(spacing=6, height=0):
                with ui.HStack(spacing=6, height=32):
                    ui.Button(
                        "Simple Scene",
                        height=32,
                        clicked_fn=lambda: self._on_create_scene("simple"),
                    )
                    ui.Button(
                        "Engine Bay",
                        height=32,
                        clicked_fn=lambda: self._on_create_scene("engine_bay"),
                    )
                ui.Label(
                    "Drag the colored spheres to move pipe endpoints.",
                    height=28, word_wrap=True,
                    style={"color": 0xFF888888, "font_size": 12},
                )

    def _build_server_section(self):
        with ui.CollapsableFrame("cuOpt Server", collapsed=False):
            with ui.VStack(spacing=4, height=0):
                ui.Label("Server URL", height=18)
                ui.StringField(model=self._server_url_model, height=22)

    def _build_config_section(self):
        with ui.CollapsableFrame("Route Configuration", collapsed=False):
            with ui.VStack(spacing=6, height=0):
                ui.Label("Grid Resolution (higher = more accurate, slower)", height=18)
                with ui.HStack(spacing=8, height=20):
                    ui.IntSlider(model=self._resolution_model, min=10, max=100)
                    ui.IntField(model=self._resolution_model, width=52)

                ui.Label("Safety Clearance", height=18)
                with ui.HStack(spacing=8, height=20):
                    ui.FloatSlider(model=self._clearance_model, min=0.5, max=20.0)
                    ui.FloatField(model=self._clearance_model, width=52)

                ui.Label("Tube Radius", height=18)
                with ui.HStack(spacing=8, height=20):
                    ui.FloatSlider(model=self._tube_radius_model, min=0.5, max=10.0)
                    ui.FloatField(model=self._tube_radius_model, width=52)

                ui.Label("Bend Penalty (0 = shortest path, higher = fewer bends)",
                         height=28, word_wrap=True)
                with ui.HStack(spacing=8, height=20):
                    ui.FloatSlider(model=self._bend_penalty_model, min=0.0, max=20.0)
                    ui.FloatField(model=self._bend_penalty_model, width=52)

    def _build_debug_section(self):
        with ui.CollapsableFrame("Debug Visualization", collapsed=True):
            with ui.VStack(spacing=6, height=0):
                with ui.HStack(spacing=8, height=22):
                    ui.CheckBox(model=self._show_grid_model, width=18)
                    ui.Label("Show occupancy grid (red = blocked)", height=22)
                with ui.HStack(spacing=8, height=22):
                    ui.CheckBox(model=self._show_graph_model, width=18)
                    ui.Label("Show free cells near obstacles (green)", height=22)

    def _build_actions_section(self):
        with ui.CollapsableFrame("Actions", collapsed=False):
            with ui.VStack(spacing=6, height=0):
                ui.Button(
                    "Route All Pipes",
                    height=36,
                    clicked_fn=self._on_route_clicked,
                    style={"Button": {"background_color": 0xFF2B7A2B}},
                )
                ui.Button(
                    "Clear Pipes",
                    height=28,
                    clicked_fn=self._on_clear,
                )

    def _build_status_section(self):
        with ui.CollapsableFrame("Status", collapsed=False):
            self._status_label = ui.Label("Ready", word_wrap=True, height=60)

    def _on_route_clicked(self):
        self._on_route_all(
            self._resolution_model.get_value_as_int(),
            self._clearance_model.get_value_as_float(),
            self._tube_radius_model.get_value_as_float(),
            self._server_url_model.get_value_as_string(),
            self._bend_penalty_model.get_value_as_float(),
            self._show_grid_model.get_value_as_bool(),
            self._show_graph_model.get_value_as_bool(),
        )

    def set_status(self, text):
        if self._status_label:
            self._status_label.text = text

    def destroy(self):
        if self._window:
            self._window.destroy()
            self._window = None
