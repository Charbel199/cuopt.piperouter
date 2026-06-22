from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException

from piperouter_solver.models import RouteResult
from piperouter_solver.solver import Solver

from .schemas import RouteOut, SolveAllBody, SolveAllOut, SolveBody
from .session_store import FilesystemSessionStore

_DEFAULT_WIRE_TYPES = Path(__file__).resolve().parents[1] / "wire_types.json"


def _setup_logging() -> None:
    """Route the 'piperouter' logger (SSSP-backend line, planner/optimizer warnings, etc.)
    to the container's stdout so it shows in `docker logs`. By default uvicorn only
    configures its own loggers, so ours otherwise vanished. Level via PIPEROUTER_LOG_LEVEL
    (default INFO)."""
    lvl = getattr(logging, os.environ.get("PIPEROUTER_LOG_LEVEL", "INFO").upper(), logging.INFO)
    log = logging.getLogger("piperouter")
    log.setLevel(lvl)
    if not any(getattr(h, "_piperouter", False) for h in log.handlers):
        h = logging.StreamHandler(sys.stdout)
        h._piperouter = True   # marker so reloads don't stack duplicate handlers
        # messages already carry the "[piperouter]" tag, so the format stays minimal
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s",
                                         datefmt="%H:%M:%S"))
        log.addHandler(h)
    log.propagate = False      # we emit directly; don't double-log via the root logger


def _backend_name() -> str:
    try:
        import cudf  # noqa: F401
        import cugraph  # noqa: F401

        return "gpu"
    except Exception:
        return "scipy"


def _route_out(res: RouteResult) -> RouteOut:
    return RouteOut(
        wire_id=res.wire_id,
        status=res.status,
        polyline=[[float(x) for x in p] for p in res.polyline],
        length_m=float(res.length_m),
        reason=getattr(res, "reason", "") or "",
        note=getattr(res, "note", "") or "",
        cells=[[int(v) for v in c] for c in getattr(res, "cells", [])],
        raw_polyline=[[float(x) for x in p]
                      for p in getattr(res, "raw_polyline", [])],
    )


def app_factory(store: FilesystemSessionStore, wire_types_path: Path) -> FastAPI:
    app = FastAPI(title="PipeRouter Solver", version="0.1.0")
    solver = Solver()

    @app.get("/health")
    def health():
        return {"status": "ok", "backend": _backend_name()}

    @app.get("/wire_types")
    def wire_types():
        return json.loads(Path(wire_types_path).read_text())

    @app.post("/solve", response_model=RouteOut)
    def solve(body: SolveBody):
        if not store.exists(body.session_id):
            raise HTTPException(status_code=404, detail="session grids not found")
        stack = store.load_stack(body.session_id)
        extra = None
        if body.locked_routes:
            from piperouter_solver.obstacles import rasterize_polylines
            routes = [
                {"polyline": lr.polyline, "radius_m": lr.outer_diameter_mm / 2000.0}
                for lr in body.locked_routes
            ]
            extra = rasterize_polylines(stack.frame, routes)
        res = solver.route_one(stack, body.route.to_route_request(), extra_obstacles=extra)
        return _route_out(res)

    @app.post("/solve_all", response_model=SolveAllOut)
    def solve_all(body: SolveAllBody):
        if not store.exists(body.session_id):
            raise HTTPException(status_code=404, detail="session grids not found")
        stack = store.load_stack(body.session_id)
        reqs = [r.to_route_request() for r in body.routes]
        report = solver.route_all(stack, reqs)
        return SolveAllOut(
            routed=report.routed,
            no_path=report.no_path,
            results=[_route_out(r) for r in report.results],
        )

    return app


_setup_logging()
app = app_factory(
    FilesystemSessionStore(),
    Path(os.environ.get("WIRE_TYPES_PATH", _DEFAULT_WIRE_TYPES)),
)
