"""Constraint-aware wire/pipe routing for automotive USD.

The Kit entry point (`extension.py`) imports omni.*, so it only loads inside a Kit
runtime. The import is guarded to keep the omni-free modules (grid_io, voxelizer,
fields, scene_ops, solver_client, router_session) importable headlessly.
"""

__version__ = "0.5.0"

try:  # only succeeds inside a Kit runtime
    from .extension import PipeRouterExtension  # noqa: F401
except Exception:  # pragma: no cover, headless import path
    pass
