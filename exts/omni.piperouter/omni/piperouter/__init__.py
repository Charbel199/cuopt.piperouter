"""omni.piperouter — constraint-aware wire/pipe routing for automotive USD.

The Kit entry point (`extension.py`) imports omni.* and is loaded only inside a
Kit runtime; the import is guarded so the omni-free modules (grid_io, voxelizer,
fields, scene_ops, solver_client, router_session) remain importable headlessly for
testing.
"""

__version__ = "0.5.0"

try:  # only succeeds inside a Kit runtime
    from .extension import PipeRouterExtension  # noqa: F401
except Exception:  # pragma: no cover - headless import path
    pass
