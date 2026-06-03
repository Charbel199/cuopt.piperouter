from __future__ import annotations

import os
from pathlib import Path

from piperouter_solver.grids import GridStack

DEFAULT_ROOT = os.environ.get("PIPEROUTER_GRID_DIR", "/dev/shm/piperouter")


def _check_session_id(session_id: str) -> None:
    if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:
        raise ValueError(f"invalid session_id: {session_id!r}")


class FilesystemSessionStore:
    """Grids live on a shared dir (typically tmpfs/`/dev/shm`) so the extension
    writes them once and the solver container reads them by handle. Only the
    session_id crosses the HTTP boundary, never the arrays."""

    def __init__(self, root: str | Path = DEFAULT_ROOT):
        self.root = Path(root)

    def grid_path(self, session_id: str) -> Path:
        _check_session_id(session_id)
        return self.root / session_id / "stack.npz"

    def save_stack(self, session_id: str, stack: GridStack) -> Path:
        p = self.grid_path(session_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        stack.save(p)
        return p

    def load_stack(self, session_id: str) -> GridStack:
        return GridStack.load(self.grid_path(session_id))

    def exists(self, session_id: str) -> bool:
        return self.grid_path(session_id).exists()
