from __future__ import annotations

import os
from pathlib import Path

from piperouter_solver.grids import GridStack

DEFAULT_ROOT = os.environ.get("PIPEROUTER_GRID_DIR", "/dev/shm/piperouter")


def _check_session_id(session_id: str) -> None:
    if not session_id or "/" in session_id or "\\" in session_id or ".." in session_id:
        raise ValueError(f"invalid session_id: {session_id!r}")


class FilesystemSessionStore:
    """Hand grid stacks to the solver through a shared directory.

    Grids live on a shared dir, typically tmpfs (`/dev/shm`): the extension writes
    them once and the solver container reads them back by handle, so only the
    session_id crosses the HTTP boundary, never the arrays.
    """

    def __init__(self, root: str | Path = DEFAULT_ROOT):
        self.root = Path(root)
        # session_id -> (mtime_ns, size, GridStack). Re-voxelizing rewrites the file,
        # so a stale hit is impossible. A hit also carries over the per-stack lazy
        # caches: scene octree, dilations, normalized cost fields.
        self._cache: dict[str, tuple[int, int, GridStack]] = {}

    def grid_path(self, session_id: str) -> Path:
        _check_session_id(session_id)
        return self.root / session_id / "stack.npz"

    def save_stack(self, session_id: str, stack: GridStack) -> Path:
        p = self.grid_path(session_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        stack.save(p)
        return p

    def load_stack(self, session_id: str) -> GridStack:
        p = self.grid_path(session_id)
        st = p.stat()
        hit = self._cache.get(session_id)
        if hit is not None and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
            return hit[2]
        stack = GridStack.load(p)
        self._cache[session_id] = (st.st_mtime_ns, st.st_size, stack)
        return stack

    def exists(self, session_id: str) -> bool:
        return self.grid_path(session_id).exists()
