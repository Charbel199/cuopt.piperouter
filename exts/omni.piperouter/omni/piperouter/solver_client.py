"""HTTP client for the solver service, stdlib-only so it works in any Kit python."""
from __future__ import annotations

import json
import urllib.request


class SolverClient:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 900.0):
        # Generous because the `dense` planner trades latency for route quality: a
        # nine-wire scene takes minutes where the corridor planner takes seconds, and a
        # socket timeout mid-solve looks like a hang rather than a slow answer.
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def _get(self, path: str):
        with urllib.request.urlopen(self.base + path, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def _post(self, path: str, body: dict):
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.base + path, data=data,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode())

    def health(self):
        return self._get("/health")

    def wire_types(self):
        return self._get("/wire_types")

    def solve(self, session_id: str, route: dict, locked_routes=None):
        body = {"session_id": session_id, "route": route}
        if locked_routes:
            body["locked_routes"] = locked_routes
        return self._post("/solve", body)

    def solve_all(self, session_id: str, routes: list):
        return self._post("/solve_all", {"session_id": session_id, "routes": routes})
