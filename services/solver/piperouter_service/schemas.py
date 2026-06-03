from __future__ import annotations

from pydantic import BaseModel, Field

from piperouter_solver.models import RouteRequest, WireType


class WireSpec(BaseModel):
    id: str
    label: str = ""
    kind: str = "wire"
    outer_diameter_mm: float
    min_bend_radius_mm: float
    cost_per_m: float = 0.0
    mass_per_m_kg: float = 0.0
    max_temp_c: float = 1.0e9
    em_sensitivity: float = 0.0
    color: list[float] = Field(default_factory=lambda: [0.8, 0.8, 0.8])

    def to_wire_type(self) -> WireType:
        c = self.color
        return WireType(
            id=self.id, label=self.label or self.id, kind=self.kind,
            outer_diameter_mm=self.outer_diameter_mm,
            min_bend_radius_mm=self.min_bend_radius_mm,
            cost_per_m=self.cost_per_m, mass_per_m_kg=self.mass_per_m_kg,
            max_temp_c=self.max_temp_c, em_sensitivity=self.em_sensitivity,
            color=(float(c[0]), float(c[1]), float(c[2])),
        )


class RouteSpec(BaseModel):
    wire: WireSpec
    start: list[float]
    end: list[float]
    waypoints: list[list[float]] = Field(default_factory=list)
    weights: dict = Field(default_factory=dict)
    connectivity: int = 26
    priority: int = 0
    clearance_m: float = 0.0

    def to_route_request(self) -> RouteRequest:
        return RouteRequest(
            wire=self.wire.to_wire_type(),
            start=tuple(self.start),
            end=tuple(self.end),
            waypoints=[tuple(w) for w in self.waypoints],
            weights=dict(self.weights),
            connectivity=self.connectivity,
            priority=self.priority,
            clearance_m=self.clearance_m,
        )


class LockedRoute(BaseModel):
    polyline: list[list[float]]
    outer_diameter_mm: float = 0.0


class SolveBody(BaseModel):
    session_id: str
    route: RouteSpec
    locked_routes: list[LockedRoute] = Field(default_factory=list)


class SolveAllBody(BaseModel):
    session_id: str
    routes: list[RouteSpec]


class RouteOut(BaseModel):
    wire_id: str
    status: str
    polyline: list[list[float]] = Field(default_factory=list)
    length_m: float = 0.0


class SolveAllOut(BaseModel):
    routed: int
    no_path: int
    results: list[RouteOut]
