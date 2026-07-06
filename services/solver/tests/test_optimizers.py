import numpy as np

from piperouter_solver import optimizers
from piperouter_solver.models import WireType


def _wire(**kw):
    base = dict(id="w", label="w", kind="wire", outer_diameter_mm=10.0,
                min_bend_radius_mm=50.0, cost_per_m=1.0, mass_per_m_kg=0.1,
                max_temp_c=200.0, em_sensitivity=1.0, color=(1.0, 0.0, 0.0))
    base.update(kw)
    return WireType(**base)


def test_every_local_optimizer_pins_endpoints_and_stays_free(empty_stack):
    s = empty_stack
    s.occupancy[5, 5, 1] = 1                      # a blocked cell next to the path
    frame = s.frame
    wire = _wire()
    blocked = s.occupancy.astype(bool)
    # a zig-zag polyline through FREE cells (y alternates 5/6; x=5 sits at y=6 -> free)
    poly = [np.asarray(frame.grid_to_world((i, 5 + (i % 2), 1)), dtype=float)
            for i in range(10)]
    fixed = [0, len(poly) - 1]

    for name, cls in optimizers.LOCAL_OPTIMIZERS.items():
        out = cls().optimize(poly, frame, blocked, wire, None, None, 1.0, fixed)
        # (fibre may densify the path, so don't assert the count - assert the invariants)
        assert len(out) >= 2, f"{name}: degenerate output"
        assert np.allclose(out[0], poly[0]) and np.allclose(out[-1], poly[-1]), \
            f"{name}: moved a pinned endpoint"
        for p in out:
            i, j, k = frame.world_to_grid((float(p[0]), float(p[1]), float(p[2])))
            assert not blocked[i, j, k], f"{name}: moved a point into a blocked cell"


def test_strength_zero_is_passthrough(empty_stack):
    frame = empty_stack.frame
    blocked = empty_stack.occupancy.astype(bool)
    wire = _wire()
    poly = [np.asarray(frame.grid_to_world((i, 5, 1)), dtype=float) for i in range(6)]
    for name in ("fibre", "trajopt", "elastic_rod"):
        out = optimizers.make_local(name).optimize(poly, frame, blocked, wire,
                                                   None, None, 0.0, [0, 5])
        assert all(np.allclose(a, b) for a, b in zip(out, poly)), f"{name}: not pass-through"
