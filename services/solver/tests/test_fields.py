import numpy as np

from piperouter_solver.fields import (
    melt_mask,
    neighbor_offsets,
    normalize,
    soft_cost_field,
    turn_penalty,
)
from piperouter_solver.models import WireType


def _wire(em_sens=0.5, max_temp=100.0):
    return WireType(
        id="t", label="t", kind="wire", outer_diameter_mm=10.0,
        min_bend_radius_mm=50.0, cost_per_m=1.0, mass_per_m_kg=0.1,
        max_temp_c=max_temp, em_sensitivity=em_sens, color=(1.0, 0.0, 0.0),
    )


# --- neighbor offsets ---

def test_connectivity_counts():
    assert len(neighbor_offsets(6)) == 6
    assert len(neighbor_offsets(18)) == 18
    assert len(neighbor_offsets(26)) == 26


def test_no_zero_offset_and_all_unique():
    for c in (6, 18, 26):
        offs = neighbor_offsets(c)
        assert (0, 0, 0) not in offs
        assert len(set(offs)) == len(offs)


def test_six_connectivity_is_axis_only():
    offs = set(neighbor_offsets(6))
    assert (1, 0, 0) in offs
    assert (1, 1, 0) not in offs  # no face-diagonals in 6-connectivity


# --- soft cost field and melt mask ---

def test_normalize_maps_to_unit_interval():
    a = np.array([0.0, 5.0, 10.0], dtype=np.float32)
    n = normalize(a)
    assert n.min() == 0.0 and n.max() == 1.0


def test_normalize_constant_field_is_zero():
    a = np.full((3, 3), 7.0, dtype=np.float32)
    assert np.all(normalize(a) == 0.0)


def test_surface_weight_makes_far_cells_cost_more(empty_stack):
    s = empty_stack
    s.surface_dist[0, 0, 0] = 0.0   # at a surface
    s.surface_dist[9, 9, 2] = 10.0  # far away
    cost = soft_cost_field(s, _wire(), {"surface": 1.0, "thermal": 0.0, "em": 0.0})
    assert cost[9, 9, 2] > cost[0, 0, 0]


def test_em_sensitivity_scales_em_cost(empty_stack):
    s = empty_stack
    s.em[4, 4, 1] = 1.0
    weights = {"surface": 0.0, "thermal": 0.0, "em": 1.0}
    sensitive = soft_cost_field(s, _wire(em_sens=1.0), weights)
    immune = soft_cost_field(s, _wire(em_sens=0.0), weights)
    assert sensitive[4, 4, 1] > 0.0
    assert immune[4, 4, 1] == 0.0


def test_melt_mask_flags_cells_above_rating(empty_stack):
    s = empty_stack
    s.thermal[2, 2, 1] = 150.0
    mask = melt_mask(s, _wire(max_temp=100.0))
    assert mask[2, 2, 1]
    assert not mask[0, 0, 0]


# --- turn penalty ---

def test_straight_travel_has_zero_penalty():
    assert turn_penalty((1, 0, 0), (1, 0, 0), min_bend_radius_mm=50.0, cell_size_mm=100.0) == 0.0


def test_sharper_turn_costs_more_than_gentler_turn():
    gentle = turn_penalty((1, 0, 0), (1, 1, 0), min_bend_radius_mm=50.0, cell_size_mm=100.0)
    sharp = turn_penalty((1, 0, 0), (-1, 0, 0), min_bend_radius_mm=50.0, cell_size_mm=100.0)
    assert sharp > gentle > 0.0


def test_sub_radius_turn_penalised_far_more_than_above_radius():
    # Same turn, same cell size. A wire whose min bend radius is under the radius the turn
    # implies pays no sub-radius penalty, so it costs less than one whose min radius is over.
    above = turn_penalty((1, 0, 0), (1, 1, 0), min_bend_radius_mm=10.0, cell_size_mm=100.0)
    below = turn_penalty((1, 0, 0), (1, 1, 0), min_bend_radius_mm=400.0, cell_size_mm=100.0)
    assert below > above
