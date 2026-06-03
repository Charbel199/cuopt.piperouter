import numpy as np

from omni.piperouter import fields


def test_thermal_peaks_at_source_and_decays_to_ambient():
    bmin, cell, res = np.zeros(3), 0.1, (10, 10, 1)
    src_pos = bmin + (np.array([5, 5, 0]) + 0.5) * cell
    # sources are (pos, temp_c, falloff_m)
    f = fields.thermal_field(bmin, cell, res, [(src_pos, 120.0, 0.3)], ambient=20.0)
    assert abs(f[5, 5, 0] - 120.0) < 1.0     # reaches the source temperature
    assert abs(f[0, 0, 0] - 20.0) < 1.0      # ambient far away
    assert f[5, 5, 0] > f[7, 5, 0] > f[0, 0, 0]


def test_no_sources_is_uniform_ambient():
    bmin, cell, res = np.zeros(3), 0.1, (4, 4, 1)
    f = fields.thermal_field(bmin, cell, res, [], ambient=20.0)
    assert np.allclose(f, 20.0)


def test_em_sums_and_is_zero_without_sources():
    bmin, cell, res = np.zeros(3), 0.1, (10, 10, 1)
    assert np.allclose(fields.em_field(bmin, cell, res, []), 0.0)
    src = bmin + (np.array([5, 5, 0]) + 0.5) * cell
    f = fields.em_field(bmin, cell, res, [(src, 1.0, 0.3)])
    assert f[5, 5, 0] > 0.0
    assert f[0, 0, 0] == 0.0
