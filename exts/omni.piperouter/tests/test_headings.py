import math

from omni.piperouter import headings


def test_none_maps_to_none():
    assert headings.axis_to_vector("None") is None


def test_axis_labels_map_to_unit_vectors():
    assert headings.axis_to_vector("+X") == (1.0, 0.0, 0.0)
    assert headings.axis_to_vector("-X") == (-1.0, 0.0, 0.0)
    assert headings.axis_to_vector("+Y") == (0.0, 1.0, 0.0)
    assert headings.axis_to_vector("-Z") == (0.0, 0.0, -1.0)


def test_options_start_with_none_and_end_with_custom():
    assert headings.HEADING_OPTIONS[0] == "None"
    # Custom is APPENDED so saved head_idx values from old sessions stay valid
    assert headings.HEADING_OPTIONS[-1] == headings.CUSTOM
    assert len(headings.HEADING_OPTIONS) == 8


def test_custom_label_maps_to_none_vector():
    # Custom's direction comes from the marker's rotation, not the label
    assert headings.axis_to_vector(headings.CUSTOM) is None


def _close(a, b, tol=1e-6):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def test_angles_to_vector_axes_zup():
    assert _close(headings.angles_to_vector(0, 0, "Z"), (1, 0, 0))
    assert _close(headings.angles_to_vector(90, 0, "Z"), (0, 1, 0))
    assert _close(headings.angles_to_vector(0, 90, "Z"), (0, 0, 1))


def test_angles_to_vector_axes_yup():
    assert _close(headings.angles_to_vector(90, 0, "Y"), (0, 0, 1))   # side = +Z
    assert _close(headings.angles_to_vector(0, 90, "Y"), (0, 1, 0))   # up = +Y


def test_angle_vector_round_trip():
    for up in ("Z", "Y"):
        for az, el in ((0, 0), (37, 12), (-120, -45), (179, 89)):
            v = headings.angles_to_vector(az, el, up)
            assert abs(math.sqrt(sum(c * c for c in v)) - 1.0) < 1e-9   # unit
            az2, el2 = headings.vector_to_angles(v, up)
            assert abs(az2 - az) < 1e-6 and abs(el2 - el) < 1e-6


def test_zero_vector_angles():
    assert headings.vector_to_angles((0, 0, 0)) == (0.0, 0.0)
