from omni.piperouter import headings


def test_none_maps_to_none():
    assert headings.axis_to_vector("None") is None


def test_axis_labels_map_to_unit_vectors():
    assert headings.axis_to_vector("+X") == (1.0, 0.0, 0.0)
    assert headings.axis_to_vector("-X") == (-1.0, 0.0, 0.0)
    assert headings.axis_to_vector("+Y") == (0.0, 1.0, 0.0)
    assert headings.axis_to_vector("-Z") == (0.0, 0.0, -1.0)


def test_options_start_with_none():
    assert headings.HEADING_OPTIONS[0] == "None"
    assert len(headings.HEADING_OPTIONS) == 7
