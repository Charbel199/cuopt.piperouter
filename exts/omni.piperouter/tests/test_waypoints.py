from omni.piperouter import waypoints


def test_move_down():
    assert waypoints.reorder(["a", "b", "c", "d"], 0, 2) == ["b", "c", "a", "d"]


def test_move_up():
    assert waypoints.reorder(["a", "b", "c", "d"], 3, 1) == ["a", "d", "b", "c"]


def test_same_index_is_noop_copy():
    src = ["a", "b", "c"]
    out = waypoints.reorder(src, 1, 1)
    assert out == ["a", "b", "c"]
    assert out is not src  # always a fresh list


def test_indices_are_clamped():
    assert waypoints.reorder(["a", "b", "c"], 0, 99) == ["b", "c", "a"]
    assert waypoints.reorder(["a", "b", "c"], -5, 0) == ["a", "b", "c"]


def test_empty_list():
    assert waypoints.reorder([], 0, 1) == []
