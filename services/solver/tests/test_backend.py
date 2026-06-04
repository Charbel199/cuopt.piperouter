import numpy as np

from piperouter_solver.backend import _walk_predecessors, shortest_path


def test_walk_predecessors_reachable():
    preds = {3: 2, 2: 1, 1: 0}          # sink 3 -> 2 -> 1 -> source 0
    assert _walk_predecessors(lambda n: preds[n], 3, 0) == [0, 1, 2, 3]


def test_walk_predecessors_unreachable_deadends_to_none():
    # cuGraph's unreachable marker: sink's predecessor is -1 (never reaches source)
    preds = {3: -1}
    assert _walk_predecessors(lambda n: preds[n], 3, 0) is None


def test_walk_predecessors_source_equals_sink():
    assert _walk_predecessors(lambda n: -1, 0, 0) == [0]


def test_finds_cheapest_of_two_paths():
    # nodes: 0 -> 1 -> 3 (cost 2) vs 0 -> 2 -> 3 (cost 10)
    src = np.array([0, 1, 0, 2], dtype=np.int32)
    dst = np.array([1, 3, 2, 3], dtype=np.int32)
    w = np.array([1.0, 1.0, 5.0, 5.0], dtype=np.float32)
    path, cost = shortest_path(src, dst, w, n_nodes=4, source_id=0, sink_id=3)
    assert path == [0, 1, 3]
    assert abs(cost - 2.0) < 1e-6


def test_returns_none_when_unreachable():
    src = np.array([0], dtype=np.int32)
    dst = np.array([1], dtype=np.int32)
    w = np.array([1.0], dtype=np.float32)
    path, cost = shortest_path(src, dst, w, n_nodes=3, source_id=0, sink_id=2)
    assert path is None
    assert cost == float("inf")
