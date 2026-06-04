from __future__ import annotations

import numpy as np


def _scipy_sssp(src, dst, weight, n_nodes, source_id, sink_id):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    graph = csr_matrix((weight, (src, dst)), shape=(n_nodes, n_nodes))
    dist, preds = dijkstra(
        graph, directed=True, indices=source_id, return_predecessors=True
    )
    if not np.isfinite(dist[sink_id]):
        return None, float("inf")
    path = []
    node = sink_id
    while node != -9999 and node != source_id:
        path.append(int(node))
        node = int(preds[node])
    path.append(source_id)
    path.reverse()
    return path, float(dist[sink_id])


def shortest_path(src, dst, weight, n_nodes, source_id, sink_id):
    """SSSP returning (node_path | None, cost). Uses cuGraph if importable."""
    try:
        import cudf  # noqa: F401
        import cugraph  # noqa: F401

        return _cugraph_sssp(src, dst, weight, n_nodes, source_id, sink_id)
    except Exception:
        return _scipy_sssp(src, dst, weight, n_nodes, source_id, sink_id)


def _cugraph_sssp(src, dst, weight, n_nodes, source_id, sink_id):
    import cudf
    import cugraph

    gdf = cudf.DataFrame(
        {
            "src": cudf.Series(np.asarray(src)),
            "dst": cudf.Series(np.asarray(dst)),
            "weight": cudf.Series(np.asarray(weight, dtype="float32")),
        }
    )
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(gdf, source="src", destination="dst", edge_attr="weight")
    res = cugraph.sssp(G, source=source_id)
    res = res.set_index("vertex")
    sink_cost = float(res.loc[sink_id, "distance"])
    if not np.isfinite(sink_cost):
        return None, float("inf")
    path = _walk_predecessors(lambda n: res.loc[n, "predecessor"], sink_id, source_id)
    if path is None:
        return None, float("inf")
    return path, sink_cost


def _walk_predecessors(pred_of, sink_id, source_id):
    """Rebuild sink->source by following predecessors; reverse to source->sink.

    Returns None if the chain dead-ends (predecessor -1) before reaching the source —
    i.e. the sink is unreachable. cuGraph marks unreachable vertices with a large FINITE
    sentinel distance (not inf) + predecessor -1, so the distance check alone can miss
    it; this catch prevents returning a bogus straight-through "route"."""
    path = []
    node = int(sink_id)
    while node != -1 and node != source_id:
        path.append(node)
        node = int(pred_of(node))
    if node != source_id:
        return None
    path.append(int(source_id))
    path.reverse()
    return path
