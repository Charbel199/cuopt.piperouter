from __future__ import annotations

import logging

import numpy as np

_log = logging.getLogger("piperouter")
_sssp_backend_logged = False


def _log_sssp_backend(name):
    """Announce the SSSP backend ONCE, so a silent cuGraph->scipy (GPU->CPU) fallback is
    visible in the logs instead of quietly costing ~600 ms/wire."""
    global _sssp_backend_logged
    if not _sssp_backend_logged:
        _sssp_backend_logged = True
        if name == "cugraph":
            _log.info("[piperouter] SSSP backend: cuGraph (GPU)")
        else:
            _log.warning("[piperouter] SSSP backend: scipy Dijkstra (CPU) — cuGraph not "
                         "available; route search will be much slower. Install "
                         "cugraph-cu12 in the solver image for the GPU path.")


def _to_host(a):
    """numpy view of an array that may be a cupy (GPU) array (from the GPU edge build)."""
    cp = getattr(a, "get", None)   # cupy arrays expose .get() -> numpy
    return a.get() if callable(cp) and type(a).__module__.startswith("cupy") else np.asarray(a)


def _scipy_sssp(src, dst, weight, n_nodes, source_id, sink_id):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import dijkstra

    src, dst, weight = _to_host(src), _to_host(dst), _to_host(weight)
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

        _log_sssp_backend("cugraph")
        return _cugraph_sssp(src, dst, weight, n_nodes, source_id, sink_id)
    except Exception:
        _log_sssp_backend("scipy")
        return _scipy_sssp(src, dst, weight, n_nodes, source_id, sink_id)


def _cugraph_sssp(src, dst, weight, n_nodes, source_id, sink_id):
    import cudf
    import cugraph

    # cudf.Series accepts a cupy (GPU) array zero-copy OR a numpy array (host->device).
    # When the lattice was built with the GPU path, src/dst/weight are already cupy
    # arrays on the device, so this builds the graph with no CPU->GPU transfer.
    gdf = cudf.DataFrame({"src": cudf.Series(src),
                          "dst": cudf.Series(dst),
                          "weight": cudf.Series(weight)})
    G = cugraph.Graph(directed=True)
    G.from_cudf_edgelist(gdf, source="src", destination="dst", edge_attr="weight")
    res = cugraph.sssp(G, source=source_id)

    # Pull the result columns to host arrays ONCE, then index by vertex. The old code
    # did res.loc[node, "predecessor"] per path node, and each cudf .loc is a GPU kernel
    # + sync — ~100ms/route just for reconstruction. A single to_numpy() + array lookup
    # is ~5-6x faster and identical.
    verts = res["vertex"].to_numpy()
    dist = res["distance"].to_numpy()
    preds = res["predecessor"].to_numpy()
    vmax = int(verts.max())
    pred_of = np.full(vmax + 1, -1, dtype=np.int64)
    pred_of[verts] = preds
    dist_of = np.full(vmax + 1, np.inf, dtype=np.float64)
    dist_of[verts] = dist

    sink_cost = float(dist_of[sink_id]) if sink_id <= vmax else float("inf")
    if not np.isfinite(sink_cost):
        return None, float("inf")
    path = _walk_predecessors(lambda n: pred_of[n], sink_id, source_id)
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
