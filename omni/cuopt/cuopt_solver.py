"""
cuOpt microservice integration for 3D pipe routing.

Converts the occupancy grid into a CSR waypoint graph, submits it to a
cuOpt self-hosted server, and extracts the optimal waypoint-level path.

The cuOpt solver treats the problem as a single-vehicle routing problem:
  - every valid (non-blocked) grid cell is a waypoint node
  - edges connect 26-connected neighbors with distance + bend penalty weights
  - one "delivery" task sits at the destination node
  - the vehicle starts at the source node

cuOpt internally computes shortest paths on this graph and returns the
full waypoint sequence, which we convert back to world coordinates.
"""

import json
import time as _time

import carb
import numpy as np

from .pathfinding import OccupancyGrid3D, _NEIGHBORS_26


def build_waypoint_graph(grid, bend_penalty=0.0):
    # turn the free cells of *grid* into a directed CSR graph.
    # fully vectorized with numpy

    ri, rj, rk = grid.res_xyz
    avg_cell = float(grid.cell_size[0])
    free = ~grid.occupied

    # assign node IDs to free cells (-1 = blocked)
    node_ids = np.full((ri, rj, rk), -1, dtype=np.int32)
    fi, fj, fk = np.where(free)
    n_nodes = len(fi)
    node_ids[fi, fj, fk] = np.arange(n_nodes, dtype=np.int32)

    cells_list = list(zip(fi.tolist(), fj.tolist(), fk.tolist()))
    lookup_dict = {c: i for i, c in enumerate(cells_list)}

    # neighbor offsets and their properties
    neighbor_offsets = []
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == 0 and dj == 0 and dk == 0:
                    continue
                dist = np.sqrt(di*di + dj*dj + dk*dk)
                axes = (di != 0) + (dj != 0) + (dk != 0)
                w = dist * avg_cell + bend_penalty * max(0, axes - 1)
                neighbor_offsets.append((di, dj, dk, w, axes > 1))

    # build all edges vectorized
    all_src = []
    all_dst = []
    all_wgt = []

    padded = np.pad(grid.occupied, 1, constant_values=True)

    for di, dj, dk, w, is_diagonal in neighbor_offsets:
        ni = fi + di
        nj = fj + dj
        nk = fk + dk

        # bounds check
        valid = (ni >= 0) & (ni < ri) & (nj >= 0) & (nj < rj) & (nk >= 0) & (nk < rk)

        # neighbor must be free
        valid_idx = np.where(valid)[0]
        dst_ids = node_ids[ni[valid_idx], nj[valid_idx], nk[valid_idx]]
        has_dst = dst_ids >= 0

        # corner-cutting check for diagonal moves
        if is_diagonal:
            oi, oj, ok = fi[valid_idx], fj[valid_idx], fk[valid_idx]
            no_cut = np.ones(len(valid_idx), dtype=bool)
            if di != 0:
                no_cut &= ~padded[oi + di + 1, oj + 1, ok + 1]
            if dj != 0:
                no_cut &= ~padded[oi + 1, oj + dj + 1, ok + 1]
            if dk != 0:
                no_cut &= ~padded[oi + 1, oj + 1, ok + dk + 1]
            has_dst &= no_cut

        keep = np.where(has_dst)[0]
        if len(keep) == 0:
            continue

        src_ids = node_ids[fi[valid_idx[keep]], fj[valid_idx[keep]], fk[valid_idx[keep]]]
        all_src.append(src_ids)
        all_dst.append(dst_ids[keep])
        all_wgt.append(np.full(len(keep), w, dtype=np.float32))

    if not all_src:
        empty = np.zeros(0, dtype=np.int32)
        return (
            np.zeros(n_nodes + 1, dtype=np.int32),
            empty, np.zeros(0, dtype=np.float32),
            cells_list, lookup_dict,
        )

    src_all = np.concatenate(all_src)
    dst_all = np.concatenate(all_dst)
    wgt_all = np.concatenate(all_wgt)

    # sort by source node to build CSR
    order = np.argsort(src_all, kind="stable")
    src_sorted = src_all[order]
    dst_sorted = dst_all[order]
    wgt_sorted = wgt_all[order]

    # build offsets from sorted source array
    offsets = np.zeros(n_nodes + 1, dtype=np.int32)
    if len(src_sorted) > 0:
        counts = np.bincount(src_sorted, minlength=n_nodes)
        offsets[1:] = np.cumsum(counts)

    carb.log_warn(
        f"[omni.cuopt] CSR graph: {n_nodes} nodes, {len(dst_sorted)} edges"
    )

    return (offsets, dst_sorted.astype(np.int32), wgt_sorted, cells_list, lookup_dict)


def solve(grid, start_world, end_world,
          server_url="http://localhost:5000", time_limit=5,
          bend_penalty=0.0):
    # find the optimal path via the cuOpt self-hosted microservice

    t0 = _time.time()
    offsets, edges, weights, cells, lookup = build_waypoint_graph(
        grid, bend_penalty=bend_penalty,
    )
    t_csr = _time.time() - t0

    start_idx = grid.world_to_grid(start_world)
    end_idx = grid.world_to_grid(end_world)

    if start_idx not in lookup:
        return None, "Start point is inside an obstacle."
    if end_idx not in lookup:
        return None, "End point is inside an obstacle."

    src = lookup[start_idx]
    dst = lookup[end_idx]
    n_nodes = len(cells)
    n_edges = len(edges)

    t1 = _time.time()
    route, msg = _call_cuopt(
        offsets, edges, weights, src, dst, server_url, time_limit,
    )
    t_cuopt = _time.time() - t1

    if route is None:
        return None, msg

    path = [grid.grid_to_world(cells[i]) for i in route if 0 <= i < n_nodes]
    if not path:
        return None, "cuOpt returned an empty route."

    carb.log_warn(
        f"[omni.cuopt] solve: csr={t_csr:.3f}s | cuopt={t_cuopt:.3f}s | "
        f"{n_nodes} nodes, {n_edges} edges, {len(path)} waypoints"
    )
    return path, f"{n_nodes} nodes, {n_edges} edges, {len(path)} waypoints"


def _call_cuopt(offsets, edges, weights, src, dst, base_url, time_limit):
    # POST the problem to the cuOpt REST API and poll for a solution
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError

    base = base_url.rstrip("/")

    t = _time.time()
    payload = json.dumps({
        "cost_waypoint_graph_data": {
            "waypoint_graph": {
                "0": {
                    "offsets": offsets.tolist(),
                    "edges":   edges.tolist(),
                    "weights": weights.tolist(),
                }
            }
        },
        "task_data": {
            "task_locations": [dst],
        },
        "fleet_data": {
            "vehicle_locations":   [[src, dst]],
            "vehicle_time_windows": [[0, 100000]],
        },
        "solver_config": {
            "time_limit": time_limit,
        },
    }).encode()
    t_json = _time.time() - t
    payload_kb = len(payload) / 1024

    headers = {"Content-Type": "application/json", "CLIENT-VERSION": "custom"}

    try:
        t = _time.time()
        req = Request(f"{base}/cuopt/request", data=payload,
                      headers=headers, method="POST")
        with urlopen(req, timeout=30) as r:
            body = json.loads(r.read())
        t_submit = _time.time() - t

        inline = _extract_solver_block(body)
        if inline:
            return _parse_solver_response(inline)

        req_id = body.get("reqId")
        if not req_id:
            return None, f"Unexpected cuOpt response: {body}"

        t = _time.time()
        polls = 0
        for _ in range(600):
            _time.sleep(0.1)
            polls += 1
            poll = Request(f"{base}/cuopt/solution/{req_id}", method="GET")
            with urlopen(poll, timeout=10) as r:
                poll_body = json.loads(r.read())
            sr = _extract_solver_block(poll_body)
            if sr is not None:
                t_solve = _time.time() - t
                carb.log_warn(
                    f"[omni.cuopt] http: json={t_json:.3f}s ({payload_kb:.0f}KB) | "
                    f"submit={t_submit:.3f}s | solve={t_solve:.3f}s ({polls} polls)"
                )
                return _parse_solver_response(sr)

        return None, "Timed out waiting for cuOpt."

    except (URLError, HTTPError) as exc:
        return None, f"Cannot reach cuOpt at {base}: {exc}"
    except Exception as exc:
        return None, f"cuOpt error: {exc}"


def _extract_solver_block(body):
    # pull the solver result dict from either API version
    resp = body.get("response", {})
    return resp.get("solver_response") or resp.get("solver_infeasible_response")


def _parse_solver_response(sr):
    # extract the forward path (Depot to Delivery) from the solver block
    status = sr.get("status", -1)
    if status != 0:
        return None, f"cuOpt infeasible (status {status})."

    # 26.x nests route per vehicle under vehicle_data.{id}
    vdata = sr.get("vehicle_data", {})
    if vdata:
        v0 = vdata.get("0", {})
        route = v0.get("route", [])
        types = v0.get("type", [])
    else:
        route = sr.get("route", [])
        types = sr.get("type", [])

    # keep nodes from Depot to first Delivery (inclusive)
    path = []
    for node, typ in zip(route, types):
        path.append(int(node))
        if typ == "Delivery":
            break

    return (path or None), ("OK" if path else "Empty route from cuOpt.")
