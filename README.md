# cuOpt Pipe Router

Omniverse extension that routes pipes through 3D environments while avoiding obstacles. Uses NVIDIA Warp for GPU voxelization and NVIDIA cuOpt for GPU-accelerated route optimization.

## How it works

```
Scene geometry (any imported mesh)
        |
        |  1. Warp GPU kernel: for each grid cell, query signed distance
        |     to nearest triangle. Inside mesh or within clearance = blocked.
        v
Occupancy grid (3D voxel grid: blocked vs free)
        |
        |  2. Convert free cells to a CSR waypoint graph.
        |     Each free cell = node. Neighboring free cells = edges.
        |     Edge weight = distance + bend penalty.
        v
CSR waypoint graph (nodes + edges + weights)
        |
        |  3. Send graph to cuOpt server via REST API.
        |     cuOpt solves optimal route on GPU.
        |     Returns waypoint-level path.
        v
Optimal path (sequence of 3D points)
        |
        |  4. Smooth with Catmull-Rom spline.
        |     Generate tube mesh in USD.
        v
Tube in the scene
```

Pipes are routed sequentially. Each completed pipe becomes an obstacle for the next one, preventing collisions between pipes.

## What is a CSR waypoint graph?

CSR (Compressed Sparse Row) packs a graph into two flat arrays:

```
offsets = [0, 3, 5, 8, ...]    # node i owns edges[offsets[i] .. offsets[i+1]]
edges   = [4, 7, 12, 0, 7, ...] # destination node of each edge
weights = [2.1, 3.5, ...]       # cost of each edge
```

Node 0 connects to nodes 4, 7, 12 (edges[0..3]). Node 1 connects to nodes 0, 7 (edges[3..5]). And so on.

## What does cuOpt optimize?

Minimizes total edge weight along the path from start to end. Edge weight is:

```
weight = euclidean_distance + bend_penalty * max(0, axes_changed - 1)
```

- `bend_penalty = 0`: shortest path, diagonal cuts allowed
- `bend_penalty > 0`: prefers axis-aligned (straight) segments, fewer bends, longer path

With multiple pipes, routing order matters. The first pipe gets the most freedom. Each subsequent pipe has more constraints because previous pipes block space. (this is what cuOpt expects as input)

## Requirements

- Omniverse Kit-based app (USD Composer, Isaac Sim, etc.)
- `omni.warp` extension enabled (for GPU voxelization)
- cuOpt server for GPU routing:
  ```
  docker run --gpus all -it --rm -p 5001:5000 nvidia/cuopt:26.4.0-cuda13.0-py3.13
  ```

## Setup

1. Open your Omniverse app
2. Window > Extensions > gear icon > add this repo's path as a search path
3. Search for "cuOpt Pipe Router" and enable it

## Usage

1. Click **Create Engine Bay Scene** or import your own geometry under `/World/Obstacles`
2. Drag the colored spheres to position pipe start/end points
3. Adjust parameters:
   - **Grid Resolution**: cells per axis (30 = fast, 60+ = accurate)
   - **Safety Clearance**: minimum distance from obstacles (units)
   - **Tube Radius**: pipe thickness
   - **Bend Penalty**: 0 = shortest, higher = straighter
4. Click **Route All Pipes**

## Debug visualization

Expand the "Debug Visualization" section in the panel:

- **Show occupancy grid**: red cubes = blocked cells (obstacle + clearance). This is what the solver sees as impassable.
- **Show waypoint graph**: green dots = free cells

Both render under `/World/Debug` and are cleared on the next run.

## Project structure

```
config/extension.toml         
omni/cuopt/
    extension.py              extension lifecycle, multi-pipe orchestration
    warp_voxelizer.py         GPU voxelization via Warp SDF queries
    cuopt_solver.py           CSR graph builder + cuOpt REST client
    pathfinding.py            occupancy grid, path smoothing
    scene_builder.py          USD geometry: obstacles, markers, tubes
    debug_viz.py              occupancy grid + waypoint graph visualization
    ui/panel.py               omni.ui panel
```
