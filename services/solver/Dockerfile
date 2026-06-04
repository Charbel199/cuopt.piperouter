# PipeRouter solver service.
# CUDA runtime base (Ubuntu 24.04 ships Python 3.12). cuGraph is installed from
# NVIDIA's wheel index for the GPU shortest-path backend; if the wheel is
# unavailable for this platform the build still succeeds and the service falls
# back to the scipy CPU dijkstra (identical results, slower on huge graphs).
FROM nvidia/cuda:12.8.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIPEROUTER_GRID_DIR=/dev/shm/piperouter \
    WIRE_TYPES_PATH=/app/wire_types.json

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# GPU backend (optional — never fails the build). cupy provides cuSPARSE/cuSOLVER
# for the fibre-neutre smoothing least-squares; cudf/cugraph for the SSSP routing.
RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com \
        cugraph-cu12 cudf-cu12 cupy-cuda12x \
    || echo "WARN: GPU wheels unavailable; service will use the scipy/numpy fallback"

COPY pyproject.toml wire_types.json ./
COPY piperouter_solver ./piperouter_solver
COPY piperouter_service ./piperouter_service
RUN pip install --no-cache-dir .

EXPOSE 8000
# The container runs as root and shares the grid dir (a bind mount) with the host
# extension (a normal user). Make it world-writable on startup so the extension can
# create session subdirs in it — otherwise the host gets EACCES on the root-owned dir.
CMD ["sh", "-c", "mkdir -p \"$PIPEROUTER_GRID_DIR\" && chmod 777 \"$PIPEROUTER_GRID_DIR\" && exec uvicorn piperouter_service.app:app --host 0.0.0.0 --port 8000"]
