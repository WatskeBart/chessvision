FROM python:3.13-slim

# Suppress debconf prompts during apt operations (TERM not set in build env).
# ARG (not ENV) so it doesn't persist into the running container.
ARG DEBIAN_FRONTEND=noninteractive

# Apply any pending security patches in the base image, then install the
# system libraries required by OpenCV (GUI + image I/O), ncnn, and libgomp.
# Package versions pinned to what ships in Debian 13 (Trixie) — update them
# when rebuilding after a Debian security advisory.
# Note: libglib2.0-0 was renamed to libglib2.0-0t64 in Debian 13.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
        libgl1=1.7.0-1+b2 \
        libglib2.0-0t64=2.84.4-3~deb13u3 \
        libsm6=2:1.2.6-1 \
        libxext6=2:1.3.4-1+b3 \
        libxrender1=1:0.9.12-1 \
        libgomp1=14.2.0-19 \
    && apt-get purge -y --allow-remove-essential --auto-remove perl-base \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Force CPU-only torch regardless of the pytorch-cu130 index override in
# pyproject.toml. This keeps the image ~200 MB instead of ~2 GB; inference
# runs via ncnn/onnxruntime so CUDA is not needed at detection time.
ENV UV_TORCH_BACKEND=cpu

# Install dependencies before copying source for better layer caching.
# --no-frozen: the uv.lock records CUDA hashes for x86_64; re-resolving here
# picks up the CPU wheels instead.
COPY pyproject.toml uv.lock ./
RUN uv sync --no-dev --no-install-project

# Copy the package source and install the project
COPY chessvision/ chessvision/
RUN uv sync --no-dev

EXPOSE 8080

# Mount models and recorded games from the host at runtime.
# Default model path (GRANDMASTER_PIECES_MODEL_PATH) expects models/ here.
VOLUME ["/app/models", "/app/games"]

# Default to web streaming — open http://localhost:8080 in a browser.
# For X11, override at runtime:
#   podman run --rm -it \
#     -e DISPLAY=$DISPLAY \
#     -v /tmp/.X11-unix:/tmp/.X11-unix \
#     ... chessvision uv run gm-detect
CMD ["uv", "run", "gm-detect", "--web"]
