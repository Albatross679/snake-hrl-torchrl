# syntax=docker/dockerfile:1
#
# Snake HRL — Hierarchical RL for snake robot predation
# CPU-only Docker image for macOS (Intel & Apple Silicon)
#
# ============================================================
# SETUP (run from the repo root on your Mac)
# ============================================================
#
# 1. Install Docker Desktop for Mac:
#       https://www.docker.com/products/docker-desktop/
#
# 2. Build the image (first time takes ~5-10 min):
#       docker build -t snake-hrl .
#
#    For a specific platform (useful on Apple Silicon if you
#    need an x86 image for compatibility):
#       docker build --platform linux/amd64 -t snake-hrl .
#
# 3. Run an interactive shell:
#       docker run -it --rm snake-hrl
#
# 4. Run with W&B credentials and mount output dir:
#       docker run -it --rm \
#         -e WANDB_API_KEY="your-key-here" \
#         -v $(pwd)/output:/app/output \
#         -v $(pwd)/data:/app/data \
#         -v $(pwd)/model:/app/model \
#         snake-hrl
#
# 5. Run training directly:
#       docker run -it --rm \
#         -e WANDB_API_KEY="your-key-here" \
#         -v $(pwd)/output:/app/output \
#         snake-hrl \
#         python3 -m locomotion_elastica.train
#
# 6. Run surrogate model training:
#       docker run -it --rm \
#         -v $(pwd)/data:/app/data \
#         -v $(pwd)/output:/app/output \
#         snake-hrl \
#         python3 -m aprx_model_elastica.train_surrogate
#
# 7. Run tests:
#       docker run --rm snake-hrl pytest
#
# ============================================================

FROM python:3.13-slim

LABEL maintainer="albatross679"
LABEL description="Snake HRL — Hierarchical RL for snake robot predation (CPU)"

# System dependencies for MuJoCo rendering and building C extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libglew-dev \
    libosmesa6-dev \
    libglfw3-dev \
    patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- dismech-python (changes rarely, cached early) ----------
COPY dismech-python-src/ ./dismech-python-src/
RUN pip install --no-cache-dir ./dismech-python-src

# ---------- Python dependencies ----------
# CPU-only PyTorch keeps the image ~2 GB smaller than CUDA variants.
#
# On x86_64: use the PyTorch CPU-only index for a lean install.
# On arm64 (Apple Silicon via Docker Desktop): the default PyPI wheel
# is already CPU-only, so we install from the standard index.
RUN pip install --no-cache-dir --upgrade pip \
    && ARCH=$(dpkg --print-architecture) \
    && if [ "$ARCH" = "amd64" ]; then \
         pip install --no-cache-dir \
           --index-url https://download.pytorch.org/whl/cpu \
           torch==2.10.0; \
       else \
         pip install --no-cache-dir torch==2.10.0; \
       fi \
    && pip install --no-cache-dir \
       torchrl==0.11.1 \
       tensordict==0.11.0 \
       gymnasium==1.2.3 \
       mujoco==3.5.0 \
       numpy==2.4.0 \
       scipy==1.17.1 \
       matplotlib==3.10.8 \
       tqdm==4.67.1 \
       pyyaml==6.0.3 \
       numba==0.64.0 \
       plotly==6.6.0 \
       pyelastica==0.3.3.post2 \
       wandb==0.25.0

# ---------- Project source ----------
COPY pyproject.toml ./
COPY src/ ./src/
COPY locomotion_elastica/ ./locomotion_elastica/
COPY aprx_model_elastica/ ./aprx_model_elastica/
COPY bing2019/ ./bing2019/
COPY snakebot-gym/ ./snakebot-gym/
COPY tests/ ./tests/

# Install the project in editable mode
RUN pip install --no-cache-dir -e .

# Prevent OpenBLAS/MKL thread over-subscription in parallel envs
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

CMD ["bash"]
