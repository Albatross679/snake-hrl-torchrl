# syntax=docker/dockerfile:1

# Multi-arch image: works on both Intel and Apple Silicon Macs
FROM python:3.13-slim

LABEL maintainer="albatross679"
LABEL description="Snake HRL — Hierarchical RL for snake robot predation (CPU)"

# System dependencies for rendering and building C extensions
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

# Install dismech-python first (changes rarely)
COPY dismech-python-src/ ./dismech-python-src/
RUN pip install --no-cache-dir ./dismech-python-src

# Install Python dependencies with pinned versions matching the dev environment.
# CPU-only PyTorch (no CUDA) — keeps image ~2 GB smaller.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       torch==2.10.0 \
    && pip install --no-cache-dir \
       torchrl==0.11.1 \
       tensordict==0.11.0 \
       gymnasium==1.2.3 \
       numpy==2.4.0 \
       scipy==1.17.1 \
       matplotlib==3.10.8 \
       tqdm==4.67.1 \
       pyyaml==6.0.3 \
       numba==0.64.0 \
       plotly==6.6.0 \
       pyelastica==0.3.3.post2 \
       wandb==0.25.0

# Copy project source
COPY pyproject.toml ./
COPY src/ ./src/
COPY locomotion_elastica/ ./locomotion_elastica/
COPY aprx_model_elastica/ ./aprx_model_elastica/
COPY bing2019/ ./bing2019/
COPY snakebot-gym/ ./snakebot-gym/
COPY tests/ ./tests/

# Install the project
RUN pip install --no-cache-dir -e .

# Default env vars for parallel worker thread control
ENV OPENBLAS_NUM_THREADS=1
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

CMD ["bash"]
