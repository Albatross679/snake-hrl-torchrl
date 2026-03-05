# syntax=docker/dockerfile:1

FROM python:3.12-slim

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

# Install dismech-python from its source directory
COPY dismech-python-src/ ./dismech-python-src/
RUN pip install --no-cache-dir -e ./dismech-python-src

# Install Python dependencies (mujoco is optional, not included)
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       torch>=2.1.0 \
       torchrl>=0.3.0 \
       tensordict>=0.3.0 \
       gymnasium>=0.29.0 \
       numpy>=1.24.0 \
       scipy>=1.11.0 \
       matplotlib>=3.8.0 \
       tensorboard>=2.15.0 \
       tqdm>=4.66.0 \
       pyyaml>=6.0 \
       numba>=0.58.0 \
       plotly>=5.18.0 \
       pyelastica>=0.3.0 \
       wandb>=0.16.0 \
       pytest>=7.4.0 \
       pytest-cov>=4.1.0

# Copy everything
COPY . .

# Install the project in editable mode
RUN pip install --no-cache-dir -e .

CMD ["bash"]
