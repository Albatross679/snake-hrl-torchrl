#!/usr/bin/env bash
#
# Snake HRL — Native macOS setup with MPS (Apple GPU) support
#
# ============================================================
# PREREQUISITES
# ============================================================
#
# 1. macOS 12.3+ on Apple Silicon (M1/M2/M3/M4)
#
# 2. Xcode Command Line Tools:
#       xcode-select --install
#
# 3. Homebrew (https://brew.sh):
#       /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
#
# 4. Python 3.13 (via Homebrew):
#       brew install python@3.13
#
# ============================================================
# SETUP
# ============================================================
#
# 1. Run this script from the repo root:
#       chmod +x script/setup_macos.sh
#       ./script/setup_macos.sh
#
# 2. Activate the virtual environment:
#       source .venv/bin/activate
#
# 3. Verify MPS is available:
#       python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
#
# ============================================================
# USAGE (after setup)
# ============================================================
#
# Activate env:
#       source .venv/bin/activate
#
# Run locomotion training:
#       python3 -m locomotion_elastica.train
#
# Run surrogate model training:
#       python3 -m aprx_model_elastica.train_surrogate
#
# Run tests:
#       pytest
#
# Set W&B credentials (once):
#       wandb login
#
# ============================================================

set -euo pipefail

PYTHON="python3.13"
VENV_DIR=".venv"

# ---------- Check prerequisites ----------

if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script is for macOS only." >&2
    exit 1
fi

if ! command -v "$PYTHON" &>/dev/null; then
    echo "Error: $PYTHON not found. Install with: brew install python@3.13" >&2
    exit 1
fi

if ! command -v brew &>/dev/null; then
    echo "Error: Homebrew not found. Install from https://brew.sh" >&2
    exit 1
fi

# ---------- System dependencies ----------

echo "==> Installing system dependencies via Homebrew..."
brew install cmake glfw glew mesa || true

# ---------- Create virtual environment ----------

if [ -d "$VENV_DIR" ]; then
    echo "==> Virtual environment already exists at $VENV_DIR, reusing it."
else
    echo "==> Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "==> Upgrading pip..."
pip install --upgrade pip

# ---------- Install dismech-python ----------

if [ -d "dismech-python-src" ]; then
    echo "==> Installing dismech-python..."
    pip install ./dismech-python-src
else
    echo "==> Warning: dismech-python-src/ not found, skipping."
fi

# ---------- Install PyTorch with MPS support ----------
# Standard PyPI torch includes MPS backend on macOS ARM64.

echo "==> Installing PyTorch (with MPS support)..."
pip install \
    torch==2.10.0 \
    torchrl==0.11.1 \
    tensordict==0.11.0

# ---------- Install remaining dependencies ----------

echo "==> Installing dependencies..."
pip install \
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

# ---------- Install the project ----------

echo "==> Installing snake-hrl in editable mode..."
pip install -e .

# ---------- Set thread control defaults ----------

echo "==> Setting thread control env vars in activate script..."
ACTIVATE_SCRIPT="$VENV_DIR/bin/activate"
if ! grep -q "OPENBLAS_NUM_THREADS" "$ACTIVATE_SCRIPT"; then
    cat >> "$ACTIVATE_SCRIPT" << 'ENVVARS'

# Snake HRL: prevent thread over-subscription in parallel envs
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
ENVVARS
fi

# ---------- Verify ----------

echo ""
echo "==> Setup complete! Verifying..."
python3 -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  MPS:      {torch.backends.mps.is_available()}')
"
python3 -c "
import torchrl
print(f'  TorchRL:  {torchrl.__version__}')
"

echo ""
echo "Done. Activate with:  source $VENV_DIR/bin/activate"
