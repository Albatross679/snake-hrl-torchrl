#!/usr/bin/env bash
#
# Snake HRL — VM setup for Vast.ai / cloud GPU instances
#
# Installs all system tools, ML Python packages, clones the repo,
# installs project deps, downloads B2 assets, and authenticates services.
#
# Usage:
#   ./script/setup.sh <repo-url> <env-file>
#
# Example:
#   ./script/setup.sh git@github.com:Albatross679/snake-hrl-torchrl.git /workspace/.env
#
# Prerequisites:
#   - CUDA-enabled base image (vastai/pytorch, nvcr.io/nvidia/pytorch, etc.)
#   - .env file with credentials (see CLAUDE.md or skill docs)

set -euo pipefail

REPO_URL="${1:?Usage: setup.sh <repo-url> <env-file>}"
ENV_FILE="${2:?Usage: setup.sh <repo-url> <env-file>}"
PROJECT_DIR="/workspace/snake-hrl-torchrl"

# ============================================================
# 0. Verify Python version (project requires >=3.12)
# ============================================================

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)
if [ "$PYTHON_MAJOR" -lt 3 ] || { [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 12 ]; }; then
    echo "Error: Python >=3.12 required, found $PYTHON_VERSION" >&2
    exit 1
fi
echo "==> Python $PYTHON_VERSION detected."

# ============================================================
# 1. Load environment variables
# ============================================================

echo "==> Loading env vars from $ENV_FILE..."
if [ ! -f "$ENV_FILE" ]; then
    echo "Error: $ENV_FILE not found." >&2
    exit 1
fi

set -a
source "$ENV_FILE"
set +a

# Persist to .bashrc so they survive new shells
while IFS='=' read -r key value; do
    # Skip comments and blank lines
    [[ -z "$key" || "$key" =~ ^# ]] && continue
    if ! grep -q "export $key=" ~/.bashrc 2>/dev/null; then
        echo "export $key=\"$value\"" >> ~/.bashrc
    fi
done < "$ENV_FILE"

echo "    Env vars loaded and persisted to ~/.bashrc."

# ============================================================
# 2. System tools
# ============================================================

echo "==> Installing system tools..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq \
    git-lfs \
    sqlite3 \
    rclone \
    tmux \
    htop \
    nvtop \
    jq \
    ripgrep \
    rsync \
    openssh-client \
    cmake \
    curl \
    wget \
    unzip \
    > /dev/null 2>&1

git lfs install --skip-repo > /dev/null 2>&1
echo "    System tools installed."

# ============================================================
# 3. Node.js, Claude Code, GitHub CLI
# ============================================================

echo "==> Installing Node.js 22..."
if ! command -v node &>/dev/null || [[ "$(node -v)" != v22* ]]; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash - > /dev/null 2>&1
    apt-get install -y -qq nodejs > /dev/null 2>&1
fi
echo "    Node.js $(node -v) installed."

echo "==> Installing Claude Code..."
npm install -g @anthropic-ai/claude-code > /dev/null 2>&1
echo "    Claude Code installed."

echo "==> Installing GitHub CLI..."
if ! command -v gh &>/dev/null; then
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg > /dev/null 2>&1
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
    apt-get update -qq > /dev/null 2>&1
    apt-get install -y -qq gh > /dev/null 2>&1
fi
echo "    GitHub CLI $(gh --version | head -1) installed."

# ============================================================
# 4. Python packages
# ============================================================

echo "==> Upgrading pip..."
pip install -q --upgrade pip

# Extra packages not in pyproject.toml but useful on cloud VMs
echo "==> Installing extra cloud/ML packages..."
pip install -q \
    transformers \
    accelerate \
    peft \
    datasets \
    bitsandbytes \
    huggingface_hub \
    numba \
    plotly \
    b2

echo "    Extra packages installed."

# ============================================================
# 5. Clone / pull project repo
# ============================================================

echo "==> Setting up project repo..."
if [ -d "$PROJECT_DIR/.git" ]; then
    echo "    Repo already exists, pulling latest..."
    cd "$PROJECT_DIR"
    git pull --ff-only
else
    echo "    Cloning repo..."
    git clone "$REPO_URL" "$PROJECT_DIR"
    cd "$PROJECT_DIR"
fi

# ============================================================
# 6. Install project dependencies
# ============================================================

echo "==> Installing project dependencies (editable + monitoring extra)..."
pip install -q -e ".[monitoring]"
echo "    Project installed."

# ============================================================
# 6b. Thread control defaults (prevent over-subscription in parallel envs)
# ============================================================

echo "==> Setting thread control env vars..."
for var in OPENBLAS_NUM_THREADS OMP_NUM_THREADS MKL_NUM_THREADS; do
    if ! grep -q "export $var=" ~/.bashrc 2>/dev/null; then
        echo "export $var=1" >> ~/.bashrc
    fi
done
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "    Thread control set (OPENBLAS/OMP/MKL = 1)."

# ============================================================
# 7. Download B2 assets
# ============================================================

if [ -f "script/b2-pull.sh" ]; then
    echo "==> Downloading assets from Backblaze B2..."
    bash script/b2-pull.sh
else
    echo "==> Skipping B2 pull (script/b2-pull.sh not found)."
fi

# ============================================================
# 8. Authenticate services
# ============================================================

echo "==> Authenticating services..."

# W&B
if [ -n "${WANDB_API_KEY:-}" ]; then
    wandb login "$WANDB_API_KEY" > /dev/null 2>&1 && echo "    W&B: authenticated." || echo "    W&B: auth failed."
else
    echo "    W&B: WANDB_API_KEY not set, skipping."
fi

# HuggingFace
if [ -n "${HF_TOKEN:-}" ]; then
    huggingface-cli login --token "$HF_TOKEN" > /dev/null 2>&1 && echo "    HF: authenticated." || echo "    HF: auth failed."
else
    echo "    HF: HF_TOKEN not set, skipping."
fi

# GitHub CLI
if [ -n "${GH_TOKEN:-}" ]; then
    echo "$GH_TOKEN" | gh auth login --with-token > /dev/null 2>&1 && echo "    GH: authenticated." || echo "    GH: auth failed."
else
    echo "    GH: GH_TOKEN not set, skipping."
fi

# ============================================================
# 9. Verify
# ============================================================

echo ""
echo "==> Verifying installation..."
python3 -c "
import torch, torchrl, tensordict
print(f'  Python:      {__import__(\"sys\").version.split()[0]}')
print(f'  PyTorch:     {torch.__version__}')
print(f'  TorchRL:     {torchrl.__version__}')
print(f'  TensorDict:  {tensordict.__version__}')
print(f'  CUDA:        {torch.version.cuda}')
print(f'  GPU count:   {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}:       {torch.cuda.get_device_name(i)}')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Project dir: $PROJECT_DIR"
echo ""
echo "  Next steps:"
echo "    tmux new -s work"
echo "    claude"
echo "============================================"
