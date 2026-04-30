#!/usr/bin/env bash
# VM Setup Script — run once after SSH into a new Vast.ai instance
# Usage: ./setup.sh <repo-url> [.env-path]
#
# Prerequisites:
#   - Any PyTorch/CUDA base image (e.g. vastai/pytorch, nvcr.io/nvidia/pytorch)
#   - A .env file with required credentials (see below)
#
# Required .env variables:
#   ANTHROPIC_API_KEY     — Claude Code
#   B2_APPLICATION_KEY_ID — Backblaze B2
#   B2_APPLICATION_KEY    — Backblaze B2
#   WANDB_API_KEY         — Weights & Biases
#   HF_TOKEN              — HuggingFace
#   GH_TOKEN              — GitHub CLI (optional, can use `gh auth login` instead)

set -euo pipefail

REPO_URL="${1:-}"
ENV_FILE="${2:-.env}"

if [[ -z "$REPO_URL" ]]; then
    echo "Usage: ./setup.sh <repo-url> [.env-path]"
    echo "  repo-url:  Git repository URL to clone"
    echo "  .env-path: Path to .env file (default: .env in current dir)"
    exit 1
fi

# --- Step 1: Load environment variables ---
echo "=== Step 1: Loading environment variables ==="
if [[ -f "$ENV_FILE" ]]; then
    set -a
    source "$ENV_FILE"
    set +a
    # Persist to .bashrc for future sessions (deduplicate on re-run)
    while IFS= read -r line; do
        [[ -z "$line" || "$line" == \#* ]] && continue
        key="${line%%=*}"
        sed -i "/^${key}=/d" ~/.bashrc 2>/dev/null || true
        echo "$line" >> ~/.bashrc
    done < "$ENV_FILE"
    echo "   Loaded from $ENV_FILE"
else
    echo "   WARNING: $ENV_FILE not found. Set env vars manually."
fi

# Validate critical env vars
MISSING=()
[[ -z "${ANTHROPIC_API_KEY:-}" ]] && MISSING+=("ANTHROPIC_API_KEY")
[[ -z "${B2_APPLICATION_KEY_ID:-}" ]] && MISSING+=("B2_APPLICATION_KEY_ID")
[[ -z "${B2_APPLICATION_KEY:-}" ]] && MISSING+=("B2_APPLICATION_KEY")
[[ -z "${WANDB_API_KEY:-}" ]] && MISSING+=("WANDB_API_KEY")

if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "   WARNING: Missing env vars: ${MISSING[*]}"
    echo "   Some steps may fail. Continuing anyway..."
fi

# --- Step 2: Install system tools ---
echo "=== Step 2: Installing system tools ==="
apt-get update && apt-get install -y \
    git-lfs \
    sqlite3 \
    rclone \
    tmux \
    htop \
    nvtop \
    jq \
    ripgrep \
    rsync \
    curl \
    wget \
    openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install
echo "   System tools installed"

# --- Step 3: Install Node.js, Claude Code, and GitHub CLI ---
echo "=== Step 3: Installing Node.js, Claude Code, and GitHub CLI ==="
if ! command -v node &>/dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
    apt-get install -y nodejs
    rm -rf /var/lib/apt/lists/*
    echo "   Node.js installed: $(node --version)"
else
    echo "   Node.js already installed: $(node --version)"
fi

if ! command -v claude &>/dev/null; then
    npm install -g @anthropic-ai/claude-code
    echo "   Claude Code installed"
else
    echo "   Claude Code already installed"
fi

if ! command -v gh &>/dev/null; then
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
        | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg 2>/dev/null
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
        > /etc/apt/sources.list.d/github-cli.list
    apt-get update && apt-get install -y gh
    rm -rf /var/lib/apt/lists/*
    echo "   GitHub CLI installed"
else
    echo "   GitHub CLI already installed: $(gh --version | head -1)"
fi

# --- Step 4: Install ML Python packages ---
echo "=== Step 4: Installing ML Python packages ==="
pip install --no-cache-dir \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    transformers \
    accelerate \
    tokenizers \
    sentencepiece \
    datasets \
    evaluate \
    peft \
    safetensors \
    einops \
    wandb \
    bitsandbytes \
    nltk \
    tqdm \
    matplotlib \
    seaborn \
    plotly \
    pytest \
    hypothesis \
    b2 \
    huggingface_hub \
    torchrl \
    tensordict \
    gymnasium
echo "   ML packages installed"

# --- Step 5: Clone project ---
echo "=== Step 5: Cloning project ==="
PROJECT_DIR="/workspace/$(basename "$REPO_URL" .git)"
if [[ -d "$PROJECT_DIR" ]]; then
    echo "   $PROJECT_DIR already exists, pulling latest..."
    cd "$PROJECT_DIR" && git pull
else
    cd /workspace
    git clone "$REPO_URL"
    cd "$PROJECT_DIR"
fi
echo "   Project at: $PROJECT_DIR"

# --- Step 6: Install project dependencies ---
echo "=== Step 6: Installing project dependencies ==="
if [[ -f "pyproject.toml" ]]; then
    pip install -e . 2>&1 | tail -3
    echo "   Installed via pyproject.toml"
elif [[ -f "requirements.txt" ]]; then
    pip install -r requirements.txt 2>&1 | tail -3
    echo "   Installed via requirements.txt"
else
    echo "   No pyproject.toml or requirements.txt found, skipping"
fi

# --- Step 7: Download large files from B2 ---
echo "=== Step 7: Downloading files from Backblaze B2 ==="
if [[ -f "script/b2-pull.sh" ]]; then
    chmod +x script/b2-pull.sh
    ./script/b2-pull.sh
else
    echo "   No script/b2-pull.sh found, skipping B2 download"
fi

# --- Step 8: Authenticate services ---
echo "=== Step 8: Authenticating services ==="

# W&B
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    wandb login "$WANDB_API_KEY" 2>/dev/null && echo "   W&B: authenticated" || echo "   W&B: failed"
fi

# GitHub CLI
if [[ -n "${GH_TOKEN:-}" ]]; then
    echo "$GH_TOKEN" | gh auth login --with-token 2>/dev/null && echo "   gh: authenticated" || echo "   gh: failed"
else
    echo "   gh: skipped (no GH_TOKEN, run 'gh auth login' manually)"
fi

# HuggingFace
if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli login --token "$HF_TOKEN" 2>/dev/null && echo "   HF: authenticated" || echo "   HF: failed"
else
    echo "   HF: skipped (no HF_TOKEN)"
fi

# --- Step 9: Verify ---
echo "=== Step 9: Verification ==="
echo "   Python: $(python3 --version)"
echo "   PyTorch: $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo "   CUDA: $(python3 -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'not found')"
echo "   GPU: $(python3 -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")' 2>/dev/null || echo 'not found')"
echo "   Node: $(node --version 2>/dev/null || echo 'not found')"
echo "   Claude: $(claude --version 2>/dev/null || echo 'not found')"
echo "   gh: $(gh --version 2>/dev/null | head -1 || echo 'not found')"
echo ""
echo "=== Setup complete! ==="
echo "   cd $PROJECT_DIR"
echo "   Start a tmux session: tmux new -s work"
