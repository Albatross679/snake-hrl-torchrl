---
name: vm-setup
description: Set up a new Vast.ai (or any cloud GPU) VM instance using a popular base image (e.g. vastai/pytorch or nvidia/pytorch). Installs all system tools and ML Python packages, handles environment variables, project cloning, pip install from pyproject.toml, Backblaze B2 file downloads, and service authentication (W&B, HF, GitHub). Use when the user asks to (1) set up a new VM or cloud instance, (2) configure a fresh Vast.ai machine, (3) run the setup script on a new instance, (4) bootstrap a project on a new GPU machine, (5) "I just spun up a new VM", (6) initialize environment on remote machine.
---

# VM Setup

Run the setup script after SSH into a new Vast.ai instance running a popular base image (e.g. `vastai/pytorch`, `nvcr.io/nvidia/pytorch`, or any CUDA-enabled image). The script installs all required system tools and ML Python packages so you don't need a custom Docker image.

## Prerequisites

1. A running Vast.ai instance using any PyTorch/CUDA base image
2. A `.env` file with credentials (create locally, scp to VM)

### Required .env format

```
ANTHROPIC_API_KEY=sk-ant-...
B2_APPLICATION_KEY_ID=...
B2_APPLICATION_KEY=...
WANDB_API_KEY=...
HF_TOKEN=hf_...
GH_TOKEN=ghp_...
```

## Usage

```bash
# 1. SCP the .env file to the VM
scp .env root@<vast-ip>:/workspace/.env

# 2. SSH into the VM
ssh root@<vast-ip> -p <port>

# 3. Run the setup script (after first git clone)
# First time: clone manually, then use the script for future VMs
git clone <repo-url> /workspace/project
cd /workspace/project
./script/setup.sh <repo-url> /workspace/.env

# Or if the script is already available:
curl -sL <raw-script-url> | bash -s -- <repo-url> /workspace/.env
```

## What the script does

1. Loads env vars from `.env` and persists to `~/.bashrc`
2. Installs system tools (git-lfs, sqlite3, rclone, tmux, htop, nvtop, jq, ripgrep, rsync, openssh-client)
3. Installs Node.js 22, Claude Code, and GitHub CLI
4. Installs common ML Python packages (transformers, accelerate, peft, datasets, wandb, bitsandbytes, torchrl, gymnasium, etc.)
5. Clones the project repo (or pulls if already cloned)
6. Installs project deps via `pip install -e .` (pyproject.toml)
7. Runs `script/b2-pull.sh` to download model weights from Backblaze B2
8. Authenticates W&B, HuggingFace, and GitHub CLI
9. Verifies Python, PyTorch, CUDA, and GPU availability

## Post-setup

```bash
# Start a tmux session (persists if SSH drops)
tmux new -s work

# Verify Claude Code works
claude

# Start training
python train_t5.py --finetune ...
```

## Adding files to B2 download

Edit `script/b2-pull.sh` to add more files. Each file is a `b2 file download` line:

```bash
b2 file download "b2://mlworkflow/path/to/file" local/path/to/file
```
