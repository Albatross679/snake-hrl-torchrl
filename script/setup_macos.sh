#!/usr/bin/env bash
# setup_macos.sh — One-command setup for running snake-hrl on macOS
#
# Usage:
#   ./script/setup_macos.sh
#
# Prerequisites:
#   - Docker Desktop for Mac (https://www.docker.com/products/docker-desktop/)

set -euo pipefail

echo "=== Snake HRL — macOS Setup ==="
echo ""

# Check Docker is running
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found."
    echo "Install Docker Desktop for Mac: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "ERROR: Docker daemon not running. Start Docker Desktop first."
    exit 1
fi

echo "[1/3] Building image (first time takes ~10 min)..."
docker compose build

echo ""
echo "[2/3] Creating output directory..."
mkdir -p output data

echo ""
echo "[3/3] Running smoke test..."
docker compose run --rm snake-hrl python3 -c "
import torch, torchrl, pyelastica
print(f'  PyTorch:    {torch.__version__}')
print(f'  TorchRL:    {torchrl.__version__}')
print(f'  PyElastica: {pyelastica.__version__}')
print(f'  Device:     cpu')
print('  Smoke test PASSED')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start training:"
echo "  docker compose run --rm snake-hrl python -m locomotion_elastica.train"
echo ""
echo "Or interactively:"
echo "  docker compose run --rm snake-hrl bash"
