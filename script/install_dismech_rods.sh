#!/usr/bin/env bash
# Install dismech-rods (C++ DER simulation with pybind11 bindings)
#
# Prerequisites: conda environment active, cmake, g++
# Result: py_dismech Python module available for import

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Installing dismech-rods dependencies ==="

# 1. System dependencies
echo "--- Installing system packages ---"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    libeigen3-dev \
    libccd-dev \
    libfcl-dev \
    freeglut3-dev \
    mesa-common-dev \
    libglu1-mesa-dev

# 2. Conda dependencies (SymEngine and MKL)
echo "--- Installing conda packages ---"
conda install -y -c conda-forge symengine mkl mkl-devel

# 3. pybind11
echo "--- Installing pybind11 ---"
pip install pybind11

# 4. Clone dismech-rods
echo "--- Cloning dismech-rods ---"
cd "$PROJECT_DIR"
if [ -d "dismech-rods" ]; then
    echo "dismech-rods directory already exists, pulling latest..."
    cd dismech-rods && git pull && cd ..
else
    git clone https://github.com/StructuresComp/dismech-rods.git
fi

# 5. Build C++ library
echo "--- Building dismech-rods ---"
cd dismech-rods
mkdir -p build
cd build
cmake ..
make -j"$(nproc)"
cd ..

# 6. Install Python bindings
echo "--- Installing Python bindings ---"
pip install -e .

cd "$PROJECT_DIR"

# 7. Verify installation
echo "=== Verifying installation ==="
python -c "import py_dismech; print('py_dismech imported successfully')"

echo "=== dismech-rods installation complete ==="
