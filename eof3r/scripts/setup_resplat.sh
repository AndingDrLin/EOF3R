#!/bin/bash
# Setup script for ReSplat environment
#
# ReSplat requires Python 3.12 + PyTorch 2.7.0 + CUDA 12.8
# This is separate from the main eof3r environment.
#
# Usage:
#     bash eof3r/scripts/setup_resplat.sh
#
# After setup:
#     conda activate resplat
#     python eof3r/scripts/train/train_phase_b.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESPLAT_DIR="$PROJECT_ROOT/baselines/resplat"

echo "=== ReSplat Environment Setup ==="
echo "Project root: $PROJECT_ROOT"
echo "ReSplat dir: $RESPLAT_DIR"
echo ""

# Check if ReSplat exists
if [ ! -d "$RESPLAT_DIR" ]; then
    echo "ERROR: ReSplat not found at $RESPLAT_DIR"
    echo "Clone: git clone https://github.com/cvg/ReSplat baselines/resplat"
    exit 1
fi

# Check CUDA version
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "unknown")
echo "CUDA version: $CUDA_VERSION"

# Create conda environment
echo ""
echo "=== Creating conda environment: resplat ==="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n resplat python=3.12
conda activate resplat

# Install PyTorch (CUDA 12.8)
echo ""
echo "=== Installing PyTorch ==="
pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

# Install ReSplat requirements
echo ""
echo "=== Installing ReSplat dependencies ==="
pip install -r "$RESPLAT_DIR/requirements.txt"

# Install gsplat (required for rendering)
echo ""
echo "=== Installing gsplat ==="
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3

# Install pointops (kNN operations)
echo ""
echo "=== Installing pointops ==="
cd "$RESPLAT_DIR/src/model/encoder/pointops"
python setup.py install
cd "$PROJECT_ROOT"

# Install eof3r training module (in development mode)
echo ""
echo "=== Installing eof3r training module ==="
pip install -e "$PROJECT_ROOT"

# Verify installation
echo ""
echo "=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')

import gsplat
print(f'gsplat: {gsplat.__version__}')

print('ReSplat environment ready!')
"

echo ""
echo "=== Setup Complete ==="
echo "Activate: conda activate resplat"
echo "Train:    python eof3r/scripts/train/train_phase_b.py"
