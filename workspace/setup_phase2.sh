#!/bin/bash
# Setup script for Phase 2 - Detectron2 training environment

set -e

echo "Setting up Phase 2 - Mask R-CNN Training Environment"
echo "=================================================="

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
    source .venv/bin/activate
fi

# Install/upgrade base dependencies
echo "Installing base dependencies..."
pip install --upgrade pip setuptools wheel

# Install PyTorch (check if CUDA is available)
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
else
    echo "No CUDA detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio
fi

# Install Detectron2
echo "Installing Detectron2..."
pip install 'git+https://github.com/facebookresearch/detectron2.git'

# Install other dependencies
echo "Installing additional dependencies..."
pip install opencv-python pycocotools wandb tqdm matplotlib seaborn scikit-learn

# Create necessary directories
echo "Creating directory structure..."
mkdir -p workspace/outputs/mask_rcnn_nh_r50
mkdir -p workspace/configs
mkdir -p workspace/qc

# Register datasets
echo "Registering datasets..."
python workspace/register_dataset.py \
    --coco-json data/niihhaa/coco_annotations.json \
    --image-dir data/niihhaa/dataset \
    --split 0.8

echo ""
echo "Setup complete!"
echo ""
echo "To start training, run:"
echo "  python workspace/train_net.py --config-file workspace/configs/mask_rcnn_nh_r50.yaml --num-gpus 1 \\"
echo "    --coco-json data/niihhaa/coco_annotations.json --image-dir data/niihhaa/dataset"
echo ""



