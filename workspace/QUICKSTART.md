# Phase 2 Quick Start Guide

## Prerequisites
- Python 3.10
- CUDA 11.7 (for GPU training)
- 8GB+ GPU memory (RTX 3060 or better)

## Quick Setup

```bash
# 1. Activate virtual environment
source .venv/bin/activate

# 2. Install dependencies (if not already done)
bash workspace/setup_phase2.sh

# OR manually:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install opencv-python pycocotools wandb tqdm matplotlib seaborn scikit-learn
```

## Training

```bash
# Register datasets
python workspace/register_dataset.py \
    --coco-json data/niihhaa/coco_annotations.json \
    --image-dir data/niihhaa/dataset

# Start training
python workspace/train_net.py \
    --config-file workspace/configs/mask_rcnn_nh_r50.yaml \
    --num-gpus 1 \
    --coco-json data/niihhaa/coco_annotations.json \
    --image-dir data/niihhaa/dataset \
    --wandb-project tooth-poc \
    --wandb-name maskrcnn_r50_nh
```

## Evaluation

```bash
python workspace/evaluate.py \
    --config-file workspace/configs/mask_rcnn_nh_r50.yaml \
    --model-path workspace/outputs/mask_rcnn_nh_r50/model_final.pth \
    --coco-json data/niihhaa/coco_annotations.json \
    --image-dir data/niihhaa/dataset
```

## Monitoring

- **Wandb Dashboard:** https://wandb.ai (check your project "tooth-poc")
- **Local Logs:** `workspace/outputs/mask_rcnn_nh_r50/`
- **Metrics:** `workspace/outputs/mask_rcnn_nh_r50/metrics.json`

## Troubleshooting

### GPU OOM (Out of Memory)
Edit `workspace/configs/mask_rcnn_nh_r50.yaml`:
```yaml
SOLVER:
  IMS_PER_BATCH: 1  # Reduce from 2 to 1
INPUT:
  MAX_SIZE_TRAIN: 512  # Reduce from 1024
```

### Dataset Not Found
Make sure to run `register_dataset.py` before training.

### Import Errors
Ensure Detectron2 is installed:
```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## Expected Training Time
- ~12,000 iterations
- ~2-4 hours on RTX 3060 (depending on batch size)
- Checkpoints saved every 2,000 iterations
