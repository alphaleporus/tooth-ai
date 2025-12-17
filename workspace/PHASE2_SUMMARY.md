# Phase 2 - Baseline Mask R-CNN Training Setup

## Overview
Setup for training Mask R-CNN (ResNet50-FPN) on Niihhaa dataset using Detectron2 framework.

## Files Created

### 1. Configuration
- **`workspace/configs/mask_rcnn_nh_r50.yaml`**
  - Detectron2 config for Mask R-CNN R50-FPN
  - 33 classes (32 teeth + background)
  - Input size: 512-1024px
  - Batch size: 2
  - Learning rate: 0.00025
  - Max iterations: 12,000

### 2. Dataset Registration
- **`workspace/register_dataset.py`**
  - Registers COCO dataset with Detectron2
  - Handles train/val split (80/20 default)
  - Maps categories to 32 tooth classes (FDI notation)

### 3. Training Script
- **`workspace/train_net.py`**
  - Main training script with wandb integration
  - Supports single/multi-GPU training
  - Automatic dataset registration
  - Mixed precision (AMP) support

### 4. Evaluation Script
- **`workspace/evaluate.py`**
  - Computes mAP@0.5 and mask IoU
  - Per-class metrics and confusion matrix
  - Saves metrics to JSON

### 5. Setup Script
- **`workspace/setup_phase2.sh`**
  - Installs all dependencies
  - Sets up directory structure
  - Registers datasets

## Training Command

```bash
# Activate environment
source .venv/bin/activate

# Register datasets (if not already done)
python workspace/register_dataset.py \
    --coco-json data/niihhaa/coco_annotations.json \
    --image-dir data/niihhaa/dataset \
    --split 0.8

# Start training
python workspace/train_net.py \
    --config-file workspace/configs/mask_rcnn_nh_r50.yaml \
    --num-gpus 1 \
    --coco-json data/niihhaa/coco_annotations.json \
    --image-dir data/niihhaa/dataset \
    --wandb-project tooth-poc \
    --wandb-name maskrcnn_r50_nh
```

## Evaluation Command

```bash
python workspace/evaluate.py \
    --config-file workspace/configs/mask_rcnn_nh_r50.yaml \
    --model-path workspace/outputs/mask_rcnn_nh_r50/model_final.pth \
    --coco-json data/niihhaa/coco_annotations.json \
    --image-dir data/niihhaa/dataset \
    --output-dir workspace/outputs/mask_rcnn_nh_r50
```

## Configuration Details

### Model Architecture
- **Backbone:** ResNet-50 with FPN
- **Head:** Standard ROI Heads
- **Mask Head:** 4-layer conv upsample
- **Pretrained:** COCO Instance Segmentation

### Training Settings
- **Batch Size:** 2 images/batch
- **Learning Rate:** 0.00025 (with warmup)
- **Optimizer:** SGD with momentum 0.9
- **Weight Decay:** 0.0001
- **LR Schedule:** Step decay at 6k and 9k iterations
- **Max Iterations:** 12,000
- **Evaluation Period:** Every 1,000 iterations
- **Checkpoint Period:** Every 2,000 iterations

### Data Augmentation
- Resize shortest edge (512-1024px)
- Random horizontal flip
- Random brightness
- Random contrast

### Input/Output
- **Input Size:** 512-1024px (maintains aspect ratio)
- **Output Directory:** `workspace/outputs/mask_rcnn_nh_r50`
- **Logs:** Saved to output directory + wandb

## Early Stopping Rules

- If validation mAP does not improve for 6 consecutive evaluations:
  - Reduce learning rate by factor of 10
  - Or stop training if already at minimum LR

## Memory Optimization

If GPU OOM occurs:
1. Reduce `IMS_PER_BATCH` to 1
2. Reduce `MAX_SIZE_TRAIN` to 512
3. Use gradient accumulation (modify training loop)
4. Reduce `NUM_WORKERS` to 2

## Expected Outputs

### Training
- Checkpoints: `workspace/outputs/mask_rcnn_nh_r50/model_*.pth`
- Final model: `workspace/outputs/mask_rcnn_nh_r50/model_final.pth`
- Training logs: `workspace/outputs/mask_rcnn_nh_r50/metrics.json`
- Wandb logs: Online dashboard

### Evaluation
- Metrics JSON: `workspace/outputs/mask_rcnn_nh_r50/metrics.json`
  - bbox mAP@0.5
  - segm mAP@0.5
  - mask mean IoU per class
  - Per-class precision/recall/F1
  - Confusion matrix

## Notes

1. **FDI Label Mapping:** Current dataset has generic labels (molar, premolar, etc.). 
   The 32-class mapping needs to be implemented based on actual FDI tooth numbers (11-48).
   For now, the system uses placeholder class names (tooth_1 to tooth_32).

2. **Dataset Split:** 80% train, 20% validation (configurable via `--split`)

3. **Wandb Integration:** Automatically logs:
   - Losses (every 20 iterations)
   - Learning rate
   - Checkpoint info
   - Config parameters

4. **Mixed Precision:** Detectron2 automatically uses AMP if available (PyTorch 1.6+)

## Next Steps

1. Run setup script: `bash workspace/setup_phase2.sh`
2. Verify dataset registration
3. Start training with command above
4. Monitor via wandb dashboard
5. Evaluate after training completes



