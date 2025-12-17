# Phase 3 - Resolution Impact Analysis Summary

## Overview
Phase 3 evaluates the impact of image resolution and tiling on Mask R-CNN performance for tooth detection. Three inference methods are compared: 512×512 direct, 1024×512 direct, and tiled 1024×1024.

## Files Created

### Configuration Files
1. **`workspace/configs/mask_rcnn_512.yaml`**
   - 512×512 training config
   - MAX_SIZE_TRAIN: 512
   - MAX_ITER: 9000
   - STEPS: (3000, 6000)

2. **`workspace/configs/mask_rcnn_1024x512.yaml`**
   - 1024×512 training config
   - MAX_SIZE_TRAIN: 1024
   - MAX_ITER: 12000
   - ROI_HEADS.BATCH_SIZE_PER_IMAGE: 128
   - NUM_WORKERS: 2

### Inference Scripts
3. **`workspace/tools/tiled_inference.py`**
   - Tiled inference for high-resolution images
   - Tile size: 1024×1024 with 256px overlap
   - Merges results using soft-NMS and centroid clustering
   - Supports upscaling if original images unavailable

4. **`workspace/tools/resolution_evaluator.py`**
   - Evaluates all three methods on validation set
   - Computes mask IoU, bbox mAP, and FDI accuracy
   - Generates comparison JSON and markdown reports

5. **`workspace/tools/hybrid_inference.py`** (Optional)
   - Hybrid strategy: selects method based on image size and confidence
   - Uses 1024×512 for images < 1500px width
   - Uses tiled inference for larger images or low confidence

### Analysis Tools
6. **`workspace/phase3/analysis_plots.ipynb`**
   - Jupyter notebook for visualization
   - Line plots, bar charts, histograms, box plots
   - Summary statistics table

7. **`workspace/tools/generate_decision_report.py`**
   - Auto-generates decision report from evaluation results
   - Answers key questions about method selection
   - Provides recommendations based on metrics

### Directory Structure
```
workspace/phase3/
├── exp_512/          # 512×512 model outputs
├── exp_1024/         # 1024×512 model outputs
├── tiled/            # Tiled inference outputs
├── reports/          # Evaluation reports
└── analysis_plots.ipynb
```

## Training Commands

### 512×512 Model
```bash
python workspace/train_net.py \
    --config-file workspace/configs/mask_rcnn_512.yaml \
    --num-gpus 1 \
    --coco-json data/niihhaa/coco_annotations.json \
    --image-dir data/niihhaa/dataset \
    --wandb-project tooth-poc \
    --wandb-name maskrcnn_r50_512
```

### 1024×512 Model
```bash
python workspace/train_net.py \
    --config-file workspace/configs/mask_rcnn_1024x512.yaml \
    --num-gpus 1 \
    --coco-json data/niihhaa/coco_annotations.json \
    --image-dir data/niihhaa/dataset \
    --wandb-project tooth-poc \
    --wandb-name maskrcnn_r50_1024x512
```

## Evaluation Command

```bash
python workspace/tools/resolution_evaluator.py \
    --model-512 workspace/phase3/exp_512/model_512_final.pth \
    --model-1024 workspace/phase3/exp_1024/model_1024_final.pth \
    --config-512 workspace/configs/mask_rcnn_512.yaml \
    --config-1024 workspace/configs/mask_rcnn_1024x512.yaml \
    --coco-json data/niihhaa/coco_annotations.json \
    --image-dir data/niihhaa/dataset \
    --output-dir workspace/phase3/reports \
    --num-samples 30
```

## Tiled Inference Command

```bash
python workspace/tools/tiled_inference.py \
    --weights workspace/phase3/exp_1024/model_1024_final.pth \
    --config workspace/configs/mask_rcnn_1024x512.yaml \
    --input data/niihhaa/dataset/XXXX.png \
    --output workspace/phase3/tiled/XXXX_tiled.png
```

## Expected Outputs

After running evaluation, the following files should exist:

1. **`workspace/phase3/exp_512/model_512_final.pth`** - 512×512 trained model
2. **`workspace/phase3/exp_1024/model_1024_final.pth`** - 1024×512 trained model
3. **`workspace/phase3/reports/resolution_comparison.json`** - Detailed metrics
4. **`workspace/phase3/reports/resolution_comparison.md`** - Human-readable report
5. **`workspace/phase3/reports/PHASE3_DECISION.md`** - Decision report (generated)

## Decision Report Generation

After evaluation completes, generate decision report:

```bash
python workspace/tools/generate_decision_report.py \
    --results-json workspace/phase3/reports/resolution_comparison.json \
    --output workspace/phase3/reports/PHASE3_DECISION.md
```

## Key Metrics Evaluated

1. **Mask IoU**: Intersection over Union for segmentation masks
2. **Bbox mAP@0.5**: Mean Average Precision for bounding boxes at IoU=0.5
3. **FDI Accuracy**: Classification accuracy for FDI tooth notation
4. **Per-image comparisons**: Winner distribution across methods

## Analysis Questions

The decision report answers:
1. Which method should be used for final model?
2. Does tiled inference significantly improve IoU (>0.02)?
3. Does FDI accuracy improve without noise?
4. Are improvements worth GPU cost?

## Next Steps

1. Train both models (512 and 1024)
2. Run evaluation on 30 validation images
3. Generate plots using analysis_plots.ipynb
4. Generate decision report
5. Implement recommended method in Phase 4

## Notes

- NUM_CLASSES corrected to 32 (background is implicit in Detectron2)
- Tiled inference uses soft-NMS and centroid clustering for merging
- Hybrid strategy available for production deployment
- All scripts support both original high-res and upscaled images



