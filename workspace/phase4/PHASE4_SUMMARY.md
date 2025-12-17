# Phase 4 - ROI Classifier & Integrated FDI Correction Summary

## Overview
Phase 4 improves tooth numbering reliability by training an EfficientNet-B0 ROI classifier and integrating it with Mask R-CNN predictions using anatomical correction logic.

## Files Created

### ROI Extraction
1. **`workspace/tools/extract_rois.py`**
   - Extracts ROI crops from Mask R-CNN predictions
   - Crops each tooth instance to 128×128
   - Creates train/val split with labels CSV

### Classifier Training
2. **`workspace/classifier/train_effnet.py`**
   - Trains EfficientNet-B0 for 32-class FDI classification
   - Handles class imbalance with weighted loss
   - Uses augmentations (rotation, brightness, contrast)
   - Saves best checkpoint and confusion matrix

### Integrated Inference
3. **`workspace/tools/integrated_inference.py`**
   - Combines Mask R-CNN + EfficientNet + Anatomical correction
   - Uses classifier for low-confidence predictions (< 0.85)
   - Applies quadrant-based sorting and FDI mapping
   - Generates visualization and JSON output

### Evaluation
4. **`workspace/tools/evaluate_integrated.py`**
   - Evaluates integrated system on validation set
   - Computes FDI accuracy before/after correction
   - Checks for consistency violations
   - Generates detailed metrics report

### Report Generation
5. **`workspace/tools/generate_phase4_report.py`**
   - Generates comprehensive Phase 4 decision report
   - Analyzes error types and improvements
   - Provides recommendations for Phase 5

## Directory Structure

```
workspace/phase4/
├── roi_dataset/
│   ├── train/          # Training ROI crops
│   ├── val/            # Validation ROI crops
│   ├── train_labels.csv
│   └── val_labels.csv
├── reports/            # Evaluation reports
├── integrated_out/     # Integrated inference outputs
└── PHASE4_SUMMARY.md
```

## Workflow

### Step 1: Extract ROIs
```bash
python workspace/tools/extract_rois.py \
    --images data/niihhaa/dataset \
    --coco data/niihhaa/coco_annotations.json \
    --model workspace/phase3/exp_1024/model_1024_final.pth \
    --config workspace/configs/mask_rcnn_1024x512.yaml \
    --out workspace/phase4/roi_dataset
```

### Step 2: Train EfficientNet Classifier
```bash
python workspace/classifier/train_effnet.py \
    --data workspace/phase4/roi_dataset \
    --epochs 35 \
    --batch-size 64 \
    --lr 1e-4 \
    --out workspace/phase4/effnet_fdi_best.pth \
    --wandb-project tooth-poc \
    --wandb-name effnet_roi_fdi
```

### Step 3: Run Integrated Inference (Test)
```bash
python workspace/tools/integrated_inference.py \
    --image data/niihhaa/dataset/XXXX.png \
    --maskrcnn workspace/phase3/exp_1024/model_1024_final.pth \
    --effnet workspace/phase4/effnet_fdi_best.pth \
    --config workspace/configs/mask_rcnn_1024x512.yaml \
    --out workspace/phase4/integrated_out/XXXX_output.png
```

### Step 4: Evaluate Integrated System
```bash
python workspace/tools/evaluate_integrated.py \
    --images data/niihhaa/dataset \
    --coco data/niihhaa/coco_annotations.json \
    --maskrcnn workspace/phase3/exp_1024/model_1024_final.pth \
    --effnet workspace/phase4/effnet_fdi_best.pth \
    --config workspace/configs/mask_rcnn_1024x512.yaml \
    --out workspace/phase4/reports/ \
    --num-samples 30
```

### Step 5: Generate Decision Report
```bash
python workspace/tools/generate_phase4_report.py \
    --integrated-metrics workspace/phase4/reports/integrated_metrics.json \
    --output workspace/phase4/PHASE4_DECISION.md
```

## Key Features

### ROI Extraction
- Automatic cropping from Mask R-CNN masks/bboxes
- 128×128 resize with padding
- Ground truth matching for labels
- Train/val split (80/20)

### EfficientNet Classifier
- EfficientNet-B0 architecture (timm)
- 32-class classification (FDI 1-32)
- Class-weighted loss
- Data augmentations (rotation, brightness, contrast)
- Cosine learning rate schedule

### Integrated Inference
- Confidence-based method selection
- Anatomical quadrant detection
- Geometric sorting and FDI mapping
- Conflict resolution logic

### Evaluation Metrics
- Mask IoU (segmentation quality)
- Bbox mAP@0.5 (detection quality)
- FDI accuracy (before/after correction)
- Consistency checks (duplicates, quadrant violations, swapped neighbors)

## Expected Outputs

After completing all steps:

1. **`workspace/phase4/roi_dataset/`** - ROI crops and labels
2. **`workspace/phase4/effnet_fdi_best.pth`** - Trained classifier
3. **`workspace/phase4/reports/integrated_metrics.json`** - Evaluation metrics
4. **`workspace/phase4/reports/integrated_metrics.md`** - Human-readable report
5. **`workspace/phase4/PHASE4_DECISION.md`** - Decision report

## Next Steps

1. Extract ROIs from best Phase 3 model
2. Train EfficientNet classifier
3. Run integrated inference on validation set
4. Evaluate and generate decision report
5. Proceed to Phase 5 with recommended unified pipeline

## Notes

- Use best-performing resolution from Phase 3 for ROI extraction
- Classifier improves FDI accuracy for low-confidence predictions
- Anatomical correction handles geometric ordering
- Integrated system provides better reliability at cost of computation



