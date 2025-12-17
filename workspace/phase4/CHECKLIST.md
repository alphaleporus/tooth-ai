# Phase 4 Artifacts Checklist

## Required Files

### ROI Extraction
- [x] `/workspace/tools/extract_rois.py` - ✅ Created
- [ ] `/workspace/phase4/roi_dataset/train/` - *Will be created after ROI extraction*
- [ ] `/workspace/phase4/roi_dataset/val/` - *Will be created after ROI extraction*
- [ ] `/workspace/phase4/roi_dataset/train_labels.csv` - *Will be created after ROI extraction*
- [ ] `/workspace/phase4/roi_dataset/val_labels.csv` - *Will be created after ROI extraction*

### Classifier Training
- [x] `/workspace/classifier/train_effnet.py` - ✅ Created
- [ ] `/workspace/phase4/effnet_fdi_best.pth` - *Will be created after training*
- [ ] `/workspace/phase4/effnet_fdi_confusion_matrix.png` - *Will be created after training*

### Integrated Inference
- [x] `/workspace/tools/integrated_inference.py` - ✅ Created
- [ ] `/workspace/phase4/integrated_out/` - *Will contain test outputs*

### Evaluation
- [x] `/workspace/tools/evaluate_integrated.py` - ✅ Created
- [ ] `/workspace/phase4/reports/integrated_metrics.json` - *Will be created after evaluation*
- [ ] `/workspace/phase4/reports/integrated_metrics.md` - *Will be created after evaluation*

### Decision Report
- [x] `/workspace/tools/generate_phase4_report.py` - ✅ Created
- [ ] `/workspace/phase4/PHASE4_DECISION.md` - *Will be created after running report generator*

## Directory Structure

- [x] `/workspace/phase4/` - ✅ Created
- [x] `/workspace/phase4/roi_dataset/` - ✅ Created
- [x] `/workspace/phase4/roi_dataset/train/` - ✅ Created
- [x] `/workspace/phase4/roi_dataset/val/` - ✅ Created
- [x] `/workspace/phase4/reports/` - ✅ Created
- [x] `/workspace/phase4/integrated_out/` - ✅ Created
- [x] `/workspace/classifier/` - ✅ Created

## Execution Order

1. **Extract ROIs** (Step 2)
   ```bash
   python workspace/tools/extract_rois.py \
       --images data/niihhaa/dataset \
       --coco data/niihhaa/coco_annotations.json \
       --model workspace/phase3/exp_1024/model_1024_final.pth \
       --config workspace/configs/mask_rcnn_1024x512.yaml \
       --out workspace/phase4/roi_dataset
   ```

2. **Train EfficientNet** (Step 3)
   ```bash
   python workspace/classifier/train_effnet.py \
       --data workspace/phase4/roi_dataset \
       --epochs 35 \
       --batch-size 64 \
       --lr 1e-4 \
       --out workspace/phase4/effnet_fdi_best.pth
   ```

3. **Test Integrated Inference** (Step 4 - optional)
   ```bash
   python workspace/tools/integrated_inference.py \
       --image data/niihhaa/dataset/XXXX.png \
       --maskrcnn workspace/phase3/exp_1024/model_1024_final.pth \
       --effnet workspace/phase4/effnet_fdi_best.pth \
       --config workspace/configs/mask_rcnn_1024x512.yaml \
       --out workspace/phase4/integrated_out/test_output.png
   ```

4. **Evaluate System** (Step 5)
   ```bash
   python workspace/tools/evaluate_integrated.py \
       --images data/niihhaa/dataset \
       --coco data/niihhaa/coco_annotations.json \
       --maskrcnn workspace/phase3/exp_1024/model_1024_final.pth \
       --effnet workspace/phase4/effnet_fdi_best.pth \
       --config workspace/configs/mask_rcnn_1024x512.yaml \
       --out workspace/phase4/reports/
   ```

5. **Generate Decision Report** (Step 6)
   ```bash
   python workspace/tools/generate_phase4_report.py \
       --integrated-metrics workspace/phase4/reports/integrated_metrics.json \
       --output workspace/phase4/PHASE4_DECISION.md
   ```

## Status

✅ **All code artifacts created and ready**
⏳ **ROI extraction, training, and evaluation pending** (requires GPU and Phase 3 models)

## Dependencies

- Phase 3 trained models (512 or 1024)
- PyTorch with CUDA
- timm library for EfficientNet
- Detectron2
- OpenCV, NumPy, etc.

## Notes

- Use best-performing model from Phase 3 for ROI extraction
- Classifier training takes ~30-60 minutes on GPU
- Evaluation on 30 images takes ~10-15 minutes
- All scripts include error handling and progress logging
