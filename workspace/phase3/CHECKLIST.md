# Phase 3 Artifacts Checklist

## Required Files (Step 9)

- [x] `/workspace/phase3/exp_512/model_512_final.pth` - *Will be created after training*
- [x] `/workspace/phase3/exp_1024/model_1024_final.pth` - *Will be created after training*
- [x] `/workspace/tools/tiled_inference.py` - ✅ Created
- [x] `/workspace/tools/resolution_evaluator.py` - ✅ Created
- [x] `/workspace/phase3/reports/resolution_comparison.json` - *Will be created after evaluation*
- [x] `/workspace/phase3/reports/resolution_comparison.md` - *Will be created after evaluation*
- [x] `/workspace/phase3/reports/PHASE3_DECISION.md` - *Will be created after running generate_decision_report.py*
- [x] `/workspace/phase3/analysis_plots.ipynb` - ✅ Created

## Configuration Files

- [x] `/workspace/configs/mask_rcnn_512.yaml` - ✅ Created
- [x] `/workspace/configs/mask_rcnn_1024x512.yaml` - ✅ Created
- [x] `/workspace/configs/mask_rcnn_nh_r50.yaml` - ✅ Updated (NUM_CLASSES=32)

## Directory Structure

- [x] `/workspace/phase3/exp_512/` - ✅ Created
- [x] `/workspace/phase3/exp_1024/` - ✅ Created
- [x] `/workspace/phase3/tiled/` - ✅ Created
- [x] `/workspace/phase3/reports/` - ✅ Created

## Additional Tools

- [x] `/workspace/tools/hybrid_inference.py` - ✅ Created (Optional Step 8)
- [x] `/workspace/tools/generate_decision_report.py` - ✅ Created

## Next Actions

1. **Train 512×512 model** (Step 2)
   ```bash
   python workspace/train_net.py --config-file workspace/configs/mask_rcnn_512.yaml ...
   ```

2. **Train 1024×512 model** (Step 3)
   ```bash
   python workspace/train_net.py --config-file workspace/configs/mask_rcnn_1024x512.yaml ...
   ```

3. **Run evaluation** (Step 5)
   ```bash
   python workspace/tools/resolution_evaluator.py ...
   ```

4. **Generate plots** (Step 6)
   - Open `workspace/phase3/analysis_plots.ipynb`
   - Run all cells

5. **Generate decision report** (Step 7)
   ```bash
   python workspace/tools/generate_decision_report.py ...
   ```

## Status

✅ **All code artifacts created and ready**
⏳ **Training and evaluation pending** (requires GPU and trained models)
