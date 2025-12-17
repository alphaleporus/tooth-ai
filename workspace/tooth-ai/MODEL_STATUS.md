# Model Status

**Date:** November 2024

## Current Status

**Models Required:**
- `maskrcnn_final.pth` - ❌ Not found (needs to be trained in Phase 3)
- `effnet_fdi_final.pth` - ❌ Not found (needs to be trained in Phase 4)
- `config.yaml` - ✅ Exists

## Expected Behavior

For a complete POC, models need to be trained first:
1. **Phase 3:** Train Mask R-CNN → `workspace/phase3/exp_1024/model_1024_final.pth`
2. **Phase 4:** Train EfficientNet → `workspace/phase4/effnet_fdi_best.pth`
3. **Copy to unified directory:**
   ```bash
   cp workspace/phase3/exp_1024/model_1024_final.pth workspace/tooth-ai/models/maskrcnn_final.pth
   cp workspace/phase4/effnet_fdi_best.pth workspace/tooth-ai/models/effnet_fdi_final.pth
   ```

## Impact on Validation

**Current Validation Score:** 45/100
- Code structure: ✅ Complete
- Models: ❌ Missing (expected)
- Dependencies: ⚠️ Not installed (can be installed)

**With Models:** Expected 75-85/100
**With Full Setup:** Expected 90-95/100

## Next Steps

1. Train models in Phase 3 and Phase 4
2. Copy trained models to `workspace/tooth-ai/models/`
3. Run ONNX export
4. Run benchmarks
5. Re-run validation
