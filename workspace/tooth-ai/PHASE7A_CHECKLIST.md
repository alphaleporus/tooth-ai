# Phase 7A Completion Checklist

## Step 7 Required Files

- [x] `/workspace/tooth-ai/Dockerfile` - ✅ Created
- [ ] `/workspace/tooth-ai/export/maskrcnn.onnx` - *Will be created after ONNX export*
- [ ] `/workspace/tooth-ai/export/effnet.onnx` - *Will be created after ONNX export*
- [ ] `/workspace/tooth-ai/export/maskrcnn_trt_fp16.plan` - *Will be created after TensorRT conversion*
- [ ] `/workspace/tooth-ai/export/effnet_trt_fp16.plan` - *Will be created after TensorRT conversion*
- [ ] `/workspace/tooth-ai/export/benchmarks/latency.json` - *Will be created after benchmarking*
- [x] `/.github/workflows/ci.yml` - ✅ Created
- [x] `/workspace/tooth-ai/repro/EXPERIMENT_MANIFEST.json` - ✅ Created
- [x] `/workspace/tooth-ai/demo.sh` - ✅ Created
- [x] `/workspace/tooth-ai/release_notes_v0.1_poc.md` - ✅ Created

## Additional Files Created

### Docker & Deployment
- [x] `Dockerfile` - ✅ Created
- [x] `docker-compose.yml` - ✅ Created
- [x] `demo.sh` - ✅ Created
- [x] `ui/requirements.txt` - ✅ Created

### Model Optimization
- [x] `export/export_all_to_onnx.py` - ✅ Created
- [x] `export/convert_to_tensorrt.sh` - ✅ Created
- [x] `export/benchmark_inference.py` - ✅ Created

### CI/CD
- [x] `.github/workflows/ci.yml` - ✅ Created

### Reproducibility
- [x] `repro/EXPERIMENT_MANIFEST.json` - ✅ Created
- [x] `repro/docker-compose.yml` - ✅ Created
- [x] `repro/reproduce_metrics.sh` - ✅ Created
- [x] `repro/requirements.txt` - ✅ Created

### Documentation
- [x] `release_notes_v0.1_poc.md` - ✅ Created
- [x] `PHASE7A_SUMMARY.md` - ✅ Created

## Execution Status

✅ **All code artifacts created**
⏳ **ONNX exports pending** (requires trained models)
⏳ **TensorRT engines pending** (requires ONNX + TensorRT)
⏳ **Benchmarks pending** (requires models)

## Next Actions

1. **Copy trained models** to `workspace/tooth-ai/models/`:
   ```bash
   cp workspace/phase3/exp_1024/model_1024_final.pth workspace/tooth-ai/models/maskrcnn_final.pth
   cp workspace/phase4/effnet_fdi_best.pth workspace/tooth-ai/models/effnet_fdi_final.pth
   ```

2. **Build Docker image**:
   ```bash
   cd workspace/tooth-ai
   docker build -t tooth-ai:poc . > docker_build.log 2>&1
   ```

3. **Export to ONNX**:
   ```bash
   python export/export_all_to_onnx.py \
       --maskrcnn models/maskrcnn_final.pth \
       --effnet models/effnet_fdi_final.pth \
       --config models/config.yaml \
       --out export/
   ```

4. **Convert to TensorRT** (if TensorRT available):
   ```bash
   bash export/convert_to_tensorrt.sh
   ```

5. **Run benchmarks**:
   ```bash
   python export/benchmark_inference.py \
       --images data/niihhaa/dataset \
       --maskrcnn models/maskrcnn_final.pth \
       --effnet models/effnet_fdi_final.pth \
       --config models/config.yaml \
       --onnx-effnet export/effnet.onnx \
       --onnx-maskrcnn export/maskrcnn.onnx \
       --out export/benchmarks/
   ```

6. **Test demo**:
   ```bash
   ./demo.sh
   ```

## Notes

- ONNX and TensorRT files will be generated when export scripts are run
- Benchmark results require trained models
- Docker build requires models to be in `models/` directory
- CI workflow will run on GitHub Actions when pushed to repository
