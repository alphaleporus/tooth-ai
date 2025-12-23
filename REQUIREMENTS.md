# Tooth-AI: Requirements & Budget

## Project Summary
| Item | Details |
|------|---------|
| Goal | Dental diagnosis from OPG radiographs |
| Model | Mask R-CNN (ResNet-50 + FPN) |
| Framework | Detectron2 |
| Dataset | final-di (Roboflow) |

---

## Dataset Details

| Split | Images | Annotations |
|-------|--------|-------------|
| Train | 2,647 | 55,404 |
| Valid | 2,129 | 35,007 |
| Test | 2,216 | 30,239 |
| **Total** | **6,992** | **120,650** |

### Classes (41 total)
- **32 teeth**: Numbered 1-32
- **9 conditions**: Caries, Crown, Filling, Implant, Prefabricated metal post, Retained root, Root canal filling, Root canal obturation, t

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Iterations | **24,000** |
| Batch size | 4 |
| Learning rate | 0.001 |
| Epochs | ~36 |

### Computation Summary
| Metric | Value |
|--------|-------|
| Total image passes | 24,000 × 4 = **96,000** |
| Epochs over train set | ~36 |
| Total annotations processed | ~**2 million** |

---

## Data Augmentation (Online)

| Transform | Parameters |
|-----------|------------|
| Multi-scale resize | 480-608 px |
| Horizontal flip | 50% |
| Brightness/Contrast/Saturation | ±20% |
| Rotation | ±10° |

**Effective images**: ~100K+ unique views (no extra storage)

---

## Storage Requirements

| Item | Size | Cost (RunPod) |
|------|------|---------------|
| Dataset | 631 MB | ~₹5 |
| Model checkpoints (6×) | ~3 GB | ~₹25 |
| Logs & artifacts | ~500 MB | ~₹5 |
| **Total storage** | **~5 GB** | **~₹35** |

> RunPod storage: ~₹7/GB/month (volume storage)

---

## GPU Training Cost

| Item | Provider | Time/Size | Cost |
|------|----------|-----------|------|
| GPU compute | RunPod A6000 | 10-12 hrs | ₹300-360 |
| Storage | RunPod Volume | 5 GB | ₹35 |
| **Total** | | | **₹335-400** |

---

## Free Services

| Service | Purpose |
|---------|---------|
| WandB | Experiment tracking |
| GitHub | Code repository |
| Roboflow | Dataset source |

---

## Fine-tuning (If Needed)

| Scenario | Additional Cost |
|----------|-----------------|
| mAP < 70% → unfreeze backbone | +₹50-100 (2-3 hrs) |
| Domain shift → hospital data | +₹100-150 |

---

## Total Budget Summary

| Scenario | Cost |
|----------|------|
| **Training only** | **₹335-400** |
| + Fine-tuning | ₹400-500 |
| + Deployment (1 month) | ₹800-1,000 |

---

## Checklist

- [x] Dataset: 6,992 images, 41 classes
- [x] Storage: ~5 GB required
- [x] Config: 24K iterations, batch=4
- [x] Augmentation: Online (6 transforms)
- [x] Local test: Passed
- [x] WandB: Configured
- [ ] GPU training
- [ ] Evaluation (target: mAP@50 ≥ 70%)
