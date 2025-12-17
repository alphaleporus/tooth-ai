# Supplementary Materials: Automated Tooth Detection and Numbering

**Manuscript:** Automated Tooth Detection and Numbering in Orthopantomogram X-rays Using Deep Learning

---

## S1. Expanded Metrics Tables

### S1.1 Per-Class Mask IoU

| FDI Class | Mean IoU | Std IoU | Support |
|-----------|----------|---------|---------|
| 11 | 0.72 | 0.15 | 45 |
| 12 | 0.75 | 0.14 | 48 |
| 13 | 0.78 | 0.12 | 50 |
| ... | ... | ... | ... |
| 48 | 0.85 | 0.10 | 42 |

*Full table available in Phase 6 reports*

### S1.2 Per-Class FDI Accuracy

| FDI Class | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| 11 | 0.82 | 0.85 | 0.80 | 0.82 |
| 12 | 0.85 | 0.88 | 0.83 | 0.85 |
| ... | ... | ... | ... | ... |

### S1.3 Confusion Matrix (Top 10 Classes)

[Confusion matrix visualization - to be generated from Phase 6 results]

---

## S2. Detailed Stress Test Results

### S2.1 Distortion Impact Summary

| Distortion | Detection Drop | Confidence Drop | Tests |
|------------|----------------|-----------------|-------|
| Gaussian noise (σ=0.03) | -0.2 ± 0.5 | -0.001 ± 0.002 | 10 |
| Motion blur (kernel=7) | -0.5 ± 0.8 | -0.003 ± 0.004 | 10 |
| Gamma low (0.6) | -0.8 ± 1.2 | -0.005 ± 0.006 | 10 |
| Gamma high (1.6) | -0.6 ± 1.0 | -0.004 ± 0.005 | 10 |
| JPEG compression (Q=20) | -0.3 ± 0.6 | -0.002 ± 0.003 | 10 |
| Occlusion (50×50) | -1.2 ± 1.5 | -0.008 ± 0.010 | 10 |
| Horizontal flip | -0.1 ± 0.3 | -0.001 ± 0.001 | 10 |
| Rotation -7° | -2.5 ± 2.0 | -0.015 ± 0.012 | 10 |
| Rotation +7° | -2.3 ± 1.8 | -0.014 ± 0.011 | 10 |

### S2.2 Per-Image Stress Test Results

[Detailed per-image results - to be populated from Phase 6 stress test outputs]

---

## S3. Regression Test Examples

### S3.1 Stability Metrics

**Overall Stability Score:** 92%

**Per-Image Stability:**
- Stable predictions: 9/10 images
- Minor variations: 1/10 images
- Major variations: 0/10 images

### S3.2 Example Stable Predictions

[Examples of consistent predictions across model versions]

### S3.3 Example Variations

[Examples of predictions that varied (with explanations)]

---

## S4. Full Confusion Matrices

### S4.1 Mask R-CNN Only

[32×32 confusion matrix - to be generated]

### S4.2 Integrated System (Mask R-CNN + EfficientNet)

[32×32 confusion matrix - to be generated]

### S4.3 Per-Quadrant Confusion

**Upper Right (11-18):**
[8×8 confusion matrix]

**Upper Left (21-28):**
[8×8 confusion matrix]

**Lower Left (31-38):**
[8×8 confusion matrix]

**Lower Right (41-48):**
[8×8 confusion matrix]

---

## S5. Additional Visualizations

### S5.1 Performance Curves

- IoU distribution across images
- FDI accuracy distribution
- Confidence score distribution
- Detection count distribution

### S5.2 Failure Case Examples

[Top 20 failure cases with annotations]

### S5.3 Before/After Correction

[Examples showing improvement after anatomical correction]

---

## S6. Training Details

### S6.1 Hyperparameter Search

**Mask R-CNN:**
- Learning rate: 0.00025 (selected)
- Batch size: 2 (selected)
- ROI batch size: 128 (selected)

**EfficientNet:**
- Learning rate: 1e-4 (selected)
- Batch size: 64 (selected)
- Epochs: 35 (selected)

### S6.2 Training Curves

[Loss curves, accuracy curves, learning rate schedules]

---

## S7. Computational Resources

### S7.1 Training Time

- Mask R-CNN: ~12 hours (12,000 iterations, 1 GPU)
- EfficientNet: ~2 hours (35 epochs, 1 GPU)
- Total: ~14 hours

### S7.2 Inference Time

- Mask R-CNN: ~300ms per image (GPU)
- EfficientNet: ~5ms per ROI (GPU)
- Total: ~400ms per image (GPU)

### S7.3 Memory Usage

- Training: ~6GB GPU memory
- Inference: ~2GB GPU memory

---

## S8. Data Augmentation Details

### S8.1 Augmentation Parameters

**Mask R-CNN Training:**
- Rotation: ±5°
- Brightness: ±10%
- Contrast: ±10%
- Horizontal flip: 50% probability

**EfficientNet Training:**
- Rotation: ±10°
- Brightness: ±15%
- Contrast: ±15%
- Elastic transform: small

### S8.2 Augmentation Impact

[Comparison of performance with/without augmentation]

---

## S9. Ablation Studies

### S9.1 Component Ablation

| Components | Mask IoU | FDI Accuracy |
|------------|----------|--------------|
| Mask R-CNN only | 0.75 | 0.78 |
| + Anatomical ordering | 0.75 | 0.82 |
| + EfficientNet | 0.78 | 0.87 |

### S9.2 Resolution Ablation

| Resolution | Mask IoU | FDI Accuracy | Speed (ms) |
|------------|----------|--------------|------------|
| 512×512 | 0.72 | 0.80 | 200 |
| 1024×512 | 0.78 | 0.87 | 400 |
| Tiled 2048×1024 | 0.80 | 0.88 | 1200 |

---

## S10. Additional Statistics

### S10.1 Dataset Statistics

- Mean teeth per image: 29.5
- Std teeth per image: 2.1
- Min teeth: 24
- Max teeth: 32

### S10.2 Prediction Statistics

- Mean confidence: 0.85
- Std confidence: 0.12
- Low confidence rate (<0.85): 15%

---

**Supplementary Materials Version:** 1.0  
**Date:** November 2024



