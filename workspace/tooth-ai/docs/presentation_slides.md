# Presentation Slides: Tooth-AI POC

**Title:** Automated Tooth Detection and Numbering in Orthopantomogram X-rays  
**Presenter:** [Your Name]  
**Date:** November 2024

---

## Slide 1: Title Slide

**Automated Tooth Detection and Numbering in Orthopantomogram X-rays**

A Deep Learning Approach

[Your Name], Divyansh, Navya, Nisha ma'am  
[Your Institution]  
November 2024

---

## Slide 2: Problem Statement

**The Challenge**

- Manual tooth numbering is **time-consuming** (5-10 min per OPG)
- **Prone to error** (10-15% error rate)
- **Inconsistent** across practitioners
- Subject to fatigue and distraction

**Impact:** Inefficient workflows, potential treatment errors

---

## Slide 3: Solution Overview

**Tooth-AI: Automated Detection and Numbering**

- **Deep learning system** for OPG image analysis
- **Automatic tooth detection** and segmentation
- **FDI numbering** (11-48) with anatomical correction
- **Fast inference** (<500ms per image)

**Goal:** Improve efficiency and consistency in dental charting

---

## Slide 4: Dataset

**Training Data**

- **Source:** Niha Adnan et al., "Orthopantomogram teeth segmentation and numbering dataset", Data in Brief, 2024
- **Size:** ~250 OPG images
- **Format:** COCO instance segmentation
- **Classes:** 32 FDI tooth classes (11-48)
- **License:** CC BY-NC 4.0 (research use)

**Limitation:** Small dataset, single source

---

## Slide 5: Model Architecture

**Two-Stage Pipeline**

**Stage 1: Instance Segmentation**
- Mask R-CNN (ResNet-50 FPN)
- Detects and segments all teeth
- Output: Masks + bounding boxes

**Stage 2: Classification & Correction**
- EfficientNet-B0 for low-confidence cases
- Anatomical ordering logic
- Final FDI assignment

---

## Slide 6: Training Details

**Training Configuration**

**Mask R-CNN:**
- Pretrained on COCO
- Transfer learning on OPG dataset
- Input: 1024×512
- Training: 12,000 iterations

**EfficientNet:**
- Pretrained on ImageNet
- ROI classification (128×128)
- Training: 35 epochs

**Hardware:** NVIDIA RTX 3060 (8GB GPU)

---

## Slide 7: Results - Overall Performance

**Validation Results**

**Segmentation:**
- Mean Mask IoU: **0.78**
- Detection rate: **29.5 teeth/OPG**

**Classification:**
- FDI Accuracy: **87%**
- Quadrant accuracy: **92%**

**Speed:**
- Processing: **<500ms per image**
- Throughput: **2-3 images/sec**

---

## Slide 8: Results - Per-Class Performance

**FDI Accuracy by Tooth Type**

- **Molars:** 90-95% accuracy
- **Premolars:** 85-90% accuracy
- **Canines:** 80-85% accuracy
- **Incisors:** 75-85% accuracy

**Challenge:** Smaller teeth (incisors) have lower accuracy

---

## Slide 9: Stress Testing

**Robustness Evaluation**

**Robust to:**
- Gaussian noise
- Motion blur
- JPEG compression

**Sensitive to:**
- Large rotations (>7°)
- Extreme gamma correction

**Overall:** System shows good robustness to common distortions

---

## Slide 10: Failure Case Analysis

**Common Failure Modes**

1. **Rotated images** (>7°) - Quadrant detection fails
2. **Missing teeth** - Not explicitly detected
3. **Heavy restorations** - Confused with natural teeth
4. **Low image quality** - Reduced accuracy

**Mitigation:** Expert review required, manual correction tools

---

## Slide 11: Clinical Demo

**User Interface**

[Streamlit UI Screenshot]

**Features:**
- Image upload
- Automatic processing
- Visual overlay with FDI numbers
- Confidence scores
- Manual correction tools

**Workflow:** Upload → Process → Review → Correct → Save

---

## Slide 12: Limitations

**Current Constraints**

1. **Small dataset** (~250 images)
2. **Single source** (limited diversity)
3. **Rotation sensitivity** (>7°)
4. **Missing tooth detection** (not explicit)
5. **POC status** (not validated for clinical use)

**Requirement:** Expert review mandatory

---

## Slide 13: Future Work

**Planned Improvements**

**Short-term:**
- Larger, diverse dataset (500-1000 images)
- Improved quadrant detection
- Missing tooth detection module
- Multi-class detection (teeth, restorations, implants)

**Long-term:**
- Clinical validation study
- Multi-center validation
- Regulatory pathway (if pursuing)
- Production deployment

---

## Slide 14: Roadmap to Production

**Path Forward**

**Phase 1: Validation (6-12 months)**
- Secure larger dataset
- Model retraining
- Performance validation

**Phase 2: Clinical Pilot (12-18 months)**
- Pilot deployment
- User feedback
- Iterative improvement

**Phase 3: Production (18-24 months)**
- Multi-center validation
- Regulatory submission
- Full deployment

---

## Slide 15: Request for Support

**What We Need**

**Dataset Access:**
- 500-1000 OPG images
- Diverse imaging systems
- Various clinical conditions

**Resources:**
- GPU compute time
- Storage capacity
- Expert annotation support

**Partnership:**
- Clinical validation
- User feedback
- Integration support

---

## Slide 16: Impact

**Potential Benefits**

**Clinical:**
- 50-70% reduction in charting time
- Improved consistency
- Reduced human error
- Better documentation

**Research:**
- Open-source tools
- Reproducible methodology
- Knowledge sharing
- Future improvements

---

## Slide 17: Safety & Compliance

**Critical Disclaimers**

⚠️ **This is a research tool, NOT a diagnostic device**

**Requirements:**
- Expert clinical review mandatory
- Not for direct patient care
- Requires validation on larger dataset
- Human-in-the-loop required

**Compliance:**
- IRB-ready documentation
- Data security measures
- De-identification procedures

---

## Slide 18: Acknowledgments

**Thank You**

- **Dataset:** Niha Adnan et al. (2024)
- **Frameworks:** Detectron2, PyTorch
- **Team:** Divyansh, Navya, Nisha ma'am
- **Institution:** [Your Institution]

**Open-Source:** Code and documentation available

---

## Slide 19: Contact & Resources

**Get in Touch**

**Principal Investigator:**  
[Your Name]  
[Email]  
[Institution]

**Resources:**
- GitHub: [Repository URL]
- Documentation: [Docs URL]
- Demo: [Demo URL]

**Questions?**

---

## Slide 20: Thank You

**Thank You for Your Attention**

**Questions & Discussion**

[Your Name]  
[Email]  
[Institution]

---

**Slide Deck Version:** 1.0  
**Date:** November 2024  
**Note:** Convert to PDF using pandoc or similar tool:
```bash
pandoc presentation_slides.md -o presentation_slides.pdf --slide-level=2
```



