# Data Access Request: Tooth-AI POC

**Project:** Automated Tooth Detection and Numbering in Orthopantomogram X-rays  
**Requesting Institution:** [Your Institution]  
**Principal Investigator:** [Your Name]  
**Date:** November 2024

---

## 1. Executive Summary

We are requesting access to a larger dataset of orthopantomogram (OPG) X-ray images to validate and improve our automated tooth detection and numbering system. Our proof-of-concept (POC) has demonstrated promising results on a publicly available dataset (~250 images), and we seek to validate and improve the system on a more diverse, clinically representative dataset.

**Current Status:** POC completed with 85%+ FDI accuracy on validation set  
**Request:** Access to 500-1000 additional OPG images for validation and model improvement  
**Timeline:** 6-12 months for data collection, annotation, and model refinement

---

## 2. Motivation for Larger Dataset

### 2.1 Current Limitations
- **Small Dataset:** Current POC uses ~250 images from Niha Adnan et al. (2024) dataset
- **Limited Diversity:** Single source, limited variation in:
  - Imaging systems (different X-ray machines)
  - Patient demographics (age ranges, ethnicities)
  - Clinical conditions (restorations, implants, missing teeth)
  - Image quality and resolution

### 2.2 Need for Validation
- **Generalization:** Current model may not generalize to different imaging systems
- **Robustness:** Need to test on diverse patient populations
- **Clinical Readiness:** Larger dataset required for clinical validation
- **Regulatory Requirements:** FDA/CE marking may require multi-center validation

### 2.3 Research Goals
- Validate model performance on diverse dataset
- Improve robustness to different imaging conditions
- Develop missing tooth detection capabilities
- Enhance quadrant detection for rotated images
- Prepare for multi-center clinical validation

---

## 3. Current Performance Summary

### 3.1 POC Results (Phase 6 Validation)

**Performance Metrics:**
- **Mean Mask IoU:** 0.75-0.85 (depending on resolution)
- **FDI Accuracy:** 85-90% (after anatomical correction)
- **Detection Rate:** 28-32 teeth per OPG (expected range)
- **Inference Speed:** <500ms per image (GPU)

**Detailed Results:** See `workspace/phase6/reports/PHASE6_FINAL_REPORT.md`

### 3.2 Strengths
- High accuracy on standard OPG images
- Fast inference suitable for clinical workflow
- Robust to minor image quality variations
- Good performance on complete dentitions

### 3.3 Limitations (Identified in Phase 6)
- Performance degrades on rotated images (>7°)
- Difficulty with missing teeth detection
- Some quadrant misclassifications
- Lower accuracy on images with heavy restorations/implants

---

## 4. Proposed Model Improvements

### 4.1 Current Architecture
- **Instance Segmentation:** Mask R-CNN (ResNet-50 FPN)
- **ROI Classifier:** EfficientNet-B0 for FDI numbering
- **Post-processing:** Anatomical ordering and geometric correction

### 4.2 Planned Improvements (With Larger Dataset)

**1. Enhanced Quadrant Detection**
- Current: Centroid-based heuristics
- Planned: Jaw line detection using deep learning
- Benefit: Better handling of rotated images

**2. Missing Tooth Detection**
- Current: Not explicitly modeled
- Planned: Explicit gap detection module
- Benefit: Accurate tooth count and numbering

**3. Restoration/Implant Handling**
- Current: May confuse restorations with natural teeth
- Planned: Multi-class detection (tooth, restoration, implant)
- Benefit: More accurate segmentation

**4. Multi-Resolution Training**
- Current: Trained on 1024×512
- Planned: Multi-scale training with data augmentation
- Benefit: Better generalization to different resolutions

**5. Active Learning**
- Current: Fixed training set
- Planned: Iterative annotation of difficult cases
- Benefit: Improved performance with fewer annotations

---

## 5. Description of Current POC System

### 5.1 Pipeline Overview

```
Input OPG Image
    ↓
Preprocessing (Resize, Normalize)
    ↓
Mask R-CNN (Instance Segmentation)
    ├── High Confidence (≥0.85) → Use Mask R-CNN prediction
    └── Low Confidence (<0.85) → EfficientNet ROI Classifier
    ↓
Anatomical Ordering
    ├── Quadrant Detection (Upper/Lower, Left/Right)
    ├── Geometric Sorting (X-centroid sorting)
    └── FDI Mapping (11-48)
    ↓
Post-processing
    ├── Duplicate Detection
    ├── Conflict Resolution
    └── Final FDI Assignment
    ↓
Output: JSON + Visualization
```

### 5.2 Technical Details

**Mask R-CNN:**
- Architecture: ResNet-50 Feature Pyramid Network
- Input: 1024×512 (configurable)
- Classes: 32 FDI tooth classes
- Pretrained: COCO instance segmentation
- Training: Transfer learning on Niha Adnan dataset

**EfficientNet-B0:**
- Architecture: EfficientNet-B0
- Input: 128×128 (tooth ROI crops)
- Classes: 32 FDI classes
- Purpose: Correct low-confidence Mask R-CNN predictions

**Anatomical Correction:**
- Quadrant detection based on Y-centroid (upper vs lower)
- X-centroid sorting within quadrants
- FDI mapping: Upper-right (11-18), Upper-left (21-28), Lower-left (31-38), Lower-right (41-48)
- Conflict resolution using geometric ordering

### 5.3 Validation Results

**Phase 6 Comprehensive Testing:**
- Batch validation: 100 images (50 val + 50 train)
- Stress testing: 9 distortion types
- Regression testing: Stability tracking
- Failure case analysis: Top 20 worst cases identified

**Key Findings:**
- Robust to Gaussian noise, motion blur, JPEG compression
- Sensitive to large rotations (>7°)
- Good performance on standard OPG images
- Requires expert review for edge cases

---

## 6. Exact Data Requirements

### 6.1 Dataset Specifications

**Number of Images:**
- **Minimum:** 500 OPG images
- **Optimal:** 1000 OPG images
- **Rationale:** 
  - 500 images: Sufficient for validation and minor improvements
  - 1000 images: Enables robust model retraining and generalization

**Age Ranges:**
- **Pediatric:** 50-100 images (ages 5-12)
- **Adolescent:** 100-200 images (ages 13-18)
- **Adult:** 300-600 images (ages 19-65)
- **Geriatric:** 50-100 images (ages 65+)
- **Rationale:** Different age groups have different tooth development stages

**Device Types:**
- **Digital OPG Systems:** 
  - Planmeca ProX
  - Carestream CS 8100
  - Vatech PaX-i3D
  - Other common systems
- **Rationale:** Different systems have different image characteristics

**Clinical Conditions:**
- **Normal Dentition:** 40-50% of dataset
- **Restorations Present:** 30-40% (fillings, crowns)
- **Implants Present:** 10-15%
- **Missing Teeth:** 20-30%
- **Orthodontic Treatment:** 5-10%
- **Rationale:** Real-world clinical diversity

**Image Quality:**
- **High Quality:** 60-70% (clear, well-exposed)
- **Moderate Quality:** 20-30% (acceptable, minor artifacts)
- **Challenging:** 5-10% (motion blur, low contrast, rotated)
- **Rationale:** Test robustness to real-world conditions

### 6.2 Data Format Requirements

**Preferred Format:**
- DICOM files (with de-identification)
- OR PNG/JPEG (if DICOM not available)

**Resolution:**
- Minimum: 512×512 pixels
- Optimal: 1024×1024 or higher
- Aspect ratio: Standard OPG format (approximately 2:1 width:height)

**Annotations:**
- Format: COCO instance segmentation JSON
- OR LabelMe JSON (we can convert)
- Classes: 32 FDI tooth classes (11-48)
- Quality: Expert-annotated or validated annotations

### 6.3 Exclusion Criteria

**Exclude:**
- Images with severe artifacts (unusable)
- Images with <20 visible teeth (too incomplete)
- Images with non-standard views (not true OPG)
- Images with patient identifiers visible (must be de-identified)

---

## 7. Compliance Summary and Security Measures

### 7.1 Data Security

**Storage:**
- Encrypted research server (AES-256)
- Access-controlled network drive
- No cloud storage without explicit approval
- Regular security audits

**Access Controls:**
- Authorized personnel only (PI + 3 co-investigators)
- Password-protected systems
- Two-factor authentication
- Access logging and audit trails

**Data Transfer:**
- Encrypted transfer (TLS 1.3)
- Secure file transfer protocols only
- No email transmission of images

### 7.2 De-Identification

**Procedures:**
- Remove all DICOM metadata containing PHI
- Replace filenames with hash-based identifiers
- Remove visible patient identifiers
- Store only image data and annotations

**Verification:**
- Automated de-identification script
- Manual review of sample images
- DICOM tag removal verification

### 7.3 Usage Restrictions

**Research Only:**
- Data used solely for research purposes
- No commercial use
- No sharing with third parties
- No publication of patient-identifiable information

**Retention:**
- As per IRB approval (typically 7 years)
- Secure deletion after retention period
- Documentation of deletion process

---

## 8. Expected Outcomes and Publications

### 8.1 Research Outcomes

**Short-term (6-12 months):**
- Validation of POC on diverse dataset
- Improved model robustness
- Identification of failure modes
- Performance benchmarks

**Long-term (12-24 months):**
- Publication-ready results
- Clinical validation study
- Open-source tool release
- Potential regulatory submission

### 8.2 Planned Publications

**Target Journals:**
- Medical Image Analysis (Elsevier)
- IEEE Transactions on Medical Imaging
- Journal of Dental Research
- Computer Methods and Programs in Biomedicine

**Publication Plan:**
1. **Technical Paper:** Model architecture, training, evaluation
2. **Clinical Validation:** Multi-center study (if access granted)
3. **Open-source Release:** Code and trained models

**Acknowledgments:**
- Dataset providers will be acknowledged
- Collaborating institutions will be co-authors (if appropriate)
- Funding sources will be disclosed

### 8.3 Data Sharing

**Open-source Components:**
- Model architecture and training code
- Inference pipeline
- Evaluation scripts
- Documentation

**Not Shared:**
- Patient images (protected by privacy)
- Annotations (if provided by hospital)
- Patient demographics or PHI

---

## 9. Timeline

### 9.1 Data Collection Phase (Months 1-3)
- IRB approval (if required)
- Data access agreement execution
- Data transfer and receipt
- De-identification verification

### 9.2 Annotation Phase (Months 2-6)
- Expert annotation of images
- Quality control and validation
- Inter-annotator agreement measurement

### 9.3 Model Development Phase (Months 4-9)
- Model retraining on expanded dataset
- Hyperparameter tuning
- Architecture improvements
- Validation and testing

### 9.4 Analysis and Publication Phase (Months 10-12)
- Comprehensive evaluation
- Statistical analysis
- Manuscript preparation
- Publication submission

---

## 10. Contact Information

**Principal Investigator:**  
[Your Name]  
[Title]  
[Institution]  
[Email]  
[Phone]

**Co-Investigators:**  
- Divyansh: [Email]
- Navya: [Email]
- Nisha ma'am: [Email]

**Institutional Review Board:**  
[IRB Contact Information]

---

## 11. Attachments

1. **POC Performance Report:** `workspace/phase6/reports/PHASE6_FINAL_REPORT.md`
2. **IRB Checklist:** `workspace/tooth-ai/compliance/IRB_CHECKLIST.md`
3. **Data Security Policy:** `workspace/tooth-ai/compliance/DATA_SECURITY_POLICY.md`
4. **Model Risk Assessment:** `workspace/tooth-ai/compliance/MODEL_RISK_ASSESSMENT.md`
5. **Technical Documentation:** `workspace/tooth-ai/FINAL_POC_REPORT.md`

---

**Document Version:** 1.0  
**Date:** November 2024  
**Status:** Ready for Submission



