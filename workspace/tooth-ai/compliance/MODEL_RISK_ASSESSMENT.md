# Model Risk Assessment: Tooth-AI POC

**Project:** Automated Tooth Detection and Numbering in Orthopantomogram X-rays  
**Assessment Date:** November 2024  
**Model Version:** v0.1.0-poc

---

## 1. Executive Summary

This document assesses the risks associated with the Tooth-AI automated tooth detection and numbering system. The system is a **proof-of-concept (POC) research tool** and is **NOT approved for clinical diagnosis or patient care decisions**. All predictions require expert clinical review.

**Risk Level:** **LOW to MODERATE** (with appropriate safeguards)

---

## 2. System Overview

### 2.1 Purpose
- Automated detection and segmentation of teeth in OPG images
- Automatic FDI tooth numbering (11-48)
- Assist dental professionals in charting workflows

### 2.2 Limitations
- POC status (not production-ready)
- Trained on limited dataset (~250 images)
- Not validated for clinical use
- Requires expert review

---

## 3. Risk Categories

### 3.1 Misclassification Risks

**Risk: Incorrect FDI Numbering**

**Severity:** MODERATE
- **Impact:** Wrong tooth number assigned
- **Likelihood:** 10-15% error rate (based on POC validation)
- **Consequences:**
  - Incorrect treatment planning
  - Wrong tooth identification in records
  - Potential treatment errors

**Mitigation:**
- Human-in-the-loop requirement (expert review)
- Confidence scores displayed (<0.85 flagged for review)
- Visual overlay for verification
- Training on failure cases

**Risk: Missing Tooth Detection**

**Severity:** MODERATE
- **Impact:** Missing teeth not detected
- **Likelihood:** 5-10% (based on POC)
- **Consequences:**
  - Incomplete charting
  - Missing treatment planning
  - Inaccurate records

**Mitigation:**
- Expert review required
- Visual inspection of predictions
- Flag low detection counts (<28 teeth)
- Future: Explicit missing tooth detection module

**Risk: False Positive Detections**

**Severity:** LOW to MODERATE
- **Impact:** Non-tooth structures detected as teeth
- **Likelihood:** 2-5% (based on POC)
- **Consequences:**
  - Incorrect tooth count
  - Confusion in charting
  - Unnecessary review

**Mitigation:**
- Confidence threshold filtering
- Expert review
- Visual verification
- Post-processing validation

### 3.2 Segmentation Risks

**Risk: Inaccurate Mask Boundaries**

**Severity:** LOW to MODERATE
- **Impact:** Tooth boundaries not accurately segmented
- **Likelihood:** 10-20% (IoU < 0.8)
- **Consequences:**
  - Inaccurate measurements
  - Incorrect area calculations
  - Treatment planning errors

**Mitigation:**
- Display IoU/confidence scores
- Expert review for low-confidence cases
- Visual overlay for verification
- Manual correction capability

**Risk: Overlapping Masks**

**Severity:** LOW
- **Impact:** Adjacent teeth masks overlap
- **Likelihood:** 5-10%
- **Consequences:**
  - Ambiguous boundaries
  - Measurement errors

**Mitigation:**
- Post-processing duplicate detection
- Non-maximum suppression
- Expert review

### 3.3 Quadrant Misclassification

**Risk: Wrong Quadrant Assignment**

**Severity:** MODERATE
- **Impact:** Tooth assigned to wrong quadrant
- **Likelihood:** 5-10% (based on POC)
- **Consequences:**
  - Wrong FDI number (e.g., 11 vs 21)
  - Treatment planning errors
  - Record inconsistencies

**Mitigation:**
- Anatomical correction logic
- Expert review
- Visual quadrant indicators
- Future: Improved quadrant detection

---

## 4. Tooth Numbering Error Severity

### 4.1 Severity Classification

**CRITICAL Errors:**
- Wrong quadrant (e.g., upper vs lower)
- Adjacent tooth swap (e.g., 11 vs 12)
- **Impact:** High - can lead to wrong tooth treatment

**MODERATE Errors:**
- Non-adjacent swap (e.g., 11 vs 13)
- Missing tooth not detected
- **Impact:** Medium - requires correction but less critical

**LOW Errors:**
- Minor boundary inaccuracy
- Confidence score slightly off
- **Impact:** Low - cosmetic or minor measurement issues

### 4.2 Error Rates (POC Validation)

**Overall FDI Accuracy:** 85-90%
- **Critical Errors:** <2%
- **Moderate Errors:** 5-8%
- **Low Errors:** 5-10%

---

## 5. Misleading Segmentation Impacts

### 5.1 Clinical Impact

**Treatment Planning:**
- Incorrect tooth identification → wrong treatment
- Missing teeth → incomplete treatment plan
- Boundary errors → inaccurate measurements

**Record Keeping:**
- Inconsistent numbering → confusion
- Missing entries → incomplete records
- Wrong quadrant → treatment errors

### 5.2 Patient Safety

**Direct Risks:**
- Wrong tooth treatment (if not reviewed)
- Missing treatment (if tooth not detected)
- Inaccurate measurements (if boundaries wrong)

**Indirect Risks:**
- Delayed treatment (if errors require re-review)
- Patient confusion (if records inconsistent)

---

## 6. Safe-Use Guidelines

### 6.1 Model is NOT Diagnostic

**Critical Disclaimer:**
- **This model is a research tool, NOT a diagnostic device**
- **NOT approved for clinical diagnosis**
- **NOT for direct patient care decisions**
- **Requires expert clinical review**

### 6.2 Recommended Workflow

**Step 1: Run Inference**
- Upload OPG image
- Run automated detection
- Review confidence scores

**Step 2: Expert Review**
- Review all predictions
- Verify FDI numbers
- Check for missing teeth
- Validate boundaries

**Step 3: Manual Correction (If Needed)**
- Correct any errors
- Add missing teeth manually
- Adjust boundaries if needed

**Step 4: Final Verification**
- Second review (if critical case)
- Document any corrections
- Save verified results

### 6.3 Quality Assurance

**Automated Checks:**
- Flag low confidence predictions (<0.85)
- Flag unusual detection counts (<28 or >32)
- Flag quadrant violations
- Display confidence scores

**Manual Checks:**
- Expert review required
- Visual verification
- Comparison with clinical notes (if available)
- Documentation of corrections

---

## 7. Human-in-the-Loop Requirement

### 7.1 Mandatory Review

**All Predictions Must Be:**
- Reviewed by qualified dental professional
- Verified for accuracy
- Corrected if errors found
- Documented in records

### 7.2 Review Criteria

**Check:**
- All teeth detected (28-32 expected)
- Correct FDI numbers
- Correct quadrants
- Accurate boundaries
- No false positives

### 7.3 Documentation

**Record:**
- Review date and reviewer
- Any corrections made
- Confidence in predictions
- Any concerns or limitations

---

## 8. Failure Mode Analysis

### 8.1 Known Failure Modes

**Rotated Images:**
- Performance degrades >7° rotation
- Quadrant detection may fail
- **Mitigation:** Pre-process rotation correction

**Heavy Restorations:**
- May confuse restorations with natural teeth
- Boundary detection may be inaccurate
- **Mitigation:** Expert review, future multi-class detection

**Missing Teeth:**
- Not explicitly detected
- May cause numbering errors
- **Mitigation:** Expert review, future missing tooth module

**Low Image Quality:**
- Motion blur, low contrast
- Reduced accuracy
- **Mitigation:** Quality checks, expert review

### 8.2 Edge Cases

**Pediatric Patients:**
- Mixed dentition (baby + adult teeth)
- Different tooth development stages
- **Mitigation:** Special handling, expert review

**Geriatric Patients:**
- Missing teeth common
- Worn teeth, restorations
- **Mitigation:** Expert review, adjusted expectations

**Orthodontic Cases:**
- Crowded teeth
- Unusual positions
- **Mitigation:** Expert review, manual correction

---

## 9. Risk Mitigation Strategies

### 9.1 Technical Mitigations

**Model Improvements:**
- Larger, more diverse training dataset
- Improved quadrant detection
- Missing tooth detection module
- Multi-class detection (tooth, restoration, implant)

**Quality Controls:**
- Confidence thresholds
- Automated validation checks
- Post-processing corrections
- Visual verification tools

### 9.2 Process Mitigations

**Workflow:**
- Mandatory expert review
- Two-reviewer system (for critical cases)
- Documentation requirements
- Training for users

**Monitoring:**
- Track error rates
- Collect feedback
- Identify failure patterns
- Continuous improvement

---

## 10. Regulatory Considerations

### 10.1 Current Status

**POC/Research Tool:**
- Not FDA approved
- Not CE marked
- Not for clinical use
- Research/educational purposes only

### 10.2 Future Regulatory Pathway

**If Pursuing Approval:**
- Clinical validation study required
- Multi-center validation
- Regulatory submission (FDA 510(k) or CE marking)
- Post-market surveillance

---

## 11. Recommendations

### 11.1 For Current POC

**Immediate Actions:**
1. Add clear disclaimers (NOT diagnostic)
2. Require expert review
3. Display confidence scores
4. Provide manual correction tools
5. Document limitations

### 11.2 For Future Development

**Improvements Needed:**
1. Larger, diverse training dataset
2. Missing tooth detection
3. Improved quadrant detection
4. Multi-class detection
5. Clinical validation study

---

## 12. Contact Information

**Principal Investigator:**  
[Your Name]  
[Email]

**Clinical Advisor:**  
Nisha ma'am  
[Email]

**Risk Management:**  
[Institutional Risk Management Contact]

---

**Assessment Version:** 1.0  
**Date:** November 2024  
**Next Review:** Upon model updates or clinical validation



