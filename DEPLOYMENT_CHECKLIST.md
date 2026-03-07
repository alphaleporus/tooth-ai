# DEPLOYMENT CHECKLIST: Emergency Fix Deployment

## ⏰ Total Estimated Time: 2-4 hours

---

## Phase 1: Pre-Deployment Validation (30 minutes)

### Step 1.1: Run Unit Tests ✅

```powershell
# Test emergency fixes
python test_emergency_fixes.py
```

**Expected Output**:
```
ALL TESTS PASSED - EMERGENCY FIXES VALIDATED ✅
Ready for production deployment
```

**If tests fail**: Fix issues before proceeding. Do NOT deploy failing code.

---

### Step 1.2: Backup Current System ✅

```powershell
# Create backup
Copy-Item app.py app.py.backup_20260128 -Force

# Verify backup
Get-Item app.py.backup_20260128
```

**Expected**: Backup file created with timestamp

---

### Step 1.3: Verify Model Availability ✅

```powershell
# Check ResNeXt-101 model exists
Test-Path "output\resnext101_cascade_60k\model_final.pth"
```

**Expected**: Should return `True`

**If False**: Model missing, need to download/train before deployment

---

## Phase 2: Code Integration (60 minutes)

Follow **INTEGRATION_GUIDE.md** exactly:

- [ ] Step 1: Add imports (2 min)
- [ ] Step 2: Insert FDIValidator class (5 min)
- [ ] Step 3: Replace process_predictions() (10 min)
- [ ] Step 4: Update function calls (10 min)
- [ ] Step 5: Remove old NMS call (2 min)
- [ ] Step 6: Add FDI validation layer (10 min)

**Validation After Each Step**:
```powershell
# Check syntax
python -m py_compile app.py
```

**Expected**: No output = no syntax errors

---

## Phase 3: Initial Testing (30 minutes)

### Step 3.1: Smoke Test (Quick Validation) ✅

```powershell
# Start app
streamlit run app.py
```

**Check in console** for debug output:
```
[DEBUG] process_predictions() called
[DEBUG] Raw detections from model: ...
[DEBUG] Post-NMS: ... teeth after NMS
```

**If missing debug logs**: process_predictions() not integrated correctly

---

### Step 3.2: Upload Test Image ✅

Upload ONE image from `data\test data\92273_YAGNESH_MHASKAR_26_M_20260108_131000.jpg`

**Check UI for**:
- Sidebar shows "FDI Confidence: XX%"
- If duplicates present: Warning messages visible
- If confidence <0.70: Warning banner
- Console shows debug logs

**Expected Behavior**:
- Teeth detected: 24-30
- FDI Confidence: ≥0.70
- No "CRITICAL" warnings

---

### Step 3.3: Baseline Metrics Capture ✅

Test on 5 priority images from TEST_DATA_ANALYSIS.md:

| Image | Teeth Detected | FDI Confidence | Warnings | Status |
|-------|---------------|----------------|----------|--------|
| 92273_YAGNESH_MHASKAR | __ | __% | __ | __ |
| 91325_GANESH_PAWAR | __ | __% | __ | __ |
| 89901_SHEJAL_KATKE | __ | __% | __ | __ |
| 91397_BHIMRAO_BHANDARI | __ | __% | __ | __ |
| 92527_RANDOLPH_ANTHONY | __ | __% | __ | __ |

**Success Criteria**:
- ✅ 4/5 images with ≥20 teeth detected
- ✅ Average FDI confidence ≥0.70
- ✅ No system crashes

**If criteria not met**: Proceed to Phase 5 (Threshold Tuning)

---

## Phase 4: Extended Validation (60 minutes)

### Step 4.1: Test on 10 Images ✅

Expand testing to 10 images from `data\test data\`:

**Performance Metrics to Track**:
```
Total teeth detected (average): ____ (target: 24-30)
FDI confidence (average): ____% (target: ≥70%)
False positives (lower jaw): ____ (target: ≤4)
False negatives (upper jaw): ____ (target: ≤2)
FDI duplicates: ____ instances (target: ≤2)
System crashes: ____ (target: 0)
```

**Document Each Image**:
- Screenshot result
- Teeth detected (upper/lower breakdown)
- FDI confidence score
- Any warnings or errors

---

### Step 4.2: Compare Before vs After ✅

**Before Fixes** (from original issue report):
- Upper jaw: ~10 teeth (29% false negatives)
- Lower jaw: ~25 teeth (79% false positives)
- FDI duplicates: 15+ instances
- Clinical usability: 0%

**After Fixes** (measure now):
- Upper jaw: ____ teeth
- Lower jaw: ____ teeth
- FDI duplicates: ____ instances
- Clinical usability: ____%

**Expected Improvement**:
- Upper jaw: 12-14 teeth (≥85% sensitivity)
- Lower jaw: 13-16 teeth (≤4 false positives)
- FDI duplicates: 0-2 instances
- Clinical usability: ≥85%

---

### Step 4.3: Error Analysis ✅

For any failing images, investigate:

**If too many teeth** (>35 detected):
- Check console: NMS applied?  
- Try more aggressive NMS: `iou_threshold=0.20` (was 0.25)
- Try higher lower jaw threshold: `base_threshold=0.50` (was 0.42)

**If too few teeth** (<20 detected):
- Check console: Threshold values used?
- Try lower thresholds: upper=0.08 (was 0.12), lower=0.35 (was 0.42)
- Check if ensemble is harming: Test single-pass

**If FDI duplicates persist**:
- Check: Was FDIValidator integrated?
- Check: Are warnings displayed?
- Review console for validation logs

---

## Phase 5: Threshold Tuning (Optional, 60 minutes)

**Only if Phase 4 results are <80% success rate**

### Step 5.1: Run Threshold Sweep ✅

Create quick sweep script:

```python
# threshold_sweep.py
import cv2
from app import load_model, process_predictions

image_path = "data/test data/92273_YAGNESH_MHASKAR_26_M_20260108_131000.jpg"
image = cv2.imread(image_path)

# Load model
predictor, cfg = load_model(CONFIGPATH, WEIGHTS_PATH)

# Run inference once
outputs = predictor(image)

# Try different thresholds
for upper_thresh in [0.08, 0.10, 0.12, 0.14, 0.16]:
    for lower_thresh in [0.35, 0.38, 0.42, 0.45, 0.48]:
        # Modify process_predictions to use these thresholds
        # ... (requires temporary code modification)
        
        teeth, anomalies = process_predictions(outputs, image.shape[1], image.shape[0])
        
        print(f"Upper={upper_thresh:.2f}, Lower={lower_thresh:.2f} → {len(teeth)} teeth")
```

**Analyze Results**:
- Find combination with teeth count closest to 28-30
- Update `emergency_fix_nms_thresholds.py` with optimal values
- Restart app and re-test

---

## Phase 6: Clinical Validation (2-4 hours)

### Step 6.1: Dental Professional Review ✅

**Requirement**: Have a dental professional (or knowledgeable reviewer) validate 20 cases

**Review Process**:
1. AI processes X-ray
2. Reviewer counts actual teeth visible
3. Reviewer verifies FDI numbers are correct
4. Reviewer checks anomaly detections

**Metrics to Collect**:
```
Sensitivity (teeth detected): ____ % (target: ≥90%)
Precision (correct detections): ____ % (target: ≥85%)
FDI accuracy (correct numbering): ____ % (target: ≥85%)
Clinical agreement (acceptable cases): ____ % (target: ≥85%)
```

**Decision Criteria**:
- If ≥3 of 4 metrics meet target → **APPROVE FOR PRODUCTION**
- If 2/4 metrics met → **DEPLOY WITH MANUAL REVIEW FLAGS**
- If <2/4 metrics met → **DO NOT DEPLOY, requires fine-tuning**

---

### Step 6.2: Document Clinical Findings ✅

Create clinical validation report:

```markdown
# Clinical Validation Report - Emergency Fixes

## Date: 2026-01-28
## Reviewer: [Name], [Credentials]

## Performance Metrics:
- Sensitivity: ___%
- Precision: ___%
- FDI Accuracy: ___%
- Clinical Agreement: ___%

## Error Analysis:
- False Negatives: [list teeth commonly missed]
- False Positives: [common phantom teeth locations]
- FDI Errors: [common numbering mistakes]

## Recommendation:
[ ] APPROVED for production deployment
[ ] APPROVED with manual review flags
[ ] NOT APPROVED - requires additional work

Signature: ________________ Date: __________
```

---

## Phase 7: Production Deployment (30 minutes)

### Step 7.1: Final Pre-Flight Checks ✅

- [ ] All unit tests passing
- [ ] No syntax errors in app.py
- [ ] Backup created and verified
- [ ] Clinical validation approved
- [ ] Deployment checklist completed
- [ ] Team notified of deployment

---

### Step 7.2: Deploy to Production ✅

**If you have a staging environment**:
1. Deploy to staging first
2. Test with real users for 1 day
3. Collect feedback
4. Deploy to production if successful

**If deploying directly to production**:
```powershell
# Stop current app (Ctrl+C)

# Verify app.py has all changes
python -m py_compile app.py

# Start production app
streamlit run app.py --server.port 8501
```

---

### Step 7.3: Monitor First 50 Cases ✅

**Track KPIs in real-time**:
```
Average teeth detected: ____ per image (target: 24-30)
FDI confidence distribution: ____% ≥0.70 (target: ≥80%)
Warning rate: ____% (target: <15%)
Crash rate: ____% (target: 0%)
User complaints: ____ (target: 0)
```

**Alert Thresholds**:
- Average teeth <20 for 1 hour → **ALERT TEAM**
- FDI confidence <0.60 for 10 consecutive images → **ALERT**
- Any system crash → **IMMEDIATE ROLLBACK**

---

## Phase 8: Post-Deployment (Ongoing)

### Step 8.1: Weekly Monitoring ✅

**Week 1 Checklist**:
- [ ] Review 50 random cases for quality
- [ ] Collect user feedback
- [ ] Analyze error patterns
- [ ] Document any issues
- [ ] Adjust thresholds if needed

**KPIs to Track**:
- Detection success rate (target: ≥85%)
- False positive rate (target: ≤10%)
- FDI confidence average (target: ≥0.75)
- User satisfaction (target: ≥4/5)

---

### Step 8.2: Fine-Tuning Decision Point ✅

**After 1 week, evaluate**:

**If system meets ALL targets** (≥85% across all metrics):
✅ **SUCCESS** - Continue normal operation
- Keep monitoring
- Collect data for future improvements

**If system meets 80-85%** (close but not quite):
⚠️ **THRESHOLD TUNING NEEDED**
- Run comprehensive threshold sweep on 50 images
- Adjust IoU or base thresholds
- Re-validate on 20 images
- Deploy updates

**If system <80%**:
🚨 **FINE-TUNING REQUIRED**
- Annotate 100-150 images from new dataset
- Fine-tune ResNeXt-101 for 5,000 iterations
- Validate to medical-grade standards
- Deploy fine-tuned model

---

## Emergency Rollback Procedure

**If critical issues arise during deployment**:

### Quick Rollback (5 minutes):

```powershell
# Stop current app (Ctrl+C)

# Restore backup
Copy-Item app.py.backup_20260128 app.py -Force

# Restart app
streamlit run app.py
```

**Notify team**:
```
Deployment rolled back due to: [reason]
System reverted to previous version
Investigation in progress
Expected resolution: [timeline]
```

---

## Success Metrics Summary

### Minimum Acceptable Performance (Deploy with Manual Review):
- Sensitivity: ≥85%
- Precision: ≥80%
- FDI Accuracy: ≥80%
- Clinical Agreement: ≥80%

### Target Performance (Deploy Autonomously):
- Sensitivity: ≥90%
- Precision: ≥85%
- FDI Accuracy: ≥85%
- Clinical Agreement: ≥90%

---

## Sign-Off

**Deployment Approved By**:

| Role | Name | Signature | Date |
|------|------|-----------|------|
| Technical Lead | __________ | __________ | __________ |
| Clinical Validator | __________ | __________ | __________ |
| QA Engineer | __________ | __________ | __________ |

---

**Deployment Status**: [ ] Pre-Flight | [ ] In Progress | [ ] Complete | [ ] Rolled Back

**Final Notes**:
____________________________________________________________
____________________________________________________________
____________________________________________________________

---

**Document Version**: 1.0  
**Last Updated**: 2026-01-28  
**Next Review**: After 1 week of production use
