# Patient Privacy and De-Identification Policy

**Project:** Tooth-AI POC  
**Effective Date:** November 2024

---

## 1. Overview

This document outlines the de-identification procedures for orthopantomogram (OPG) X-ray images to ensure patient privacy and HIPAA compliance. All patient-identifiable information (PHI) must be removed before images are used for research.

---

## 2. Required De-Identification Steps

### 2.1 DICOM Metadata Removal

**DICOM Tags to Remove:**

**Patient Information (Group 0010):**
- (0010,0010) Patient's Name
- (0010,0020) Patient ID
- (0010,0030) Patient's Birth Date
- (0010,0040) Patient's Sex
- (0010,1000) Other Patient IDs
- (0010,1001) Other Patient Names
- (0010,1005) Patient's Birth Name
- (0010,1010) Patient's Age
- (0010,1040) Patient's Address
- (0010,1060) Patient's Mother's Birth Name
- (0010,2150) Country of Residence
- (0010,2152) Region of Residence
- (0010,2154) Patient's Telephone Numbers

**Study Information (Group 0020):**
- (0020,0010) Study ID
- (0020,0011) Series Number
- (0020,0013) Instance Number

**Equipment Information (Group 0008):**
- (0008,0050) Accession Number
- (0008,0080) Institution Name (may be kept if not identifying)
- (0008,0090) Referring Physician's Name
- (0008,1040) Institutional Department Name
- (0008,1050) Performing Physician's Name
- (0008,1060) Name of Physician(s) Reading Study

**Other Identifying Tags:**
- All private tags (odd group numbers)
- Any tags containing dates (except study date, which may be anonymized)
- Any tags containing names or IDs

**Tools for DICOM De-identification:**
- `dicom-anonymizer` (Python)
- `gdcm` (DICOM anonymizer)
- Custom scripts using `pydicom`

### 2.2 Visible Patient Identifiers

**Remove from Images:**
- Patient name watermarks
- Medical record numbers
- Dates of birth
- Institution names (if identifying)
- Any text overlays containing PHI
- Barcodes or QR codes (may contain PHI)

**Image Processing:**
- Use image editing tools to remove visible text
- Crop or mask identifying regions
- Verify no PHI remains visible

### 2.3 Filename Anonymization

**Hashing Rules:**
- Replace original filenames with SHA-256 hash
- Format: `{hash}.{extension}` (e.g., `a3f5b2c1...png`)
- Maintain separate secure mapping file (encrypted) for research purposes only
- Mapping file stored separately from images

**Example:**
```python
import hashlib
original_filename = "Patient_John_Doe_20240101.png"
hash_obj = hashlib.sha256(original_filename.encode())
anonymized = hash_obj.hexdigest() + ".png"
# Result: "a3f5b2c1d4e6f7g8h9i0j1k2l3m4n5o6p7q8r9s0t1u2v3w4x5y6z7.png"
```

### 2.4 Annotation File De-identification

**COCO JSON Format:**
- Remove any patient identifiers from filenames
- Use anonymized filenames in annotations
- Remove any metadata containing PHI
- Verify no PHI in annotation text fields

---

## 3. No PHI Storage

### 3.1 Prohibited Information

**Never Store:**
- Patient names
- Medical record numbers
- Social Security Numbers
- Dates of birth
- Addresses
- Phone numbers
- Email addresses
- Insurance information
- Any other 18 HIPAA identifiers

### 3.2 Allowed Information

**May Store (If De-identified):**
- Age ranges (e.g., "50-60 years")
- Sex (if not identifying)
- Study date (anonymized, e.g., "2024-Q1")
- Image characteristics (resolution, format)
- Clinical findings (anonymized, e.g., "missing teeth count")

---

## 4. De-Identification Verification

### 4.1 Automated Checks

**Scripts to Run:**
- DICOM tag verification (check for removed tags)
- Filename hash verification
- Image text detection (OCR to find visible text)
- Metadata scan (check JSON/annotation files)

### 4.2 Manual Review

**Sample Review:**
- Review 10% of images manually
- Check for visible identifiers
- Verify DICOM metadata removal
- Confirm filename anonymization

**Documentation:**
- Document verification process
- Record any issues found
- Correct issues before use
- Maintain verification log

---

## 5. Re-identification Risk Assessment

### 5.1 Risk Factors

**Low Risk:**
- Fully de-identified images (no PHI)
- Hash-based filenames
- No linkage to patient records

**Medium Risk:**
- Images with unique characteristics (rare conditions)
- Small dataset (easier to identify)
- Specific date ranges

**High Risk:**
- Images with visible unique features
- Linkage to other datasets possible
- Small, specific population

### 5.2 Mitigation Strategies

**For Low Risk:**
- Standard de-identification sufficient
- Hash-based filenames
- No additional measures needed

**For Medium/High Risk:**
- Additional anonymization (blur unique features)
- Larger dataset (dilute uniqueness)
- Statistical disclosure control
- Expert review before release

---

## 6. Data Use Agreements

### 6.1 Personnel Agreements

**All Personnel Must:**
- Sign data use agreement
- Complete HIPAA training
- Complete research ethics training
- Agree to confidentiality
- Report any breaches immediately

### 6.2 Agreement Contents

**Must Include:**
- Purpose of data use (research only)
- Prohibited uses (no clinical use, no sharing)
- Security requirements
- Breach reporting procedures
- Penalties for violations

---

## 7. HIPAA Compliance

### 7.1 HIPAA Safe Harbor Method

**18 Identifiers to Remove:**
1. Names
2. Geographic subdivisions smaller than state
3. Dates (except year)
4. Telephone numbers
5. Fax numbers
6. Email addresses
7. Social Security Numbers
8. Medical record numbers
9. Health plan beneficiary numbers
10. Account numbers
11. Certificate/license numbers
12. Vehicle identifiers
13. Device identifiers
14. Web URLs
15. IP addresses
16. Biometric identifiers
17. Full face photos
18. Any other unique identifying number

**If All Removed:**
- Data is considered de-identified
- No HIPAA authorization required
- Can be used for research

### 7.2 Expert Determination Method

**Alternative Approach:**
- Expert certifies re-identification risk is very small
- Document expert determination
- Maintain certification records

---

## 8. Documentation Requirements

### 8.1 De-identification Log

**Maintain Records:**
- Date of de-identification
- Method used
- Personnel performing de-identification
- Verification results
- Any issues encountered

### 8.2 Mapping File (If Needed)

**If Maintaining Linkage:**
- Store separately (encrypted)
- Access-controlled
- Audit logged
- Destroy after research completion

---

## 9. Training Requirements

### 9.1 Required Training

**All Personnel Must Complete:**
- HIPAA Privacy Rule training
- Research ethics training
- Data security training
- De-identification procedures training

### 9.2 Training Frequency

- Initial training before data access
- Annual refresher training
- Updates when procedures change

---

## 10. Contact Information

**Privacy Officer:**  
[Privacy Officer Contact]

**Principal Investigator:**  
[Your Name]  
[Email]

**IRB Office:**  
[IRB Contact]

---

**Document Version:** 1.0  
**Last Updated:** November 2024



