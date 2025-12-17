# IRB Checklist for Tooth-AI POC

**Project Title:** Automated Tooth Detection and Numbering in Orthopantomogram X-rays  
**Principal Investigator:** [Your Name]  
**Co-Investigators:** Divyansh, Navya, Nisha ma'am  
**Date:** November 2024  
**Status:** Proof of Concept (POC)

---

## 1. Project Summary

### 1.1 Research Objectives
- Develop an automated system for tooth detection and numbering in orthopantomogram (OPG) X-ray images
- Improve efficiency and consistency of dental charting workflows
- Validate performance on a publicly available dataset (Niha Adnan et al., 2024)

### 1.2 Clinical Relevance
- **Problem:** Manual tooth numbering in OPG images is time-consuming and prone to human error
- **Solution:** AI-assisted tool to automate detection and numbering using FDI notation
- **Impact:** Potential to reduce charting time by 50-70% while maintaining accuracy

### 1.3 Dataset Source
- **Primary Dataset:** Niha Adnan et al., "Orthopantomogram teeth segmentation and numbering dataset", Data in Brief, 2024
- **License:** CC BY-NC 4.0 (non-commercial research use)
- **Size:** ~250 OPG images with manual annotations
- **Format:** LabelMe JSON annotations converted to COCO instance segmentation format

---

## 2. Annotation Process Summary

### 2.1 Current Dataset (Niha Adnan et al., 2024)
- **Annotation Method:** Manual polygon annotation using LabelMe tool
- **Annotation Classes:** 32 FDI tooth classes (teeth 11-48)
- **Quality Control:** QC audit performed (Phase 1) - <15% suspicious samples
- **Validation:** Cross-validation with expert review

### 2.2 Future Dataset (If Hospital Access Granted)
- **Annotation Plan:** 
  - Expert dental radiologists will annotate using same LabelMe/COCO format
  - Inter-annotator agreement will be measured (target: >90% IoU)
  - Quality control checks will be performed on all annotations
- **Annotation Guidelines:** 
  - Follow FDI numbering system (11-48)
  - Include all visible teeth
  - Mark missing teeth explicitly
  - Note restorations/implants if present

---

## 3. Data De-Identification Procedure

### 3.1 Current Dataset (Public)
- Niha Adnan dataset is already de-identified (publicly available)
- No PHI (Protected Health Information) present
- Filenames are anonymized

### 3.2 Future Dataset (If Hospital Access)
- **De-identification Steps:**
  1. Remove all DICOM metadata containing patient information
  2. Replace original filenames with hash-based identifiers (SHA-256)
  3. Remove any visible patient identifiers from images (watermarks, text overlays)
  4. Store only image data and annotations (no patient demographics)
  5. Maintain separate secure mapping file (encrypted) for research purposes only

### 3.3 PHI Removal Checklist
- [ ] Patient name
- [ ] Medical record number
- [ ] Date of birth
- [ ] Social Security Number
- [ ] Address
- [ ] Phone number
- [ ] Email
- [ ] Insurance information
- [ ] DICOM metadata tags (0010,0010; 0010,0020; etc.)

---

## 4. Storage and Retention Details

### 4.1 Current Storage
- **Location:** Local research server / cloud storage (encrypted)
- **Format:** Images (PNG/JPEG) + COCO JSON annotations
- **Access:** Research team only (password-protected, encrypted drives)

### 4.2 Future Storage (If Hospital Access)
- **Storage Location:** 
  - Encrypted research server
  - Access-controlled network drive
  - No cloud storage without explicit IRB approval
- **Encryption:** 
  - At-rest: AES-256 encryption
  - In-transit: TLS 1.3 for data transfer
- **Backup:** 
  - Encrypted backups stored separately
  - Regular backup verification
- **Retention Period:** 
  - As per IRB approval (typically 7 years post-study completion)
  - Secure deletion after retention period

### 4.3 Access Controls
- **Authorized Personnel:** 
  - Principal Investigator
  - Co-Investigators (Divyansh, Navya, Nisha ma'am)
  - IT support (minimal access, as needed)
- **Access Logging:** 
  - All data access logged with timestamp and user ID
  - Regular audit of access logs
- **Training:** 
  - All team members completed HIPAA/research ethics training
  - Signed data use agreements

---

## 5. Investigator Roles

### 5.1 Principal Investigator
- **Name:** [Your Name]
- **Role:** Overall project oversight, IRB submission, data access approval
- **Responsibilities:** 
  - Ensure compliance with IRB protocols
  - Approve all data access requests
  - Review and approve publications

### 5.2 Co-Investigators

**Divyansh:**
- **Role:** Technical development, model training, evaluation
- **Responsibilities:** 
  - Model architecture development
  - Training pipeline implementation
  - Performance evaluation

**Navya:**
- **Role:** Data processing, annotation quality control
- **Responsibilities:** 
  - Data preprocessing and conversion
  - Annotation validation
  - Quality control audits

**Nisha ma'am:**
- **Role:** Clinical advisor, expert review
- **Responsibilities:** 
  - Clinical validation of predictions
  - Expert annotation review
  - Clinical workflow integration guidance

---

## 6. Risk/Benefit Analysis

### 6.1 Risks to Patient Data (If Hospital Access Granted)

**Minimal Risk:**
- **Data Breach Risk:** Low - data stored on encrypted, access-controlled systems
- **Re-identification Risk:** Very Low - all PHI removed, images anonymized
- **Misuse Risk:** Low - data used only for research, not shared externally

**Mitigation Strategies:**
- Encryption at rest and in transit
- Access controls and audit logging
- Regular security assessments
- Data use agreements with all personnel
- Secure deletion after retention period

### 6.2 Benefits

**Research Benefits:**
- Advancement of AI in dental imaging
- Improved understanding of automated tooth detection
- Potential for future clinical applications

**Clinical Benefits (Future):**
- Reduced charting time
- Improved consistency
- Reduced human error
- Better patient care through efficiency

**Societal Benefits:**
- Open-source tools for research community
- Reproducible research methodology
- Educational value

### 6.3 Risk Level Assessment
- **Overall Risk:** **Minimal** (for retrospective image analysis with de-identified data)
- **IRB Category:** Likely exempt or expedited review (retrospective, de-identified data)

---

## 7. Compliance Checklist

### 7.1 Pre-Submission
- [x] Project summary completed
- [x] Data de-identification procedures documented
- [x] Storage and security measures defined
- [x] Investigator roles assigned
- [x] Risk/benefit analysis completed

### 7.2 IRB Submission Requirements
- [ ] IRB application form completed
- [ ] Protocol document prepared
- [ ] Data use agreement template ready
- [ ] Informed consent waiver requested (if applicable for retrospective data)
- [ ] HIPAA authorization waiver requested (if applicable)
- [ ] Investigator CVs and training certificates
- [ ] Data security plan submitted

### 7.3 Post-Approval
- [ ] IRB approval letter received
- [ ] Data access agreements signed by all personnel
- [ ] Data storage systems configured
- [ ] Access controls implemented
- [ ] Training completed for all team members
- [ ] Regular compliance audits scheduled

---

## 8. References

1. Niha Adnan et al., "Orthopantomogram teeth segmentation and numbering dataset", Data in Brief, 2024
2. FDA Guidance: "Clinical Decision Support Software" (2019)
3. HIPAA Privacy Rule (45 CFR Parts 160 and 164)
4. Common Rule (45 CFR Part 46)

---

## 9. Contact Information

**Principal Investigator:**  
[Your Name]  
[Email]  
[Institution]

**IRB Office:**  
[IRB Contact Information]

**Data Security Officer:**  
[DSO Contact Information]

---

**Document Version:** 1.0  
**Last Updated:** November 2024  
**Next Review:** Upon IRB submission



