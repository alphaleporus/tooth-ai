# Data Security Policy: Tooth-AI POC

**Project:** Automated Tooth Detection and Numbering in Orthopantomogram X-rays  
**Effective Date:** November 2024  
**Review Date:** Annually or as needed

---

## 1. Purpose

This policy establishes security measures for handling research data in the Tooth-AI project, ensuring compliance with HIPAA, institutional policies, and research ethics standards.

---

## 2. Scope

This policy applies to:
- All research data (images, annotations, model outputs)
- All personnel (Principal Investigator, Co-Investigators, IT support)
- All systems (servers, workstations, storage devices)
- All data transfers and sharing

---

## 3. Local Storage Rules

### 3.1 Approved Storage Locations

**Primary Storage:**
- Encrypted research server (institutional)
- Access-controlled network drive
- Encrypted local drives (for temporary work only)

**Prohibited Storage:**
- Personal devices (laptops, phones) without encryption
- Public cloud storage (Dropbox, Google Drive) without approval
- Unencrypted USB drives
- Email attachments

### 3.2 Storage Requirements

**Encryption:**
- At-rest: AES-256 encryption required
- Full disk encryption on all devices
- Encrypted containers for sensitive data

**Access Controls:**
- Password-protected systems (strong passwords required)
- Two-factor authentication where available
- Role-based access (only authorized personnel)

**Backup:**
- Encrypted backups only
- Stored separately from primary data
- Regular backup verification
- Secure backup deletion after retention period

---

## 4. Encryption Guidance

### 4.1 Data at Rest

**Requirements:**
- All research data must be encrypted
- Use AES-256 encryption standard
- Full disk encryption on laptops/workstations
- Encrypted file containers for sensitive data

**Tools:**
- BitLocker (Windows)
- FileVault (macOS)
- LUKS (Linux)
- VeraCrypt (cross-platform file containers)

### 4.2 Data in Transit

**Requirements:**
- TLS 1.3 for all network transfers
- SFTP/SCP for file transfers (not FTP)
- Encrypted email for metadata (not images)
- VPN for remote access

**Prohibited:**
- Unencrypted file transfers
- Email transmission of images
- Public Wi-Fi without VPN

### 4.3 Key Management

**Requirements:**
- Encryption keys stored separately from data
- Key rotation every 12 months
- Secure key backup (encrypted, separate location)
- Key destruction upon project completion

---

## 5. Research-Only Usage Clause

### 5.1 Permitted Uses

**Research Activities:**
- Model training and evaluation
- Performance analysis
- Publication preparation
- Educational presentations (de-identified data only)

### 5.2 Prohibited Uses

**Commercial Use:**
- No commercial product development
- No licensing to third parties
- No sale of data or models

**Clinical Use:**
- No direct clinical diagnosis
- No patient care decisions
- No integration into clinical systems without IRB approval

**Sharing:**
- No sharing with unauthorized personnel
- No public release of patient images
- No sharing with third parties without IRB approval

---

## 6. Access Controls

### 6.1 Authorized Personnel

**Principal Investigator:**
- Full access to all data
- Approval authority for data access requests
- Responsible for compliance oversight

**Co-Investigators:**
- Access to data needed for assigned tasks
- No access to patient identifiers (if any)
- Must complete training and sign agreements

**IT Support:**
- Minimal access (system maintenance only)
- No access to research data
- Must sign confidentiality agreements

### 6.2 Access Management

**Account Management:**
- Unique accounts for each user
- Strong password requirements
- Regular account review and deactivation of unused accounts

**Access Logging:**
- All data access logged (timestamp, user, action)
- Regular audit of access logs
- Investigation of suspicious activity

**Training:**
- All personnel must complete:
  - HIPAA training
  - Research ethics training
  - Data security training
  - Annual refresher training

---

## 7. Cleanup Policies

### 7.1 Data Retention

**Retention Period:**
- As per IRB approval (typically 7 years post-study completion)
- Or as required by institutional policy
- Whichever is longer

**Retention Documentation:**
- Maintain inventory of all stored data
- Document retention period for each dataset
- Track retention expiration dates

### 7.2 Secure Deletion

**Deletion Procedures:**
1. Verify retention period has expired
2. Obtain PI approval for deletion
3. Securely delete all copies:
   - Primary storage
   - Backups
   - Temporary files
   - Cache files
4. Document deletion (date, method, personnel)
5. Verify deletion completion

**Deletion Methods:**
- Cryptographic erasure (overwrite with random data)
- Physical destruction (for physical media)
- Secure deletion software (DBAN, etc.)
- Verify deletion cannot be recovered

### 7.3 Equipment Disposal

**Before Disposal:**
- Securely erase all data
- Verify deletion
- Document disposal
- Follow institutional IT disposal procedures

---

## 8. Incident Response

### 8.1 Security Incident Definition

**Incidents Include:**
- Unauthorized access to data
- Data breach or exposure
- Loss or theft of devices containing data
- Malware infection
- Physical security breach

### 8.2 Incident Response Procedures

**Immediate Actions:**
1. Contain the incident (isolate affected systems)
2. Assess the scope and impact
3. Notify Principal Investigator immediately
4. Notify IT security (if institutional)
5. Document all actions taken

**Reporting:**
- Report to IRB within 24 hours (if required)
- Report to institutional security office
- Report to data provider (if applicable)
- Follow institutional incident response procedures

**Remediation:**
- Address vulnerabilities
- Update security measures
- Provide additional training if needed
- Review and update policies

---

## 9. Compliance Monitoring

### 9.1 Regular Audits

**Frequency:**
- Quarterly access log reviews
- Annual security assessments
- Ad-hoc audits as needed

**Audit Scope:**
- Access logs review
- Storage location verification
- Encryption verification
- Training compliance
- Policy adherence

### 9.2 Documentation

**Maintain Records:**
- Access logs
- Security incidents
- Training completion
- Policy updates
- Audit reports

---

## 10. Policy Updates

**Review Schedule:**
- Annual review
- Updates as needed (regulatory changes, incidents)
- Version control of policy documents

**Approval:**
- Principal Investigator approval required
- IRB notification (if significant changes)
- Team notification and training on updates

---

## 11. Contact Information

**Principal Investigator:**  
[Your Name]  
[Email]  
[Phone]

**IT Security:**  
[IT Security Contact]

**IRB Office:**  
[IRB Contact]

---

**Policy Version:** 1.0  
**Last Updated:** November 2024  
**Next Review:** November 2025



