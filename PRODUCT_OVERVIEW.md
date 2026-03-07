# 🦷 Tooth-AI: Product Overview

**Tooth-AI** is a premium, AI-powered dental diagnostic system designed to assist dental professionals in analyzing panoramic X-rays. By leveraging advanced computer vision models (Detectron2), it automates the detection of teeth and pathological anomalies with high precision.

## 🌟 Key Features

### 1. Multi-Model Intelligence
Tooth-AI offers flexibility through specialized AI models tailored for different clinical needs:
*   **Hybrid 9-Class Model (Standard)**:
    *   **Purpose**: Complete dental charting.
    *   **Detects**: Both **Teeth** (for FDI numbering) and **Anomalies**.
    *   **Best For**: General checkups and generating full dental charts.
*   **RTX4060 48k Model (High-Sensitivity)**:
    *   **Purpose**: Focused pathology detection.
    *   **Detects**: **Anomalies Only** (Caries, Crowns, Fillings, Root Canals).
    *   **Feature**: Automatically filters out normal teeth (IDs 0-33) to focus purely on issues.
    *   **Best For**: Second opinions on pathology and identifying subtle issues.

### 2. Smart Ensemble Inference 🧠
The system doesn't just look at the image once. It employs a **"Smart Ensemble"** strategy to ensure robustness:
*   **Standard Pass**: Analyzes the original image.
*   **CLAHE Pass**: Applies *Contrast Limited Adaptive Histogram Equalization* to see through spinal shadows (anterior teeth).
*   **Gamma Pass**: Brightens dark areas to recover details in underexposed quadrants.
*   **Flip Augmentation**: Flips the image horizontally to break directional bias, then fuses results.
*   **Weighted Box Fusion (WBF)**: Combines all these passes into a single, highly accurate set of predictions.

### 3. Advanced Visualization 🎨
We prioritize clarity and readability:
*   **Mask-Only Display**: No clutter from bounding boxes. We use precise segmentation masks conforming to the shape of the tooth/anomaly.
*   **Color Theory**: 
    *   🟢 **Teeth**: Subtle Green (Healthy/Calming).
    *   🔴 **Caries**: Red (Alert/Urgent).
    *   🟡 **Crowns**: Gold/Yellow (Restoration).
    *   🟠 **Fillings**: Orange (Treatment).
*   **Smart Labeling**: Labels automatically shift positions to avoid overlapping each other, ensuring legibility even in crowded areas.

### 4. Automated Reporting 📋
Generates professional clinical reports instantly:
*   **FDI Dental Chart**: Automatically assigns numbers (ISO 3950 notation) to detected teeth.
*   **Findings Table**: detailed list of every anomaly with confidence scores.
*   **Unmarked Anomalies**: Lists detected issues that couldn't be linked to a specific tooth (e.g., in edentulous areas).
*   **PDF Download**: One-click generation of a patient-ready PDF report including the processed X-ray image.

---

## 🔍 Supported Detections

| Class Name | Type | Description |
| :--- | :--- | :--- |
| **Tooth** | Anatomy | Normal dentition (filtered in RTX mode) |
| **Caries** | Pathology | Tooth decay / cavities |
| **Crown** | Restoration | Artificial tooth cap (Gold/Metal/Ceramic) |
| **Filling** | Restoration | Restorative material (Amalgam/Composite) |
| **Root Canal** | Treatment | Endodontic treatment (Obturation/Filling) |
| **Implant** | Restoration | Artificial tooth root |
| **Retained Root** | Pathology | Root fragment left after extraction |
| **Metal Post** | Restoration | Post-and-core restoration |

---

## 🛠️ Technical Workflow

1.  **Image Upload**: User provides a panoramic X-ray (JPG/PNG).
2.  **Preprocessing**: Image is converted to BGR, resized if necessary.
3.  **AI Inference**: The selected model (Hybrid or RTX4060) processes the image using the Smart Ensemble pipeline.
4.  **Geometric Analysis** (Hybrid Mode):
    *   **NMS**: Removes duplicate detections.
    *   **Jaw Splitting**: Uses a Y-Median split to separate Upper vs. Lower jaw.
    *   **FDI Sorting**: spatially sorts teeth to assign correct quadrant numbers (11-18, 21-28, etc.).
5.  **Anomaly Mapping**: Anomalies are geometrically mapped to the nearest overlapping tooth.
6.  **Reporting**: Final results are compiled into visual overlays and PDF documents.

---

## 🚀 How to Use

1.  **Launch the App**: `streamlit run app.py`
2.  **Select Model**: Open the Sidebar ⚙️ and choose:
    *   *Hybrid 9-Class* for full charting.
    *   *RTX4060 48k* for pathology check.
3.  **Upload Image**: Drag & Drop your X-ray.
4.  **Review**: Check the "AI Analysis Results" panel for visualization and metrics.
5.  **Download Report**: Click "Download Report (PDF)" to save the findings.
