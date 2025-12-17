# Tooth-AI v0.1.0 POC - Release Notes

**Release Date:** November 2024  
**Version:** 0.1.0-poc  
**Status:** Proof of Concept

---

## Overview

Tooth-AI is an automated tooth detection and numbering system for orthopantomogram (OPG) X-ray images. This proof-of-concept (POC) release demonstrates end-to-end functionality including instance segmentation, FDI tooth numbering, and a clinician-friendly interface.

---

## Features

### Core Functionality

- **Instance Segmentation**: Mask R-CNN (ResNet-50 FPN) for tooth detection and segmentation
- **FDI Numbering**: Automatic tooth numbering using FDI notation (11-48)
- **ROI Classification**: EfficientNet-B0 classifier for improved numbering accuracy
- **Anatomical Ordering**: Geometric and anatomical logic for correct tooth sequence
- **Unified Pipeline**: Integrated inference engine combining all components

### Deployment & Infrastructure

- **Docker Support**: Production-ready containerization with GPU support
- **FastAPI Server**: RESTful API for programmatic access
- **Streamlit UI**: Interactive web interface for clinicians
- **ONNX Export**: Model optimization for deployment
- **TensorRT Support**: High-performance inference engines (FP16)
- **CI/CD**: GitHub Actions workflow for automated testing

### Validation & Testing

- **Batch Validation**: Comprehensive metrics on validation and training sets
- **Stress Testing**: Robustness evaluation with 9 distortion types
- **Regression Testing**: Stability tracking across model updates
- **Failure Case Analysis**: Identification and visualization of worst cases

---

## Model Cards

### Mask R-CNN (ResNet-50 FPN)

- **Architecture**: Mask R-CNN with ResNet-50 Feature Pyramid Network
- **Input Size**: 1024×512 (configurable)
- **Classes**: 32 FDI tooth classes
- **Pretrained**: COCO instance segmentation
- **Training**: Transfer learning on Niha Adnan et al. dataset
- **Performance**: 
  - Mean Mask IoU: TBD (see Phase 6 report)
  - Mean FDI Accuracy: TBD

### EfficientNet-B0 ROI Classifier

- **Architecture**: EfficientNet-B0
- **Input Size**: 128×128 (tooth ROI crops)
- **Classes**: 32 FDI classes
- **Pretrained**: ImageNet
- **Training**: Fine-tuned on extracted tooth ROIs
- **Purpose**: Correct low-confidence Mask R-CNN predictions

---

## Benchmarks

### Inference Performance

| Engine | Latency (ms) | Throughput (imgs/sec) | Memory (MB) |
|--------|--------------|----------------------|--------------|
| PyTorch (Mask R-CNN) | TBD | TBD | TBD |
| PyTorch (EfficientNet) | TBD | TBD | TBD |
| ONNX Runtime | TBD | TBD | TBD |
| TensorRT FP16 | TBD | TBD | TBD |

*Benchmarks will be populated after running `benchmark_inference.py`*

### Accuracy Metrics

- **Mask IoU**: TBD (see Phase 6 validation report)
- **FDI Accuracy**: TBD
- **Detection Rate**: TBD (expected: 28-32 teeth per OPG)

---

## Installation

### Prerequisites

- Docker with GPU support (NVIDIA Container Toolkit)
- CUDA 11.7+ (for GPU inference)
- Python 3.10+ (for local development)

### Quick Start with Docker

```bash
# Clone repository
git clone https://github.com/your-org/tooth-ai.git
cd tooth-ai

# Build and run demo
./demo.sh
```

The demo script will:
1. Build Docker image
2. Start container with GPU support
3. Launch FastAPI (port 8000) and Streamlit UI (port 8501)
4. Open browser to Streamlit interface

### Local Development

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r api/requirements.txt
pip install streamlit

# Run API
uvicorn api.server:app --host 0.0.0.0 --port 8000

# Run UI (separate terminal)
streamlit run ui/app.py --server.port 8501
```

---

## Usage Examples

### CLI Tool

```bash
python cli/predict.py \
    --image path/to/opg.png \
    --output result.json \
    --visualization result.png
```

### API Request

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@opg.png" \
     -F "return_visualization=true" \
     -o result.json
```

### Python API

```python
from inference.engine import load_engine

# Load engine
engine = load_engine("models/")

# Run inference
result = engine.predict("opg.png", return_visualization=True)

# Access results
for tooth in result['teeth']:
    print(f"FDI: {tooth['fdi']}, Confidence: {tooth['score']:.2f}")
```

---

## Limitations

### Dataset Constraints

- **Small Dataset**: ~250 OPG images (Niha Adnan et al., 2024)
- **Single Source**: Limited diversity in imaging systems
- **Class Imbalance**: Some FDI classes underrepresented
- **Annotation Quality**: Manual annotations may have inconsistencies

### Technical Limitations

- **Quadrant Detection**: Relies on centroid-based heuristics
- **Missing Teeth**: System doesn't explicitly detect gaps
- **Rotation Sensitivity**: Performance may degrade on rotated images
- **Resolution**: Trained on 1024×512, may not generalize to all resolutions

### Clinical Limitations

- **POC Status**: Not validated for clinical use
- **No Regulatory Approval**: Research/educational purposes only
- **Expert Review Required**: All predictions should be reviewed by clinicians
- **No Medical Advice**: System is a tool, not a replacement for clinical judgment

---

## Next Steps

### Short-term Improvements

1. **Larger Dataset**: Collect >1000 diverse OPG images
2. **Robust Quadrant Detection**: Implement jaw line detection
3. **Missing Tooth Detection**: Explicit gap detection
4. **Multi-center Validation**: Test across different imaging systems

### Long-term Goals

1. **Clinical Validation**: Multi-center study with expert annotations
2. **Regulatory Pathway**: FDA/CE marking preparation
3. **Production Deployment**: Cloud-based service
4. **Integration**: PACS/DICOM integration

---

## Citation

### Dataset

```
Niha Adnan et al., "Orthopantomogram teeth segmentation and numbering dataset",
Data in Brief, 2024.
```

### Code

```
Tooth-AI POC v0.1.0
MIT License
Copyright (c) 2024
```

---

## License

- **Code**: MIT License
- **Dataset**: CC BY-NC 4.0 (Niha Adnan et al., 2024)
- **Models**: Trained weights are provided for research use

---

## Support & Contributing

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Documentation**: See `README.md` and `docs/` directory
- **Contributing**: Contributions welcome! Please see CONTRIBUTING.md

---

## Acknowledgments

- **Dataset**: Niha Adnan et al. for the OPG dataset
- **Frameworks**: Detectron2, PyTorch, Streamlit, FastAPI
- **Community**: Open source contributors and reviewers

---

## Changelog

### v0.1.0-poc (2024-11-16)

- Initial POC release
- Mask R-CNN instance segmentation
- EfficientNet ROI classifier
- Unified inference pipeline
- Docker deployment
- FastAPI server
- Streamlit UI
- ONNX/TensorRT export
- Comprehensive validation suite

---

**For detailed technical documentation, see:**
- `README.md` - Project overview
- `FINAL_POC_REPORT.md` - Complete POC report
- `docs/` - Additional documentation



