# Tooth-AI: Unified Inference Pipeline

Unified pipeline for automated tooth detection and FDI numbering from orthopantomogram (OPG) images.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r api/requirements.txt
pip install streamlit  # For UI
```

### Basic Usage

**CLI:**
```bash
python cli/predict.py --image path/to/image.png --output result.json --vis result.png
```

**API:**
```bash
python api/server.py --port 8000
```

**UI:**
```bash
streamlit run ui/app.py
```

## Project Structure

- `inference/` - Core inference engine
- `api/` - FastAPI server for deployment
- `ui/` - Streamlit clinician interface
- `cli/` - Command-line tool
- `export/` - Model export utilities
- `models/` - Trained model weights

## Model Requirements

Place trained models in `models/` directory:
- `maskrcnn_final.pth` - Mask R-CNN model
- `effnet_fdi_final.pth` - EfficientNet classifier
- `config.yaml` - Detectron2 config file

## Documentation

See `FINAL_POC_REPORT.md` for complete documentation.

## Citation

Dataset: Niha Adnan et al., "Orthopantomogram teeth segmentation and numbering dataset", Data in Brief, 2024.



