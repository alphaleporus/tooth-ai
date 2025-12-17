# Quick UI Start Guide

## Option 1: Using Preview Script

```bash
cd workspace/tooth-ai
python3 preview_ui.py
```

This will:
- Check dependencies
- Launch Streamlit UI
- Open browser at http://localhost:8501

## Option 2: Direct Streamlit Command

```bash
cd workspace/tooth-ai
streamlit run ui/app.py
```

Or with custom port:
```bash
streamlit run ui/app.py --server.port 8502
```

## Option 3: Using Python Module

```bash
cd workspace/tooth-ai
python3 -m streamlit run ui/app.py
```

## Install Streamlit (if needed)

```bash
# User installation (recommended)
pip3 install --user streamlit

# Or in virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install streamlit
```

## Access the UI

Once started, open your browser to:
- **http://localhost:8501** (default)
- Or the port shown in the terminal

## Expected UI Features

1. **Image Upload Section**
   - Drag and drop OPG images
   - Browse file system
   - Support for PNG, JPG, JPEG

2. **Results Display**
   - Original image
   - Overlaid tooth masks
   - FDI numbering
   - Confidence scores

3. **Sidebar Controls**
   - Model loading status
   - View toggles (masks, boxes, labels)
   - Export options
   - Feedback collection

4. **Statistics Panel**
   - Detection count
   - Method used (Mask R-CNN vs EfficientNet)
   - Correction statistics

## Without Models

The UI will:
- ✅ Load and display interface
- ✅ Accept image uploads
- ⚠️ Show error when attempting inference
- ✅ Display helpful error message about missing models

This is expected behavior until models are trained.

## Troubleshooting

**Port already in use:**
```bash
# Use different port
streamlit run ui/app.py --server.port 8502
```

**Import errors:**
```bash
# Install missing dependencies
pip install --user streamlit numpy opencv-python Pillow
```

**UI won't open browser:**
- Manually navigate to http://localhost:8501
- Check terminal for the exact URL

