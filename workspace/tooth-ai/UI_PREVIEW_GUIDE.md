# UI Preview Guide

## Quick Start

The Streamlit UI has been launched. Access it at:

**http://localhost:8501**

## UI Features

### Main Interface
- **Image Upload:** Drag and drop or browse to upload OPG images
- **Automatic Processing:** Runs inference when image is uploaded
- **Visual Results:** Displays image with tooth detection overlays
- **FDI Numbering:** Shows tooth numbers with confidence scores

### Sidebar Controls
- **Model Configuration:** Load models from custom directory
- **View Toggles:** Show/hide masks, bounding boxes, labels
- **Export Options:** Download results as PNG or JSON
- **Feedback:** Flag errors for review

### Expected Behavior (Without Models)

Since models are not yet trained, you'll see:
- ⚠️ Warning message about missing models
- UI interface fully functional
- Upload and visualization features available
- Error message when attempting inference

## Manual Launch

If you need to restart the UI:

```bash
cd workspace/tooth-ai
streamlit run ui/app.py
```

Or with custom port:

```bash
streamlit run ui/app.py --server.port 8502
```

## Troubleshooting

**UI won't start:**
- Check if port 8501 is already in use
- Install streamlit: `pip install streamlit`
- Check Python version: `python3 --version` (should be 3.10+)

**Models not loading:**
- This is expected - models need to be trained first
- UI will show error message but still be functional
- Once models are trained, copy them to `models/` directory

**Import errors:**
- Install dependencies: `pip install -r requirements_full.txt`
- Check that all required packages are installed

## UI Screenshots

Once models are available, the UI will show:
- Original OPG image
- Overlaid tooth masks (colored by FDI number)
- Bounding boxes around each tooth
- FDI labels with confidence scores
- Statistics panel (detection count, method used, etc.)

