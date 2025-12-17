"""
FastAPI server for tooth detection inference API.
"""

import os
import sys
import base64
import io
from pathlib import Path
from typing import List, Optional
import uvicorn

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import cv2
from PIL import Image

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.engine import ToothDetectionEngine, load_engine
from inference.visualize import visualize_predictions, save_visualization


# Initialize FastAPI app
app = FastAPI(
    title="Tooth-AI Inference API",
    description="API for tooth detection and FDI numbering",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine
engine: Optional[ToothDetectionEngine] = None


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    teeth: List[dict]
    num_detections: int
    image_shape: List[int]
    metadata: dict


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    image_paths: List[str]


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    global engine
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    
    try:
        if os.path.exists(os.path.join(model_dir, 'maskrcnn_final.pth')):
            engine = load_engine(model_dir)
            print("✓ Models loaded successfully")
        else:
            print("⚠ Models not found, API will return errors until models are loaded")
    except Exception as e:
        print(f"⚠ Error loading models: {e}")


@app.get("/ping")
async def ping():
    """Health check endpoint."""
    return {"status": "ok", "engine_loaded": engine is not None}


@app.get("/version")
async def version():
    """Get API version."""
    return {
        "version": "1.0.0",
        "api_name": "Tooth-AI Inference API",
        "engine_loaded": engine is not None
    }


@app.get("/model_info")
async def model_info():
    """Get model information."""
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {
        "maskrcnn_loaded": True,
        "effnet_loaded": True,
        "num_classes": engine.num_classes,
        "confidence_threshold": engine.confidence_threshold,
        "device": engine.device
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), return_visualization: bool = False):
    """
    Run inference on uploaded image.
    
    Args:
        file: Uploaded image file
        return_visualization: Whether to include visualization in response
    
    Returns:
        Prediction results
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Please load models first.")
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Save temporarily
    temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tmp')
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"temp_{file.filename}")
    cv2.imwrite(temp_path, image)
    
    try:
        # Run inference
        results = engine.predict(temp_path, return_visualization=return_visualization)
        
        # Clean up temp file
        os.remove(temp_path)
        
        return results
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(request: BatchPredictionRequest):
    """
    Run inference on multiple images.
    
    Args:
        request: Batch prediction request with image paths
    
    Returns:
        List of prediction results
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    results = engine.predict_batch(request.image_paths)
    return {"results": results, "num_images": len(results)}


@app.post("/predict_from_base64")
async def predict_from_base64(image_base64: str, return_visualization: bool = False):
    """
    Run inference on base64-encoded image.
    
    Args:
        image_base64: Base64-encoded image string
        return_visualization: Whether to include visualization
    
    Returns:
        Prediction results
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Decode base64
        image_bytes = base64.b64decode(image_base64)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Save temporarily
        temp_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tmp')
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, "temp_base64.png")
        cv2.imwrite(temp_path, image)
        
        # Run inference
        results = engine.predict(temp_path, return_visualization=return_visualization)
        
        # Clean up
        os.remove(temp_path)
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.get("/visualization/{image_id}")
async def get_visualization(image_id: str):
    """
    Get visualization for a previously processed image.
    Note: This is a placeholder - in production, store visualizations with IDs.
    """
    return {"message": "Visualization endpoint - implement storage system"}


def main():
    """Run the API server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Tooth-AI Inference API')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8000, help='Port number')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == '__main__':
    main()

