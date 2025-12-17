"""
Unified inference engine for tooth detection pipeline.
Combines Mask R-CNN, EfficientNet classifier, and anatomical ordering.
"""

import os
import json
import base64
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from torchvision import transforms

# Optional imports: allow module import without full runtime deps so that
# the UI and other components can load, even if inference cannot run yet.
try:
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.structures import Instances

    _HAS_DETECTRON2 = True
except ImportError:
    # Detectron2 is required only when actually running inference.
    _HAS_DETECTRON2 = False

import timm
from pycocotools import mask as mask_utils

from .preprocess import preprocess_for_maskrcnn, preprocess_roi_for_classifier
from .postprocess import (
    extract_roi,
    select_low_confidence_instances,
    postprocess_predictions,
)


class ToothDetectionEngine:
    """Unified inference engine for tooth detection."""
    
    def __init__(self, maskrcnn_model_path: str, effnet_model_path: str,
                 config_path: str, num_classes: int = 32,
                 confidence_threshold: float = 0.85,
                 device: str = 'cuda'):
        """
        Initialize inference engine.
        
        Args:
            maskrcnn_model_path: Path to Mask R-CNN model
            effnet_model_path: Path to EfficientNet classifier
            config_path: Path to Detectron2 config
            num_classes: Number of FDI classes
            confidence_threshold: Threshold for using classifier
            device: Device for inference
        """
        # Detectron2 is required for inference; if it's not installed,
        # fail fast with a clear, user-friendly error message.
        if not _HAS_DETECTRON2:
            raise ImportError(
                "Detectron2 is not installed, but is required to run Mask R-CNN inference.\n\n"
                "To install Detectron2, please follow the official instructions for your platform:\n"
                "  https://detectron2.readthedocs.io/en/latest/tutorials/install.html\n\n"
                "For CPU-only (example):\n"
                "  pip install 'git+https://github.com/facebookresearch/detectron2.git'\n\n"
                "Note: You can still import the module and load the UI without Detectron2,\n"
                "but actual inference will not work until Detectron2 is installed."
            )

        self.confidence_threshold = confidence_threshold
        self.device = device
        self.num_classes = num_classes
        
        # Load Mask R-CNN
        print("Loading Mask R-CNN model...")
        self.cfg = get_cfg()
        self.cfg.merge_from_file(config_path)
        self.cfg.MODEL.WEIGHTS = maskrcnn_model_path
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        self.maskrcnn_predictor = DefaultPredictor(self.cfg)
        
        # Load EfficientNet
        print("Loading EfficientNet classifier...")
        self.effnet_model = self._load_effnet(effnet_model_path)
        self.effnet_model.eval()
        
        # ROI preprocessing transform
        self.roi_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _load_effnet(self, model_path: str):
        """Load EfficientNet model."""
        model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=self.num_classes)
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        return model
    
    def _classify_roi(self, roi_image: np.ndarray) -> Tuple[int, float]:
        """Classify ROI using EfficientNet."""
        # Preprocess
        roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        roi_resized = cv2.resize(roi_rgb, (128, 128))
        
        # Convert to tensor
        roi_tensor = self.roi_transform(roi_resized).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.effnet_model(roi_tensor)
            probs = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
            # Map to FDI (0-31 -> 1-32)
            predicted_class = predicted.item() + 1
            confidence_score = confidence.item()
        
        return predicted_class, confidence_score
    
    def predict(self, image_path: str, return_visualization: bool = True) -> Dict:
        """
        Run unified inference on an image.
        
        Args:
            image_path: Path to input image
            return_visualization: Whether to generate visualization
        
        Returns:
            Dictionary with predictions and metadata
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        original_image = image.copy()
        h, w = image.shape[:2]
        
        # Step 1: Run Mask R-CNN
        outputs = self.maskrcnn_predictor(image)
        instances = outputs["instances"]
        
        # Extract instance data
        instance_data = []
        
        for i in range(len(instances)):
            mask = instances.pred_masks[i].cpu().numpy()
            box = instances.pred_boxes.tensor[i].cpu().numpy()
            score = instances.scores[i].item()
            pred_class = instances.pred_classes[i].item()
            
            # Calculate centroid
            y_coords, x_coords = np.where(mask)
            if len(y_coords) > 0 and len(x_coords) > 0:
                centroid = (float(x_coords.mean()), float(y_coords.mean()))
            else:
                centroid = (float((box[0] + box[2]) / 2), float((box[1] + box[3]) / 2))
            
            instance_data.append({
                'index': i,
                'bbox': box.tolist(),
                'mask': mask,
                'centroid': centroid,
                'maskrcnn_class': pred_class + 1,  # Map to 1-32
                'confidence': score
            })
        
        # Step 2: Identify low-confidence instances
        low_conf_indices = select_low_confidence_instances(instance_data, self.confidence_threshold)
        
        # Step 3: Run classifier on low-confidence instances
        classifier_predictions = {}
        for idx in low_conf_indices:
            inst = instance_data[idx]
            roi = extract_roi(image, inst['mask'], np.array(inst['bbox']))
            fdi, conf = self._classify_roi(roi)
            classifier_predictions[idx] = (fdi, conf)
        
        # Step 4: Post-process (anatomical ordering, fusion, duplicate resolution)
        processed = postprocess_predictions(instance_data, image, classifier_predictions, self.confidence_threshold)
        
        # Step 5: Format output
        teeth = []
        for inst in processed:
            # Convert mask to RLE
            mask_rle = self._mask_to_rle(inst['mask'])
            
            teeth.append({
                'fdi': int(inst['fdi']),
                'score': float(inst.get('final_confidence', inst['confidence'])),
                'bbox': inst['bbox'],
                'mask': mask_rle,
                'centroid': inst['centroid'],
                'method_used': 'effnet' if inst['index'] in classifier_predictions else 'maskrcnn',
                'correction_applied': inst.get('correction_applied', False)
            })
        
        result = {
            'teeth': teeth,
            'num_detections': len(teeth),
            'image_shape': [h, w],
            'metadata': {
                'maskrcnn_used': sum(1 for t in teeth if t['method_used'] == 'maskrcnn'),
                'effnet_used': sum(1 for t in teeth if t['method_used'] == 'effnet'),
                'corrections_applied': sum(1 for t in teeth if t['correction_applied'])
            }
        }
        
        # Generate visualization if requested
        if return_visualization:
            from .visualize import visualize_predictions
            vis_image = visualize_predictions(original_image, processed)
            result['visualization'] = self._image_to_base64(vis_image)
        
        return result
    
    def _mask_to_rle(self, mask: np.ndarray) -> str:
        """Convert binary mask to RLE string."""
        rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """Convert image to base64 string."""
        _, buffer = cv2.imencode('.png', image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        return image_base64
    
    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Run inference on multiple images."""
        results = []
        for image_path in image_paths:
            try:
                result = self.predict(image_path, return_visualization=False)
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({'error': str(e)})
        return results


def load_engine(model_dir: str, device: str = 'cuda') -> ToothDetectionEngine:
    """
    Load inference engine from model directory.
    
    Args:
        model_dir: Directory containing models and config
        device: Device for inference
    
    Returns:
        Initialized ToothDetectionEngine
    
    Raises:
        FileNotFoundError: If required model files are missing
    """
    maskrcnn_path = os.path.join(model_dir, 'maskrcnn_final.pth')
    effnet_path = os.path.join(model_dir, 'effnet_fdi_final.pth')
    config_path = os.path.join(model_dir, 'config.yaml')
    
    # Check for missing files
    missing_files = []
    if not os.path.exists(maskrcnn_path):
        missing_files.append(f"maskrcnn_final.pth (expected at {maskrcnn_path})")
    if not os.path.exists(effnet_path):
        missing_files.append(f"effnet_fdi_final.pth (expected at {effnet_path})")
    if not os.path.exists(config_path):
        missing_files.append(f"config.yaml (expected at {config_path})")
    
    if missing_files:
        error_msg = f"MODEL MISSING — cannot run inference.\n\n"
        error_msg += f"Missing files in {model_dir}:\n"
        for f in missing_files:
            error_msg += f"  - {f}\n"
        error_msg += "\nTo obtain models:\n"
        error_msg += "1. Train models in Phase 3 (Mask R-CNN) and Phase 4 (EfficientNet)\n"
        error_msg += "2. Copy trained models:\n"
        error_msg += "   cp workspace/phase3/exp_1024/model_1024_final.pth workspace/tooth-ai/models/maskrcnn_final.pth\n"
        error_msg += "   cp workspace/phase4/effnet_fdi_best.pth workspace/tooth-ai/models/effnet_fdi_final.pth\n"
        error_msg += "   cp workspace/configs/mask_rcnn_1024x512.yaml workspace/tooth-ai/models/config.yaml\n"
        raise FileNotFoundError(error_msg)
    
    return ToothDetectionEngine(
        maskrcnn_path, effnet_path, config_path,
        device=device
    )

