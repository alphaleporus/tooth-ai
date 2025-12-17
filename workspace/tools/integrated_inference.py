#!/usr/bin/env python3
"""
Integrated inference pipeline combining Mask R-CNN and EfficientNet classifier
with anatomical FDI correction logic.
"""

import argparse
import os
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import torch
import torch.nn.functional as F
from torchvision import transforms

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
import timm


# FDI quadrant mapping
FDI_QUADRANTS = {
    'upper_right': list(range(11, 19)),  # 11-18
    'upper_left': list(range(21, 29)),  # 21-28
    'lower_left': list(range(31, 39)),   # 31-38
    'lower_right': list(range(41, 49))   # 41-48
}


def load_effnet_model(model_path: str, num_classes: int = 32, device: str = 'cuda'):
    """Load trained EfficientNet model."""
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def preprocess_roi(image: np.ndarray, size: Tuple[int, int] = (128, 128)) -> torch.Tensor:
    """Preprocess ROI image for EfficientNet."""
    # Resize
    image = cv2.resize(image, size)
    
    # Convert BGR to RGB
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image).unsqueeze(0)


def classify_roi(model, roi_image: np.ndarray, device: str = 'cuda') -> Tuple[int, float]:
    """
    Classify ROI using EfficientNet.
    
    Returns:
        (predicted_class, confidence)
    """
    with torch.no_grad():
        input_tensor = preprocess_roi(roi_image).to(device)
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        
        # Map back to FDI (assuming model outputs 0-31, map to 1-32)
        predicted_class = predicted.item() + 1
        confidence_score = confidence.item()
        
        return predicted_class, confidence_score


def determine_quadrant(centroid_y: float, centroid_x: float, 
                       image_height: int, image_width: int) -> str:
    """
    Determine tooth quadrant based on centroid position.
    
    Args:
        centroid_y: Y coordinate of centroid
        centroid_x: X coordinate of centroid
        image_height: Image height
        image_width: Image width
    
    Returns:
        Quadrant name: 'upper_right', 'upper_left', 'lower_left', 'lower_right'
    """
    # Split by vertical center (upper vs lower)
    vertical_split = image_height / 2
    is_upper = centroid_y < vertical_split
    
    # Split by horizontal center (left vs right)
    horizontal_split = image_width / 2
    is_right = centroid_x > horizontal_split
    
    if is_upper and is_right:
        return 'upper_right'
    elif is_upper and not is_right:
        return 'upper_left'
    elif not is_upper and not is_right:
        return 'lower_left'
    else:
        return 'lower_right'


def map_to_fdi_sequence(quadrant: str, sorted_indices: List[int]) -> Dict[int, int]:
    """
    Map sorted tooth indices to FDI numbers based on quadrant.
    
    Args:
        quadrant: Quadrant name
        sorted_indices: List of indices sorted by x-coordinate
        num_teeth: Number of teeth in quadrant
    
    Returns:
        Dictionary mapping instance index to FDI number
    """
    fdi_sequence = FDI_QUADRANTS[quadrant]
    
    # Map indices to FDI numbers
    fdi_mapping = {}
    for i, idx in enumerate(sorted_indices):
        if i < len(fdi_sequence):
            fdi_mapping[idx] = fdi_sequence[i]
        else:
            # If more teeth than expected, assign to closest available
            fdi_mapping[idx] = fdi_sequence[-1]
    
    return fdi_mapping


def apply_anatomical_correction(instances: List[Dict], image_shape: Tuple[int, int]) -> Dict[int, int]:
    """
    Apply anatomical sorting and FDI correction.
    
    Args:
        instances: List of instance dictionaries with bbox, centroid, etc.
        image_shape: (height, width) of image
    
    Returns:
        Dictionary mapping instance index to corrected FDI number
    """
    h, w = image_shape
    
    # Group by quadrant
    quadrant_groups = defaultdict(list)
    
    for i, inst in enumerate(instances):
        centroid = inst['centroid']
        quadrant = determine_quadrant(centroid[1], centroid[0], h, w)
        quadrant_groups[quadrant].append((i, inst))
    
    # Sort each quadrant by x-coordinate and assign FDI
    final_fdi_mapping = {}
    
    for quadrant, group in quadrant_groups.items():
        # Sort by x-coordinate (left to right)
        sorted_group = sorted(group, key=lambda x: x[1]['centroid'][0])
        sorted_indices = [idx for idx, _ in sorted_group]
        
        # Map to FDI sequence
        fdi_mapping = map_to_fdi_sequence(quadrant, sorted_indices)
        final_fdi_mapping.update(fdi_mapping)
    
    return final_fdi_mapping


def integrated_inference(image_path: str, maskrcnn_predictor: DefaultPredictor,
                         effnet_model, image_shape: Tuple[int, int],
                         confidence_threshold: float = 0.85,
                         device: str = 'cuda') -> Dict:
    """
    Run integrated inference combining Mask R-CNN and EfficientNet.
    
    Args:
        image_path: Path to input image
        maskrcnn_predictor: Detectron2 predictor
        effnet_model: EfficientNet classifier model
        image_shape: (height, width) of image
        confidence_threshold: Threshold for using classifier
        device: Device for EfficientNet
    
    Returns:
        Dictionary with predictions and metadata
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    h, w = image.shape[:2]
    
    # Run Mask R-CNN inference
    outputs = maskrcnn_predictor(image)
    instances = outputs["instances"]
    
    # Extract instances
    instance_data = []
    
    for i in range(len(instances)):
        mask = instances.pred_masks[i].cpu().numpy()
        box = instances.pred_boxes.tensor[i].cpu().numpy()
        score = instances.scores[i].item()
        pred_class = instances.pred_classes[i].item()
        
        # Calculate centroid
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > 0 and len(x_coords) > 0:
            centroid = (x_coords.mean(), y_coords.mean())
        else:
            # Use bbox center
            centroid = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        
        # Crop ROI
        x1, y1, x2, y2 = box.astype(int)
        x1 = max(0, x1 - 10)
        y1 = max(0, y1 - 10)
        x2 = min(w, x2 + 10)
        y2 = min(h, y2 + 10)
        roi = image[y1:y2, x1:x2]
        
        # Decide whether to use classifier
        use_classifier = score < confidence_threshold
        
        if use_classifier:
            # Run EfficientNet classifier
            effnet_class, effnet_conf = classify_roi(effnet_model, roi, device)
            final_class = effnet_class
            final_confidence = effnet_conf
            method_used = 'effnet'
        else:
            # Use Mask R-CNN prediction
            final_class = pred_class + 1  # Map to 1-32
            final_confidence = score
            method_used = 'maskrcnn'
        
        instance_data.append({
            'index': i,
            'bbox': box.tolist(),
            'mask': mask,
            'centroid': centroid,
            'maskrcnn_class': pred_class + 1,
            'maskrcnn_confidence': score,
            'final_class': final_class,
            'final_confidence': final_confidence,
            'method_used': method_used,
            'roi': roi
        })
    
    # Apply anatomical correction
    anatomical_fdi = apply_anatomical_correction(instance_data, (h, w))
    
    # Merge classifier predictions with anatomical logic
    final_predictions = []
    for inst in instance_data:
        idx = inst['index']
        classifier_fdi = inst['final_class']
        anatomical_fdi_val = anatomical_fdi.get(idx, classifier_fdi)
        
        # If classifier and anatomical disagree, prefer anatomical for ordering
        # but keep classifier confidence
        if abs(classifier_fdi - anatomical_fdi_val) > 2:  # Significant disagreement
            final_fdi = anatomical_fdi_val
            correction_applied = True
        else:
            final_fdi = classifier_fdi
            correction_applied = False
        
        final_predictions.append({
            'index': idx,
            'fdi_label': int(final_fdi),
            'confidence': float(inst['final_confidence']),
            'bbox': inst['bbox'],
            'mask_area': float(inst['mask'].sum()),
            'centroid': inst['centroid'],
            'method_used': inst['method_used'],
            'correction_applied': correction_applied,
            'classifier_fdi': classifier_fdi,
            'anatomical_fdi': anatomical_fdi_val
        })
    
    return {
        'predictions': final_predictions,
        'num_detections': len(final_predictions),
        'image_shape': (h, w)
    }


def visualize_results(image: np.ndarray, predictions: List[Dict], output_path: str):
    """Visualize predictions on image."""
    vis_image = image.copy()
    
    for pred in predictions:
        bbox = pred['bbox']
        fdi = pred['fdi_label']
        confidence = pred['confidence']
        centroid = pred['centroid']
        
        # Draw bounding box
        x1, y1, x2, y2 = [int(x) for x in bbox]
        color = (0, 255, 0) if pred['method_used'] == 'maskrcnn' else (255, 0, 0)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_text = f"FDI:{fdi} ({confidence:.2f})"
        if pred['correction_applied']:
            label_text += " [CORR]"
        
        cv2.putText(vis_image, label_text, (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw centroid
        cv2.circle(vis_image, (int(centroid[0]), int(centroid[1])), 3, color, -1)
    
    cv2.imwrite(output_path, vis_image)


def main():
    parser = argparse.ArgumentParser(description='Integrated inference with FDI correction')
    parser.add_argument('--image', type=str, required=True, help='Input image path')
    parser.add_argument('--maskrcnn', type=str, required=True, help='Path to Mask R-CNN model')
    parser.add_argument('--effnet', type=str, required=True, help='Path to EfficientNet model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--out', type=str, required=True, help='Output image path')
    parser.add_argument('--confidence-threshold', type=float, default=0.85,
                       help='Confidence threshold for using classifier')
    parser.add_argument('--num-classes', type=int, default=32, help='Number of FDI classes')
    parser.add_argument('--device', type=str, default='cuda', help='Device for inference')
    
    args = parser.parse_args()
    
    # Setup Mask R-CNN predictor
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.maskrcnn
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    maskrcnn_predictor = DefaultPredictor(cfg)
    
    # Load EfficientNet model
    print("Loading EfficientNet model...")
    effnet_model = load_effnet_model(args.effnet, args.num_classes, args.device)
    
    # Load image to get shape
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Could not load image: {args.image}")
    h, w = image.shape[:2]
    
    # Run integrated inference
    print("Running integrated inference...")
    results = integrated_inference(
        args.image, maskrcnn_predictor, effnet_model,
        (h, w), args.confidence_threshold, args.device
    )
    
    # Visualize
    print("Generating visualization...")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    visualize_results(image, results['predictions'], args.out)
    
    # Save JSON data
    json_path = args.out.replace('.png', '_data.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nInference complete!")
    print(f"  Detections: {results['num_detections']}")
    print(f"  Visualization: {args.out}")
    print(f"  Data JSON: {json_path}")
    
    # Print summary
    maskrcnn_count = sum(1 for p in results['predictions'] if p['method_used'] == 'maskrcnn')
    effnet_count = sum(1 for p in results['predictions'] if p['method_used'] == 'effnet')
    corrected_count = sum(1 for p in results['predictions'] if p['correction_applied'])
    
    print(f"\nSummary:")
    print(f"  Mask R-CNN used: {maskrcnn_count}")
    print(f"  EfficientNet used: {effnet_count}")
    print(f"  Anatomical corrections: {corrected_count}")


if __name__ == '__main__':
    main()



