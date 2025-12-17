"""
Visualization utilities for tooth detection results.
"""

import cv2
import numpy as np
from typing import List, Dict
import os


def get_color_for_fdi(fdi: int) -> tuple:
    """Get color for FDI label."""
    # Color palette for different tooth types
    colors = [
        (255, 0, 0),    # Red
        (0, 255, 0),    # Green
        (0, 0, 255),    # Blue
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
    ]
    
    # Use FDI number to select color
    color_idx = (fdi - 1) % len(colors)
    return colors[color_idx]


def visualize_predictions(image: np.ndarray, predictions: List[Dict],
                         show_boxes: bool = True, show_labels: bool = True,
                         show_masks: bool = True, alpha: float = 0.5) -> np.ndarray:
    """
    Visualize predictions on image.
    
    Args:
        image: Original image
        predictions: List of prediction dictionaries
        show_boxes: Whether to show bounding boxes
        show_labels: Whether to show FDI labels
        show_masks: Whether to show mask overlays
        alpha: Transparency for mask overlay
    
    Returns:
        Visualization image
    """
    vis_image = image.copy()
    
    # Create overlay for masks
    if show_masks:
        overlay = vis_image.copy()
    
    for pred in predictions:
        fdi = pred.get('fdi', 0)
        bbox = pred.get('bbox', [])
        mask = pred.get('mask')
        confidence = pred.get('final_confidence', pred.get('confidence', 0.0))
        centroid = pred.get('centroid', (0, 0))
        method_used = pred.get('method_used', 'maskrcnn')
        correction_applied = pred.get('correction_applied', False)
        
        color = get_color_for_fdi(fdi)
        
        # Draw mask overlay
        if show_masks and mask is not None:
            if isinstance(mask, dict):
                # RLE format - decode
                from pycocotools import mask as mask_utils
                mask_array = mask_utils.decode(mask)
                mask_bool = mask_array.astype(bool)
            else:
                mask_bool = mask.astype(bool) if isinstance(mask, np.ndarray) else None
            
            if mask_bool is not None:
                overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + np.array(color) * alpha
        
        # Draw bounding box
        if show_boxes and len(bbox) == 4:
            x1, y1, x2, y2 = [int(x) for x in bbox]
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        if show_labels:
            label_text = f"FDI:{fdi}"
            if confidence > 0:
                label_text += f" ({confidence:.2f})"
            
            if correction_applied:
                label_text += " [CORR]"
            
            # Method indicator
            if method_used == 'effnet':
                label_text += " [E]"
            
            # Position label
            x1, y1 = int(bbox[0]), int(bbox[1]) if len(bbox) == 4 else (0, 0)
            
            # Background for text
            (text_width, text_height), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(vis_image, (x1, y1 - text_height - 5),
                         (x1 + text_width, y1), color, -1)
            
            # Text
            cv2.putText(vis_image, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw centroid
        if centroid:
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.circle(vis_image, (cx, cy), 3, color, -1)
    
    # Blend overlay
    if show_masks:
        vis_image = cv2.addWeighted(overlay, alpha, vis_image, 1 - alpha, 0)
    
    return vis_image


def save_visualization(image: np.ndarray, output_path: str):
    """Save visualization image."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def create_comparison_view(original: np.ndarray, predictions: List[Dict],
                          output_path: str):
    """Create side-by-side comparison view."""
    vis_image = visualize_predictions(original, predictions)
    
    # Resize if needed
    h, w = original.shape[:2]
    if w > 1200:
        scale = 1200 / w
        new_w = 1200
        new_h = int(h * scale)
        original = cv2.resize(original, (new_w, new_h))
        vis_image = cv2.resize(vis_image, (new_w, new_h))
    
    # Concatenate side by side
    comparison = np.hstack([original, vis_image])
    
    save_visualization(comparison, output_path)



