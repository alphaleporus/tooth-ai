"""
Post-processing utilities for tooth detection pipeline.
Includes ROI extraction, confidence-based selection, and anatomical ordering.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


# FDI quadrant mapping
FDI_QUADRANTS = {
    'upper_right': list(range(11, 19)),  # 11-18
    'upper_left': list(range(21, 29)),   # 21-28
    'lower_left': list(range(31, 39)),   # 31-38
    'lower_right': list(range(41, 49))  # 41-48
}


def extract_roi(image: np.ndarray, mask: np.ndarray, bbox: np.ndarray,
               padding: int = 10) -> np.ndarray:
    """
    Extract ROI from image using mask and bounding box.
    
    Args:
        image: Full image (H, W, C)
        mask: Binary mask (H, W)
        bbox: Bounding box [x1, y1, x2, y2]
        padding: Padding around bbox in pixels
    
    Returns:
        Cropped ROI image
    """
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = image.shape[:2]
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    # Crop
    roi = image[y1:y2, x1:x2].copy()
    
    # Apply mask
    mask_roi = mask[y1:y2, x1:x2]
    if roi.shape[:2] != mask_roi.shape:
        mask_roi = cv2.resize(mask_roi.astype(np.uint8),
                            (roi.shape[1], roi.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # Set background to white
    roi[~mask_roi] = 255
    
    return roi


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
        Quadrant name
    """
    vertical_split = image_height / 2
    horizontal_split = image_width / 2
    
    is_upper = centroid_y < vertical_split
    is_right = centroid_x > horizontal_split
    
    if is_upper and is_right:
        return 'upper_right'
    elif is_upper and not is_right:
        return 'upper_left'
    elif not is_upper and not is_right:
        return 'lower_left'
    else:
        return 'lower_right'


def apply_anatomical_ordering(instances: List[Dict], image_shape: Tuple[int, int]) -> Dict[int, int]:
    """
    Apply anatomical sorting and FDI mapping.
    
    Args:
        instances: List of instance dictionaries with bbox, centroid, etc.
        image_shape: (height, width) of image
    
    Returns:
        Dictionary mapping instance index to FDI number
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
        fdi_sequence = FDI_QUADRANTS[quadrant]
        for i, idx in enumerate(sorted_indices):
            if i < len(fdi_sequence):
                final_fdi_mapping[idx] = fdi_sequence[i]
            else:
                # If more teeth than expected, assign to last available
                final_fdi_mapping[idx] = fdi_sequence[-1]
    
    return final_fdi_mapping


def select_low_confidence_instances(instances: List[Dict],
                                   confidence_threshold: float = 0.85) -> List[int]:
    """
    Select instances with confidence below threshold for classifier refinement.
    
    Args:
        instances: List of instance dictionaries
        confidence_threshold: Confidence threshold
    
    Returns:
        List of instance indices that need classifier
    """
    low_confidence_indices = []
    
    for i, inst in enumerate(instances):
        if inst.get('confidence', 0.0) < confidence_threshold:
            low_confidence_indices.append(i)
    
    return low_confidence_indices


def fuse_classifier_predictions(instances: List[Dict], classifier_predictions: Dict[int, Tuple[int, float]],
                               anatomical_fdi: Dict[int, int]) -> List[Dict]:
    """
    Fuse classifier predictions with anatomical ordering.
    
    Args:
        instances: List of instance dictionaries
        classifier_predictions: Dict mapping instance index to (fdi, confidence)
        anatomical_fdi: Dict mapping instance index to anatomical FDI
    
    Returns:
        Updated instances with final FDI labels
    """
    updated_instances = []
    
    for i, inst in enumerate(instances):
        classifier_fdi = inst.get('maskrcnn_class', None)
        classifier_conf = inst.get('confidence', 0.0)
        
        # Get classifier prediction if available
        if i in classifier_predictions:
            classifier_fdi, classifier_conf = classifier_predictions[i]
        
        # Get anatomical FDI
        anatomical_fdi_val = anatomical_fdi.get(i, classifier_fdi)
        
        # Decision logic: if classifier and anatomical disagree significantly, prefer anatomical
        if classifier_fdi and abs(classifier_fdi - anatomical_fdi_val) > 2:
            final_fdi = anatomical_fdi_val
            correction_applied = True
        else:
            final_fdi = classifier_fdi if classifier_fdi else anatomical_fdi_val
            correction_applied = False
        
        # Update instance
        updated_inst = inst.copy()
        updated_inst['fdi'] = final_fdi
        updated_inst['final_confidence'] = classifier_conf
        updated_inst['correction_applied'] = correction_applied
        updated_inst['anatomical_fdi'] = anatomical_fdi_val
        
        updated_instances.append(updated_inst)
    
    return updated_instances


def detect_duplicates(instances: List[Dict]) -> List[Tuple[int, int]]:
    """
    Detect duplicate FDI labels.
    
    Args:
        instances: List of instances with FDI labels
    
    Returns:
        List of (index1, index2) pairs with duplicate FDI
    """
    duplicates = []
    fdi_to_indices = defaultdict(list)
    
    for i, inst in enumerate(instances):
        fdi = inst.get('fdi')
        if fdi:
            fdi_to_indices[fdi].append(i)
    
    for fdi, indices in fdi_to_indices.items():
        if len(indices) > 1:
            # All pairs are duplicates
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    duplicates.append((indices[i], indices[j]))
    
    return duplicates


def resolve_duplicates(instances: List[Dict], duplicates: List[Tuple[int, int]],
                      anatomical_fdi: Dict[int, int]) -> List[Dict]:
    """
    Resolve duplicate FDI labels using anatomical ordering.
    
    Args:
        instances: List of instances
        duplicates: List of duplicate pairs
        anatomical_fdi: Anatomical FDI mapping
    
    Returns:
        Updated instances with resolved duplicates
    """
    resolved = instances.copy()
    
    for idx1, idx2 in duplicates:
        # Use anatomical FDI for both
        if idx1 in anatomical_fdi:
            resolved[idx1]['fdi'] = anatomical_fdi[idx1]
        if idx2 in anatomical_fdi:
            resolved[idx2]['fdi'] = anatomical_fdi[idx2]
        
        # Mark as corrected
        resolved[idx1]['correction_applied'] = True
        resolved[idx2]['correction_applied'] = True
    
    return resolved


def postprocess_predictions(maskrcnn_instances: List[Dict], image: np.ndarray,
                           classifier_predictions: Optional[Dict[int, Tuple[int, float]]] = None,
                           confidence_threshold: float = 0.85) -> List[Dict]:
    """
    Complete post-processing pipeline.
    
    Args:
        maskrcnn_instances: Raw Mask R-CNN predictions
        image: Original image
        classifier_predictions: Optional classifier predictions
        confidence_threshold: Threshold for using classifier
    
    Returns:
        Post-processed instances with final FDI labels
    """
    if classifier_predictions is None:
        classifier_predictions = {}
    
    h, w = image.shape[:2]
    
    # Apply anatomical ordering
    anatomical_fdi = apply_anatomical_ordering(maskrcnn_instances, (h, w))
    
    # Fuse classifier predictions
    processed = fuse_classifier_predictions(maskrcnn_instances, classifier_predictions, anatomical_fdi)
    
    # Detect and resolve duplicates
    duplicates = detect_duplicates(processed)
    if duplicates:
        processed = resolve_duplicates(processed, duplicates, anatomical_fdi)
    
    return processed

