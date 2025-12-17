"""
Image preprocessing utilities for tooth detection pipeline.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def convert_to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert image to RGB format (3-channel).
    
    Args:
        image: Input image (can be grayscale or BGR)
    
    Returns:
        RGB image (H, W, 3)
    """
    if len(image.shape) == 2:
        # Grayscale to RGB
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3:
        if image.shape[2] == 1:
            # Single channel to RGB
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            # Assume BGR, convert to RGB
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            # RGBA to RGB
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    return image


def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                maintain_aspect: bool = True) -> Tuple[np.ndarray, float, float]:
    """
    Resize image to target size.
    
    Args:
        image: Input image
        target_size: (width, height) target size
        maintain_aspect: Whether to maintain aspect ratio
    
    Returns:
        (resized_image, scale_x, scale_y)
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    if maintain_aspect:
        # Calculate scale to fit within target size
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size if needed
        if new_w != target_w or new_h != target_h:
            padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
            padded[:new_h, :new_w] = resized
            resized = padded
        
        return resized, scale, scale
    else:
        resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        scale_x = target_w / w
        scale_y = target_h / h
        return resized, scale_x, scale_y


def normalize_image(image: np.ndarray, mean: Optional[np.ndarray] = None,
                   std: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normalize image for model input.
    
    Args:
        image: Input image (0-255 range)
        mean: Mean values for normalization (default: ImageNet)
        std: Std values for normalization (default: ImageNet)
    
    Returns:
        Normalized image (0-1 range)
    """
    if mean is None:
        mean = np.array([0.485, 0.456, 0.406])
    if std is None:
        std = np.array([0.229, 0.224, 0.225])
    
    # Convert to float and normalize
    image_float = image.astype(np.float32) / 255.0
    
    # Normalize
    if len(image.shape) == 3:
        for i in range(3):
            image_float[:, :, i] = (image_float[:, :, i] - mean[i]) / std[i]
    
    return image_float


def preprocess_for_maskrcnn(image: np.ndarray, target_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, dict]:
    """
    Preprocess image for Mask R-CNN inference.
    
    Args:
        image: Input image (BGR or RGB)
        target_size: Optional target size (width, height)
    
    Returns:
        (preprocessed_image, metadata_dict)
    """
    original_shape = image.shape[:2]
    
    # Convert to RGB
    rgb_image = convert_to_rgb(image)
    
    # Resize if target size specified
    metadata = {
        'original_shape': original_shape,
        'scale_x': 1.0,
        'scale_y': 1.0
    }
    
    if target_size:
        resized, scale_x, scale_y = resize_image(rgb_image, target_size, maintain_aspect=True)
        metadata['scale_x'] = scale_x
        metadata['scale_y'] = scale_y
        metadata['resized_shape'] = resized.shape[:2]
    else:
        resized = rgb_image
    
    return resized, metadata


def preprocess_roi_for_classifier(roi: np.ndarray, target_size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """
    Preprocess ROI crop for EfficientNet classifier.
    
    Args:
        roi: ROI crop image
        target_size: Target size (width, height)
    
    Returns:
        Preprocessed ROI (normalized, ready for model)
    """
    # Resize to target size
    resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_LINEAR)
    
    # Convert to RGB if needed
    rgb_roi = convert_to_rgb(resized)
    
    # Normalize
    normalized = normalize_image(rgb_roi)
    
    return normalized

