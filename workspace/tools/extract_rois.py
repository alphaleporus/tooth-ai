#!/usr/bin/env python3
"""
Extract ROI crops from Mask R-CNN predictions for training EfficientNet classifier.
Crops each detected tooth instance and saves as 128x128 images with labels.
"""

import argparse
import os
import json
import csv
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import random
from collections import defaultdict

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes


def load_coco_annotations(coco_json_path: str) -> Dict:
    """Load COCO annotations."""
    with open(coco_json_path, 'r') as f:
        return json.load(f)


def find_image_path(filename: str, image_dir: str) -> str:
    """Find image file in directory (recursive search)."""
    import glob
    
    base_name = os.path.splitext(filename)[0]
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        pattern = os.path.join(image_dir, '**', base_name + ext)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
        pattern = os.path.join(image_dir, '**', filename)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    return None


def crop_roi_from_mask(image: np.ndarray, mask: np.ndarray, bbox: np.ndarray,
                      target_size: Tuple[int, int] = (128, 128),
                      padding: int = 10) -> np.ndarray:
    """
    Crop ROI from image using mask and bounding box.
    
    Args:
        image: Full image (H, W, C)
        mask: Binary mask (H, W)
        bbox: Bounding box [x1, y1, x2, y2]
        target_size: Target size for resized ROI
        padding: Padding around bbox in pixels
    
    Returns:
        Cropped and resized ROI image
    """
    x1, y1, x2, y2 = bbox.astype(int)
    h, w = image.shape[:2]
    
    # Add padding
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    # Crop image
    roi = image[y1:y2, x1:x2].copy()
    
    # Apply mask to ROI
    mask_roi = mask[y1:y2, x1:x2]
    if roi.shape[:2] != mask_roi.shape:
        # Handle size mismatch
        mask_roi = cv2.resize(mask_roi.astype(np.uint8), 
                            (roi.shape[1], roi.shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    
    # Apply mask (set background to white or transparent)
    roi[~mask_roi] = 255  # White background
    
    # Resize to target size
    roi_resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_LINEAR)
    
    return roi_resized


def extract_rois_from_image(predictor: DefaultPredictor, image_path: str,
                           coco_data: Dict, image_info: Dict,
                           output_dir: str, split: str,
                           labels_csv_path: str) -> List[Dict]:
    """
    Extract ROIs from a single image using Mask R-CNN predictions.
    
    Args:
        predictor: Detectron2 predictor
        image_path: Path to image file
        coco_data: COCO annotations data
        image_info: Image metadata from COCO
        output_dir: Output directory for ROI images
        split: 'train' or 'val'
        labels_csv_path: Path to CSV file for labels
    
    Returns:
        List of ROI metadata dictionaries
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return []
    
    h, w = image.shape[:2]
    
    # Run inference
    outputs = predictor(image)
    instances = outputs["instances"]
    
    # Get ground truth annotations for this image
    img_id = image_info['id']
    gt_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == img_id]
    
    # Create mapping from prediction to ground truth
    roi_metadata = []
    
    for i in range(len(instances)):
        # Get prediction
        mask = instances.pred_masks[i].cpu().numpy()
        box = instances.pred_boxes.tensor[i].cpu().numpy()
        score = instances.scores[i].item()
        pred_class = instances.pred_classes[i].item()
        
        # Find matching ground truth by IoU
        best_iou = 0.0
        best_gt = None
        
        for gt_ann in gt_annotations:
            # Get GT mask
            seg = gt_ann.get('segmentation', [])
            if isinstance(seg, list) and len(seg) > 0:
                if isinstance(seg[0], list):
                    # Polygon - convert to mask
                    from pycocotools import mask as mask_utils
                    rle = mask_utils.frPyObjects(seg, h, w)
                    gt_mask = mask_utils.decode(rle)
                    if len(gt_mask.shape) == 3:
                        gt_mask = gt_mask[:, :, 0]
                    gt_mask = gt_mask.astype(bool)
                else:
                    # RLE
                    from pycocotools import mask as mask_utils
                    gt_mask = mask_utils.decode(seg).astype(bool)
            else:
                continue
            
            # Compute IoU
            intersection = np.logical_and(mask, gt_mask).sum()
            union = np.logical_or(mask, gt_mask).sum()
            iou = intersection / union if union > 0 else 0.0
            
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_ann
        
        # Use ground truth label if IoU > 0.5, otherwise use prediction
        if best_iou > 0.5 and best_gt is not None:
            fdi_label = best_gt.get('category_id', pred_class)
        else:
            fdi_label = pred_class
        
        # Crop ROI
        roi = crop_roi_from_mask(image, mask, box, target_size=(128, 128))
        
        # Calculate bbox area
        bbox_area = (box[2] - box[0]) * (box[3] - box[1])
        
        # Save ROI
        roi_filename = f"{image_info['id']}_{i:03d}.png"
        roi_path = os.path.join(output_dir, split, roi_filename)
        cv2.imwrite(roi_path, roi)
        
        # Store metadata
        roi_metadata.append({
            'filename': roi_filename,
            'fdi_label': int(fdi_label),
            'confidence_score': float(score),
            'bbox_area': float(bbox_area),
            'iou_with_gt': float(best_iou),
            'image_id': image_info['id'],
            'instance_id': i
        })
    
    return roi_metadata


def extract_rois(images_dir: str, coco_json: str, model_path: str, config_path: str,
                 output_dir: str, train_split: float = 0.8, min_confidence: float = 0.3):
    """
    Extract ROIs from all images using Mask R-CNN predictions.
    
    Args:
        images_dir: Directory containing images
        coco_json: Path to COCO annotations JSON
        model_path: Path to trained Mask R-CNN model
        config_path: Path to config file
        output_dir: Output directory for ROI dataset
        train_split: Ratio for train/val split
        min_confidence: Minimum confidence threshold for predictions
    """
    # Setup predictor
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min_confidence
    
    predictor = DefaultPredictor(cfg)
    
    # Load COCO data
    print("Loading COCO annotations...")
    coco_data = load_coco_annotations(coco_json)
    images = coco_data['images']
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Split images
    random.seed(42)
    random.shuffle(images)
    split_idx = int(len(images) * train_split)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    print(f"Processing {len(train_images)} training images and {len(val_images)} validation images...")
    
    # Extract ROIs
    all_train_metadata = []
    all_val_metadata = []
    
    # Process training images
    for i, image_info in enumerate(train_images):
        print(f"Processing train image {i+1}/{len(train_images)}: {image_info['file_name']}")
        image_path = find_image_path(image_info['file_name'], images_dir)
        if image_path:
            metadata = extract_rois_from_image(
                predictor, image_path, coco_data, image_info,
                output_dir, 'train', None
            )
            all_train_metadata.extend(metadata)
    
    # Process validation images
    for i, image_info in enumerate(val_images):
        print(f"Processing val image {i+1}/{len(val_images)}: {image_info['file_name']}")
        image_path = find_image_path(image_info['file_name'], images_dir)
        if image_path:
            metadata = extract_rois_from_image(
                predictor, image_path, coco_data, image_info,
                output_dir, 'val', None
            )
            all_val_metadata.extend(metadata)
    
    # Save labels CSV
    print("\nSaving labels...")
    
    # Train labels
    train_csv_path = os.path.join(output_dir, 'train_labels.csv')
    with open(train_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'fdi_label', 'confidence_score', 'bbox_area'])
        writer.writeheader()
        for meta in all_train_metadata:
            writer.writerow({
                'filename': meta['filename'],
                'fdi_label': meta['fdi_label'],
                'confidence_score': meta['confidence_score'],
                'bbox_area': meta['bbox_area']
            })
    
    # Val labels
    val_csv_path = os.path.join(output_dir, 'val_labels.csv')
    with open(val_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'fdi_label', 'confidence_score', 'bbox_area'])
        writer.writeheader()
        for meta in all_val_metadata:
            writer.writerow({
                'filename': meta['filename'],
                'fdi_label': meta['fdi_label'],
                'confidence_score': meta['confidence_score'],
                'bbox_area': meta['bbox_area']
            })
    
    # Summary
    print(f"\nROI extraction complete!")
    print(f"  Training ROIs: {len(all_train_metadata)}")
    print(f"  Validation ROIs: {len(all_val_metadata)}")
    print(f"  Train labels: {train_csv_path}")
    print(f"  Val labels: {val_csv_path}")
    
    # Class distribution
    train_classes = defaultdict(int)
    val_classes = defaultdict(int)
    for meta in all_train_metadata:
        train_classes[meta['fdi_label']] += 1
    for meta in all_val_metadata:
        val_classes[meta['fdi_label']] += 1
    
    print(f"\nClass distribution (train):")
    for cls_id in sorted(train_classes.keys()):
        print(f"  Class {cls_id}: {train_classes[cls_id]}")
    
    print(f"\nClass distribution (val):")
    for cls_id in sorted(val_classes.keys()):
        print(f"  Class {cls_id}: {val_classes[cls_id]}")


def main():
    parser = argparse.ArgumentParser(description='Extract ROIs from Mask R-CNN predictions')
    parser.add_argument('--images', type=str, required=True, help='Image directory')
    parser.add_argument('--coco', type=str, required=True, help='COCO annotations JSON')
    parser.add_argument('--model', type=str, required=True, help='Path to Mask R-CNN model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--out', type=str, required=True, help='Output directory for ROI dataset')
    parser.add_argument('--train-split', type=float, default=0.8, help='Train/val split ratio')
    parser.add_argument('--min-confidence', type=float, default=0.3, help='Minimum confidence threshold')
    
    args = parser.parse_args()
    
    extract_rois(
        args.images, args.coco, args.model, args.config,
        args.out, args.train_split, args.min_confidence
    )


if __name__ == '__main__':
    main()



