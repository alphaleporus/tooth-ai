#!/usr/bin/env python3
"""
Batch validation script for unified tooth detection pipeline.
Runs inference on validation and training sets, collects comprehensive metrics.
"""

import argparse
import os
import sys
import json
import csv
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import cv2

# Add paths for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.engine import ToothDetectionEngine, load_engine
from pycocotools import mask as mask_utils


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
    return None


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def polygon_to_mask(polygon: List[float], height: int, width: int) -> np.ndarray:
    """Convert polygon to binary mask."""
    if len(polygon) < 6:
        return np.zeros((height, width), dtype=bool)
    
    x_coords = polygon[::2]
    y_coords = polygon[1::2]
    points = np.array([[x, y] for x, y in zip(x_coords, y_coords)], dtype=np.int32)
    
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(bool)


def compute_metrics_for_image(engine: ToothDetectionEngine, image_path: str,
                              gt_annotations: List[Dict], image_info: Dict) -> Dict:
    """
    Compute metrics for a single image.
    
    Returns:
        Dictionary with metrics
    """
    # Run inference
    results = engine.predict(image_path, return_visualization=False)
    
    h, w = image_info['height'], image_info['width']
    
    # Prepare ground truth
    gt_masks = []
    gt_classes = []
    
    for ann in gt_annotations:
        seg = ann.get('segmentation', [])
        if isinstance(seg, list) and len(seg) > 0:
            if isinstance(seg[0], list):
                mask = polygon_to_mask(seg[0], h, w)
            else:
                gt_mask = mask_utils.decode(seg)
                if len(gt_mask.shape) == 3:
                    gt_mask = gt_mask[:, :, 0]
                mask = gt_mask.astype(bool)
        else:
            continue
        
        gt_masks.append(mask)
        gt_classes.append(ann.get('category_id', 0))
    
    # Match predictions to ground truth
    pred_masks = []
    pred_fdi = []
    pred_scores = []
    
    for tooth in results['teeth']:
        # Decode mask if RLE
        mask = tooth.get('mask')
        if isinstance(mask, dict):
            mask_array = mask_utils.decode(mask)
            if len(mask_array.shape) == 3:
                mask_array = mask_array[:, :, 0]
            mask = mask_array.astype(bool)
        else:
            mask = mask.astype(bool) if isinstance(mask, np.ndarray) else None
        
        if mask is not None:
            pred_masks.append(mask)
            pred_fdi.append(tooth['fdi'])
            pred_scores.append(tooth['score'])
    
    # Compute mask IoU
    matched_gt = set()
    ious = []
    fdi_matches_before = []
    fdi_matches_after = []
    
    for i, pred_mask in enumerate(pred_masks):
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt_mask in enumerate(gt_masks):
            if j in matched_gt:
                continue
            iou = mask_iou(pred_mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= 0.5 and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            ious.append(best_iou)
            
            # Check FDI accuracy
            gt_fdi = gt_classes[best_gt_idx]
            pred_fdi_val = pred_fdi[i]
            fdi_matches_before.append((pred_fdi_val, gt_fdi))
            fdi_matches_after.append((pred_fdi_val, gt_fdi))  # Same for now
    
    mean_iou = np.mean(ious) if ious else 0.0
    
    # Compute FDI accuracy
    fdi_acc_before = sum(1 for p, g in fdi_matches_before if p == g) / len(fdi_matches_before) if fdi_matches_before else 0.0
    fdi_acc_after = sum(1 for p, g in fdi_matches_after if p == g) / len(fdi_matches_after) if fdi_matches_after else 0.0
    
    # Confidence distribution
    confidences = pred_scores if pred_scores else []
    conf_mean = np.mean(confidences) if confidences else 0.0
    conf_std = np.std(confidences) if confidences else 0.0
    conf_min = np.min(confidences) if confidences else 0.0
    conf_max = np.max(confidences) if confidences else 0.0
    
    return {
        'image_id': image_info['id'],
        'filename': image_info['file_name'],
        'num_detections': results['num_detections'],
        'num_gt': len(gt_annotations),
        'mask_iou': float(mean_iou),
        'fdi_accuracy_before': float(fdi_acc_before),
        'fdi_accuracy_after': float(fdi_acc_after),
        'confidence_mean': float(conf_mean),
        'confidence_std': float(conf_std),
        'confidence_min': float(conf_min),
        'confidence_max': float(conf_max),
        'maskrcnn_used': results['metadata']['maskrcnn_used'],
        'effnet_used': results['metadata']['effnet_used'],
        'corrections_applied': results['metadata']['corrections_applied']
    }


def batch_validate(images_dir: str, coco_json: str, engine: ToothDetectionEngine,
                  output_dir: str, num_val: int = 50, num_train: int = 50,
                  seed: int = 42):
    """
    Run batch validation on validation and training sets.
    
    Args:
        images_dir: Directory containing images
        coco_json: Path to COCO annotations
        engine: Inference engine
        output_dir: Output directory for results
        num_val: Number of validation images
        num_train: Number of training images
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Load COCO data
    print("Loading COCO annotations...")
    coco_data = load_coco_annotations(coco_json)
    images = coco_data['images']
    annotations = coco_data['annotations']
    
    # Create mapping
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann['image_id']].append(ann)
    
    # Split into train/val (80/20)
    random.shuffle(images)
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Sample images
    if len(val_images) > num_val:
        val_samples = random.sample(val_images, num_val)
    else:
        val_samples = val_images
    
    if len(train_images) > num_train:
        train_samples = random.sample(train_images, num_train)
    else:
        train_samples = train_images
    
    print(f"Validating {len(val_samples)} validation images and {len(train_samples)} training images...")
    
    # Process validation set
    val_results = []
    for i, image_info in enumerate(val_samples):
        print(f"Processing val image {i+1}/{len(val_samples)}: {image_info['file_name']}")
        image_path = find_image_path(image_info['file_name'], images_dir)
        if image_path:
            try:
                gt_anns = img_to_anns[image_info['id']]
                metrics = compute_metrics_for_image(engine, image_path, gt_anns, image_info)
                metrics['split'] = 'val'
                val_results.append(metrics)
            except Exception as e:
                print(f"  Error: {e}")
                continue
    
    # Process training set
    train_results = []
    for i, image_info in enumerate(train_samples):
        print(f"Processing train image {i+1}/{len(train_samples)}: {image_info['file_name']}")
        image_path = find_image_path(image_info['file_name'], images_dir)
        if image_path:
            try:
                gt_anns = img_to_anns[image_info['id']]
                metrics = compute_metrics_for_image(engine, image_path, gt_anns, image_info)
                metrics['split'] = 'train'
                train_results.append(metrics)
            except Exception as e:
                print(f"  Error: {e}")
                continue
    
    # Combine results
    all_results = val_results + train_results
    
    # Compute summary statistics
    summary = {
        'num_images': len(all_results),
        'num_val': len(val_results),
        'num_train': len(train_results),
        'mean_mask_iou': float(np.mean([r['mask_iou'] for r in all_results])),
        'std_mask_iou': float(np.std([r['mask_iou'] for r in all_results])),
        'mean_fdi_acc_before': float(np.mean([r['fdi_accuracy_before'] for r in all_results])),
        'mean_fdi_acc_after': float(np.mean([r['fdi_accuracy_after'] for r in all_results])),
        'mean_num_detections': float(np.mean([r['num_detections'] for r in all_results])),
        'mean_confidence': float(np.mean([r['confidence_mean'] for r in all_results])),
        'val_mean_iou': float(np.mean([r['mask_iou'] for r in val_results])) if val_results else 0.0,
        'train_mean_iou': float(np.mean([r['mask_iou'] for r in train_results])) if train_results else 0.0,
        'val_mean_fdi_acc': float(np.mean([r['fdi_accuracy_after'] for r in val_results])) if val_results else 0.0,
        'train_mean_fdi_acc': float(np.mean([r['fdi_accuracy_after'] for r in train_results])) if train_results else 0.0
    }
    
    # Save JSON
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, 'batch_metrics.json')
    with open(json_path, 'w') as f:
        json.dump({
            'summary': summary,
            'per_image': all_results
        }, f, indent=2)
    
    # Save CSV
    csv_path = os.path.join(output_dir, 'batch_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    
    print(f"\nBatch validation complete!")
    print(f"  Results saved to: {json_path}")
    print(f"  CSV saved to: {csv_path}")
    print(f"\nSummary:")
    print(f"  Mean Mask IoU: {summary['mean_mask_iou']:.4f} ± {summary['std_mask_iou']:.4f}")
    print(f"  Mean FDI Accuracy: {summary['mean_fdi_acc_after']:.4f}")
    print(f"  Mean Detections: {summary['mean_num_detections']:.2f}")


def main():
    parser = argparse.ArgumentParser(description='Batch validation for tooth detection pipeline')
    parser.add_argument('--images', type=str, required=True, help='Image directory')
    parser.add_argument('--coco', type=str, required=True, help='COCO annotations JSON')
    parser.add_argument('--engine', type=str, default=None, help='Path to engine (or use model-dir)')
    parser.add_argument('--model-dir', type=str, default=None, help='Model directory')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--num-val', type=int, default=50, help='Number of validation images')
    parser.add_argument('--num-train', type=int, default=50, help='Number of training images')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load engine
    print("Loading inference engine...")
    if args.model_dir:
        engine = load_engine(args.model_dir)
    elif args.engine:
        # Load from engine path (if provided as module path)
        model_dir = os.path.join(os.path.dirname(os.path.dirname(args.engine)), 'models')
        engine = load_engine(model_dir)
    else:
        # Default to workspace/tooth-ai/models
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tooth-ai', 'models')
        engine = load_engine(model_dir)
    
    batch_validate(
        args.images, args.coco, engine,
        args.out, args.num_val, args.num_train, args.seed
    )


if __name__ == '__main__':
    main()



