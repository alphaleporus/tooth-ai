#!/usr/bin/env python3
"""
Evaluate integrated inference system (Mask R-CNN + EfficientNet + Anatomical Correction).
Computes FDI accuracy, consistency checks, and error analysis.
"""

import argparse
import os
import json
import random
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import cv2

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from pycocotools import mask as mask_utils

from tools.integrated_inference import (
    integrated_inference, load_effnet_model, determine_quadrant
)


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


def box_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes [x1, y1, x2, y2]."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


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


def evaluate_single_image(maskrcnn_predictor: DefaultPredictor, effnet_model,
                          image_path: str, gt_annotations: List[Dict],
                          image_info: Dict, coco_data: Dict,
                          device: str = 'cuda') -> Dict:
    """
    Evaluate integrated inference on a single image.
    
    Returns:
        Dictionary with metrics
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    h, w = image.shape[:2]
    
    # Prepare ground truth
    gt_masks = []
    gt_boxes = []
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
        
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            x, y, bw, bh = bbox
            box = np.array([x, y, x + bw, y + bh])
        else:
            continue
        
        gt_masks.append(mask)
        gt_boxes.append(box)
        gt_classes.append(ann.get('category_id', 0))
    
    # Run integrated inference
    results = integrated_inference(
        image_path, maskrcnn_predictor, effnet_model,
        (h, w), confidence_threshold=0.85, device=device
    )
    
    predictions = results['predictions']
    
    # Match predictions to ground truth
    pred_masks = [p['mask'] for p in predictions]
    pred_boxes = [np.array(p['bbox']) for p in predictions]
    pred_fdi_before = [p['classifier_fdi'] for p in predictions]  # Before correction
    pred_fdi_after = [p['fdi_label'] for p in predictions]  # After correction
    
    # Compute mask IoU
    matched_gt = set()
    ious = []
    
    for pred_mask in pred_masks:
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
    
    mean_iou = np.mean(ious) if ious else 0.0
    
    # Compute bbox mAP (simplified)
    matched_boxes = set()
    tp = 0
    fp = 0
    
    for pred_box in pred_boxes:
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_boxes):
            if j in matched_boxes:
                continue
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= 0.5 and best_gt_idx >= 0:
            matched_boxes.add(best_gt_idx)
            tp += 1
        else:
            fp += 1
    
    fn = len(gt_boxes) - len(matched_boxes)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    map_score = precision if recall > 0 else 0.0
    
    # Compute FDI accuracy (before and after correction)
    # Match predictions to GT by IoU
    fdi_matches_before = []
    fdi_matches_after = []
    
    matched_for_fdi = set()
    for i, (pred_box, fdi_before, fdi_after) in enumerate(zip(pred_boxes, pred_fdi_before, pred_fdi_after)):
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_boxes):
            if j in matched_for_fdi:
                continue
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= 0.5 and best_gt_idx >= 0:
            matched_for_fdi.add(best_gt_idx)
            gt_fdi = gt_classes[best_gt_idx]
            fdi_matches_before.append((fdi_before, gt_fdi))
            fdi_matches_after.append((fdi_after, gt_fdi))
    
    fdi_acc_before = sum(1 for p, g in fdi_matches_before if p == g) / len(fdi_matches_before) if fdi_matches_before else 0.0
    fdi_acc_after = sum(1 for p, g in fdi_matches_after if p == g) / len(fdi_matches_after) if fdi_matches_after else 0.0
    
    # Consistency checks
    fdi_labels_after = [p['fdi_label'] for p in predictions]
    
    # Check for duplicates
    duplicates = len(fdi_labels_after) != len(set(fdi_labels_after))
    
    # Check for quadrant violations
    quadrant_violations = 0
    for pred in predictions:
        fdi = pred['fdi_label']
        centroid = pred['centroid']
        quadrant = determine_quadrant(centroid[1], centroid[0], h, w)
        
        # Check if FDI matches quadrant
        valid_fdis = {
            'upper_right': list(range(11, 19)),
            'upper_left': list(range(21, 29)),
            'lower_left': list(range(31, 39)),
            'lower_right': list(range(41, 49))
        }
        
        if fdi not in valid_fdis[quadrant]:
            quadrant_violations += 1
    
    # Check for swapped neighbors (simplified - check if adjacent teeth have swapped FDI)
    swapped_neighbors = 0
    if len(predictions) > 1:
        sorted_by_x = sorted(predictions, key=lambda p: p['centroid'][0])
        for i in range(len(sorted_by_x) - 1):
            fdi1 = sorted_by_x[i]['fdi_label']
            fdi2 = sorted_by_x[i + 1]['fdi_label']
            # Check if they're in same quadrant and should be sequential
            if abs(fdi1 - fdi2) > 2 and (fdi1 // 10 == fdi2 // 10):
                swapped_neighbors += 1
    
    return {
        'image_id': image_info['id'],
        'filename': image_info['file_name'],
        'mask_iou': float(mean_iou),
        'bbox_map': float(map_score),
        'fdi_accuracy_before': float(fdi_acc_before),
        'fdi_accuracy_after': float(fdi_acc_after),
        'num_detections': len(predictions),
        'num_gt': len(gt_annotations),
        'duplicate_fdi': duplicates,
        'quadrant_violations': quadrant_violations,
        'swapped_neighbors': swapped_neighbors,
        'corrections_applied': sum(1 for p in predictions if p['correction_applied']),
        'effnet_used': sum(1 for p in predictions if p['method_used'] == 'effnet'),
        'maskrcnn_used': sum(1 for p in predictions if p['method_used'] == 'maskrcnn')
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate integrated inference system')
    parser.add_argument('--images', type=str, required=True, help='Image directory')
    parser.add_argument('--coco', type=str, required=True, help='COCO annotations JSON')
    parser.add_argument('--maskrcnn', type=str, required=True, help='Path to Mask R-CNN model')
    parser.add_argument('--effnet', type=str, required=True, help='Path to EfficientNet model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--out', type=str, required=True, help='Output directory for reports')
    parser.add_argument('--num-samples', type=int, default=30, help='Number of images to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-classes', type=int, default=32, help='Number of FDI classes')
    parser.add_argument('--device', type=str, default='cuda', help='Device for inference')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load COCO data
    print("Loading COCO annotations...")
    coco_data = load_coco_annotations(args.coco)
    images = coco_data['images']
    annotations = coco_data['annotations']
    
    # Create mapping from image_id to annotations
    img_to_anns = defaultdict(list)
    for ann in annotations:
        img_to_anns[ann['image_id']].append(ann)
    
    # Sample validation images
    val_images = [img for img in images if img['id'] in img_to_anns]
    if len(val_images) > args.num_samples:
        val_images = random.sample(val_images, args.num_samples)
    
    print(f"Evaluating {len(val_images)} images...")
    
    # Setup predictors
    print("Loading models...")
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.maskrcnn
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
    maskrcnn_predictor = DefaultPredictor(cfg)
    
    effnet_model = load_effnet_model(args.effnet, args.num_classes, args.device)
    
    # Evaluate each image
    all_results = []
    
    for i, image_info in enumerate(val_images):
        print(f"\nProcessing image {i+1}/{len(val_images)}: {image_info['file_name']}")
        image_path = find_image_path(image_info['file_name'], args.images)
        if image_path is None:
            print(f"  Warning: Could not find image")
            continue
        
        gt_anns = img_to_anns[image_info['id']]
        
        try:
            result = evaluate_single_image(
                maskrcnn_predictor, effnet_model,
                image_path, gt_anns, image_info, coco_data, args.device
            )
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Compute summary statistics
    summary = {
        'num_images': len(all_results),
        'mean_mask_iou': np.mean([r['mask_iou'] for r in all_results]),
        'mean_bbox_map': np.mean([r['bbox_map'] for r in all_results]),
        'mean_fdi_acc_before': np.mean([r['fdi_accuracy_before'] for r in all_results]),
        'mean_fdi_acc_after': np.mean([r['fdi_accuracy_after'] for r in all_results]),
        'fdi_improvement': np.mean([r['fdi_accuracy_after'] - r['fdi_accuracy_before'] for r in all_results]),
        'total_duplicates': sum(1 for r in all_results if r['duplicate_fdi']),
        'total_quadrant_violations': sum(r['quadrant_violations'] for r in all_results),
        'total_swapped_neighbors': sum(r['swapped_neighbors'] for r in all_results),
        'total_corrections': sum(r['corrections_applied'] for r in all_results),
        'total_effnet_used': sum(r['effnet_used'] for r in all_results),
        'total_maskrcnn_used': sum(r['maskrcnn_used'] for r in all_results)
    }
    
    # Save results
    os.makedirs(args.out, exist_ok=True)
    
    output_json = os.path.join(args.out, 'integrated_metrics.json')
    with open(output_json, 'w') as f:
        json.dump({
            'summary': summary,
            'per_image': all_results
        }, f, indent=2)
    
    # Generate markdown report
    output_md = os.path.join(args.out, 'integrated_metrics.md')
    with open(output_md, 'w') as f:
        f.write("# Integrated Inference System Evaluation\n\n")
        f.write(f"**Number of images evaluated:** {summary['num_images']}\n\n")
        f.write("## Summary Statistics\n\n")
        f.write("| Metric | Value |\n")
        f.write("|--------|-------|\n")
        f.write(f"| Mean Mask IoU | {summary['mean_mask_iou']:.4f} |\n")
        f.write(f"| Mean bbox mAP@0.5 | {summary['mean_bbox_map']:.4f} |\n")
        f.write(f"| Mean FDI Accuracy (Before) | {summary['mean_fdi_acc_before']:.4f} |\n")
        f.write(f"| Mean FDI Accuracy (After) | {summary['mean_fdi_acc_after']:.4f} |\n")
        f.write(f"| FDI Accuracy Improvement | {summary['fdi_improvement']:+.4f} |\n\n")
        f.write("## Consistency Checks\n\n")
        f.write(f"- Images with duplicate FDI labels: {summary['total_duplicates']}\n")
        f.write(f"- Total quadrant violations: {summary['total_quadrant_violations']}\n")
        f.write(f"- Total swapped neighbors: {summary['total_swapped_neighbors']}\n")
        f.write(f"- Total anatomical corrections applied: {summary['total_corrections']}\n\n")
        f.write("## Method Usage\n\n")
        f.write(f"- Mask R-CNN used: {summary['total_maskrcnn_used']} detections\n")
        f.write(f"- EfficientNet used: {summary['total_effnet_used']} detections\n\n")
        f.write("## Per-Image Results\n\n")
        f.write("| Image ID | Mask IoU | FDI Acc (Before) | FDI Acc (After) | Corrections |\n")
        f.write("|----------|----------|-----------------|----------------|-------------|\n")
        for r in all_results:
            f.write(f"| {r['image_id']} | {r['mask_iou']:.4f} | {r['fdi_accuracy_before']:.4f} | {r['fdi_accuracy_after']:.4f} | {r['corrections_applied']} |\n")
    
    print(f"\nEvaluation complete!")
    print(f"  Results saved to: {output_json}")
    print(f"  Report saved to: {output_md}")
    print(f"\nSummary:")
    print(f"  Mean FDI Accuracy (Before): {summary['mean_fdi_acc_before']:.4f}")
    print(f"  Mean FDI Accuracy (After): {summary['mean_fdi_acc_after']:.4f}")
    print(f"  Improvement: {summary['fdi_improvement']:+.4f}")


if __name__ == '__main__':
    main()



