#!/usr/bin/env python3
"""
Resolution evaluation script for comparing 512×512, 1024×512, and tiled inference.
Computes mask IoU, bbox mAP, and per-FDI accuracy for each method.
"""

import argparse
import json
import os
import random
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, Instances
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Import tiled inference
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from tools.tiled_inference import run_tiled_inference


def load_coco_annotations(coco_json_path: str) -> Dict:
    """Load COCO annotations."""
    with open(coco_json_path, 'r') as f:
        return json.load(f)


def find_image_path(filename: str, image_dir: str) -> Optional[str]:
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
    box2_area = (x2_max - x2_min) * (y2_max - y1_min)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def polygon_to_mask(polygon: List[float], height: int, width: int) -> np.ndarray:
    """Convert polygon to binary mask."""
    if len(polygon) < 6:
        return np.zeros((height, width), dtype=bool)
    
    # Extract x, y coordinates
    x_coords = polygon[::2]
    y_coords = polygon[1::2]
    
    # Create polygon points
    points = np.array([[x, y] for x, y in zip(x_coords, y_coords)], dtype=np.int32)
    
    # Create mask
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    
    return mask.astype(bool)


def compute_mask_iou_for_image(pred_masks: List[np.ndarray], gt_masks: List[np.ndarray],
                              iou_threshold: float = 0.5) -> Tuple[float, int, int]:
    """
    Compute average mask IoU for an image.
    
    Returns:
        (mean_iou, num_matched, num_total_gt)
    """
    if len(gt_masks) == 0:
        return 0.0, 0, 0
    
    if len(pred_masks) == 0:
        return 0.0, 0, len(gt_masks)
    
    # Match predictions to ground truth
    matched = set()
    ious = []
    
    for i, pred_mask in enumerate(pred_masks):
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt_mask in enumerate(gt_masks):
            if j in matched:
                continue
            
            iou = mask_iou(pred_mask, gt_mask)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched.add(best_gt_idx)
            ious.append(best_iou)
    
    mean_iou = np.mean(ious) if ious else 0.0
    return mean_iou, len(matched), len(gt_masks)


def compute_bbox_map(pred_boxes: List[np.ndarray], gt_boxes: List[np.ndarray],
                    pred_scores: List[float], iou_threshold: float = 0.5) -> float:
    """
    Compute bbox mAP@iou_threshold for an image.
    Simplified version - full mAP requires all images.
    """
    if len(gt_boxes) == 0:
        return 1.0 if len(pred_boxes) == 0 else 0.0
    
    if len(pred_boxes) == 0:
        return 0.0
    
    # Sort predictions by score
    sorted_indices = sorted(range(len(pred_boxes)), key=lambda i: pred_scores[i], reverse=True)
    
    # Match predictions to ground truth
    matched_gt = set()
    tp = 0
    fp = 0
    
    for idx in sorted_indices:
        pred_box = pred_boxes[idx]
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, gt_box in enumerate(gt_boxes):
            if j in matched_gt:
                continue
            
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            tp += 1
        else:
            fp += 1
    
    fn = len(gt_boxes) - len(matched_gt)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Approximate AP as precision (simplified)
    return precision if recall > 0 else 0.0


def compute_fdi_accuracy(pred_classes: List[int], gt_classes: List[int],
                        pred_boxes: List[np.ndarray], gt_boxes: List[np.ndarray],
                        iou_threshold: float = 0.5) -> float:
    """
    Compute FDI classification accuracy (only for correctly detected teeth).
    """
    if len(gt_classes) == 0:
        return 1.0 if len(pred_classes) == 0 else 0.0
    
    if len(pred_classes) == 0:
        return 0.0
    
    # Match predictions to ground truth by IoU
    matched_gt = set()
    correct_classifications = 0
    total_matched = 0
    
    for i, (pred_class, pred_box) in enumerate(zip(pred_classes, pred_boxes)):
        best_iou = 0.0
        best_gt_idx = -1
        
        for j, (gt_class, gt_box) in enumerate(zip(gt_classes, gt_boxes)):
            if j in matched_gt:
                continue
            
            iou = box_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= iou_threshold and best_gt_idx >= 0:
            matched_gt.add(best_gt_idx)
            total_matched += 1
            if pred_class == gt_classes[best_gt_idx]:
                correct_classifications += 1
    
    return correct_classifications / total_matched if total_matched > 0 else 0.0


def evaluate_single_image(predictor_512: DefaultPredictor, predictor_1024: DefaultPredictor,
                          image_path: str, gt_annotations: List[Dict],
                          coco_data: Dict, image_info: Dict) -> Dict:
    """
    Evaluate a single image with all three methods.
    
    Returns:
        Dictionary with metrics for 512, 1024, and tiled methods
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    h, w = image.shape[:2]
    
    # Prepare ground truth
    gt_masks = []
    gt_boxes = []
    gt_classes = []
    
    for ann in gt_annotations:
        # Get segmentation
        seg = ann.get('segmentation', [])
        if isinstance(seg, list) and len(seg) > 0:
            if isinstance(seg[0], list):
                # Polygon format
                mask = polygon_to_mask(seg[0], h, w)
            else:
                # RLE format
                mask = mask_utils.decode(seg)
                mask = mask.astype(bool)
        else:
            continue
        
        # Get bounding box
        bbox = ann.get('bbox', [])
        if len(bbox) == 4:
            # Convert from [x, y, w, h] to [x1, y1, x2, y2]
            x, y, bw, bh = bbox
            box = np.array([x, y, x + bw, y + bh])
        else:
            continue
        
        gt_masks.append(mask)
        gt_boxes.append(box)
        gt_classes.append(ann.get('category_id', 0))
    
    results = {}
    
    # Method A: 512×512 inference
    print("  Running 512×512 inference...")
    image_512 = cv2.resize(image, (512, 512))
    outputs_512 = predictor_512(image_512)
    instances_512 = outputs_512["instances"]
    
    pred_masks_512 = []
    pred_boxes_512 = []
    pred_classes_512 = []
    pred_scores_512 = []
    
    for i in range(len(instances_512)):
        mask = instances_512.pred_masks[i].cpu().numpy()
        # Resize mask back to original size
        mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        pred_masks_512.append(mask_resized)
        
        box = instances_512.pred_boxes.tensor[i].cpu().numpy()
        # Scale box back to original size
        scale_x = w / 512
        scale_y = h / 512
        box_scaled = box.copy()
        box_scaled[0] *= scale_x
        box_scaled[2] *= scale_x
        box_scaled[1] *= scale_y
        box_scaled[3] *= scale_y
        pred_boxes_512.append(box_scaled)
        
        pred_classes_512.append(instances_512.pred_classes[i].item())
        pred_scores_512.append(instances_512.scores[i].item())
    
    iou_512, matched_512, total_gt = compute_mask_iou_for_image(pred_masks_512, gt_masks)
    map_512 = compute_bbox_map(pred_boxes_512, gt_boxes, pred_scores_512)
    fdi_512 = compute_fdi_accuracy(pred_classes_512, gt_classes, pred_boxes_512, gt_boxes)
    
    results["512_iou"] = float(iou_512)
    results["512_map"] = float(map_512)
    results["512_fdi_accuracy"] = float(fdi_512)
    
    # Method B: 1024×512 inference (maintain aspect ratio)
    print("  Running 1024×512 inference...")
    # Resize maintaining aspect ratio
    target_w = 1024
    target_h = int(h * target_w / w)
    image_1024 = cv2.resize(image, (target_w, target_h))
    outputs_1024 = predictor_1024(image_1024)
    instances_1024 = outputs_1024["instances"]
    
    pred_masks_1024 = []
    pred_boxes_1024 = []
    pred_classes_1024 = []
    pred_scores_1024 = []
    
    for i in range(len(instances_1024)):
        mask = instances_1024.pred_masks[i].cpu().numpy()
        mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        pred_masks_1024.append(mask_resized)
        
        box = instances_1024.pred_boxes.tensor[i].cpu().numpy()
        scale_x = w / target_w
        scale_y = h / target_h
        box_scaled = box.copy()
        box_scaled[0] *= scale_x
        box_scaled[2] *= scale_x
        box_scaled[1] *= scale_y
        box_scaled[3] *= scale_y
        pred_boxes_1024.append(box_scaled)
        
        pred_classes_1024.append(instances_1024.pred_classes[i].item())
        pred_scores_1024.append(instances_1024.scores[i].item())
    
    iou_1024, matched_1024, _ = compute_mask_iou_for_image(pred_masks_1024, gt_masks)
    map_1024 = compute_bbox_map(pred_boxes_1024, gt_boxes, pred_scores_1024)
    fdi_1024 = compute_fdi_accuracy(pred_classes_1024, gt_classes, pred_boxes_1024, gt_boxes)
    
    results["1024_iou"] = float(iou_1024)
    results["1024_map"] = float(map_1024)
    results["1024_fdi_accuracy"] = float(fdi_1024)
    
    # Method C: Tiled inference
    print("  Running tiled inference...")
    try:
        tiled_results = run_tiled_inference(
            predictor_1024,
            image_path,
            tile_size=1024,
            overlap=256,
            output_path=None,
            visualize=False
        )
        
        pred_masks_tiled = tiled_results["merged_masks"]
        pred_boxes_tiled = [np.array(box) for box in tiled_results["merged_boxes"]]
        pred_classes_tiled = tiled_results["merged_FDI_labels"]
        pred_scores_tiled = tiled_results["merged_scores"]
        
        iou_tiled, matched_tiled, _ = compute_mask_iou_for_image(pred_masks_tiled, gt_masks)
        map_tiled = compute_bbox_map(pred_boxes_tiled, gt_boxes, pred_scores_tiled)
        fdi_tiled = compute_fdi_accuracy(pred_classes_tiled, gt_classes, pred_boxes_tiled, gt_boxes)
        
        results["tiled_iou"] = float(iou_tiled)
        results["tiled_map"] = float(map_tiled)
        results["tiled_fdi_accuracy"] = float(fdi_tiled)
    except Exception as e:
        print(f"  Error in tiled inference: {e}")
        results["tiled_iou"] = 0.0
        results["tiled_map"] = 0.0
        results["tiled_fdi_accuracy"] = 0.0
    
    # Determine winner
    if results["tiled_iou"] > results["1024_iou"] and results["tiled_iou"] > results["512_iou"]:
        results["winner"] = "tiled"
    elif results["1024_iou"] > results["512_iou"]:
        results["winner"] = "1024"
    else:
        results["winner"] = "512"
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate resolution impact on Mask R-CNN')
    parser.add_argument('--model-512', type=str, required=True, help='Path to 512×512 model')
    parser.add_argument('--model-1024', type=str, required=True, help='Path to 1024×512 model')
    parser.add_argument('--config-512', type=str, required=True, help='Path to 512 config')
    parser.add_argument('--config-1024', type=str, required=True, help='Path to 1024 config')
    parser.add_argument('--coco-json', type=str, required=True, help='Path to COCO JSON')
    parser.add_argument('--image-dir', type=str, required=True, help='Image directory')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory for reports')
    parser.add_argument('--num-samples', type=int, default=30, help='Number of validation images to evaluate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Load COCO data
    print("Loading COCO annotations...")
    coco_data = load_coco_annotations(args.coco_json)
    
    # Get validation images
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data.get('categories', [])
    
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
    cfg_512 = get_cfg()
    cfg_512.merge_from_file(args.config_512)
    cfg_512.MODEL.WEIGHTS = args.model_512
    cfg_512.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor_512 = DefaultPredictor(cfg_512)
    
    cfg_1024 = get_cfg()
    cfg_1024.merge_from_file(args.config_1024)
    cfg_1024.MODEL.WEIGHTS = args.model_1024
    cfg_1024.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor_1024 = DefaultPredictor(cfg_1024)
    
    # Evaluate each image
    all_results = []
    
    for i, image_info in enumerate(val_images):
        print(f"\nProcessing image {i+1}/{len(val_images)}: {image_info['file_name']}")
        
        # Find image path
        image_path = find_image_path(image_info['file_name'], args.image_dir)
        if image_path is None:
            print(f"  Warning: Could not find image {image_info['file_name']}")
            continue
        
        # Get ground truth annotations
        gt_anns = img_to_anns[image_info['id']]
        
        try:
            results = evaluate_single_image(
                predictor_512, predictor_1024,
                image_path, gt_anns, coco_data, image_info
            )
            results["image_id"] = image_info['id']
            results["filename"] = image_info['file_name']
            all_results.append(results)
        except Exception as e:
            print(f"  Error evaluating image: {e}")
            continue
    
    # Compute summary statistics
    summary = {
        "num_images": len(all_results),
        "mean_512_iou": np.mean([r["512_iou"] for r in all_results]),
        "mean_1024_iou": np.mean([r["1024_iou"] for r in all_results]),
        "mean_tiled_iou": np.mean([r["tiled_iou"] for r in all_results]),
        "mean_512_map": np.mean([r["512_map"] for r in all_results]),
        "mean_1024_map": np.mean([r["1024_map"] for r in all_results]),
        "mean_tiled_map": np.mean([r["tiled_map"] for r in all_results]),
        "mean_512_fdi": np.mean([r["512_fdi_accuracy"] for r in all_results]),
        "mean_1024_fdi": np.mean([r["1024_fdi_accuracy"] for r in all_results]),
        "mean_tiled_fdi": np.mean([r["tiled_fdi_accuracy"] for r in all_results]),
        "winner_counts": {
            "512": sum(1 for r in all_results if r["winner"] == "512"),
            "1024": sum(1 for r in all_results if r["winner"] == "1024"),
            "tiled": sum(1 for r in all_results if r["winner"] == "tiled")
        }
    }
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_json = os.path.join(args.output_dir, "resolution_comparison.json")
    with open(output_json, 'w') as f:
        json.dump({
            "summary": summary,
            "per_image": all_results
        }, f, indent=2)
    
    # Generate markdown report
    output_md = os.path.join(args.output_dir, "resolution_comparison.md")
    with open(output_md, 'w') as f:
        f.write("# Resolution Comparison Report\n\n")
        f.write(f"**Number of images evaluated:** {summary['num_images']}\n\n")
        f.write("## Summary Statistics\n\n")
        f.write("| Metric | 512×512 | 1024×512 | Tiled |\n")
        f.write("|--------|---------|----------|-------|\n")
        f.write(f"| Mean Mask IoU | {summary['mean_512_iou']:.4f} | {summary['mean_1024_iou']:.4f} | {summary['mean_tiled_iou']:.4f} |\n")
        f.write(f"| Mean bbox mAP@0.5 | {summary['mean_512_map']:.4f} | {summary['mean_1024_map']:.4f} | {summary['mean_tiled_map']:.4f} |\n")
        f.write(f"| Mean FDI Accuracy | {summary['mean_512_fdi']:.4f} | {summary['mean_1024_fdi']:.4f} | {summary['mean_tiled_fdi']:.4f} |\n\n")
        f.write("## Winner Distribution\n\n")
        f.write(f"- 512×512: {summary['winner_counts']['512']} images\n")
        f.write(f"- 1024×512: {summary['winner_counts']['1024']} images\n")
        f.write(f"- Tiled: {summary['winner_counts']['tiled']} images\n\n")
        f.write("## Per-Image Results\n\n")
        f.write("| Image ID | Filename | 512 IoU | 1024 IoU | Tiled IoU | Winner |\n")
        f.write("|----------|----------|---------|----------|-----------|-------|\n")
        for r in all_results:
            f.write(f"| {r['image_id']} | {r['filename']} | {r['512_iou']:.4f} | {r['1024_iou']:.4f} | {r['tiled_iou']:.4f} | {r['winner']} |\n")
    
    print(f"\nEvaluation complete!")
    print(f"  Results saved to: {output_json}")
    print(f"  Report saved to: {output_md}")
    print(f"\nSummary:")
    print(f"  Mean IoU - 512: {summary['mean_512_iou']:.4f}, 1024: {summary['mean_1024_iou']:.4f}, Tiled: {summary['mean_tiled_iou']:.4f}")


if __name__ == '__main__':
    main()



