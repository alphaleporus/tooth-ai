#!/usr/bin/env python3
"""
Tiled inference for high-resolution images using Mask R-CNN.
Splits large images into overlapping tiles, runs inference on each,
and merges results using soft-NMS and centroid clustering.
"""

import argparse
import os
import numpy as np
import cv2
import torch
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
import json

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
from detectron2.structures import Boxes, Instances
import detectron2.data.transforms as T


def upscale_image_if_needed(image: np.ndarray, target_width: int = 2048) -> np.ndarray:
    """
    Upscale image to target width if it's smaller.
    
    Args:
        image: Input image (H, W, C)
        target_width: Target width in pixels
    
    Returns:
        Upscaled image
    """
    h, w = image.shape[:2]
    
    if w < target_width:
        scale = target_width / w
        new_h = int(h * scale)
        new_w = target_width
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        print(f"Upscaled image from {w}x{h} to {new_w}x{new_h}")
    
    return image


def create_tiles(image: np.ndarray, tile_size: int = 1024, overlap: int = 256) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
    """
    Create overlapping tiles from an image.
    
    Args:
        image: Input image (H, W, C)
        tile_size: Size of each tile (square)
        overlap: Overlap between tiles in pixels
    
    Returns:
        List of (tile_image, (x_offset, y_offset)) tuples
    """
    h, w = image.shape[:2]
    tiles = []
    stride = tile_size - overlap
    
    y = 0
    while y < h:
        x = 0
        while x < w:
            # Extract tile
            y_end = min(y + tile_size, h)
            x_end = min(x + tile_size, w)
            
            tile = image[y:y_end, x:x_end]
            
            # Pad if tile is smaller than tile_size
            if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                padded_tile = np.zeros((tile_size, tile_size, image.shape[2]), dtype=image.dtype)
                padded_tile[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded_tile
            
            tiles.append((tile, (x, y)))
            
            x += stride
            if x >= w:
                break
        
        y += stride
        if y >= h:
            break
    
    return tiles


def compute_iou_mask(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def compute_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Compute centroid of a binary mask."""
    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] > 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return (cx, cy)
    return (0.0, 0.0)


def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """Compute Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def soft_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4, 
             sigma: float = 0.5, score_threshold: float = 0.3) -> np.ndarray:
    """
    Soft Non-Maximum Suppression.
    
    Args:
        boxes: (N, 4) array of boxes [x1, y1, x2, y2]
        scores: (N,) array of scores
        iou_threshold: IoU threshold for suppression
        sigma: Soft-NMS parameter
        score_threshold: Minimum score to keep
    
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    # Convert boxes to (x1, y1, x2, y2) format if needed
    if boxes.shape[1] == 4:
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    else:
        # Assume (x, y, w, h) format
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        
        if len(order) == 1:
            break
        
        # Compute IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Soft-NMS: decay scores instead of removing
        decay = np.exp(-(iou**2) / sigma)
        scores[order[1:]] *= decay
        
        # Remove boxes with low scores
        remaining = np.where(scores[order[1:]] >= score_threshold)[0]
        order = order[remaining + 1]
    
    return np.array(keep, dtype=np.int32)


def cluster_by_centroid(masks: List[np.ndarray], boxes: List[np.ndarray], 
                       scores: List[float], distance_threshold: float = 20.0) -> List[int]:
    """
    Cluster masks by centroid distance.
    
    Args:
        masks: List of binary masks
        boxes: List of bounding boxes
        scores: List of scores
        distance_threshold: Maximum distance for clustering
    
    Returns:
        List of cluster IDs for each mask
    """
    if len(masks) == 0:
        return []
    
    # Compute centroids
    centroids = [compute_centroid(mask) for mask in masks]
    
    # Simple clustering: assign to nearest cluster if within threshold
    clusters = []
    cluster_centers = []
    
    for i, centroid in enumerate(centroids):
        if len(cluster_centers) == 0:
            clusters.append(0)
            cluster_centers.append(centroid)
        else:
            # Find nearest cluster
            distances = [euclidean_distance(centroid, cc) for cc in cluster_centers]
            min_dist = min(distances)
            min_idx = distances.index(min_dist)
            
            if min_dist <= distance_threshold:
                clusters.append(min_idx)
            else:
                # New cluster
                clusters.append(len(cluster_centers))
                cluster_centers.append(centroid)
    
    return clusters


def merge_tile_predictions(all_predictions: List[Dict], image_shape: Tuple[int, int],
                          tile_offsets: List[Tuple[int, int]], iou_threshold: float = 0.4) -> Dict:
    """
    Merge predictions from multiple tiles.
    
    Args:
        all_predictions: List of prediction dicts from each tile
        image_shape: (height, width) of full image
        tile_offsets: List of (x, y) offsets for each tile
        iou_threshold: IoU threshold for merging
    
    Returns:
        Merged predictions dictionary
    """
    # Collect all detections with global coordinates
    all_boxes = []
    all_scores = []
    all_classes = []
    all_masks = []
    
    for pred, (x_off, y_off) in zip(all_predictions, tile_offsets):
        instances = pred["instances"]
        
        for i in range(len(instances)):
            box = instances.pred_boxes.tensor[i].cpu().numpy()
            # Convert to global coordinates
            box[0] += x_off
            box[2] += x_off
            box[1] += y_off
            box[3] += y_off
            
            mask = instances.pred_masks[i].cpu().numpy()
            # Shift mask to global coordinates
            h, w = image_shape[:2]
            global_mask = np.zeros((h, w), dtype=bool)
            mask_h, mask_w = mask.shape
            
            y_start = y_off
            y_end = min(y_off + mask_h, h)
            x_start = x_off
            x_end = min(x_off + mask_w, w)
            
            mask_y_start = 0
            mask_y_end = y_end - y_start
            mask_x_start = 0
            mask_x_end = x_end - x_start
            
            if y_end > y_start and x_end > x_start:
                global_mask[y_start:y_end, x_start:x_end] = mask[mask_y_start:mask_y_end, mask_x_start:mask_x_end]
            
            all_boxes.append(box)
            all_scores.append(instances.scores[i].item())
            all_classes.append(instances.pred_classes[i].item())
            all_masks.append(global_mask)
    
    if len(all_boxes) == 0:
        return {
            "merged_masks": [],
            "merged_boxes": [],
            "merged_FDI_labels": [],
            "merged_scores": []
        }
    
    # Convert to numpy arrays
    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_classes = np.array(all_classes)
    
    # Apply soft-NMS
    keep_indices = soft_nms(all_boxes, all_scores, iou_threshold=iou_threshold)
    
    # Cluster by centroid for remaining detections
    kept_masks = [all_masks[i] for i in keep_indices]
    kept_boxes = all_boxes[keep_indices]
    kept_scores = all_scores[keep_indices]
    kept_classes = all_classes[keep_indices]
    
    if len(kept_masks) > 0:
        clusters = cluster_by_centroid(kept_masks, kept_boxes, kept_scores, distance_threshold=20.0)
        
        # Merge masks within same cluster
        merged_masks = []
        merged_boxes = []
        merged_classes = []
        merged_scores = []
        
        cluster_groups = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            cluster_groups[cluster_id].append(i)
        
        for cluster_id, indices in cluster_groups.items():
            if len(indices) == 1:
                # Single detection, keep as is
                idx = indices[0]
                merged_masks.append(kept_masks[idx])
                merged_boxes.append(kept_boxes[idx])
                merged_classes.append(kept_classes[idx])
                merged_scores.append(kept_scores[idx])
            else:
                # Merge multiple detections in cluster
                cluster_masks = [kept_masks[i] for i in indices]
                cluster_scores = [kept_scores[i] for i in indices]
                
                # Use weighted average based on scores
                combined_mask = np.zeros_like(kept_masks[indices[0]], dtype=np.float32)
                total_score = sum(cluster_scores)
                
                for mask, score in zip(cluster_masks, cluster_scores):
                    combined_mask += mask.astype(np.float32) * (score / total_score)
                
                # Threshold to get binary mask
                combined_mask = (combined_mask > 0.5).astype(bool)
                
                # Get bounding box
                y_coords, x_coords = np.where(combined_mask)
                if len(y_coords) > 0 and len(x_coords) > 0:
                    x1, y1 = x_coords.min(), y_coords.min()
                    x2, y2 = x_coords.max(), y_coords.max()
                    box = np.array([x1, y1, x2, y2])
                    
                    merged_masks.append(combined_mask)
                    merged_boxes.append(box)
                    merged_classes.append(kept_classes[indices[0]])  # Use class of highest score
                    merged_scores.append(max(cluster_scores))
    else:
        merged_masks = []
        merged_boxes = []
        merged_classes = []
        merged_scores = []
    
    return {
        "merged_masks": merged_masks,
        "merged_boxes": merged_boxes,
        "merged_FDI_labels": merged_classes,  # These are class indices, map to FDI later
        "merged_scores": merged_scores
    }


def run_tiled_inference(predictor: DefaultPredictor, image_path: str, 
                       tile_size: int = 1024, overlap: int = 256,
                       output_path: Optional[str] = None,
                       visualize: bool = True) -> Dict:
    """
    Run tiled inference on a high-resolution image.
    
    Args:
        predictor: Detectron2 predictor
        image_path: Path to input image
        tile_size: Size of each tile
        overlap: Overlap between tiles
        output_path: Path to save visualization (optional)
        visualize: Whether to create visualization
    
    Returns:
        Dictionary with merged predictions
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    original_shape = image.shape[:2]
    print(f"Original image shape: {image.shape}")
    
    # Upscale if needed
    image = upscale_image_if_needed(image, target_width=2048)
    
    # Create tiles
    tiles = create_tiles(image, tile_size=tile_size, overlap=overlap)
    print(f"Created {len(tiles)} tiles")
    
    # Run inference on each tile
    all_predictions = []
    tile_offsets = []
    
    for i, (tile, (x_off, y_off)) in enumerate(tiles):
        print(f"Processing tile {i+1}/{len(tiles)} at offset ({x_off}, {y_off})")
        
        # Run inference
        outputs = predictor(tile)
        all_predictions.append(outputs)
        tile_offsets.append((x_off, y_off))
    
    # Merge predictions
    print("Merging predictions...")
    merged = merge_tile_predictions(all_predictions, image.shape, tile_offsets, iou_threshold=0.4)
    
    # Resize back to original if upscaled
    if image.shape[:2] != original_shape:
        scale_h = original_shape[0] / image.shape[0]
        scale_w = original_shape[1] / image.shape[1]
        
        # Resize masks and boxes
        resized_masks = []
        resized_boxes = []
        
        for mask, box in zip(merged["merged_masks"], merged["merged_boxes"]):
            # Resize mask
            resized_mask = cv2.resize(mask.astype(np.uint8), 
                                     (original_shape[1], original_shape[0]),
                                     interpolation=cv2.INTER_NEAREST).astype(bool)
            resized_masks.append(resized_mask)
            
            # Resize box
            resized_box = box.copy()
            resized_box[0] *= scale_w
            resized_box[1] *= scale_h
            resized_box[2] *= scale_w
            resized_box[3] *= scale_h
            resized_boxes.append(resized_box)
        
        merged["merged_masks"] = resized_masks
        merged["merged_boxes"] = resized_boxes
    
    # Create visualization if requested
    if visualize and output_path:
        vis_image = image.copy()
        if image.shape[:2] != original_shape:
            vis_image = cv2.resize(vis_image, (original_shape[1], original_shape[0]))
        
        for mask, box, score in zip(merged["merged_masks"], merged["merged_boxes"], merged["merged_scores"]):
            # Draw mask
            color = np.random.randint(0, 255, 3).tolist()
            vis_image[mask] = vis_image[mask] * 0.7 + np.array(color) * 0.3
            
            # Draw box
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_image, f"{score:.2f}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(output_path, vis_image)
        print(f"Visualization saved to {output_path}")
    
    return merged


def main():
    parser = argparse.ArgumentParser(description='Tiled inference for high-resolution images')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output visualization path')
    parser.add_argument('--tile-size', type=int, default=1024, help='Tile size (default: 1024)')
    parser.add_argument('--overlap', type=int, default=256, help='Overlap between tiles (default: 256)')
    parser.add_argument('--no-vis', action='store_true', help='Skip visualization')
    
    args = parser.parse_args()
    
    # Setup config and predictor
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    predictor = DefaultPredictor(cfg)
    
    # Run tiled inference
    results = run_tiled_inference(
        predictor,
        args.input,
        tile_size=args.tile_size,
        overlap=args.overlap,
        output_path=args.output if not args.no_vis else None,
        visualize=not args.no_vis
    )
    
    # Save results JSON
    output_json = args.output.replace('.png', '.json').replace('.jpg', '.json')
    with open(output_json, 'w') as f:
        json.dump({
            "num_detections": len(results["merged_masks"]),
            "boxes": [box.tolist() for box in results["merged_boxes"]],
            "scores": results["merged_scores"],
            "classes": results["merged_FDI_labels"]
        }, f, indent=2)
    
    print(f"\nResults saved:")
    print(f"  Visualizations: {args.output}")
    print(f"  JSON: {output_json}")
    print(f"  Detections: {len(results['merged_masks'])}")


if __name__ == '__main__':
    main()



