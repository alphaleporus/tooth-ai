#!/usr/bin/env python3
"""
Failure case miner - identifies worst-performing images for analysis.
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import cv2
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.engine import ToothDetectionEngine, load_engine
from inference.visualize import visualize_predictions, save_visualization


def load_batch_metrics(metrics_path: str) -> Dict:
    """Load batch validation metrics."""
    with open(metrics_path, 'r') as f:
        return json.load(f)


def find_worst_cases(metrics: Dict, num_cases: int = 20) -> List[Dict]:
    """
    Identify worst cases by various criteria.
    
    Returns:
        List of worst case dictionaries
    """
    per_image = metrics.get('per_image', [])
    
    if not per_image:
        return []
    
    # Score each image by multiple criteria
    scored_cases = []
    
    for img_data in per_image:
        # Compute composite score (lower is worse)
        iou = img_data.get('mask_iou', 0.0)
        fdi_acc = img_data.get('fdi_accuracy_after', 0.0)
        num_det = img_data.get('num_detections', 0)
        num_gt = img_data.get('num_gt', 0)
        
        # Penalties
        iou_penalty = 1.0 - iou  # Lower IoU = higher penalty
        fdi_penalty = 1.0 - fdi_acc  # Lower accuracy = higher penalty
        detection_mismatch = abs(num_det - num_gt)  # Mismatch penalty
        detection_count_issue = 0
        if num_det < 28 or num_det > 32:
            detection_count_issue = 10  # Significant penalty
        
        composite_score = iou_penalty + fdi_penalty + detection_mismatch * 0.1 + detection_count_issue
        
        scored_cases.append({
            **img_data,
            'composite_score': composite_score,
            'iou_penalty': iou_penalty,
            'fdi_penalty': fdi_penalty,
            'detection_mismatch': detection_mismatch
        })
    
    # Sort by composite score (worst first)
    scored_cases.sort(key=lambda x: x['composite_score'], reverse=True)
    
    # Return worst N cases
    return scored_cases[:num_cases]


def visualize_failcase(engine: ToothDetectionEngine, image_path: str,
                       failcase_data: Dict, output_path: str):
    """Create visualization for a failcase."""
    # Run inference
    results = engine.predict(image_path, return_visualization=False)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        return
    
    # Create visualization
    predictions = [
        {
            'fdi': t['fdi'],
            'bbox': t['bbox'],
            'mask': t.get('mask'),
            'centroid': t['centroid'],
            'final_confidence': t['score'],
            'method_used': t['method_used'],
            'correction_applied': t['correction_applied']
        } for t in results['teeth']
    ]
    
    vis_image = visualize_predictions(image, predictions)
    
    # Add text annotations
    text_lines = [
        f"IoU: {failcase_data['mask_iou']:.3f}",
        f"FDI Acc: {failcase_data['fdi_accuracy_after']:.3f}",
        f"Detections: {failcase_data['num_detections']}/{failcase_data['num_gt']}",
        f"Score: {failcase_data['composite_score']:.3f}"
    ]
    
    y_offset = 30
    for line in text_lines:
        cv2.putText(vis_image, line, (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        y_offset += 30
    
    save_visualization(vis_image, output_path)


def mine_failcases(metrics_path: str, images_dir: str, engine: ToothDetectionEngine,
                  output_dir: str, num_cases: int = 20):
    """
    Mine failure cases from batch metrics.
    
    Args:
        metrics_path: Path to batch_metrics.json
        images_dir: Directory containing images
        engine: Inference engine
        output_dir: Output directory
    """
    # Load metrics
    print("Loading batch metrics...")
    metrics = load_batch_metrics(metrics_path)
    
    # Find worst cases
    print(f"Identifying worst {num_cases} cases...")
    worst_cases = find_worst_cases(metrics, num_cases)
    
    print(f"Found {len(worst_cases)} worst cases")
    
    # Create visualizations
    os.makedirs(output_dir, exist_ok=True)
    failcase_list = []
    
    for i, failcase in enumerate(worst_cases):
        print(f"Processing failcase {i+1}/{len(worst_cases)}: {failcase['filename']}")
        
        # Find image
        image_files = glob.glob(os.path.join(images_dir, '**', failcase['filename']), recursive=True)
        if not image_files:
            # Try with different extensions
            base_name = os.path.splitext(failcase['filename'])[0]
            image_files = glob.glob(os.path.join(images_dir, '**', f"{base_name}.*"), recursive=True)
        
        if image_files:
            image_path = image_files[0]
            
            # Create visualization
            vis_path = os.path.join(output_dir, f"failcase_{i+1:02d}_{failcase['filename']}")
            visualize_failcase(engine, image_path, failcase, vis_path)
            
            failcase_list.append({
                'rank': i + 1,
                'image_id': failcase['image_id'],
                'filename': failcase['filename'],
                'mask_iou': failcase['mask_iou'],
                'fdi_accuracy': failcase['fdi_accuracy_after'],
                'num_detections': failcase['num_detections'],
                'num_gt': failcase['num_gt'],
                'composite_score': failcase['composite_score'],
                'issues': [
                    f"IoU: {failcase['mask_iou']:.3f}",
                    f"FDI Accuracy: {failcase['fdi_accuracy_after']:.3f}",
                    f"Detection mismatch: {failcase['num_detections']} vs {failcase['num_gt']}"
                ],
                'visualization': vis_path
            })
    
    # Save failcases JSON
    failcases_path = os.path.join(output_dir, 'failcases.json')
    with open(failcases_path, 'w') as f:
        json.dump({
            'num_failcases': len(failcase_list),
            'failcases': failcase_list
        }, f, indent=2)
    
    print(f"\nFailcase mining complete!")
    print(f"  Failcases saved to: {failcases_path}")
    print(f"  Visualizations saved to: {output_dir}")
    
    # Print summary
    print(f"\nTop 5 worst cases:")
    for i, fc in enumerate(failcase_list[:5]):
        print(f"  {i+1}. {fc['filename']}: IoU={fc['mask_iou']:.3f}, FDI={fc['fdi_accuracy']:.3f}")


def main():
    parser = argparse.ArgumentParser(description='Mine failure cases from batch metrics')
    parser.add_argument('--metrics', type=str, required=True, help='Path to batch_metrics.json')
    parser.add_argument('--images', type=str, required=True, help='Image directory')
    parser.add_argument('--engine', type=str, default=None, help='Engine path')
    parser.add_argument('--model-dir', type=str, default=None, help='Model directory')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--num-cases', type=int, default=20, help='Number of worst cases to mine')
    
    args = parser.parse_args()
    
    # Load engine
    print("Loading inference engine...")
    if args.model_dir:
        engine = load_engine(args.model_dir)
    else:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tooth-ai', 'models')
        engine = load_engine(model_dir)
    
    mine_failcases(args.metrics, args.images, engine, args.out, args.num_cases)


if __name__ == '__main__':
    main()



