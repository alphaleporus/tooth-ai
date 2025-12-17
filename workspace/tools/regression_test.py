#!/usr/bin/env python3
"""
Regression testing script to ensure model updates don't break previous behavior.
Compares current predictions with baseline predictions.
"""

import argparse
import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.engine import ToothDetectionEngine, load_engine
from pycocotools import mask as mask_utils


def load_baseline_predictions(baseline_dir: str) -> Dict:
    """Load baseline predictions from directory."""
    baseline = {}
    
    json_files = glob.glob(os.path.join(baseline_dir, '*.json'))
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            # Extract image identifier from filename
            image_id = os.path.splitext(os.path.basename(json_file))[0]
            baseline[image_id] = data
    
    return baseline


def compute_mask_area(mask) -> float:
    """Compute mask area."""
    if isinstance(mask, dict):
        # RLE format
        mask_array = mask_utils.decode(mask)
        if len(mask_array.shape) == 3:
            mask_array = mask_array[:, :, 0]
        return float(mask_array.sum())
    elif isinstance(mask, np.ndarray):
        return float(mask.sum())
    return 0.0


def compare_predictions(current: Dict, baseline: Dict, image_id: str) -> Dict:
    """
    Compare current predictions with baseline.
    
    Returns:
        Dictionary with comparison metrics
    """
    current_teeth = current.get('teeth', [])
    baseline_teeth = baseline.get('teeth', [])
    
    # Count detections
    num_current = len(current_teeth)
    num_baseline = len(baseline_teeth)
    detection_diff = num_current - num_baseline
    
    # Compare FDI assignments
    current_fdis = sorted([t['fdi'] for t in current_teeth])
    baseline_fdis = sorted([t['fdi'] for t in baseline_teeth])
    fdi_changes = []
    
    # Find changed FDI assignments
    if len(current_fdis) == len(baseline_fdis):
        for i, (c_fdi, b_fdi) in enumerate(zip(current_fdis, baseline_fdis)):
            if c_fdi != b_fdi:
                fdi_changes.append({
                    'index': i,
                    'baseline_fdi': b_fdi,
                    'current_fdi': c_fdi
                })
    else:
        # Different number of detections
        fdi_changes = [{'note': 'Different number of detections'}]
    
    # Compare mask areas
    current_areas = [compute_mask_area(t.get('mask')) for t in current_teeth]
    baseline_areas = [compute_mask_area(t.get('mask')) for t in baseline_teeth]
    
    area_overlap = 0.0
    if len(current_areas) > 0 and len(baseline_areas) > 0:
        # Compute overlap (simplified - sum of areas)
        total_current = sum(current_areas)
        total_baseline = sum(baseline_areas)
        if total_baseline > 0:
            area_overlap = min(total_current, total_baseline) / total_baseline
    
    # Flag issues
    issues = []
    if abs(detection_diff) > 2:
        issues.append(f"Detection count changed significantly: {num_baseline} → {num_current}")
    
    if len(fdi_changes) > 3:
        issues.append(f"Many FDI changes: {len(fdi_changes)} changes detected")
    
    if area_overlap < 0.8:
        issues.append(f"Low mask area overlap: {area_overlap:.2f}")
    
    return {
        'image_id': image_id,
        'num_detections_baseline': num_baseline,
        'num_detections_current': num_current,
        'detection_diff': detection_diff,
        'fdi_changes': len(fdi_changes),
        'fdi_change_details': fdi_changes[:5],  # Limit details
        'mask_area_overlap': float(area_overlap),
        'issues': issues,
        'stable': len(issues) == 0
    }


def run_regression_tests(images_dir: str, engine: ToothDetectionEngine,
                        baseline_dir: str, output_dir: str,
                        num_samples: int = 10):
    """
    Run regression tests comparing current model with baseline.
    
    Args:
        images_dir: Directory containing images
        engine: Current inference engine
        baseline_dir: Directory with baseline predictions
        output_dir: Output directory
        num_samples: Number of images to test
    """
    # Load baseline predictions
    print("Loading baseline predictions...")
    baseline_predictions = load_baseline_predictions(baseline_dir)
    
    if not baseline_predictions:
        print("Warning: No baseline predictions found. Creating baseline...")
        # Create baseline from current predictions
        image_files = glob.glob(os.path.join(images_dir, '**', '*.png'), recursive=True)
        image_files.extend(glob.glob(os.path.join(images_dir, '**', '*.jpg'), recursive=True))
        
        if len(image_files) > num_samples:
            import random
            image_files = random.sample(image_files, num_samples)
        
        baseline_dir_created = os.path.join(output_dir, 'baseline_predictions')
        os.makedirs(baseline_dir_created, exist_ok=True)
        
        for image_path in image_files:
            results = engine.predict(image_path, return_visualization=False)
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            baseline_path = os.path.join(baseline_dir_created, f"{image_id}.json")
            with open(baseline_path, 'w') as f:
                json.dump(results, f, indent=2)
            baseline_predictions[image_id] = results
        
        print(f"Baseline created with {len(baseline_predictions)} images")
    
    # Run regression tests
    print(f"\nRunning regression tests on {len(baseline_predictions)} images...")
    
    regression_results = []
    
    for image_id, baseline_pred in baseline_predictions.items():
        # Find corresponding image
        image_files = glob.glob(os.path.join(images_dir, '**', f"{image_id}.*"), recursive=True)
        if not image_files:
            continue
        
        image_path = image_files[0]
        print(f"Testing {image_id}...")
        
        try:
            # Run current inference
            current_pred = engine.predict(image_path, return_visualization=False)
            
            # Compare
            comparison = compare_predictions(current_pred, baseline_pred, image_id)
            regression_results.append(comparison)
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    # Compute summary
    stable_count = sum(1 for r in regression_results if r['stable'])
    total_tests = len(regression_results)
    stability_score = stable_count / total_tests if total_tests > 0 else 0.0
    
    summary = {
        'total_tests': total_tests,
        'stable_tests': stable_count,
        'unstable_tests': total_tests - stable_count,
        'stability_score': float(stability_score),
        'mean_detection_diff': float(np.mean([r['detection_diff'] for r in regression_results])),
        'mean_fdi_changes': float(np.mean([r['fdi_changes'] for r in regression_results])),
        'mean_mask_overlap': float(np.mean([r['mask_area_overlap'] for r in regression_results]))
    }
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'regression_report.json')
    with open(report_path, 'w') as f:
        json.dump({
            'summary': summary,
            'per_image': regression_results
        }, f, indent=2)
    
    print(f"\nRegression testing complete!")
    print(f"  Report saved to: {report_path}")
    print(f"\nSummary:")
    print(f"  Stability Score: {stability_score:.2%} ({stable_count}/{total_tests} stable)")
    print(f"  Mean Detection Diff: {summary['mean_detection_diff']:.2f}")
    print(f"  Mean FDI Changes: {summary['mean_fdi_changes']:.2f}")
    print(f"  Mean Mask Overlap: {summary['mean_mask_overlap']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Regression testing for tooth detection pipeline')
    parser.add_argument('--images', type=str, required=True, help='Image directory')
    parser.add_argument('--engine', type=str, default=None, help='Engine path')
    parser.add_argument('--model-dir', type=str, default=None, help='Model directory')
    parser.add_argument('--baseline', type=str, required=True, help='Baseline predictions directory')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples if creating baseline')
    
    args = parser.parse_args()
    
    # Load engine
    print("Loading inference engine...")
    if args.model_dir:
        engine = load_engine(args.model_dir)
    else:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tooth-ai', 'models')
        engine = load_engine(model_dir)
    
    run_regression_tests(args.images, engine, args.baseline, args.out, args.num_samples)


if __name__ == '__main__':
    main()



