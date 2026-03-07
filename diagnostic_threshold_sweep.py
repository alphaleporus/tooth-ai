#!/usr/bin/env python3
"""
Diagnostic Script 2: Threshold Sweep Analysis
==============================================
Tests range of thresholds to find optimal values for tooth/anomaly detection.
Creates visualization showing detection counts vs threshold.

Usage:
    python diagnostic_threshold_sweep.py path/to/image.jpg --output sweep_results.png
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict
import json

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import torch


def load_model(model_path, config_path):
    """Load Detectron2 model."""
    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001  # Minimal threshold
    
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def threshold_sweep(image_path, model_name="resnet50_9class_20k", 
                   min_thresh=0.01, max_thresh=0.50, step=0.01):
    """
    Sweep through threshold values and count detections.
    
    Args:
        image_path: Path to test image
        model_name: Model directory name
        min_thresh: Minimum threshold to test
        max_thresh: Maximum threshold to test
        step: Threshold step size
        
    Returns:
        dict with sweep results
    """
    # Model paths
    model_dir = Path(f"output/{model_name}")
    config_path = model_dir / "config.yaml"
    weights_path = model_dir / "model_final.pth"
    
    if not weights_path.exists():
        print(f"❌ Model not found: {weights_path}")
        return None
    
    print(f"Loading model: {model_name}")
    predictor, cfg = load_model(weights_path, config_path)
    
    # Load image
    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to load image")
        return None
    
    # Run inference once
    print("Running inference...")
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    
    num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    is_9_class = (num_classes == 9)
    
    print(f"Total raw predictions: {len(instances)}")
    print(f"Sweeping thresholds from {min_thresh} to {max_thresh} (step={step})")
    
    # Sweep thresholds
    thresholds = np.arange(min_thresh, max_thresh + step, step)
    results = {
        "thresholds": [],
        "total_detections": [],
        "tooth_detections": [],
        "anomaly_detections": [],
        "image_width": image.shape[1]
    }
    
    # Zone-based analysis (matching app.py logic)
    image_width = image.shape[1]
    center_start = image_width * 0.35
    center_end = image_width * 0.65
    
    # Get box centers
    boxes = instances.pred_boxes.tensor.numpy()
    x_centers = (boxes[:, 0] + boxes[:, 2]) / 2
    in_center_zone = (x_centers > center_start) & (x_centers < center_end)
    
    results["center_zone_detections"] = []
    results["outer_zone_detections"] = []
    
    for threshold in thresholds:
        passing = scores >= threshold
        passing_classes = classes[passing]
        passing_centers = in_center_zone[passing]
        
        # Count teeth vs anomalies
        if is_9_class:
            tooth_mask = passing_classes == 0
        else:
            tooth_mask = passing_classes < 33
        
        tooth_count = tooth_mask.sum()
        anomaly_count = (~tooth_mask).sum()
        total = len(passing_classes)
        
        # Zone-based counts for teeth
        center_teeth = (tooth_mask & passing_centers).sum()
        outer_teeth = (tooth_mask & ~passing_centers).sum()
        
        results["thresholds"].append(float(threshold))
        results["total_detections"].append(int(total))
        results["tooth_detections"].append(int(tooth_count))
        results["anomaly_detections"].append(int(anomaly_count))
        results["center_zone_detections"].append(int(center_teeth))
        results["outer_zone_detections"].append(int(outer_teeth))
    
    return results


def plot_sweep_results(results, output_path="threshold_sweep.png"):
    """Create visualization of threshold sweep results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Threshold Sweep Analysis", fontsize=16, fontweight='bold')
    
    thresholds = results["thresholds"]
    
    # Plot 1: Total detections vs threshold
    ax = axes[0, 0]
    ax.plot(thresholds, results["total_detections"], 'b-', linewidth=2, label="Total")
    ax.plot(thresholds, results["tooth_detections"], 'g-', linewidth=2, label="Teeth")
    ax.plot(thresholds, results["anomaly_detections"], 'r-', linewidth=2, label="Anomalies")
    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Number of Detections", fontsize=12)
    ax.set_title("Detection Count vs Threshold", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Tooth detections (log scale to see low values)
    ax = axes[0, 1]
    ax.semilogy(thresholds, np.array(results["tooth_detections"]) + 1, 'g-', linewidth=2)
    ax.axhline(y=28, color='orange', linestyle='--', label="Expected (28 teeth)", alpha=0.7)
    ax.axhline(y=20, color='yellow', linestyle='--', label="Minimum acceptable (20)", alpha=0.7)
    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Tooth Detections (log scale)", fontsize=12)
    ax.set_title("Tooth Detection Sensitivity", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Zone-based tooth detections
    ax = axes[1, 0]
    ax.plot(thresholds, results["center_zone_detections"], 'purple', linewidth=2, 
            label="Center Zone (35-65%)")
    ax.plot(thresholds, results["outer_zone_detections"], 'cyan', linewidth=2,
            label="Outer Zones")
    ax.set_xlabel("Confidence Threshold", fontsize=12)
    ax.set_ylabel("Tooth Detections", fontsize=12)
    ax.set_title("Zone-Based Tooth Detection", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Recommended thresholds (annotation)
    ax = axes[1, 1]
    ax.axis('off')
    
    # Find optimal thresholds
    tooth_counts = np.array(results["tooth_detections"])
    
    # Find threshold giving closest to 28 teeth
    optimal_idx = np.argmin(np.abs(tooth_counts - 28))
    optimal_threshold = thresholds[optimal_idx]
    optimal_count = tooth_counts[optimal_idx]
    
    # Find threshold giving at least 20 teeth
    valid_mask = tooth_counts >= 20
    if valid_mask.any():
        conservative_idx = np.where(valid_mask)[0][-1]  # Highest threshold with ≥20 teeth
        conservative_threshold = thresholds[conservative_idx]
        conservative_count = tooth_counts[conservative_idx]
    else:
        conservative_threshold = min_thresh
        conservative_count = tooth_counts[0]
    
    # Display recommendations
    recommendations = f"""
    THRESHOLD RECOMMENDATIONS
    {'='*40}
    
    🎯 Optimal (closest to 28 teeth):
       Threshold: {optimal_threshold:.3f}
       Detections: {optimal_count} teeth
    
    🛡️  Conservative (minimum 20 teeth):
       Threshold: {conservative_threshold:.3f}
       Detections: {conservative_count} teeth
    
    📊 Current app.py settings:
       Center zone: 0.03
       Outer zones: 0.35
       Anomalies: 0.45
    
    💡 Analysis:
    """
    
    # Add diagnostic insights
    center_at_003 = results["center_zone_detections"][thresholds.index(0.03)] if 0.03 in thresholds else 0
    outer_at_035 = results["outer_zone_detections"][thresholds.index(0.35)] if 0.35 in thresholds else 0
    
    if tooth_counts.max() == 0:
        recommendations += "\n    ❌ NO teeth detected at any threshold!"
        recommendations += "\n    → Model is NOT producing tooth predictions"
        recommendations += "\n    → Try alternative model architecture"
    elif optimal_count < 15:
        recommendations += f"\n    ⚠️  Maximum {tooth_counts.max()} teeth detected"
        recommendations += "\n    → Severe underfitting or domain shift"
        recommendations += "\n    → Consider model fine-tuning"
    else:
        recommendations += f"\n    ✅ Model capable of detecting {tooth_counts.max()} teeth"
        recommendations += f"\n    → Use threshold ~{optimal_threshold:.3f}"
    
    ax.text(0.05, 0.95, recommendations, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {output_path}")
    
    return {
        "optimal_threshold": float(optimal_threshold),
        "optimal_count": int(optimal_count),
        "conservative_threshold": float(conservative_threshold),
        "conservative_count": int(conservative_count)
    }


def main():
    parser = argparse.ArgumentParser(description="Threshold sweep analysis")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="resnet50_9class_20k",
                       help="Model to test")
    parser.add_argument("--output", type=str, default="threshold_sweep.png",
                       help="Output plot filename")
    parser.add_argument("--min", type=float, default=0.01, help="Min threshold")
    parser.add_argument("--max", type=float, default=0.50, help="Max threshold")
    parser.add_argument("--step", type=float, default=0.01, help="Threshold step")
    
    args = parser.parse_args()
    
    print("="*60)
    print("THRESHOLD SWEEP DIAGNOSTIC")
    print("="*60)
    
    results = threshold_sweep(args.image, args.model, args.min, args.max, args.step)
    
    if results is None:
        print("❌ Sweep failed!")
        sys.exit(1)
    
    # Create visualization
    recommendations = plot_sweep_results(results, args.output)
    
    # Save numerical results
    output_json = Path(args.output).with_suffix('.json')
    combined_results = {**results, **recommendations}
    with open(output_json, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"✅ Data saved to: {output_json}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Recommended threshold: {recommendations['optimal_threshold']:.3f}")
    print(f"Expected detections: {recommendations['optimal_count']} teeth")
    print(f"\nConservative threshold: {recommendations['conservative_threshold']:.3f}")
    print(f"Expected detections: {recommendations['conservative_count']} teeth")
    print("="*60)


if __name__ == "__main__":
    main()
