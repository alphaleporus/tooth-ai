#!/usr/bin/env python3
"""
Diagnostic Script 1: Raw Model Predictions Inspector
====================================================
Inspects raw model outputs BEFORE any thresholding or filtering.
Helps identify if the model is producing tooth predictions at all.

Usage:
    python diagnostic_raw_predictions.py path/to/image.jpg --model resnet50_9class_20k
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
from collections import Counter
import json

# Detectron2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
import torch


def load_model(model_path, config_path):
    """Load Detectron2 model with minimal thresholding."""
    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(model_path)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Use MINIMAL threshold to capture everything
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001  # Ultra-low
    
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def analyze_raw_predictions(image_path, model_name="resnet50_9class_20k"):
    """
    Run inference and dump ALL raw predictions with detailed statistics.
    
    Returns:
        dict with comprehensive diagnostic information
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
        print(f"❌ Failed to load image: {image_path}")
        return None
    
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Pixel range: [{image.min()}, {image.max()}]")
    
    # Run inference
    print("\n" + "="*60)
    print("Running inference...")
    print("="*60)
    
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    
    # Extract raw predictions
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
    scores = instances.scores.numpy() if instances.has("scores") else []
    
    # Class names (assuming 9-class or 41-class model)
    num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    print(f"\nModel configured for {num_classes} classes")
    
    # 9-Class schema
    CLASSES_9 = [
        "Tooth", "Caries", "Crown", "Filling", "Implant",
        "Prefabricated metal post", "Retained root",
        "Root canal filling", "Root canal obturation"
    ]
    
    # 41-Class schema (33 teeth + 8 anomalies)
    CLASSES_41 = [f"Tooth_{i}" for i in range(33)] + CLASSES_9[1:]
    
    class_names = CLASSES_9 if num_classes == 9 else CLASSES_41
    
    # Diagnostic Report
    report = {
        "model_name": model_name,
        "image_path": str(image_path),
        "image_shape": image.shape,
        "num_classes": num_classes,
        "total_predictions": len(instances),
        "class_distribution": {},
        "score_statistics": {},
        "predictions_by_threshold": {},
        "detailed_predictions": []
    }
    
    print(f"\n{'='*60}")
    print(f"RAW PREDICTIONS SUMMARY")
    print(f"{'='*60}")
    print(f"Total predictions: {len(instances)}")
    
    if len(instances) == 0:
        print("\n⚠️  WARNING: Model produced ZERO predictions!")
        print("   This suggests:")
        print("   1. Model weights may be corrupted")
        print("   2. Input image is incompatible with model")
        print("   3. Image preprocessing is breaking the input")
        return report
    
    # Class distribution
    class_counter = Counter(classes)
    report["class_distribution"] = {
        class_names[cls]: int(count) for cls, count in class_counter.items()
    }
    
    print(f"\nClass Distribution:")
    for cls, count in sorted(class_counter.items()):
        cls_name = class_names[cls] if cls < len(class_names) else f"Class_{cls}"
        print(f"  {cls_name}: {count} predictions")
    
    # Score statistics
    if len(scores) > 0:
        report["score_statistics"] = {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "median": float(np.median(scores)),
            "std": float(scores.std())
        }
        
        print(f"\nScore Statistics:")
        print(f"  Min:    {scores.min():.4f}")
        print(f"  Max:    {scores.max():.4f}")
        print(f"  Mean:   {scores.mean():.4f}")
        print(f"  Median: {np.median(scores):.4f}")
        print(f"  Std:    {scores.std():.4f}")
    
    # Threshold analysis
    thresholds = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    print(f"\nPredictions Passing Different Thresholds:")
    print(f"{'Threshold':<12} {'Total':<8} {'Tooth':<8} {'Anomalies':<12}")
    print("-" * 50)
    
    for threshold in thresholds:
        passing = scores >= threshold
        passing_classes = classes[passing]
        
        # Count teeth vs anomalies
        if num_classes == 9:
            tooth_count = np.sum(passing_classes == 0)  # Class 0 = Tooth
            anomaly_count = np.sum(passing_classes > 0)
        else:
            tooth_count = np.sum(passing_classes < 33)  # Classes 0-32 = Teeth
            anomaly_count = np.sum(passing_classes >= 33)
        
        total = passing.sum()
        print(f"{threshold:<12.2f} {total:<8} {tooth_count:<8} {anomaly_count:<12}")
        
        report["predictions_by_threshold"][f"{threshold:.2f}"] = {
            "total": int(total),
            "teeth": int(tooth_count),
            "anomalies": int(anomaly_count)
        }
    
    # Detailed predictions (top 20 by score)
    print(f"\nTop 20 Predictions (sorted by confidence):")
    print(f"{'Rank':<6} {'Class':<30} {'Score':<8} {'BBox (x1,y1,x2,y2)':<30}")
    print("-" * 80)
    
    sorted_idx = np.argsort(-scores)[:20]
    for rank, idx in enumerate(sorted_idx, 1):
        cls_name = class_names[classes[idx]] if classes[idx] < len(class_names) else f"Class_{classes[idx]}"
        box = boxes[idx]
        score = scores[idx]
        
        print(f"{rank:<6} {cls_name:<30} {score:<8.4f} ({box[0]:.0f}, {box[1]:.0f}, {box[2]:.0f}, {box[3]:.0f})")
        
        report["detailed_predictions"].append({
            "rank": rank,
            "class": cls_name,
            "class_id": int(classes[idx]),
            "score": float(score),
            "bbox": box.tolist()
        })
    
    # CRITICAL DIAGNOSTIC: Teeth predictions analysis
    if num_classes == 9:
        tooth_mask = classes == 0
    else:
        tooth_mask = classes < 33
    
    tooth_scores = scores[tooth_mask]
    
    print(f"\n{'='*60}")
    print(f"TOOTH DETECTION ANALYSIS")
    print(f"{'='*60}")
    print(f"Total tooth predictions: {len(tooth_scores)}")
    
    if len(tooth_scores) == 0:
        print("\n❌ CRITICAL: NO TOOTH PREDICTIONS FOUND!")
        print("   This is the root cause of your detection failure.")
        print("   Possible reasons:")
        print("   1. Model was trained only on anomalies (unlikely)")
        print("   2. Domain shift: new images are too different from training")
        print("   3. Preprocessing is corrupting the input")
        print("   4. Model weights are from wrong checkpoint")
    else:
        print(f"Tooth score range: [{tooth_scores.min():.4f}, {tooth_scores.max():.4f}]")
        print(f"Tooth score mean: {tooth_scores.mean():.4f}")
        
        # Check if tooth scores are suspiciously low
        if tooth_scores.max() < 0.05:
            print("\n⚠️  WARNING: All tooth scores < 0.05!")
            print("   Model is detecting teeth but with very low confidence.")
            print("   This suggests domain shift - model doesn't recognize these X-rays.")
            print("\n   RECOMMENDATION: Try a different model architecture.")
        elif tooth_scores.max() < 0.15:
            print("\n⚠️  All tooth scores < 0.15 - confidence is low but detectable.")
            print("   Lowering threshold to 0.03 should help.")
        else:
            print("\n✅ Tooth predictions look normal. Threshold tuning should work.")
    
    # Save report to JSON
    output_path = Path("diagnostic_report.json")
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Diagnostic report saved to: {output_path}")
    print(f"{'='*60}")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Analyze raw model predictions")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--model", type=str, default="resnet50_9class_20k",
                       choices=["resnet50_9class_20k", "resnext101_cascade_60k", 
                               "rtx4060_48k", "cascade_r101_70k"],
                       help="Model to test")
    
    args = parser.parse_args()
    
    report = analyze_raw_predictions(args.image, args.model)
    
    if report:
        print("\n✅ Diagnostic complete!")
        
        # Quick assessment
        total = report["total_predictions"]
        if total == 0:
            print("\n🔴 STATUS: CRITICAL - No predictions at all")
        elif "Tooth" in report["class_distribution"]:
            tooth_count = report["class_distribution"]["Tooth"]
            if tooth_count == 0:
                print("\n🔴 STATUS: CRITICAL - No tooth predictions")
            elif tooth_count < 10:
                print("\n🟡 STATUS: WARNING - Very few tooth predictions")
            else:
                print("\n🟢 STATUS: OK - Tooth predictions found")
    else:
        print("\n❌ Diagnostic failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
