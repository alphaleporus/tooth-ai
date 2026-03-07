#!/usr/bin/env python3
"""
Model Comparison Harness
=========================
Easy way to test multiple models on the same image and compare results.

Usage:
    python model_comparison.py path/to/image.jpg --models all
    python model_comparison.py path/to/image.jpg --models resnet50_9class_20k resnext101_cascade_60k
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
import time
from collections import defaultdict
import json

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
import torch


# Model Registry
MODELS = {
    "resnet50_9class_20k": {
        "path": "output/resnet50_9class_20k",
        "description": "ResNet-50 + FPN (9-class, 20k iter)",
        "classes": 9,
        "architecture": "Mask R-CNN",
        "speed": "Fast (~1.6s)"
    },
    "resnext101_cascade_60k": {
        "path": "output/resnext101_cascade_60k",
        "description": "ResNeXt-101 + Cascade (41-class, 60k iter)",
        "classes": 41,
        "architecture": "Cascade Mask R-CNN",
        "speed": "Slow (~6.4s)"
    },
    "rtx4060_48k": {
        "path": "output/rtx4060_48k",
        "description": "ResNet-50 + FPN (41-class, 48k iter)",
        "classes": 41,
        "architecture": "Mask R-CNN",
        "speed": "Medium (~2.5s)"
    },
    "cascade_r101_70k": {
        "path": "output/cascade_r101_70k",
        "description": "ResNet-101 + Cascade (41-class, 70k iter)",
        "classes": 41,
        "architecture": "Cascade Mask R-CNN",
        "speed": "Slow (~5.0s)",
        "note": "May be incomplete (check for model_final.pth)"
    },
    "resnext101_9class_40k": {
        "path": "output/resnext101_9class_40k",
        "description": "ResNeXt-101 + Cascade (9-class, 40k iter)",
        "classes": 9,
        "architecture": "Cascade Mask R-CNN",
        "speed": "Slow (~6.0s)",
        "note": "Status unclear"
    }
}


def load_model(model_name, threshold=0.05):
    """Load a model from the registry."""
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_info = MODELS[model_name]
    model_dir = Path(model_info["path"])
    config_path = model_dir / "config.yaml"
    weights_path = model_dir / "model_final.pth"
    
    if not weights_path.exists():
        print(f"⚠️  Model weights not found: {weights_path}")
        # Try to find checkpoint
        checkpoints = list(model_dir.glob("model_*.pth"))
        if checkpoints:
            weights_path = sorted(checkpoints)[-1]
            print(f"   Using checkpoint: {weights_path.name}")
        else:
            raise FileNotFoundError(f"No weights found for {model_name}")
    
    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    
    predictor = DefaultPredictor(cfg)
    
    return predictor, cfg, model_info


def run_model_inference(predictor, image, model_info):
    """Run inference and extract statistics."""
    start_time = time.time()
    outputs = predictor(image)
    inference_time = time.time() - start_time
    
    instances = outputs["instances"].to("cpu")
    
    # Extract predictions
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
    scores = instances.scores.numpy() if instances.has("scores") else []
    
    # Count teeth vs anomalies
    num_classes = model_info["classes"]
    if num_classes == 9:
        tooth_mask = classes == 0
    else:
        tooth_mask = classes < 33
    
    num_teeth = tooth_mask.sum()
    num_anomalies = (~tooth_mask).sum()
    
    # Score statistics
    if len(scores) > 0:
        score_stats = {
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "median": float(np.median(scores))
        }
        
        if num_teeth > 0:
            tooth_scores = scores[tooth_mask]
            tooth_stats = {
                "min": float(tooth_scores.min()),
                "max": float(tooth_scores.max()),
                "mean": float(tooth_scores.mean())
            }
        else:
            tooth_stats = None
    else:
        score_stats = None
        tooth_stats = None
    
    return {
        "total_detections": len(instances),
        "teeth_detected": int(num_teeth),
        "anomalies_detected": int(num_anomalies),
        "inference_time": inference_time,
        "score_stats": score_stats,
        "tooth_score_stats": tooth_stats
    }


def compare_models(image_path, model_names, threshold=0.05):
    """Compare multiple models on the same image."""
    print("="*70)
    print("MODEL COMPARISON ANALYSIS")
    print("="*70)
    
    # Load image
    print(f"\nLoading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"❌ Failed to load image")
        return None
    
    print(f"Image shape: {image.shape}")
    print(f"Threshold: {threshold}")
    
    results = {}
    
    for model_name in model_names:
        print(f"\n{'='*70}")
        print(f"Testing: {model_name}")
        print(f"{'='*70}")
        
        try:
            # Load model
            predictor, cfg, model_info = load_model(model_name, threshold)
            
            print(f"Architecture: {model_info['architecture']}")
            print(f"Classes: {model_info['classes']}")
            print(f"Expected speed: {model_info['speed']}")
            
            # Run inference
            print("Running inference...")
            result = run_model_inference(predictor, image, model_info)
            
            # Display results
            print(f"\n📊 RESULTS:")
            print(f"   Total detections: {result['total_detections']}")
            print(f"   Teeth detected: {result['teeth_detected']}")
            print(f"   Anomalies detected: {result['anomalies_detected']}")
            print(f"   Inference time: {result['inference_time']:.2f}s")
            
            if result['score_stats']:
                print(f"\n   Overall scores: [{result['score_stats']['min']:.3f}, {result['score_stats']['max']:.3f}]")
                print(f"      Mean: {result['score_stats']['mean']:.3f}")
            
            if result['tooth_score_stats']:
                print(f"\n   Tooth scores: [{result['tooth_score_stats']['min']:.3f}, {result['tooth_score_stats']['max']:.3f}]")
                print(f"      Mean: {result['tooth_score_stats']['mean']:.3f}")
            else:
                print(f"\n   ⚠️  No tooth predictions!")
            
            # Assessment
            if result['teeth_detected'] >= 20:
                print(f"\n   ✅ GOOD: Detected {result['teeth_detected']} teeth")
            elif result['teeth_detected'] >= 10:
                print(f"\n   ⚠️  MODERATE: Only {result['teeth_detected']} teeth")
            elif result['teeth_detected'] > 0:
                print(f"\n   ⚠️  POOR: Only {result['teeth_detected']} teeth")
            else:
                print(f"\n   ❌ CRITICAL: No teeth detected!")
            
            results[model_name] = {
                "model_info": model_info,
                "results": result,
                "status": "success"
            }
            
        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            results[model_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    
    print(f"\n{'Model':<30} {'Teeth':<8} {'Anomalies':<12} {'Time':<10} {'Status':<10}")
    print("-" * 70)
    
    for model_name, data in results.items():
        if data["status"] == "success":
            r = data["results"]
            teeth = r["teeth_detected"]
            anomalies = r["anomalies_detected"]
            time_str = f"{r['inference_time']:.2f}s"
            
            # Status symbol
            if teeth >= 20:
                status = "✅ Good"
            elif teeth >= 10:
                status = "⚠️  Moderate"
            elif teeth > 0:
                status = "⚠️  Poor"
            else:
                status = "❌ Failed"
            
            print(f"{model_name:<30} {teeth:<8} {anomalies:<12} {time_str:<10} {status:<10}")
        else:
            print(f"{model_name:<30} {'N/A':<8} {'N/A':<12} {'N/A':<10} {'❌ Error':<10}")
    
    # Recommendation
    print(f"\n{'='*70}")
    print("RECOMMENDATION")
    print(f"{'='*70}")
    
    best_model = None
    best_teeth = -1
    
    for model_name, data in results.items():
        if data["status"] == "success":
            teeth = data["results"]["teeth_detected"]
            if teeth > best_teeth:
                best_teeth = teeth
                best_model = model_name
    
    if best_model:
        print(f"\n🏆 Best Model: {best_model}")
        print(f"   Detected {best_teeth} teeth")
        print(f"\n   To use this model in app.py:")
        print(f"   1. Change MODEL_DIR = Path('output/{best_model}')")
        print(f"   2. Restart the Streamlit app")
    else:
        print("\n❌ None of the models worked!")
        print("   This suggests a fundamental issue:")
        print("   • Domain shift: New images are too different from training data")
        print("   • All models need fine-tuning on new dataset")
        print("   • Consider collecting 50-100 images and fine-tuning best model")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare multiple models")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--models", type=str, nargs="+", default=["all"],
                       help="Models to test (or 'all')")
    parser.add_argument("--threshold", type=float, default=0.05,
                       help="Detection threshold")
    
    args = parser.parse_args()
    
    # Determine which models to test
    if "all" in args.models:
        model_names = ["resnet50_9class_20k", "resnext101_cascade_60k", "rtx4060_48k"]
        print("Testing all primary models (excluding incomplete models)")
    else:
        model_names = args.models
    
    results = compare_models(args.image, model_names, args.threshold)
    
    if results is None:
        sys.exit(1)
    
    # Save results
    output_path = Path("model_comparison.json")
    
    # Convert to JSON-serializable format
    json_results = {}
    for model_name, data in results.items():
        if data["status"] == "success":
            json_results[model_name] = {
                "teeth_detected": data["results"]["teeth_detected"],
                "anomalies_detected": data["results"]["anomalies_detected"],
                "inference_time": data["results"]["inference_time"],
                "total_detections": data["results"]["total_detections"]
            }
        else:
            json_results[model_name] = {
                "status": "failed",
                "error": data.get("error", "Unknown error")
            }
    
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n📄 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
