#!/usr/bin/env python3
"""
Integrated Diagnostic Using Existing App Infrastructure
========================================================
This script uses the existing app.py model loading to avoid dependency issues.
Works within your current environment setup.

Usage:
    python integrated_diagnostic.py
"""

import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Import from existing app.py
sys.path.insert(0, str(Path(__file__).parent))

print("Loading Tooth-AI infrastructure...")
print("="*70)

# Import necessary components from app.py
try:
    import streamlit as st
    import numpy as np
    from PIL import Image
    import cv2
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    print("✅ All dependencies loaded successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("\nPlease ensure you're using the same Python environment as the Streamlit app.")
    sys.exit(1)


# Model registry
MODELS_TO_TEST = {
    "resnet50_9class_20k": {
        "path": "output/resnet50_9class_20k",
        "type": "9-class",
        "description": "Current production model"
    },
    "resnext101_cascade_60k": {
        "path": "output/resnext101_cascade_60k",
        "type": "41-class",
        "description": "ResNeXt-101 Cascade (recommended alternative)"
    },
    "rtx4060_48k": {
        "path": "output/rtx4060_48k",
        "type": "41-class",
        "description": "RTX4060 optimized model"
    }
}


def load_model_simple(model_name):
    """Load a model using Detectron2."""
    model_info = MODELS_TO_TEST[model_name]
    model_dir = Path(model_info["path"])
    
    config_path = model_dir / "config.yaml"
    weights_path = model_dir / "model_final.pth"
    
    if not weights_path.exists():
        print(f"⚠️  Model not found: {weights_path}")
        return None, None
    
    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    
    predictor = DefaultPredictor(cfg)
    
    return predictor, cfg


def count_detections(predictor, image_path, model_type):
    """Run inference and count teeth vs anomalies."""
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    # Run inference
    outputs = predictor(image)
    instances = outputs["instances"].to("cpu")
    
    if len(instances) == 0:
        return {
            'total': 0,
            'teeth': 0,
            'anomalies': 0,
            'tooth_scores': []
        }
    
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    
    # Count based on model type
    if model_type == "9-class":
        tooth_mask = classes == 0
    else:  # 41-class
        tooth_mask = classes <= 32
    
    teeth_count = tooth_mask.sum()
    anomaly_count = (~tooth_mask).sum()
    
    tooth_scores = scores[tooth_mask].tolist() if teeth_count > 0 else []
    
    return {
        'total': len(instances),
        'teeth': int(teeth_count),
        'anomalies': int(anomaly_count),
        'tooth_scores': tooth_scores,
        'score_range': [float(min(tooth_scores)), float(max(tooth_scores))] if tooth_scores else [0, 0]
    }


def test_image_on_all_models(image_path):
    """Test a single image on all available models."""
    print(f"\n{'='*70}")
    print(f"Testing: {Path(image_path).name}")
    print(f"{'='*70}")
    
    results = {}
    
    for model_name, model_info in MODELS_TO_TEST.items():
        print(f"\n🔍 Testing {model_name}...")
        
        predictor, cfg = load_model_simple(model_name)
        
        if predictor is None:
            print(f"   ❌ Model not available")
            results[model_name] = {'status': 'unavailable'}
            continue
        
        try:
            detection_results = count_detections(predictor, image_path, model_info['type'])
            
            if detection_results is None:
                print(f"   ❌ Failed to process image")
                results[model_name] = {'status': 'failed'}
                continue
            
            # Display results
            teeth = detection_results['teeth']
            anomalies = detection_results['anomalies']
            
            print(f"   Teeth detected: {teeth}")
            print(f"   Anomalies detected: {anomalies}")
            
            if teeth > 0:
                score_range = detection_results['score_range']
                print(f"   Tooth score range: [{score_range[0]:.3f}, {score_range[1]:.3f}]")
            
            # Status
            if teeth >= 24:
                status = "✅ EXCELLENT"
            elif teeth >= 20:
                status = "✅ GOOD"
            elif teeth >= 15:
                status = "⚠️  MODERATE"
            elif teeth > 0:
                status = "⚠️  POOR"
            else:
                status = "❌ FAILED"
            
            print(f"   Status: {status}")
            
            results[model_name] = {
                'status': 'success',
                'teeth': teeth,
                'anomalies': anomalies,
                'score_range': score_range if teeth > 0 else None,
                'assessment': status
            }
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            results[model_name] = {'status': 'error', 'error': str(e)}
    
    return results


def run_batch_diagnostic(image_paths, max_images=5):
    """Run diagnostic on multiple images."""
    print(f"\n{'='*70}")
    print(f"BATCH DIAGNOSTIC - Testing {min(len(image_paths), max_images)} images")
    print(f"{'='*70}")
    
    all_results = []
    
    for i, img_path in enumerate(image_paths[:max_images]):
        result = {
            'image': str(img_path),
            'models': test_image_on_all_models(img_path)
        }
        all_results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 SUMMARY ACROSS ALL TESTED IMAGES")
    print(f"{'='*70}")
    
    # Aggregate by model
    for model_name in MODELS_TO_TEST.keys():
        teeth_counts = []
        for result in all_results:
            if model_name in result['models']:
                model_result = result['models'][model_name]
                if model_result.get('status') == 'success':
                    teeth_counts.append(model_result['teeth'])
        
        if teeth_counts:
            avg_teeth = np.mean(teeth_counts)
            min_teeth = np.min(teeth_counts)
            max_teeth = np.max(teeth_counts)
            
            print(f"\n{model_name}:")
            print(f"   Average teeth: {avg_teeth:.1f}")
            print(f"   Range: {min_teeth} - {max_teeth}")
            print(f"   Images tested: {len(teeth_counts)}")
            
            if avg_teeth >= 20:
                print(f"   ✅ RECOMMENDED FOR DEPLOYMENT")
            elif avg_teeth >= 15:
                print(f"   ⚠️  Moderate performance")
            else:
                print(f"   ❌ Not recommended")
    
    # Save results
    output_file = f"diagnostic_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        # Convert to serializable format
        serializable_results = []
        for r in all_results:
            serializable_results.append({
                'image': r['image'],
                'models': {k: v for k, v in r['models'].items()}
            })
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📄 Results saved to: {output_file}")
    
    return all_results


def main():
    print("="*70)
    print("🚨 EMERGENCY DIAGNOSTIC SYSTEM")
    print("="*70)
    print(f"Using existing Streamlit app infrastructure")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*70)
    
    # Check for test images
    test_data_dir = Path("data/test data")
    validation_dir = Path("data/final-di-remapped/valid")
    
    test_images = list(test_data_dir.glob("*.jpg")) if test_data_dir.exists() else []
    val_images = list(validation_dir.glob("*.jpg"))[:3] if validation_dir.exists() else []
    
    print(f"\nFound {len(test_images)} images in test data")
    print(f"Found {len(val_images)} images in validation set")
    
    # Prioritize test data (unannotated real-world data)
    if test_images:
        print(f"\n🎯 Testing on REAL PRODUCTION DATA (test data)")
        images_to_test = test_images[:5]
    elif val_images:
        print(f"\n🎯 Testing on validation data")
        images_to_test = val_images
    else:
        print(f"\n❌ No test images found!")
        return
    
    # Run diagnostics
    results = run_batch_diagnostic(images_to_test)
    
    # Final recommendation
    print(f"\n{'='*70}")
    print("🏥 CLINICAL RECOMMENDATION")
    print(f"{'='*70}")
    
    # Find best model
    best_model = None
    best_avg_teeth = 0
    
    for model_name in MODELS_TO_TEST.keys():
        teeth_counts = []
        for result in results:
            if model_name in result['models']:
                model_result = result['models'][model_name]
                if model_result.get('status') == 'success':
                    teeth_counts.append(model_result['teeth'])
        
        if teeth_counts:
            avg_teeth = np.mean(teeth_counts)
            if avg_teeth > best_avg_teeth:
                best_avg_teeth = avg_teeth
                best_model = model_name
    
    if best_model and best_avg_teeth >= 20:
        print(f"\n✅ DEPLOY {best_model}")
        print(f"   Average performance: {best_avg_teeth:.1f} teeth detected")
        print(f"\n   ACTION REQUIRED:")
        print(f"   1. Edit app.py line 23:")
        print(f'      MODEL_DIR = Path("output/{best_model}")')
        print(f"   2. Restart Streamlit app")
        print(f"   3. Validate on production data")
        print(f"\n   ESTIMATED RESOLUTION TIME: 5 minutes")
    elif best_model:
        print(f"\n⚠️  BEST AVAILABLE: {best_model}")
        print(f"   Performance: {best_avg_teeth:.1f} teeth (suboptimal)")
        print(f"   Consider threshold tuning or fine-tuning")
    else:
        print(f"\n❌ CRITICAL: All models failed")
        print(f"   Fine-tuning required (3-5 days)")
        print(f"   SUSPEND clinical use immediately")


if __name__ == "__main__":
    main()
