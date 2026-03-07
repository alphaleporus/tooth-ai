#!/usr/bin/env python3
"""
Debug Visualization Script
Draw raw model outputs with integer class IDs to diagnose mapping issues.
"""

import os
import sys
import random
import cv2
import numpy as np

# Force UTF-8 for Windows console
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    # Import Detectron2
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    
    print("="*60)
    print("DEBUG VISUAL: RAW CLASS ID MAPPING")
    print("="*60)
    
    # Load SOTA model
    model_dir = "output/resnext101_cascade_60k"
    config_path = os.path.join(model_dir, "config.yaml")
    weights_path = os.path.join(model_dir, "model_final.pth")
    
    print(f"Loading model from: {model_dir}")
    
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # Very low threshold
    
    predictor = DefaultPredictor(cfg)
    print("Model loaded!")
    
    # Load a random test image
    test_dir = "data/final-di-stratified/test"
    images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
    
    # Pick a random image
    random.seed(42)
    img_name = random.choice(images)
    img_path = os.path.join(test_dir, img_name)
    
    print(f"Test image: {img_name}")
    
    # Load image
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Run inference
    print("\nRunning inference...")
    outputs = predictor(img)
    
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    
    print(f"\nTotal detections: {len(boxes)}")
    print(f"Class ID range: {classes.min()} to {classes.max()}")
    
    # Count by class ID
    from collections import Counter
    class_counts = Counter(classes)
    print("\nClass ID distribution:")
    for cls_id in sorted(class_counts.keys()):
        print(f"  Class {cls_id:2d}: {class_counts[cls_id]} detections")
    
    # Create visualization
    viz = img.copy()
    
    # Color palette for different class ID ranges
    def get_color(cls_id):
        if cls_id <= 32:  # Should be teeth (IDs 0-32)
            return (0, 255, 0)  # Green
        else:  # Anomalies (IDs 33-40)
            return (0, 0, 255)  # Red
    
    # Draw ALL boxes with RAW class IDs
    print("\n" + "-"*60)
    print("Drawing boxes with RAW class IDs...")
    
    for i, (box, cls_id, score) in enumerate(zip(boxes, classes, scores)):
        x1, y1, x2, y2 = map(int, box)
        color = get_color(cls_id)
        
        # Draw box
        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
        
        # Create label with RAW class ID
        label = f"ID:{cls_id} | {score:.0%}"
        
        # Background for label
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(viz, (x1, y1 - 18), (x1 + tw + 4, y1), color, -1)
        cv2.putText(viz, label, (x1 + 2, y1 - 4), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add legend
    legend_y = 30
    cv2.putText(viz, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(viz, "GREEN = Class ID 0-32 (Expected: Teeth)", (10, legend_y + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(viz, "RED = Class ID 33-40 (Expected: Anomalies)", (10, legend_y + 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(viz, f"Total Detections: {len(boxes)}", (10, legend_y + 75), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save output
    output_path = "debug_mapping.jpg"
    cv2.imwrite(output_path, viz)
    print(f"\nSaved visualization to: {output_path}")
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS:")
    print("="*60)
    teeth_count = sum(1 for c in classes if c <= 32)
    anomaly_count = sum(1 for c in classes if c > 32)
    print(f"Detections with Class ID 0-32 (teeth range): {teeth_count}")
    print(f"Detections with Class ID 33-40 (anomaly range): {anomaly_count}")
    
    if teeth_count == 0 and anomaly_count > 0:
        print("\n[DIAGNOSIS] Model is NOT outputting teeth class IDs.")
        print("The 'anomaly' detections might be teeth mislabeled during training.")
        print("Check if boxes labeled 'anomaly' are actually around teeth shapes.")
    elif teeth_count > 0:
        print("\n[DIAGNOSIS] Model IS outputting some teeth class IDs!")
        print("The app may have a filtering or mapping bug.")
    
    print("\nOpen 'debug_mapping.jpg' to visually inspect the detections.")


if __name__ == "__main__":
    main()
