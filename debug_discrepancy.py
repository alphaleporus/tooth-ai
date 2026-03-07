#!/usr/bin/env python3
"""
Debug the inference discrepancy:
- debug_visuals.py showed 126 teeth
- forensic_analysis.py showed 0 teeth
What's different?
"""

import os
import sys
import cv2
import numpy as np
import random
from collections import Counter

# Force UTF-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    
    print("="*70)
    print("INFERENCE DISCREPANCY DEBUG")
    print("="*70)
    
    # Load SOTA model
    cfg = get_cfg()
    cfg.merge_from_file('output/resnext101_cascade_60k/config.yaml')
    cfg.MODEL.WEIGHTS = 'output/resnext101_cascade_60k/model_final.pth'
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    
    print(f"\nConfig NUM_CLASSES: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
    
    predictor = DefaultPredictor(cfg)
    
    # Use the EXACT same image that debug_visuals.py used (seed 42)
    random.seed(42)
    
    test_dir = 'data/final-di-stratified/test'
    images = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
    img_name = random.choice(images)
    
    print(f"Test image (seed 42): {img_name}")
    
    # Load with cv2
    img_path = os.path.join(test_dir, img_name)
    img_cv2 = cv2.imread(img_path)
    print(f"Image shape: {img_cv2.shape}")
    
    # Run inference
    outputs = predictor(img_cv2)
    classes = outputs['instances'].pred_classes.cpu().numpy()
    scores = outputs['instances'].scores.cpu().numpy()
    
    print("\n" + "-"*50)
    print("RESULTS ON TEST IMAGE (seed 42):")
    print("-"*50)
    print(f"Total detections: {len(classes)}")
    
    if len(classes) > 0:
        print(f"Class ID range: {classes.min()} to {classes.max()}")
        teeth_count = sum(classes <= 32)
        anom_count = sum(classes > 32)
        print(f"Teeth (ID 0-32): {teeth_count}")
        print(f"Anomalies (ID 33-40): {anom_count}")
        
        print("\nClass distribution:")
        counts = Counter(classes)
        for cls_id, count in counts.most_common(15):
            category = "TOOTH" if cls_id <= 32 else "ANOMALY"
            print(f"  Class {cls_id:2d} ({category:7s}): {count}")
    else:
        print("NO DETECTIONS!")
    
    # Test on multiple images
    print("\n" + "="*70)
    print("TESTING ON 5 RANDOM TEST IMAGES")
    print("="*70)
    
    random.seed(12345)
    test_samples = random.sample(images, min(5, len(images)))
    
    for img_name in test_samples:
        img = cv2.imread(os.path.join(test_dir, img_name))
        out = predictor(img)
        cls = out['instances'].pred_classes.cpu().numpy()
        teeth = sum(cls <= 32)
        anom = sum(cls > 32)
        print(f"{img_name[:40]:40s} | Teeth: {teeth:3d} | Anomalies: {anom:3d}")
    
    # Test on validation images
    print("\n" + "="*70)
    print("TESTING ON 5 VALIDATION IMAGES")
    print("="*70)
    
    val_dir = 'data/final-di-stratified/valid'
    val_images = [f for f in os.listdir(val_dir) if f.endswith('.jpg')]
    val_samples = random.sample(val_images, min(5, len(val_images)))
    
    for img_name in val_samples:
        img = cv2.imread(os.path.join(val_dir, img_name))
        out = predictor(img)
        cls = out['instances'].pred_classes.cpu().numpy()
        teeth = sum(cls <= 32)
        anom = sum(cls > 32)
        print(f"{img_name[:40]:40s} | Teeth: {teeth:3d} | Anomalies: {anom:3d}")


if __name__ == "__main__":
    main()
