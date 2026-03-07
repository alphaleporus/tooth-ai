#!/usr/bin/env python3
"""
Diagnostic: Analyze tooth detection patterns across multiple images
Focus on front teeth (incisors/canines) vs back teeth (molars/premolars)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
import cv2
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

print("="*70)
print("TOOTH DETECTION DIAGNOSTIC")
print("="*70)

# Load model
cfg = get_cfg()
cfg.merge_from_file('output/resnet50_9class_20k/config.yaml')
cfg.MODEL.WEIGHTS = 'output/resnet50_9class_20k/model_final.pth'
cfg.MODEL.DEVICE = 'cuda'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

predictor = DefaultPredictor(cfg)

# Test on 20 random images
test_dir = Path('data/final-di-remapped/test')
images = list(test_dir.glob('*.jpg'))[:20]

print(f"\nTesting on {len(images)} images...")
print("-"*70)

# Track detection patterns
all_teeth_counts = []
all_positions = []  # (relative_x, relative_y) for each tooth

for img_path in images:
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    
    outputs = predictor(img)
    instances = outputs['instances'].to('cpu')
    
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    
    # Count teeth (class 0) with score > 0.15
    teeth_mask = (classes == 0) & (scores > 0.15)
    teeth_boxes = boxes[teeth_mask]
    teeth_scores = scores[teeth_mask]
    
    all_teeth_counts.append(len(teeth_boxes))
    
    # Track relative positions
    for box, score in zip(teeth_boxes, teeth_scores):
        cx = (box[0] + box[2]) / 2 / w  # Relative X (0-1)
        cy = (box[1] + box[3]) / 2 / h  # Relative Y (0-1)
        all_positions.append((cx, cy, score))

print(f"\nTEETH COUNT STATISTICS:")
print(f"  Min: {min(all_teeth_counts)}")
print(f"  Max: {max(all_teeth_counts)}")
print(f"  Mean: {np.mean(all_teeth_counts):.1f}")
print(f"  Median: {np.median(all_teeth_counts):.1f}")

# Analyze position distribution
positions = np.array(all_positions)
if len(positions) > 0:
    print(f"\nPOSITION ANALYSIS ({len(positions)} total detections):")
    
    # Divide image into 5 horizontal zones
    zones = {
        'Left edge (0-20%)': (0.0, 0.2),
        'Left (20-40%)': (0.2, 0.4),
        'CENTER (40-60%)': (0.4, 0.6),  # Front teeth should be here!
        'Right (60-80%)': (0.6, 0.8),
        'Right edge (80-100%)': (0.8, 1.0)
    }
    
    print("\n  Horizontal distribution:")
    for zone_name, (x_min, x_max) in zones.items():
        count = sum(1 for x, y, s in positions if x_min <= x < x_max)
        pct = count / len(positions) * 100
        bar = '#' * int(pct / 2)
        print(f"    {zone_name:25s}: {count:4d} ({pct:5.1f}%) {bar}")
    
    # Score distribution by zone
    print("\n  Average confidence by zone:")
    for zone_name, (x_min, x_max) in zones.items():
        zone_scores = [s for x, y, s in positions if x_min <= x < x_max]
        if zone_scores:
            avg_score = np.mean(zone_scores)
            print(f"    {zone_name:25s}: {avg_score:.2f}")
        else:
            print(f"    {zone_name:25s}: NO DETECTIONS")

# DIAGNOSIS
print("\n" + "="*70)
print("DIAGNOSIS")
print("="*70)

center_count = sum(1 for x, y, s in positions if 0.4 <= x < 0.6)
edge_count = sum(1 for x, y, s in positions if x < 0.2 or x >= 0.8)
total = len(positions)

if center_count < total * 0.15:
    print("\n[PROBLEM] CENTER ZONE UNDER-DETECTION")
    print("  Front teeth (incisors) are in the center of panoramic X-rays")
    print("  but only {:.1f}% of detections are in this zone.".format(center_count/total*100))
    print("\n  POSSIBLE CAUSES:")
    print("  1. Training data may have fewer annotations in center region")
    print("  2. Front teeth have different visual characteristics (overlapping, smaller)")
    print("  3. Model may be biased toward larger back teeth")
else:
    print("\n[OK] Center zone detection appears normal ({:.1f}%)".format(center_count/total*100))

print("\n" + "="*70)
print("DONE")
