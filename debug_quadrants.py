#!/usr/bin/env python3
"""
Deep Debug: Why are lower jaw teeth not being detected?
Analysis of quadrant assignment and detection distribution.
"""

import os
import sys
import cv2
import numpy as np
from collections import Counter

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    
    print("="*70)
    print("DEEP DEBUG: Lower Jaw Teeth Detection")
    print("="*70)
    
    # Load model
    cfg = get_cfg()
    cfg.merge_from_file('output/resnext101_cascade_60k/config.yaml')
    cfg.MODEL.WEIGHTS = 'output/resnext101_cascade_60k/model_final.pth'
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    
    predictor = DefaultPredictor(cfg)
    
    # Load test image (use first test image)
    test_dir = 'data/final-di-stratified/test'
    images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    img_path = os.path.join(test_dir, images[0])
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    
    print(f"\nImage: {images[0]}")
    print(f"Dimensions: {w}x{h}")
    
    # Calculate quadrant boundaries (same as app.py)
    midline_x = w / 2
    jaw_split_y = h * 0.45  # This might be the problem!
    
    print(f"\nQuadrant Split:")
    print(f"  Midline X: {midline_x:.0f}")
    print(f"  Jaw Split Y (45%): {jaw_split_y:.0f}")
    print(f"  Upper jaw: Y < {jaw_split_y:.0f}")
    print(f"  Lower jaw: Y >= {jaw_split_y:.0f}")
    
    # Run inference
    outputs = predictor(img)
    instances = outputs['instances'].to('cpu')
    
    boxes = instances.pred_boxes.tensor.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    
    # Separate teeth vs anomalies
    teeth_mask = classes <= 32
    anom_mask = classes > 32
    
    print(f"\n" + "-"*50)
    print("RAW DETECTION COUNTS:")
    print("-"*50)
    print(f"Total detections: {len(boxes)}")
    print(f"Teeth (class 0-32): {sum(teeth_mask)}")
    print(f"Anomalies (class 33-40): {sum(anom_mask)}")
    
    # Analyze teeth by vertical position
    teeth_boxes = boxes[teeth_mask]
    teeth_scores = scores[teeth_mask]
    
    print(f"\n" + "-"*50)
    print("TEETH DISTRIBUTION BY Y-POSITION:")
    print("-"*50)
    
    upper_count = 0
    lower_count = 0
    
    for box, score in zip(teeth_boxes, teeth_scores):
        cy = (box[1] + box[3]) / 2
        if cy < jaw_split_y:
            upper_count += 1
        else:
            lower_count += 1
    
    print(f"Upper jaw (Y < {jaw_split_y:.0f}): {upper_count} teeth")
    print(f"Lower jaw (Y >= {jaw_split_y:.0f}): {lower_count} teeth")
    
    # Analyze Y coordinates of all teeth
    print(f"\n" + "-"*50)
    print("TEETH Y-COORDINATE ANALYSIS:")
    print("-"*50)
    
    y_centers = [(box[1] + box[3]) / 2 for box in teeth_boxes]
    if y_centers:
        print(f"Min Y center: {min(y_centers):.0f}")
        print(f"Max Y center: {max(y_centers):.0f}")
        print(f"Mean Y center: {np.mean(y_centers):.0f}")
        print(f"Jaw split at: {jaw_split_y:.0f}")
        
        # Count by Y range
        y_ranges = {
            f"0-{int(h*0.25)}": 0,
            f"{int(h*0.25)}-{int(h*0.45)}": 0,
            f"{int(h*0.45)}-{int(h*0.65)}": 0,
            f"{int(h*0.65)}-{h}": 0
        }
        
        for y in y_centers:
            if y < h * 0.25:
                y_ranges[f"0-{int(h*0.25)}"] += 1
            elif y < h * 0.45:
                y_ranges[f"{int(h*0.25)}-{int(h*0.45)}"] += 1
            elif y < h * 0.65:
                y_ranges[f"{int(h*0.45)}-{int(h*0.65)}"] += 1
            else:
                y_ranges[f"{int(h*0.65)}-{h}"] += 1
        
        print("\nY-range distribution:")
        for range_key, count in y_ranges.items():
            print(f"  {range_key}: {count} teeth")
    
    # Visualize jaw split line
    viz = img.copy()
    cv2.line(viz, (0, int(jaw_split_y)), (w, int(jaw_split_y)), (0, 0, 255), 3)
    cv2.line(viz, (int(midline_x), 0), (int(midline_x), h), (0, 0, 255), 2)
    
    # Draw all teeth detections with quadrant colors
    for box, score in zip(teeth_boxes, teeth_scores):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Quadrant color
        if cy < jaw_split_y:
            if cx < midline_x:
                color = (255, 0, 0)  # Q1 Blue
            else:
                color = (0, 255, 0)  # Q2 Green
        else:
            if cx >= midline_x:
                color = (0, 255, 255)  # Q3 Yellow
            else:
                color = (255, 0, 255)  # Q4 Magenta
        
        cv2.rectangle(viz, (x1, y1), (x2, y2), color, 2)
    
    # Add legend
    cv2.putText(viz, "RED LINE = Jaw Split (45%)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(viz, "Q1:Blue Q2:Green Q3:Yellow Q4:Magenta", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imwrite('debug_quadrants.jpg', viz)
    print(f"\nSaved visualization: debug_quadrants.jpg")
    
    # DIAGNOSIS
    print("\n" + "="*70)
    print("DIAGNOSIS:")
    print("="*70)
    
    if lower_count < upper_count / 2:
        print("[PROBLEM] Lower jaw has significantly fewer detections than upper jaw")
        print("[CAUSE] Possible issues:")
        print("  1. Model bias: Trained more on upper jaw images")
        print("  2. Jaw split position: 45% may be too high/low for this image")
        print("  3. Lower jaw teeth are harder to detect (root exposure, contrast)")
    
    if len(teeth_boxes) < 20:
        print("\n[PROBLEM] Total tooth count is low (should be ~24-28 for healthy adult)")
        print("[CAUSE] Possible issues:")
        print("  1. Threshold too high for less confident detections")
        print("  2. Model not trained well on certain tooth positions")


if __name__ == "__main__":
    main()
