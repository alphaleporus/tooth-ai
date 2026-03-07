#!/usr/bin/env python3
"""Debug script to find where app.py is stuck."""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np
import time
import glob

print("=" * 60)
print("DEBUGGING APP PIPELINE")
print("=" * 60)

# Load model
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances, Boxes
import torch

cfg = get_cfg()
cfg.merge_from_file('output/resnet50_9class_20k/config.yaml')
cfg.MODEL.WEIGHTS = 'output/resnet50_9class_20k/model_final.pth'
cfg.MODEL.DEVICE = 'cuda'
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05

predictor = DefaultPredictor(cfg)
print("Model loaded")

# Test image
img_path = list(glob.glob('data/final-di-remapped/test/*.jpg'))[5]
img = cv2.imread(img_path)
print(f"Image: {img.shape}")

# === TEST EACH FUNCTION ===

# 1. Apply CLAHE
print("\n1. Testing apply_clahe...")
try:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enh = clahe.apply(l)
    lab_enh = cv2.merge([l_enh, a, b])
    img_clahe = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)
    print("   PASS: apply_clahe works")
except Exception as e:
    print(f"   FAIL: {e}")

# 2. Apply Gamma
print("\n2. Testing apply_gamma...")
try:
    inv_gamma = 1.0 / 2.5
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    img_gamma = cv2.LUT(img, table)
    print("   PASS: apply_gamma works")
except Exception as e:
    print(f"   FAIL: {e}")

# 3. Run 4 passes
print("\n3. Testing 4-pass inference...")
try:
    out1 = predictor(img)
    out2 = predictor(img_clahe)
    out3 = predictor(img_gamma)
    out4 = predictor(cv2.flip(img, 1))
    print(f"   PASS: 4 passes, got {len(out1['instances'])}, {len(out2['instances'])}, {len(out3['instances'])}, {len(out4['instances'])} detections")
except Exception as e:
    print(f"   FAIL: {e}")

# 4. Flip instances
print("\n4. Testing flip_instances_horizontal...")
try:
    instances = out4["instances"]
    if len(instances) > 0:
        boxes = instances.pred_boxes.tensor.clone()
        old_x1 = boxes[:, 0].clone()
        old_x2 = boxes[:, 2].clone()
        boxes[:, 0] = img.shape[1] - old_x2
        boxes[:, 2] = img.shape[1] - old_x1
        instances.pred_boxes.tensor = boxes
        if instances.has("pred_masks"):
            masks = instances.pred_masks
            instances.pred_masks = masks.flip(dims=[2])
    print("   PASS: flip_instances works")
except Exception as e:
    print(f"   FAIL: {e}")
    import traceback
    traceback.print_exc()

# 5. Weighted Box Fusion
print("\n5. Testing weighted_box_fusion...")
try:
    outputs_list = [out1, out2, out3, out4]
    
    all_boxes = []
    all_scores = []
    all_classes = []
    
    for outputs in outputs_list:
        instances = outputs["instances"].to("cpu")
        if len(instances) == 0:
            continue
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        
        for box, score, cls in zip(boxes, scores, classes):
            if score >= 0.05:
                all_boxes.append(box)
                all_scores.append(score)
                all_classes.append(cls)
    
    print(f"   Collected {len(all_boxes)} boxes")
    
    if all_boxes:
        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        all_classes = np.array(all_classes)
        
        # Simple NMS-like fusion
        fused_boxes = []
        fused_scores = []
        fused_classes = []
        
        used = [False] * len(all_boxes)
        sorted_idx = np.argsort(-all_scores)
        
        for i in sorted_idx:
            if used[i]:
                continue
            fused_boxes.append(all_boxes[i])
            fused_scores.append(all_scores[i])
            fused_classes.append(all_classes[i])
            used[i] = True
            
            for j in sorted_idx:
                if used[j]:
                    continue
                # Simple IoU check
                b1, b2 = all_boxes[i], all_boxes[j]
                x1 = max(b1[0], b2[0])
                y1 = max(b1[1], b2[1])
                x2 = min(b1[2], b2[2])
                y2 = min(b1[3], b2[3])
                if x1 < x2 and y1 < y2:
                    inter = (x2-x1) * (y2-y1)
                    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
                    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
                    iou = inter / (a1 + a2 - inter)
                    if iou > 0.4:
                        used[j] = True
        
        print(f"   Fused to {len(fused_boxes)} boxes")
        print("   PASS: WBF works")
    else:
        print("   PASS: No boxes to fuse")
except Exception as e:
    print(f"   FAIL: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)
