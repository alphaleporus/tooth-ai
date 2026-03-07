#!/usr/bin/env python3
"""
FORENSIC ANALYSIS: Find the "Smoking Gun" in Model Inference
Tests: Color Space, Preprocessing, Config Mismatch
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image

# Force UTF-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

def main():
    print("="*70)
    print("FORENSIC ANALYSIS: Model Inference Bug Hunt")
    print("="*70)
    
    # Import Detectron2
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.data import transforms as T
    
    model_dir = "output/resnext101_cascade_60k"
    config_path = os.path.join(model_dir, "config.yaml")
    weights_path = os.path.join(model_dir, "model_final.pth")
    
    # Load config
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    
    print("\n[1] CONFIG ANALYSIS")
    print("-"*50)
    print(f"INPUT.FORMAT: {cfg.INPUT.FORMAT}")  # Should be "BGR"
    print(f"MIN_SIZE_TRAIN: {cfg.INPUT.MIN_SIZE_TRAIN}")
    print(f"MAX_SIZE_TRAIN: {cfg.INPUT.MAX_SIZE_TRAIN}")
    print(f"MIN_SIZE_TEST: {cfg.INPUT.MIN_SIZE_TEST}")
    print(f"MAX_SIZE_TEST: {cfg.INPUT.MAX_SIZE_TEST}")
    print(f"PIXEL_MEAN: {cfg.MODEL.PIXEL_MEAN}")
    print(f"PIXEL_STD: {cfg.MODEL.PIXEL_STD}")
    print(f"NUM_CLASSES: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
    
    # Create predictor
    predictor = DefaultPredictor(cfg)
    
    # Get test image
    test_dir = "data/final-di-stratified/test"
    img_files = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
    img_path = os.path.join(test_dir, img_files[0])
    
    print(f"\n[2] TEST IMAGE ANALYSIS")
    print("-"*50)
    print(f"Image: {img_files[0]}")
    
    # Method 1: cv2.imread (BGR - what Detectron2 expects)
    img_cv2 = cv2.imread(img_path)
    print(f"cv2.imread shape: {img_cv2.shape}, dtype: {img_cv2.dtype}")
    print(f"cv2 pixel sample [0,0]: {img_cv2[0,0]} (BGR order)")
    
    # Method 2: PIL (RGB - what Streamlit uses)
    img_pil = Image.open(img_path)
    img_pil_np = np.array(img_pil)
    print(f"PIL shape: {img_pil_np.shape}, dtype: {img_pil_np.dtype}")
    print(f"PIL pixel sample [0,0]: {img_pil_np[0,0]} (RGB order)")
    
    # Check if they're different
    if len(img_pil_np.shape) == 2:
        print("Image is GRAYSCALE")
        is_different = False
    else:
        is_different = not np.allclose(img_cv2[0,0], img_pil_np[0,0])
        print(f"BGR vs RGB difference: {is_different}")
    
    print(f"\n[3] INFERENCE COMPARISON")
    print("-"*50)
    
    # Test 1: Feed BGR (correct for Detectron2)
    print("\n>>> Test A: cv2.imread (BGR) - CORRECT for Detectron2")
    outputs_bgr = predictor(img_cv2)
    classes_bgr = outputs_bgr['instances'].pred_classes.cpu().numpy()
    scores_bgr = outputs_bgr['instances'].scores.cpu().numpy()
    teeth_bgr = sum(classes_bgr <= 32)
    anom_bgr = sum(classes_bgr > 32)
    print(f"Total: {len(classes_bgr)}, Teeth(0-32): {teeth_bgr}, Anomalies(33-40): {anom_bgr}")
    if len(scores_bgr) > 0:
        print(f"Score range: {scores_bgr.min():.3f} - {scores_bgr.max():.3f}")
    
    # Test 2: Feed RGB (WRONG - what might be happening in app)
    print("\n>>> Test B: PIL RGB (NOT converted) - WRONG for Detectron2")
    if len(img_pil_np.shape) == 2:
        img_rgb_test = cv2.cvtColor(img_pil_np, cv2.COLOR_GRAY2BGR)
    elif img_pil_np.shape[2] == 4:
        img_rgb_test = cv2.cvtColor(img_pil_np, cv2.COLOR_RGBA2RGB)
    else:
        img_rgb_test = img_pil_np  # Keep as RGB (WRONG)
    outputs_rgb = predictor(img_rgb_test)
    classes_rgb = outputs_rgb['instances'].pred_classes.cpu().numpy()
    scores_rgb = outputs_rgb['instances'].scores.cpu().numpy()
    teeth_rgb = sum(classes_rgb <= 32)
    anom_rgb = sum(classes_rgb > 32)
    print(f"Total: {len(classes_rgb)}, Teeth(0-32): {teeth_rgb}, Anomalies(33-40): {anom_rgb}")
    if len(scores_rgb) > 0:
        print(f"Score range: {scores_rgb.min():.3f} - {scores_rgb.max():.3f}")
    
    # Test 3: PIL RGB -> BGR conversion (what app.py SHOULD do)
    print("\n>>> Test C: PIL RGB -> BGR conversion - SHOULD MATCH Test A")
    img_pil_bgr = cv2.cvtColor(img_pil_np, cv2.COLOR_RGB2BGR)
    outputs_pil_bgr = predictor(img_pil_bgr)
    classes_pil_bgr = outputs_pil_bgr['instances'].pred_classes.cpu().numpy()
    scores_pil_bgr = outputs_pil_bgr['instances'].scores.cpu().numpy()
    teeth_pil_bgr = sum(classes_pil_bgr <= 32)
    anom_pil_bgr = sum(classes_pil_bgr > 32)
    print(f"Total: {len(classes_pil_bgr)}, Teeth(0-32): {teeth_pil_bgr}, Anomalies(33-40): {anom_pil_bgr}")
    if len(scores_pil_bgr) > 0:
        print(f"Score range: {scores_pil_bgr.min():.3f} - {scores_pil_bgr.max():.3f}")
    
    print(f"\n[4] VERDICT")
    print("-"*50)
    
    # Analyze results
    if teeth_bgr > 0 and teeth_rgb == 0:
        print("[SMOKING GUN] RGB/BGR MISMATCH CONFIRMED!")
        print(">>> The app is feeding RGB when Detectron2 expects BGR.")
        print(">>> FIX: Ensure proper cv2.cvtColor(image, cv2.COLOR_RGB2BGR)")
    elif teeth_bgr == 0 and teeth_rgb == 0:
        print("[NEGATIVE] Color space is NOT the issue.")
        print(">>> Both BGR and RGB produce 0 teeth.")
        print(">>> The model truly has low teeth confidence.")
    elif teeth_bgr > 0 and teeth_pil_bgr == teeth_bgr:
        print("[CONFIRMED] PIL->BGR conversion works correctly.")
        print(">>> app.py color handling is correct.")
    
    print("\n" + "="*70)
    print("FORENSIC ANALYSIS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
