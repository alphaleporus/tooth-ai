#!/usr/bin/env python3
"""
Tooth-AI: Hybrid Inference Engine
9-Class ResNet-50 Model + Geometric FDI Assignment
Professional Dental Diagnostic Tool
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

import streamlit as st
import numpy as np
from PIL import Image
import cv2

# ============================================
# CONFIGURATION
# ============================================

# Model paths - 41-Class ResNeXt-101 Cascade (UPDATED for better performance)
MODEL_DIR = Path("output/resnext101_cascade_60k")
CONFIG_PATH = MODEL_DIR / "config.yaml"
WEIGHTS_PATH = MODEL_DIR / "model_final.pth"
MODEL_VERSION = "ResNeXt-101 Cascade 41-Class v2.0 (60k iter)"

# 9-Class Schema
CLASSES = [
    "Tooth",                      # ID 0
    "Caries",                     # ID 1
    "Crown",                      # ID 2
    "Filling",                    # ID 3
    "Implant",                    # ID 4
    "Prefabricated metal post",   # ID 5
    "Retained root",              # ID 6
    "Root canal filling",         # ID 7
    "Root canal obturation"       # ID 8
]

ANOMALY_CLASSES = set(CLASSES[1:])  # All except "Tooth"

# Color scheme
TOOTH_COLOR = (0, 255, 0)  # Green (BGR)
ANOMALY_COLOR = (0, 0, 255)  # Red (BGR)
ANOMALY_COLORS = {
    "Caries": (0, 0, 255),            # Red
    "Crown": (0, 215, 255),           # Gold
    "Filling": (255, 191, 0),         # Deep sky blue
    "Implant": (211, 0, 148),         # Violet
    "Prefabricated metal post": (0, 140, 255),  # Orange
    "Retained root": (60, 20, 220),   # Crimson
    "Root canal filling": (255, 255, 0),  # Cyan
    "Root canal obturation": (50, 205, 50)  # Lime
}

# FDI Chart (Standard Dental View)
FDI_UPPER = [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]
FDI_LOWER = [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]


# ============================================
# DENTAL GEOMETRIC ENGINE
# ============================================

class DentalGeometricEngine:
    """Brain of the system: Converts raw detections to dental chart."""
    
    @staticmethod
    def get_box_center(box):
        """Get center point of a bounding box."""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @staticmethod
    def get_box_width(box):
        """Get width of bounding box."""
        return box[2] - box[0]
    
    @staticmethod
    def split_jaws(teeth_boxes):
        """
        Split teeth into upper and lower jaw using simple Y-median splitting.
        Fast and reliable for panoramic X-rays where jaws are clearly separated.
        
        Args:
            teeth_boxes: List of dicts with 'box' key
            
        Returns:
            (upper_jaw, lower_jaw) - lists of tooth detections
        """
        if len(teeth_boxes) < 2:
            # Not enough teeth to split
            return teeth_boxes, []
        
        # Get Y centers
        y_centers = [DentalGeometricEngine.get_box_center(t['box'])[1] for t in teeth_boxes]
        
        # Simple median split (much faster than KMeans)
        median_y = np.median(y_centers)
        
        upper_jaw = [t for t, y in zip(teeth_boxes, y_centers) if y < median_y]
        lower_jaw = [t for t, y in zip(teeth_boxes, y_centers) if y >= median_y]
        
        return upper_jaw, lower_jaw
    
    @staticmethod
    def sort_and_assign_fdi(jaw_teeth, jaw_type='upper'):
        """
        Sort teeth left-to-right and assign FDI numbers with gap detection.
        
        Args:
            jaw_teeth: List of tooth detections for one jaw
            jaw_type: 'upper' or 'lower'
            
        Returns:
            List of dicts with 'box', 'fdi', 'score', 'quadrant'
        """
        if not jaw_teeth:
            return []
        
        # Sort by X center (left to right)
        sorted_teeth = sorted(jaw_teeth, 
                             key=lambda t: DentalGeometricEngine.get_box_center(t['box'])[0])
        
        # FDI sequence for this jaw
        fdi_sequence = FDI_UPPER if jaw_type == 'upper' else FDI_LOWER
        
        # Assign FDI with gap detection
        result = []
        fdi_index = 0
        
        for i, tooth in enumerate(sorted_teeth):
            # Gap detection: check distance to previous tooth
            if i > 0 and fdi_index < len(fdi_sequence):
                prev_tooth = sorted_teeth[i - 1]
                curr_center_x = DentalGeometricEngine.get_box_center(tooth['box'])[0]
                prev_center_x = DentalGeometricEngine.get_box_center(prev_tooth['box'])[0]
                
                gap = curr_center_x - prev_center_x
                local_width = DentalGeometricEngine.get_box_width(tooth['box'])
                
                # If gap > 1.6x local width, assume missing tooth(s)
                if gap > local_width * 1.6:
                    missing_count = int(gap / local_width) - 1
                    missing_count = min(missing_count, 3)  # Cap at 3 missing
                    fdi_index += missing_count
            
            # Assign FDI number
            if fdi_index < len(fdi_sequence):
                fdi_number = fdi_sequence[fdi_index]
            else:
                fdi_number = fdi_sequence[-1]  # Fallback to last
            
            # Determine quadrant from FDI
            quadrant = fdi_number // 10
            
            result.append({
                'box': tooth['box'],
                'fdi': fdi_number,
                'score': tooth['score'],
                'quadrant': quadrant,
                'mask': tooth.get('mask'),
                'findings': []  # Will be populated by anomaly mapping
            })
            
            fdi_index += 1
        
        return result
    
    @staticmethod
    def compute_iou(box1, box2):
        """Compute Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x1 >= x2 or y1 >= y2:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    @staticmethod
    def map_anomalies(teeth, anomalies, iou_threshold=0.15):
        """
        Map anomalies to their associated teeth based on IoU overlap.
        
        Args:
            teeth: List of tooth dicts with 'box', 'fdi', etc.
            anomalies: List of anomaly dicts with 'box', 'class_name', 'score'
            iou_threshold: Minimum IoU to associate anomaly with tooth
            
        Returns:
            Updated teeth list with 'findings' populated
        """
        for anomaly in anomalies:
            anom_box = anomaly['box']
            best_tooth = None
            best_iou = iou_threshold
            
            for tooth in teeth:
                iou = DentalGeometricEngine.compute_iou(anom_box, tooth['box'])
                if iou > best_iou:
                    best_iou = iou
                    best_tooth = tooth
            
            if best_tooth:
                finding = {
                    'type': anomaly['class_name'],
                    'score': anomaly['score'],
                    'box': anom_box,
                    'mask': anomaly.get('mask')  # Include mask for precise visualization
                }
                best_tooth['findings'].append(finding)
                anomaly['associated_fdi'] = best_tooth['fdi']
            else:
                # Anomaly without associated tooth - add as standalone (unmarked)
                anomaly['associated_fdi'] = None
        
        return teeth
    
    @staticmethod
    def compute_diou(box1, box2):
        """
        Compute Distance-IoU between two boxes.
        
        DIoU = IoU - (ρ²(b, b_gt) / c²)
        
        Where:
          ρ² = squared Euclidean distance between box centers
          c² = squared diagonal of the smallest enclosing box
        
        This penalizes detections whose centers are close together,
        even if their standard IoU is low (critical for packed molars).
        """
        # Standard IoU components
        iou = DentalGeometricEngine.compute_iou(box1, box2)
        
        # Center distance (ρ²)
        cx1, cy1 = (box1[0] + box1[2]) / 2, (box1[1] + box1[3]) / 2
        cx2, cy2 = (box2[0] + box2[2]) / 2, (box2[1] + box2[3]) / 2
        rho_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
        
        # Enclosing box diagonal (c²)
        enc_x1 = min(box1[0], box2[0])
        enc_y1 = min(box1[1], box2[1])
        enc_x2 = max(box1[2], box2[2])
        enc_y2 = max(box1[3], box2[3])
        c_sq = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2
        
        if c_sq == 0:
            return iou
        
        diou = iou - (rho_sq / c_sq)
        return diou
    
    @staticmethod
    def apply_nms(detections, iou_threshold=0.4):
        """
        Apply Distance-IoU (DIoU) based Non-Maximum Suppression.
        
        Uses DIoU instead of standard IoU to better handle tightly packed
        molars where center proximity matters more than overlap area.
        
        DIoU = IoU - (center_distance² / enclosing_diagonal²)
        
        Args:
            detections: List of detection dicts with 'box' and 'score'
            iou_threshold: DIoU threshold for suppression (0.4 recommended)
        """
        if not detections:
            return []
        
        # Sort by score (highest first)
        detections = sorted(detections, key=lambda x: x['score'], reverse=True)
        
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            
            # Remove detections with high DIoU (overlapping OR close centers)
            remaining = []
            for det in detections:
                diou = DentalGeometricEngine.compute_diou(best['box'], det['box'])
                if diou < iou_threshold:
                    remaining.append(det)
            detections = remaining
        
        return keep


# ============================================
# MODEL LOADING
# ============================================

# Correct class mapping from COCO annotations (verified via inspect_classes.py)
# IDs 0-32: teeth (t, 1-32), IDs 33-40: anomalies
PATHOLOGY_CLASS_MAP = {
    **{i: "Tooth" for i in range(33)},  # IDs 0-32 map to "Tooth"
    33: "Caries",
    34: "Crown",
    35: "Filling",
    36: "Implant",
    37: "Prefabricated metal post",
    38: "Retained root",
    39: "Root canal filling",
    40: "Root canal obturation",
}

# Model Configurations
MODEL_OPTIONS = {
    "Complete Scan": {
        "config": Path("output/resnet50_9class_20k/config.yaml"),
        "weights": Path("output/resnet50_9class_20k/model_final.pth"),
        "type": "hybrid",
        "description": "Full dental checkup (Teeth + Anomalies)",
        # Hybrid mode: uses 9-class model for teeth + 41-class model for anomalies
        "pathology_config": Path("output/rtx4060_48k/config.yaml"),
        "pathology_weights": Path("output/rtx4060_48k/model_final.pth"),
    },
    "Pathology Focus": {
        "config": Path("output/rtx4060_48k/config.yaml"),
        "weights": Path("output/rtx4060_48k/model_final.pth"),
        "type": "anomalies_only",
        "description": "High-sensitivity anomaly detection only",
        "class_map": PATHOLOGY_CLASS_MAP,
    }
}

# Add colors for new inferred classes
ANOMALY_COLORS.update({
    "Caries (Severe)": (0, 0, 255),    # Red
    "Filling": (255, 140, 0),          # Orange
    "Root Canal": (200, 200, 0),       # Cyan-yellow
    "Crown": (0, 200, 255),            # Gold/Yellow
    # Dynamic anomaly colors will be handled by default fallback
})

@st.cache_resource
def load_model(config_path, weights_path):
    """Load Detectron2 model with specified config and weights."""
    import torch
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    
    # Determine device
    # Check for CUDA availability and test if it's actually working
    device = "cpu"
    device_name = "CPU"
    
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            # Quick smoke test — catches corrupted CUDA state
            test = torch.zeros(1, device="cuda")
            del test
            torch.cuda.empty_cache()
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
        except RuntimeError:
            print("⚠️ CUDA test failed — GPU state corrupted. Using CPU.")
            device = "cpu"
            device_name = "CPU (CUDA unavailable)"
    
    # Load config
    cfg = get_cfg()
    cfg.merge_from_file(str(config_path))
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05  # Low threshold, filter later
    
    predictor = DefaultPredictor(cfg)
    
    return predictor, device_name


def run_inference(predictor, image_bgr, threshold=0.05):
    """
    FOCUSED 2-PASS INFERENCE PIPELINE
    
    Uses two complementary passes for robust detection:
    1. Original image - catches clear teeth and anomalies
    2. CLAHE enhanced - recovers teeth hidden in spinal shadow zone
    
    Results are fused using Weighted Box Fusion (WBF) with tight merging.
    
    Note: Previous 4-pass approach (+ gamma + H-flip) caused severe
    over-detection (59 teeth instead of ~28-32) due to noise amplification.
    
    Args:
        predictor: Detectron2 predictor
        image_bgr: BGR image (numpy array)
        threshold: Base detection threshold
        
    Returns:
        outputs: Detectron2 Instances with fused detections
    """
    predictor.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    
    # === PASS 1: Original Image ===
    outputs_original = predictor(image_bgr)
    
    # === PASS 2: CLAHE Enhanced (recovers spinal shadow zone) ===
    clahe_image = apply_clahe(image_bgr, clip_limit=2.5, tile_size=(8, 8))
    outputs_clahe = predictor(clahe_image)
    
    # === FUSION via Weighted Box Fusion ===
    # Higher IoU threshold (0.55) ensures overlapping boxes from both passes
    # are properly merged rather than kept as separate detections
    fused_outputs = weighted_box_fusion(
        [outputs_original, outputs_clahe],
        iou_threshold=0.55,
        skip_box_threshold=threshold
    )
    
    return fused_outputs


def apply_clahe(image_bgr, clip_limit=2.5, tile_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance
    local contrast in X-ray images, particularly in the spinal shadow zone.
    
    Args:
        image_bgr: BGR image
        clip_limit: Threshold for contrast limiting (2.0-3.0 optimal for X-ray)
        tile_size: Size of grid for histogram equalization
        
    Returns:
        Enhanced BGR image
    """
    # Convert to LAB color space (L = lightness)
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    l_enhanced = clahe.apply(l_channel)
    
    # Merge and convert back
    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return result


def apply_gamma(image_bgr, gamma=2.0):
    """
    Apply gamma correction to brighten dark regions.
    Gamma > 1 brightens the image, useful for recovering underexposed areas.
    
    Args:
        image_bgr: BGR image
        gamma: Gamma value (>1 brightens, <1 darkens)
        
    Returns:
        Gamma-corrected BGR image
    """
    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 
        for i in np.arange(0, 256)
    ]).astype("uint8")
    
    return cv2.LUT(image_bgr, table)


def flip_instances_horizontal(outputs, image_width):
    """Flip detected boxes back to original orientation after H-flip inference."""
    instances = outputs["instances"]
    
    if len(instances) == 0:
        return outputs
    
    # Flip box coordinates: new_x1 = width - old_x2, new_x2 = width - old_x1
    boxes = instances.pred_boxes.tensor.clone()
    old_x1 = boxes[:, 0].clone()
    old_x2 = boxes[:, 2].clone()
    boxes[:, 0] = image_width - old_x2
    boxes[:, 2] = image_width - old_x1
    instances.pred_boxes.tensor = boxes
    
    # Flip masks if present
    if instances.has("pred_masks"):
        masks = instances.pred_masks
        instances.pred_masks = masks.flip(dims=[2])  # Flip along width axis
    
    outputs["instances"] = instances
    return outputs


def weighted_box_fusion(outputs_list, iou_threshold=0.55, skip_box_threshold=0.05):
    """
    Consensus-Weighted Box Fusion (WBF) for 2-pass ensemble.
    
    Tracks which pass each detection came from. Consensus logic:
      - Detection in BOTH passes (2/2): Full confidence → max(scores)
      - Detection in ONLY ONE pass (1/2): Penalized → score × 0.6
    
    Single-pass detections are likely noise from one augmentation and
    should not have the same weight as consensus detections.
    
    Args:
        outputs_list: List of Detectron2 outputs from each pass
        iou_threshold: IoU threshold for matching boxes across passes
        skip_box_threshold: Minimum score to consider a box
        
    Returns:
        Fused Detectron2 outputs with consensus-weighted scores
    """
    import torch
    from detectron2.structures import Instances, Boxes
    
    SINGLE_PASS_PENALTY = 0.6  # 40% penalty for single-pass detections
    num_passes = len(outputs_list)
    
    # Collect all detections WITH pass origin tracking
    all_boxes = []
    all_scores = []
    all_classes = []
    all_masks = []
    all_pass_ids = []  # Track which pass each detection came from
    
    for pass_id, outputs in enumerate(outputs_list):
        instances = outputs["instances"].to("cpu")
        if len(instances) == 0:
            continue
            
        boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        classes = instances.pred_classes.numpy()
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None
        
        for i, (box, score, cls) in enumerate(zip(boxes, scores, classes)):
            if score >= skip_box_threshold:
                all_boxes.append(box)
                all_scores.append(score)
                all_classes.append(cls)
                all_pass_ids.append(pass_id)
                if masks is not None:
                    all_masks.append(masks[i])
    
    if not all_boxes:
        # Return empty instances
        empty = Instances(outputs_list[0]["instances"].image_size)
        empty.pred_boxes = Boxes(torch.empty(0, 4))
        empty.scores = torch.empty(0)
        empty.pred_classes = torch.empty(0, dtype=torch.int64)
        return {"instances": empty}
    
    all_boxes = np.array(all_boxes)
    all_scores = np.array(all_scores)
    all_classes = np.array(all_classes)
    all_pass_ids = np.array(all_pass_ids)
    
    # Group by class and apply consensus-weighted WBF within each class
    fused_boxes = []
    fused_scores = []
    fused_classes = []
    fused_masks = []
    
    unique_classes = np.unique(all_classes)
    
    for cls in unique_classes:
        cls_mask = all_classes == cls
        cls_boxes = all_boxes[cls_mask]
        cls_scores = all_scores[cls_mask]
        cls_pass_ids = all_pass_ids[cls_mask]
        cls_indices = np.where(cls_mask)[0]
        
        # Sort by score
        sorted_idx = np.argsort(-cls_scores)
        cls_boxes = cls_boxes[sorted_idx]
        cls_scores = cls_scores[sorted_idx]
        cls_pass_ids = cls_pass_ids[sorted_idx]
        cls_indices = cls_indices[sorted_idx]
        
        used = [False] * len(cls_boxes)
        
        for i in range(len(cls_boxes)):
            if used[i]:
                continue
            
            # Find all overlapping boxes from any pass
            cluster_boxes = [cls_boxes[i]]
            cluster_scores = [cls_scores[i]]
            cluster_pass_ids = {cls_pass_ids[i]}
            cluster_mask_indices = [cls_indices[i]]
            used[i] = True
            
            for j in range(i + 1, len(cls_boxes)):
                if used[j]:
                    continue
                    
                iou = compute_iou(cls_boxes[i], cls_boxes[j])
                if iou > iou_threshold:
                    cluster_boxes.append(cls_boxes[j])
                    cluster_scores.append(cls_scores[j])
                    cluster_pass_ids.add(cls_pass_ids[j])
                    cluster_mask_indices.append(cls_indices[j])
                    used[j] = True
            
            # Weighted average of cluster boxes
            cluster_boxes = np.array(cluster_boxes)
            cluster_scores = np.array(cluster_scores)
            weights = cluster_scores / cluster_scores.sum()
            
            fused_box = np.average(cluster_boxes, axis=0, weights=weights)
            
            # CONSENSUS SCORING:
            # - Both passes agree → full max(score)
            # - Single pass only → penalize by 40% (likely noise)
            num_contributing_passes = len(cluster_pass_ids)
            base_score = float(np.max(cluster_scores))
            
            if num_contributing_passes >= min(2, num_passes):
                # Consensus: detection confirmed by both passes
                fused_score = base_score
            else:
                # Single-pass only: apply penalty
                fused_score = base_score * SINGLE_PASS_PENALTY
            
            fused_boxes.append(fused_box)
            fused_scores.append(fused_score)
            fused_classes.append(cls)
            
            # Use mask from highest scoring box
            if all_masks:
                best_idx = cluster_mask_indices[0]
                fused_masks.append(all_masks[best_idx])
    
    # Create fused Instances
    image_size = outputs_list[0]["instances"].image_size
    fused_instances = Instances(image_size)
    fused_instances.pred_boxes = Boxes(torch.tensor(np.array(fused_boxes), dtype=torch.float32))
    fused_instances.scores = torch.tensor(np.array(fused_scores), dtype=torch.float32)
    fused_instances.pred_classes = torch.tensor(np.array(fused_classes), dtype=torch.int64)
    
    if fused_masks:
        fused_instances.pred_masks = torch.tensor(np.array(fused_masks))
    
    return {"instances": fused_instances}


def compute_iou(box1, box2):
    """Compute IoU between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x1 >= x2 or y1 >= y2:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0


# ============================================
# ANATOMICAL PLAUSIBILITY ENFORCEMENT
# ============================================

def resolve_fdi_conflicts(teeth):
    """
    Subsystem 3: Global Anatomical Constraint (GAC)
    
    Enforces Winner-Takes-All: no FDI number can appear more than once.
    
    Algorithm:
      1. Group teeth by FDI number
      2. For each duplicate group, the highest-confidence detection keeps the FDI
      3. Losers are reassigned to the nearest unoccupied FDI slot in the
         same quadrant, or discarded if no valid slot exists
    
    Args:
        teeth: List of tooth dicts with 'fdi', 'score', 'box', 'quadrant'
    
    Returns:
        Deduplicated teeth list with unique FDI assignments
    """
    if not teeth:
        return teeth
    
    # All valid FDI numbers for adult dentition
    VALID_FDIS = set(FDI_UPPER + FDI_LOWER)
    
    # Step 1: Find occupied FDI numbers and detect conflicts
    fdi_groups = {}
    for tooth in teeth:
        fdi = tooth['fdi']
        if fdi not in fdi_groups:
            fdi_groups[fdi] = []
        fdi_groups[fdi].append(tooth)
    
    resolved = []
    occupied_fdis = set()
    
    # Step 2: Process each FDI group
    for fdi, group in fdi_groups.items():
        if len(group) == 1:
            # No conflict — keep as-is
            resolved.append(group[0])
            occupied_fdis.add(fdi)
        else:
            # CONFLICT: Multiple teeth claim the same FDI
            # Winner-Takes-All: highest confidence keeps the label
            group_sorted = sorted(group, key=lambda t: t['score'], reverse=True)
            winner = group_sorted[0]
            resolved.append(winner)
            occupied_fdis.add(fdi)
            
            # Losers: try to reassign to nearest unoccupied FDI in same quadrant
            for loser in group_sorted[1:]:
                quadrant = loser['quadrant']
                # Get valid FDIs in this quadrant, sorted by distance to claimed FDI
                quadrant_fdis = [f for f in VALID_FDIS 
                                if f // 10 == quadrant and f not in occupied_fdis]
                
                if quadrant_fdis:
                    # Assign to nearest unoccupied slot
                    nearest = min(quadrant_fdis, key=lambda f: abs(f - fdi))
                    loser['fdi'] = nearest
                    loser['quadrant'] = nearest // 10
                    resolved.append(loser)
                    occupied_fdis.add(nearest)
                # else: discard — no valid slot available (anatomically impossible)
    
    return resolved


def suppress_metal_artifacts(teeth, anomalies, proximity_px=5, penalty=0.5):
    """
    Subsystem 4: Metal Artifact Suppression (MAS)
    
    Radiographic flare from metal dental work (crowns, fillings, root canals)
    causes the model to hallucinate Caries near these structures.
    
    Rule: If a Caries detection overlaps with (IoU > 0.1) or is within
    `proximity_px` pixels of a radiopaque material, its confidence score
    is penalized by `penalty` (50%).
    
    Args:
        teeth: List of tooth dicts (with 'findings')
        anomalies: List of raw anomaly dicts
        proximity_px: Pixel proximity threshold for flare detection
        penalty: Score multiplier for penalized caries (0.5 = 50% reduction)
    
    Returns:
        Updated (teeth, anomalies) with penalized caries scores
    """
    METAL_CLASSES = {"Crown", "Filling", "Root canal filling", "Root canal obturation",
                     "Implant", "Prefabricated metal post"}
    
    def boxes_within_proximity(box1, box2, px):
        """Check if two boxes are within `px` pixels of each other."""
        # Expand box1 by px in all directions and check overlap
        expanded = [box1[0] - px, box1[1] - px, box1[2] + px, box1[3] + px]
        return compute_iou(expanded, box2) > 0
    
    # Collect all metal anomaly boxes (from both teeth findings and raw anomalies)
    metal_boxes = []
    
    # From teeth findings
    for tooth in teeth:
        for finding in tooth.get('findings', []):
            if finding['type'] in METAL_CLASSES:
                metal_boxes.append(finding['box'])
    
    # From raw anomalies
    for anom in anomalies:
        if anom.get('class_name', '') in METAL_CLASSES:
            metal_boxes.append(anom['box'])
    
    if not metal_boxes:
        return teeth, anomalies
    
    # Penalize caries in teeth findings
    for tooth in teeth:
        penalized_findings = []
        for finding in tooth.get('findings', []):
            if finding['type'] == 'Caries':
                near_metal = False
                for metal_box in metal_boxes:
                    iou = compute_iou(finding['box'], metal_box)
                    if iou > 0.1 or boxes_within_proximity(finding['box'], metal_box, proximity_px):
                        near_metal = True
                        break
                if near_metal:
                    finding['score'] *= penalty
                    finding['type'] = f"Caries (near metal, penalized)"
            penalized_findings.append(finding)
        tooth['findings'] = penalized_findings
    
    # Penalize caries in raw anomalies
    for anom in anomalies:
        if anom.get('class_name', '') == 'Caries':
            for metal_box in metal_boxes:
                iou = compute_iou(anom['box'], metal_box)
                if iou > 0.1 or boxes_within_proximity(anom['box'], metal_box, proximity_px):
                    anom['score'] *= penalty
                    anom['class_name'] = f"Caries (near metal, penalized)"
                    break
    
    return teeth, anomalies


def recursive_prune_teeth(teeth, max_teeth=32, step=0.02, max_threshold=0.90):
    """
    Subsystem 5: Anatomical Count Check — Recursive Pruning
    
    If the teeth count exceeds the anatomical maximum (32 for adults),
    incrementally raise the confidence floor and remove the lowest-scoring
    teeth until the count is plausible.
    
    Mathematical logic:
      threshold_n = threshold_0 + n × step
      teeth_n = {t ∈ teeth | t.score ≥ threshold_n}
      Repeat until |teeth_n| ≤ max_teeth or threshold_n > max_threshold
    
    Args:
        teeth: List of tooth dicts with 'score'
        max_teeth: Maximum anatomically valid count (32)
        step: Confidence threshold increment per iteration
        max_threshold: Safety cap to prevent over-pruning
    
    Returns:
        Pruned teeth list with count ≤ max_teeth
    """
    if len(teeth) <= max_teeth:
        return teeth
    
    # Start from the minimum score in the current set
    current_threshold = min(t['score'] for t in teeth)
    
    while len(teeth) > max_teeth and current_threshold < max_threshold:
        current_threshold += step
        teeth = [t for t in teeth if t['score'] >= current_threshold]
    
    # If still over limit (unlikely), hard-prune by keeping top-N by score
    if len(teeth) > max_teeth:
        teeth = sorted(teeth, key=lambda t: t['score'], reverse=True)[:max_teeth]
    
    return teeth


def validate_anatomy(teeth, anomalies, anomaly_threshold=0.15):
    """
    Subsystem 6: Master Anatomical Validation Orchestrator
    
    Called immediately before report generation. Applies all anatomical
    plausibility checks in the correct order:
    
    Pipeline:
      1. Recursive Prune     → teeth count ≤ 32
      2. Resolve FDI Conflicts → no duplicate FDI numbers
      3. Metal Artifact Suppression → penalize caries near metal
      4. Re-deduplicate anomalies  → remove penalized below threshold
    
    Args:
        teeth: List of tooth dicts (post-FDI assignment)
        anomalies: List of raw anomaly dicts
        anomaly_threshold: Minimum score for anomalies after penalties
    
    Returns:
        (validated_teeth, validated_anomalies) — anatomically plausible results
    """
    # Step 1: Enforce maximum tooth count (32 for adult dentition)
    teeth = recursive_prune_teeth(teeth, max_teeth=32)
    
    # Step 2: Resolve duplicate FDI assignments (Winner-Takes-All)
    teeth = resolve_fdi_conflicts(teeth)
    
    # Step 3: Suppress radiographic flare artifacts near metal work
    teeth, anomalies = suppress_metal_artifacts(teeth, anomalies)
    
    # Step 4: Remove anomalies whose scores fell below threshold after penalties
    for tooth in teeth:
        tooth['findings'] = [f for f in tooth.get('findings', []) 
                            if f['score'] >= anomaly_threshold]
    anomalies = [a for a in anomalies if a['score'] >= anomaly_threshold]
    
    return teeth, anomalies


def process_predictions(outputs, image_width, teeth_threshold=0.15, anomaly_threshold=0.25, class_map=None):
    """
    Process raw model outputs into teeth and anomaly lists.
    Uses DYNAMIC ZONE-BASED THRESHOLDING to compensate for center zone weakness.
    
    Args:
        outputs: Raw Detectron2 outputs
        image_width: Width of the input image for zone calculation
        teeth_threshold: Base threshold (used for anomalies)
        anomaly_threshold: Minimum confidence for anomalies
        class_map: Optional dict mapping class_id -> class_name
        
    Returns:
        (teeth, anomalies) - lists of detection dicts
    """
    instances = outputs["instances"].to("cpu")
    
    boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []
    classes = instances.pred_classes.numpy() if instances.has("pred_classes") else []
    scores = instances.scores.numpy() if instances.has("scores") else []
    masks = instances.pred_masks.numpy() if instances.has("pred_masks") else []
    
    teeth = []
    anomalies = []
    
    # Define "Center Zone" (where spine shadow causes weak detection)
    center_start = image_width * 0.35
    center_end = image_width * 0.65
    
    for i, (box, cls_id, score) in enumerate(zip(boxes, classes, scores)):
        # Determine class name using specific map or default global CLASSES
        if class_map and cls_id in class_map:
            class_name = class_map[cls_id]
        elif class_map is None and cls_id < len(CLASSES):
            class_name = CLASSES[cls_id]
        else:
            class_name = f"Unknown_{cls_id}"
        
        # Calculate box center X
        x_center = (box[0] + box[2]) / 2
        is_center_zone = (x_center > center_start) and (x_center < center_end)
        
        detection = {
            'box': box.tolist(),
            'class_id': int(cls_id),
            'class_name': class_name,
            'score': float(score),
            'mask': masks[i] if len(masks) > i else None,
            'is_center_zone': is_center_zone  # Track for debugging
        }
        
        # DYNAMIC THRESHOLDING FOR TEETH
        # Center zone (35-65%): Slightly relaxed threshold (model scores lower here)
        # Outer zones: Use slider value as-is (back teeth are clear)
        # The user's slider value (teeth_threshold) is always respected as minimum floor
        
        # Handle both 9-class (class_name == "Tooth") and 41-class (cls_id 0-32) models
        is_tooth = False
        if class_name == "Tooth":
            is_tooth = True  # 9-class model
        elif cls_id <= 32:  # 41-class model: IDs 0-32 are individual teeth (11,12...48)
            is_tooth = True
        
        if is_tooth:
            # Center zone gets a small relaxation (70% of slider), but never below 0.08
            # This prevents ghost detections while still recovering spinal-shadow teeth
            if is_center_zone:
                threshold = max(teeth_threshold * 0.7, 0.08)
            else:
                threshold = teeth_threshold
            if score >= threshold:
                teeth.append(detection)
        elif class_name in ANOMALY_CLASSES or (class_map and class_name != "Tooth"):
            # Structural anomalies (Crown, Implant, Metal Post) are visually distinctive -
            # use a lower threshold so they aren't filtered out
            STRUCTURAL_ANOMALIES = {"Crown", "Implant", "Prefabricated metal post"}
            if class_name in STRUCTURAL_ANOMALIES:
                effective_threshold = max(anomaly_threshold * 0.6, 0.10)
            else:
                effective_threshold = anomaly_threshold
            
            if score >= effective_threshold:
                anomalies.append(detection)
    
    # Deduplicate overlapping anomalies (keep highest scoring)
    anomalies = deduplicate_anomalies(anomalies, iou_threshold=0.6)
    
    return teeth, anomalies


def deduplicate_anomalies(anomalies, iou_threshold=0.6):
    """
    Remove duplicate anomaly detections by keeping only the highest scoring
    box when multiple anomalies overlap significantly.
    
    Solves: "Filling (86%)" overlapping with "Root Canal (97%)" on same tooth
    
    Args:
        anomalies: List of anomaly detection dicts
        iou_threshold: IoU above which boxes are considered duplicates
        
    Returns:
        Deduplicated list of anomalies
    """
    if len(anomalies) <= 1:
        return anomalies
    
    # Sort by score (highest first)
    anomalies = sorted(anomalies, key=lambda x: x['score'], reverse=True)
    
    keep = []
    used = [False] * len(anomalies)
    
    for i in range(len(anomalies)):
        if used[i]:
            continue
        
        keep.append(anomalies[i])
        used[i] = True
        
        # Mark overlapping boxes as used (they will be discarded)
        for j in range(i + 1, len(anomalies)):
            if used[j]:
                continue
            
            iou = compute_iou(anomalies[i]['box'], anomalies[j]['box'])
            if iou > iou_threshold:
                used[j] = True  # Discard lower-scoring overlapping box
    
    return keep


def clean_clinical_findings(teeth):
    """
    Apply hierarchy rules to clean up findings per tooth.
    
    Hierarchy Rules:
    1. Root Canal > Filling (remove Filling) - Root canal implies filling material
    2. Crown > Filling (remove Filling) - Crown covers filling
    3. Keep highest scoring of same type (e.g. duplicate Caries)
    
    Args:
        teeth: List of tooth dicts with 'findings'
        
    Returns:
        Teeth list with cleaned findings
    """
    for tooth in teeth:
        findings = tooth.get('findings', [])
        if not findings:
            continue
            
        # 1. Deduplicate same type (keep max score)
        # Sort by score descending
        findings.sort(key=lambda x: x['score'], reverse=True)
        unique_findings = []
        seen_types = set()
        
        for f in findings:
            if f['type'] not in seen_types:
                unique_findings.append(f)
                seen_types.add(f['type'])
        
        # 2. Apply Hierarchy Rules
        types_present = seen_types
        clean_findings = []
        
        for f in unique_findings:
            t = f['type']
            
            # Rule: Root Canal hides Filling
            if t == 'Filling' and ('Root canal filling' in types_present or 'Root canal obturation' in types_present):
                continue
                
            # Rule: Crown hides Filling
            if t == 'Filling' and 'Crown' in types_present:
                continue
            
            clean_findings.append(f)
            
        tooth['findings'] = clean_findings
        
    return teeth


# ============================================
# VISUALIZATION
# ============================================

# Color scheme based on color theory:
# - Teeth: Cool color (cyan/teal) - calming, healthy
# - Anomalies: Warm color (red/orange) - attention-grabbing, warning
# - Labels: High contrast white on dark background

# Single consistent color for all teeth (subtle, semi-transparent)
TOOTH_COLOR = (180, 220, 180)  # Soft green - healthy connotation

# Anomaly colors - warm spectrum for attention (red = severe, orange = moderate)
ANOMALY_COLORS = {
    "Caries": (50, 50, 255),           # Red - decay/damage
    "Retained root": (50, 50, 200),    # Dark red - serious
    "Crown": (50, 200, 255),           # Yellow/Gold - restoration
    "Filling": (50, 180, 255),         # Orange - treatment
    "Root canal filling": (100, 255, 255),  # Bright yellow - treatment
    "Root canal obturation": (100, 255, 200),  # Yellow-green
    "Implant": (255, 150, 50),         # Blue - artificial
    "Prefabricated metal post": (255, 100, 100),  # Light blue
}


def draw_detections(image, teeth, anomalies, show_masks=True, opacity=0.35):
    """
    Draw teeth and anomaly masks with clean, readable visualization.
    
    Design principles:
    - Teeth: Subtle green overlay (healthy = green)
    - Anomalies: Red/orange overlay (attention = warm colors)
    - Labels: White text on dark pill-shaped backgrounds
    - No overlapping labels
    """
    overlay = image.copy()
    h, w = image.shape[:2]
    
    # Track label positions to avoid overlap
    used_label_positions = []
    
    def get_safe_label_position(x, y, label_w, label_h):
        """Find a position that doesn't overlap with existing labels."""
        # Try original position first
        for offset_y in [0, -25, 25, -50, 50]:
            test_y = y + offset_y
            overlap = False
            for (lx, ly, lw, lh) in used_label_positions:
                if not (x + label_w < lx or x > lx + lw or test_y + label_h < ly or test_y > ly + lh):
                    overlap = True
                    break
            if not overlap:
                used_label_positions.append((x, test_y, label_w, label_h))
                return x, test_y
        return x, y  # Fallback
    
    # === FIRST PASS: Draw all tooth masks (subtle green) ===
    for tooth in teeth:
        if not show_masks or tooth.get('mask') is None:
            continue
            
        mask = tooth['mask'].astype(np.uint8)
        mask_area = mask > 0
        
        if mask_area.any():
            # Apply subtle green overlay
            colored_mask = np.zeros_like(overlay)
            colored_mask[:, :] = TOOTH_COLOR
            overlay[mask_area] = cv2.addWeighted(
                overlay[mask_area], 1 - opacity,
                colored_mask[mask_area], opacity, 0
            )
    
    # === SECOND PASS: Draw anomaly overlays (red/orange) ===
    for tooth in teeth:
        for finding in tooth.get('findings', []):
            anom_color = ANOMALY_COLORS.get(finding['type'], (50, 50, 255))
            anom_mask = finding.get('mask')
            
            if anom_mask is not None and show_masks:
                mask = anom_mask.astype(np.uint8)
                mask_area = mask > 0
                
                if mask_area.any():
                    colored_mask = np.zeros_like(overlay)
                    colored_mask[:, :] = anom_color
                    overlay[mask_area] = cv2.addWeighted(
                        overlay[mask_area], 0.5,
                        colored_mask[mask_area], 0.5, 0
                    )
            else:
                # Fallback to box
                box = finding.get('box')
                if box is not None:
                    x1, y1, x2, y2 = map(int, box)
                    if x2 > x1 and y2 > y1:
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), anom_color, 2)
    
    # === THIRD PASS: Draw FDI labels (clear, readable) ===
    for tooth in teeth:
        fdi = tooth.get('fdi', '?')
        mask = tooth.get('mask')
        
        if mask is not None:
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
            else:
                box = tooth['box']
                cx = int((box[0] + box[2]) / 2)
                cy = int((box[1] + box[3]) / 2)
        else:
            box = tooth['box']
            cx = int((box[0] + box[2]) / 2)
            cy = int((box[1] + box[3]) / 2)
        
        # Draw FDI number with pill-shaped background
        label = str(fdi)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        
        padding = 6
        label_w = tw + padding * 2
        label_h = th + padding * 2
        
        # Get safe position
        lx = cx - label_w // 2
        ly = cy - label_h // 2
        lx, ly = get_safe_label_position(lx, ly, label_w, label_h)
        
        # Draw rounded rectangle background (dark)
        cv2.rectangle(overlay, (lx, ly), (lx + label_w, ly + label_h), (30, 30, 30), -1)
        cv2.rectangle(overlay, (lx, ly), (lx + label_w, ly + label_h), (255, 255, 255), 1)
        
        # Draw white text
        text_x = lx + padding
        text_y = ly + padding + th
        cv2.putText(overlay, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    # === FOURTH PASS: Draw anomaly labels for associated findings ===
    for tooth in teeth:
        findings = tooth.get('findings', [])
        if not findings:
            continue
            
        # Get position below the tooth
        box = tooth['box']
        x1, y2 = int(box[0]), int(box[3])
        
        for i, finding in enumerate(findings):
            label = finding['type'][:12]
            score = finding['score']
            text = f"{label}"
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            thickness = 1
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Position below tooth, stacked
            label_y = y2 + 12 + i * 16
            
            if label_y < h - 20:  # Don't draw off screen
                # Red background for anomaly
                anom_color = ANOMALY_COLORS.get(finding['type'], (50, 50, 255))
                cv2.rectangle(overlay, (x1, label_y - th - 2), (x1 + tw + 6, label_y + 2), anom_color, -1)
                cv2.putText(overlay, text, (x1 + 3, label_y - 2), font, font_scale, (255, 255, 255), thickness)
    
    # === FIFTH PASS: Draw unmarked anomalies ===
    # === FIFTH PASS: Draw unmarked anomalies ===
    for anomaly in anomalies:
        if anomaly.get('associated_fdi') is not None:
            continue
        
        color = ANOMALY_COLORS.get(anomaly['class_name'], (50, 50, 255))
        anom_mask = anomaly.get('mask')
        
        if anom_mask is not None and show_masks:
            mask = anom_mask.astype(np.uint8)
            mask_area = mask > 0
            if mask_area.any():
                colored_mask = np.zeros_like(overlay)
                colored_mask[:, :] = color
                # Slightly higher opacity for better visibility as requested
                overlay[mask_area] = cv2.addWeighted(
                    overlay[mask_area], 0.3,
                    colored_mask[mask_area], 0.7, 0
                )
        else:
            box = anomaly['box']
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        
        # Label with smart positioning
        box = anomaly['box']
        cx = int((box[0] + box[2]) / 2)
        cy = int(box[1]) # Start at top edge
        
        label = f"{anomaly['class_name']}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        padding = 4
        label_w = tw + padding * 2
        label_h = th + padding * 2
        
        # Center label horizontally on box
        lx = cx - label_w // 2
        ly = cy - label_h - 4 # Default: above box
        
        # Use smart positioning to avoid overlap
        lx, ly = get_safe_label_position(lx, ly, label_w, label_h)
        
        # Draw Label Background (Dark with transparency effect via solid color for now)
        # Using a distinct color for standalone anomalies
        cv2.rectangle(overlay, (lx, ly), (lx + label_w, ly + label_h), (20, 20, 20), -1)
        cv2.rectangle(overlay, (lx, ly), (lx + label_w, ly + label_h), color, 1)
        
        # Draw Text
        text_x = lx + padding
        text_y = ly + padding + th
        cv2.putText(overlay, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
    
    return overlay


def generate_clinical_report(teeth, anomalies, model_name="Complete Scan"):
    """
    Generate comprehensive clinical findings report with full FDI charting.
    Shows all 32 adult tooth positions with status and anomalies.
    """
    import datetime
    
    report = []
    
    # Header
    report.append("# 🦷 Dental Diagnostic Report\n")
    report.append(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**Model:** {model_name}")
    report.append(f"**Analysis Type:** {'Full Checkup' if model_name == 'Complete Scan' else 'Pathology Focus'}\n")
    
    # Build detected FDI lookup
    detected_fdi = {t['fdi']: t for t in teeth}
    
    # Calculate metrics
    teeth_findings_count = sum(len(t.get('findings', [])) for t in teeth)
    unmarked_anomalies = [a for a in anomalies if a.get('associated_fdi') is None]
    total_anomalies = teeth_findings_count + len(unmarked_anomalies)
    
    # Ambiguity check
    ambiguous_teeth = [t for t in teeth if t.get('score', 1) < 0.20]
    scan_issues = []
    if len(teeth) < 10:
        scan_issues.append("⚠️ Very few teeth detected - possible partial scan")
    if ambiguous_teeth:
        scan_issues.append(f"⚠️ {len(ambiguous_teeth)} teeth with low confidence (<20%)")
    
    # Summary
    report.append("## 📋 Summary\n")
    report.append("| Metric | Value |")
    report.append("|--------|-------|")
    report.append(f"| Teeth Detected | {len(teeth)} / 32 |")
    report.append(f"| Anomalies Found | {total_anomalies} |")
    report.append(f"| Scan Quality | {'⚠️ Review Needed' if scan_issues else '✅ Good'} |")
    report.append("")
    
    if scan_issues:
        report.append("### Scan Quality Notes")
        for issue in scan_issues:
            report.append(f"- {issue}")
        report.append("")
    
    # Full 32-tooth FDI Template
    FDI_TEMPLATE = {
        1: [18, 17, 16, 15, 14, 13, 12, 11],  # Upper Right (Q1)
        2: [21, 22, 23, 24, 25, 26, 27, 28],  # Upper Left (Q2)
        3: [31, 32, 33, 34, 35, 36, 37, 38],  # Lower Left (Q3)
        4: [48, 47, 46, 45, 44, 43, 42, 41],  # Lower Right (Q4)
    }
    
    QUADRANT_NAMES = {
        1: "Upper Right (Q1)",
        2: "Upper Left (Q2)",
        3: "Lower Left (Q3)",
        4: "Lower Right (Q4)"
    }
    
    # Visual Dental Chart
    report.append("## 🦷 Dental Chart\n")
    report.append("```")
    report.append("        UPPER JAW")
    report.append("   R                      L")
    
    # Upper row
    upper_row = "   "
    upper_status = "   "
    for fdi in FDI_TEMPLATE[1]:
        upper_row += f"{fdi:>3}"
        if fdi in detected_fdi:
            t = detected_fdi[fdi]
            if t.get('findings'):
                upper_status += "  ⚠"
            elif t.get('score', 1) < 0.20:
                upper_status += "  ?"
            else:
                upper_status += "  ✓"
        else:
            upper_status += "  ─"
    upper_row += " │"
    upper_status += " │"
    for fdi in FDI_TEMPLATE[2]:
        upper_row += f"{fdi:>3}"
        if fdi in detected_fdi:
            t = detected_fdi[fdi]
            if t.get('findings'):
                upper_status += "  ⚠"
            elif t.get('score', 1) < 0.20:
                upper_status += "  ?"
            else:
                upper_status += "  ✓"
        else:
            upper_status += "  ─"
    report.append(upper_row)
    report.append(upper_status)
    report.append("   " + "─" * 50)
    
    # Lower row
    lower_row = "   "
    lower_status = "   "
    for fdi in FDI_TEMPLATE[4]:
        lower_row += f"{fdi:>3}"
        if fdi in detected_fdi:
            t = detected_fdi[fdi]
            if t.get('findings'):
                lower_status += "  ⚠"
            elif t.get('score', 1) < 0.20:
                lower_status += "  ?"
            else:
                lower_status += "  ✓"
        else:
            lower_status += "  ─"
    lower_row += " │"
    lower_status += " │"
    for fdi in FDI_TEMPLATE[3]:
        lower_row += f"{fdi:>3}"
        if fdi in detected_fdi:
            t = detected_fdi[fdi]
            if t.get('findings'):
                lower_status += "  ⚠"
            elif t.get('score', 1) < 0.20:
                lower_status += "  ?"
            else:
                lower_status += "  ✓"
        else:
            lower_status += "  ─"
    report.append(lower_row)
    report.append(lower_status)
    report.append("        LOWER JAW")
    report.append("```")
    report.append("\n**Legend:** ✓=Healthy  ⚠=Anomaly  ─=Not Detected  ?=Low Confidence\n")
    
    # Quadrant-based detailed findings
    report.append("## 📝 Detailed Findings by Quadrant\n")
    
    for q_num, fdi_list in FDI_TEMPLATE.items():
        report.append(f"### {QUADRANT_NAMES[q_num]}\n")
        report.append("| FDI | Status | Findings |")
        report.append("|-----|--------|----------|")
        
        for fdi in fdi_list:
            if fdi in detected_fdi:
                tooth = detected_fdi[fdi]
                confidence = tooth.get('score', 0)
                
                if confidence < 0.20:
                    status = f"⚠️ Low Conf ({confidence:.0%})"
                else:
                    status = f"✅ Detected ({confidence:.0%})"
                
                findings = tooth.get('findings', [])
                if findings:
                    findings_str = ", ".join([f"{f['type']} ({f['score']:.0%})" for f in findings])
                else:
                    findings_str = "Healthy"
            else:
                status = "─ Not Detected"
                findings_str = "—"
            
            report.append(f"| {fdi} | {status} | {findings_str} |")
        
        report.append("")
    
    # Unmarked Anomalies
    if unmarked_anomalies:
        report.append("## ⚠️ Additional Anomalies (Unassociated)\n")
        report.append("*These anomalies were detected but could not be linked to a specific tooth.*\n")
        report.append("| Condition | Confidence | Region |")
        report.append("|-----------|------------|--------|")
        for anomaly in unmarked_anomalies:
            box = anomaly['box']
            x_center = (box[0] + box[2]) / 2
            location = "Center" if 0.35 < x_center / 1000 < 0.65 else ("Left" if x_center / 1000 < 0.35 else "Right")
            report.append(f"| {anomaly['class_name']} | {anomaly['score']:.0%} | {location} |")
        report.append("")
    
    # Legend
    report.append("---")
    report.append("## 📖 Reference\n")
    report.append("**FDI Numbering:** Q1 (11-18) Upper Right • Q2 (21-28) Upper Left • Q3 (31-38) Lower Left • Q4 (41-48) Lower Right")
    
    return "\n".join(report)


def generate_anomaly_report(anomalies, model_name="Pathology Focus"):
    """
    Generate focused anomaly-only report for Pathology Focus mode.
    """
    import datetime
    
    report = []
    
    # Header
    report.append("# 🔍 Pathology Detection Report\n")
    report.append(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report.append(f"**Model:** {model_name}")
    report.append(f"**Analysis Type:** High-Sensitivity Anomaly Detection\n")
    
    # Summary
    report.append("## 📋 Summary\n")
    report.append(f"**Total Anomalies Detected:** {len(anomalies)}\n")
    
    if not anomalies:
        report.append("✅ **No pathological findings detected.**\n")
        return "\n".join(report)
    
    # Anomaly Table
    report.append("## 🔍 Detected Anomalies\n")
    report.append("| # | Condition | Confidence | Region |")
    report.append("|---|-----------|------------|--------|")
    
    for i, anomaly in enumerate(anomalies, 1):
        box = anomaly['box']
        x_center = (box[0] + box[2]) / 2
        location = "Center" if 0.35 < x_center / 1000 < 0.65 else ("Left" if x_center / 1000 < 0.35 else "Right")
        report.append(f"| {i} | {anomaly['class_name']} | {anomaly['score']:.0%} | {location} |")
    
    report.append("")
    report.append("---")
    report.append("*This report focuses on pathological findings only. For complete dental charting, use Complete Scan mode.*")
    
    return "\n".join(report)


def generate_pdf_report(teeth, anomalies, original_image_rgb, processed_image_rgb, model_name="Complete Scan"):
    """
    Generate a comprehensive PDF report with clinical findings and both OPG images.
    
    Args:
        teeth: List of teeth with findings
        anomalies: List of anomalies
        original_image_rgb: RGB numpy array of original X-ray
        processed_image_rgb: RGB numpy array of processed image with overlays
        model_name: Name of the model used
        
    Returns:
        bytes: PDF content
    """
    from io import BytesIO
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from PIL import Image as PILImage
    import tempfile
    import os
    import datetime
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
    styles = getSampleStyleSheet()
    story = []
    temp_files = []  # Track temp files for cleanup
    
    is_pathology_mode = model_name == "Pathology Focus"
    
    # Title
    title_text = "🔍 Pathology Detection Report" if is_pathology_mode else "🦷 Complete Dental Report"
    title_style = ParagraphStyle('Title', parent=styles['Title'], fontSize=22, textColor=colors.darkblue)
    story.append(Paragraph(title_text, title_style))
    story.append(Spacer(1, 0.15*inch))
    
    # Date and model info
    story.append(Paragraph(f"<b>Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Paragraph(f"<b>Model:</b> {model_name}", styles['Normal']))
    story.append(Paragraph(f"<b>Analysis Type:</b> {'Anomaly Detection Only' if is_pathology_mode else 'Full Dental Checkup'}", styles['Normal']))
    story.append(Spacer(1, 0.2*inch))
    
    # Summary metrics
    teeth_findings_count = sum(len(t.get('findings', [])) for t in teeth)
    unmarked_anomalies = [a for a in anomalies if a.get('associated_fdi') is None]
    total_anomalies = teeth_findings_count + len(unmarked_anomalies)
    
    # Scan quality check
    ambiguous_teeth = [t for t in teeth if t.get('score', 1) < 0.20]
    scan_quality = "⚠️ Review Needed" if (len(teeth) < 10 or ambiguous_teeth) else "✅ Good"
    
    summary_data = [
        ['Metric', 'Value'],
        ['Model', model_name],
        ['Teeth Detected', f"{len(teeth)} / 32" if not is_pathology_mode else "N/A"],
        ['Anomalies Found', str(total_anomalies)],
        ['Scan Quality', scan_quality if not is_pathology_mode else "N/A"]
    ]
    summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ]))
    story.append(Paragraph("<b>Summary</b>", styles['Heading2']))
    story.append(summary_table)
    story.append(Spacer(1, 0.25*inch))
    
    # Images Section - Side by Side
    story.append(Paragraph("<b>X-Ray Analysis</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    # Save both images to temp files
    pil_original = PILImage.fromarray(original_image_rgb)
    pil_processed = PILImage.fromarray(processed_image_rgb)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp1:
        pil_original.save(tmp1.name)
        temp_files.append(tmp1.name)
        img_original = Image(tmp1.name, width=3*inch, height=1.8*inch)
        
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp2:
        pil_processed.save(tmp2.name)
        temp_files.append(tmp2.name)
        img_processed = Image(tmp2.name, width=3*inch, height=1.8*inch)
    
    # Create side-by-side image table
    img_table = Table([
        [Paragraph("<b>Original OPG</b>", styles['Normal']), Paragraph("<b>AI Analysis</b>", styles['Normal'])],
        [img_original, img_processed]
    ], colWidths=[3.2*inch, 3.2*inch])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(img_table)
    story.append(Spacer(1, 0.3*inch))
    
    # MODE-SPECIFIC CONTENT
    if is_pathology_mode:
        # ============ PATHOLOGY FOCUS MODE ============
        if total_anomalies == 0:
            story.append(Paragraph("✅ <b>No pathological findings detected.</b>", styles['Normal']))
        else:
            story.append(Paragraph("<b>Detected Anomalies</b>", styles['Heading2']))
            
            anomaly_data = [['#', 'Condition', 'Confidence', 'Region']]
            all_anomalies = unmarked_anomalies  # In pathology mode, all are unmarked
            
            for i, anomaly in enumerate(all_anomalies, 1):
                box = anomaly['box']
                x_center = (box[0] + box[2]) / 2
                location = "Center" if 0.35 < x_center / 1000 < 0.65 else ("Left" if x_center / 1000 < 0.35 else "Right")
                anomaly_data.append([
                    str(i),
                    anomaly['class_name'],
                    f"{anomaly['score']:.0%}",
                    location
                ])
            
            anomaly_table = Table(anomaly_data, colWidths=[0.5*inch, 2.5*inch, 1*inch, 1*inch])
            anomaly_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.red),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            story.append(anomaly_table)
    else:
        # ============ COMPLETE SCAN MODE ============
        # FDI Template
        FDI_TEMPLATE = {
            1: [18, 17, 16, 15, 14, 13, 12, 11],
            2: [21, 22, 23, 24, 25, 26, 27, 28],
            3: [31, 32, 33, 34, 35, 36, 37, 38],
            4: [48, 47, 46, 45, 44, 43, 42, 41],
        }
        QUADRANT_NAMES = {1: "Upper Right (Q1)", 2: "Upper Left (Q2)", 3: "Lower Left (Q3)", 4: "Lower Right (Q4)"}
        
        detected_fdi = {t['fdi']: t for t in teeth}
        
        # Quadrant findings tables
        for q_num, fdi_list in FDI_TEMPLATE.items():
            story.append(Paragraph(f"<b>{QUADRANT_NAMES[q_num]}</b>", styles['Heading3']))
            
            q_data = [['FDI', 'Status', 'Findings']]
            for fdi in fdi_list:
                if fdi in detected_fdi:
                    tooth = detected_fdi[fdi]
                    conf = tooth.get('score', 0)
                    status = "✓ Detected" if conf >= 0.20 else "? Low Conf"
                    findings = tooth.get('findings', [])
                    findings_str = ", ".join([f['type'] for f in findings]) if findings else "Healthy"
                else:
                    status = "─ Not Detected"
                    findings_str = "—"
                q_data.append([str(fdi), status, findings_str])
            
            q_table = Table(q_data, colWidths=[0.6*inch, 1.5*inch, 2.5*inch])
            q_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            story.append(q_table)
            story.append(Spacer(1, 0.15*inch))
        
        # Additional unmarked anomalies
        if unmarked_anomalies:
            story.append(Paragraph("<b>⚠️ Additional Anomalies (Unassociated)</b>", styles['Heading2']))
            
            unmarked_data = [['Condition', 'Confidence', 'Region']]
            for anomaly in unmarked_anomalies:
                box = anomaly['box']
                x_center = (box[0] + box[2]) / 2
                location = "Center" if 0.35 < x_center / 1000 < 0.65 else ("Left" if x_center / 1000 < 0.35 else "Right")
                unmarked_data.append([anomaly['class_name'], f"{anomaly['score']:.0%}", location])
            
            unmarked_table = Table(unmarked_data, colWidths=[2*inch, 1*inch, 1.5*inch])
            unmarked_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.orange),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(unmarked_table)
    
    # Build PDF
    doc.build(story)
    
    # Clean up temp files
    for tmp_path in temp_files:
        try:
            os.unlink(tmp_path)
        except:
            pass
    
    return buffer.getvalue()


# ============================================
# STREAMLIT UI
# ============================================

def main():
    st.set_page_config(
        page_title="Dent-AI | AI Dental Diagnostics",
        page_icon="🦷",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Comprehensive Professional Theme - Warm Off-White
    # Inspired by medical/dental SaaS like Dentrix, Open Dental
    st.markdown("""
    <style>
    /* ===== RESET & GLOBAL - Warm Off-White Theme ===== */
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main app background - Warm cream/off-white */
    .stApp {
        background-color: #fafaf9 !important;
    }
    
    /* Main container */
    .main .block-container {
        background-color: #fafaf9 !important;
        padding-top: 2rem !important;
    }
    
    /* All text should be dark for readability */
    .stApp, .stApp p, .stApp span, .stApp div, .stApp label {
        color: #374151 !important;
    }
    
    /* ===== HEADER STYLES ===== */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0891b2 !important;
        text-align: center;
        margin-bottom: 0.2rem;
        padding-top: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #6b7280 !important;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    /* ===== SIDEBAR ===== */
    [data-testid="stSidebar"] {
        background-color: #f5f5f4 !important;
        border-right: 1px solid #e7e5e4 !important;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        background-color: #f5f5f4 !important;
    }
    
    /* Sidebar text */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown {
        color: #374151 !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4 {
        color: #1f2937 !important;
    }
    
    /* Sidebar selectbox */
    [data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #ffffff !important;
        border: 1px solid #d6d3d1 !important;
        color: #374151 !important;
    }
    
    /* ===== HEADINGS ===== */
    h1, h2, h3, h4, h5, h6 {
        color: #1f2937 !important;
    }
    
    /* ===== CARDS & METRICS ===== */
    [data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #e7e5e4 !important;
        border-radius: 8px !important;
        padding: 0.75rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04) !important;
    }
    
    [data-testid="stMetric"] label {
        color: #6b7280 !important;
        font-size: 0.8rem !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0891b2 !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button, .stDownloadButton > button {
        background-color: #0891b2 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
    }
    
    .stButton > button:hover, .stDownloadButton > button:hover {
        background-color: #0e7490 !important;
    }
    
    /* ===== FILE UPLOADER - COMPREHENSIVE FIX ===== */
    [data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 2px dashed #d6d3d1 !important;
        border-radius: 8px !important;
    }
    
    [data-testid="stFileUploader"] > div {
        background-color: #ffffff !important;
    }
    
    [data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
    }
    
    [data-testid="stFileUploader"] section > div {
        background-color: #ffffff !important;
    }
    
    /* The drag-drop zone */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] {
        background-color: #f5f5f4 !important;
        border: none !important;
    }
    
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] {
        color: #374151 !important;
    }
    
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] div {
        color: #374151 !important;
    }
    
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: #6b7280 !important;
    }
    
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] small {
        color: #9ca3af !important;
    }
    
    /* Browse button in uploader */
    [data-testid="stFileUploader"] button {
        background-color: #374151 !important;
        color: #ffffff !important;
        border: none !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #374151 !important;
    }
    
    [data-testid="stFileUploader"] small {
        color: #6b7280 !important;
    }
    
    /* Target all divs inside file uploader */
    [data-testid="stFileUploader"] div[data-testid] {
        background-color: transparent !important;
    }
    
    [data-testid="stFileUploaderDropzone"] {
        background-color: #f5f5f4 !important;
    }
    
    /* ===== EXPANDERS ===== */
    .streamlit-expanderHeader {
        background-color: #f5f5f4 !important;
        color: #374151 !important;
        border-radius: 6px !important;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff !important;
        border: 1px solid #e7e5e4 !important;
    }
    
    /* ===== SUCCESS/INFO/ERROR ALERTS ===== */
    [data-testid="stAlert"] {
        border-radius: 6px !important;
    }
    
    /* Success */
    .stSuccess, [data-baseweb="notification"][kind="positive"] {
        background-color: #ecfdf5 !important;
        border: 1px solid #a7f3d0 !important;
        color: #065f46 !important;
    }
    
    /* Info */
    .stInfo, [data-baseweb="notification"][kind="info"] {
        background-color: #ecfeff !important;
        border: 1px solid #a5f3fc !important;
        color: #0e7490 !important;
    }
    
    /* Error */
    .stError, [data-baseweb="notification"][kind="negative"] {
        background-color: #fef2f2 !important;
        border: 1px solid #fecaca !important;
        color: #991b1b !important;
    }
    
    /* ===== IMAGES ===== */
    .stImage {
        border-radius: 8px !important;
        overflow: hidden !important;
        border: 1px solid #e7e5e4 !important;
    }
    
    .stImage img {
        border-radius: 8px !important;
    }
    
    /* Image captions */
    .stImage figcaption {
        color: #6b7280 !important;
        font-size: 0.85rem !important;
    }
    
    /* ===== DIVIDERS ===== */
    hr {
        border-color: #e7e5e4 !important;
        margin: 1.5rem 0 !important;
    }
    
    /* ===== SLIDERS ===== */
    .stSlider label {
        color: #374151 !important;
    }
    
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        color: #6b7280 !important;
    }
    
    /* ===== CHECKBOXES ===== */
    .stCheckbox label {
        color: #374151 !important;
    }
    
    /* ===== MARKDOWN & TEXT ===== */
    .stMarkdown, .stMarkdown p, .stText {
        color: #374151 !important;
    }
    
    /* Code blocks */
    .stMarkdown code {
        background-color: #f5f5f4 !important;
        color: #374151 !important;
        border: 1px solid #e7e5e4 !important;
    }
    
    .stMarkdown pre {
        background-color: #f5f5f4 !important;
        border: 1px solid #e7e5e4 !important;
    }
    
    /* ===== TABLES ===== */
    .stMarkdown table {
        border-collapse: collapse !important;
    }
    
    .stMarkdown th {
        background-color: #f5f5f4 !important;
        color: #1f2937 !important;
        border: 1px solid #e7e5e4 !important;
    }
    
    .stMarkdown td {
        color: #374151 !important;
        border: 1px solid #e7e5e4 !important;
    }
    
    /* ===== SPINNER ===== */
    .stSpinner > div {
        border-top-color: #0891b2 !important;
    }
    
    /* ===== CAPTIONS ===== */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #6b7280 !important;
    }
    
    /* ===== SELECT BOXES ===== */
    .stSelectbox label {
        color: #374151 !important;
    }
    
    .stSelectbox > div > div {
        background-color: #ffffff !important;
        border-color: #d6d3d1 !important;
        color: #374151 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🦷 Dent-AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Dental Anomaly Detection System</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        # Logo/Title
        st.markdown("### 🦷 Dent-AI")
        st.caption("v2.0 | Dental Diagnostics")
        st.divider()
        
        # Model Selection
        st.markdown("#### 🧠 Model")
        model_name = st.selectbox(
            "Select Analysis Mode",
            options=list(MODEL_OPTIONS.keys()),
            index=0,
            label_visibility="collapsed"
        )
        model_config = MODEL_OPTIONS[model_name]
        
        # Model info badge
        if model_config.get("type") == "anomalies_only":
            st.info(f"🔍 **{model_name}**\n\n{model_config['description']}")
        else:
            st.success(f"✅ **{model_name}**\n\n{model_config['description']}")
        
        # Load model
        try:
            predictor, device_name = load_model(model_config["config"], model_config["weights"])
            st.caption(f"🖥️ Device: {device_name}")
        except Exception as e:
            st.error(f"❌ Model Error: {e}")
            return
        
        st.divider()
        
        # Detection Settings in Expander
        with st.expander("🎯 Detection Settings", expanded=False):
            teeth_conf = st.slider(
                "Teeth Confidence",
                min_value=0.10, max_value=0.50, value=0.20, step=0.05,
                help="Higher = fewer false positives, Lower = more detections"
            )
            anomaly_conf = st.slider(
                "Anomaly Confidence",
                min_value=0.10, max_value=0.80, value=0.15, step=0.05,
                help="Minimum confidence to report an anomaly"
            )
        
        # Visualization Settings in Expander
        with st.expander("🎨 Visualization", expanded=False):
            show_masks = st.checkbox("Show Segmentation Masks", True)
            mask_opacity = st.slider(
                "Mask Opacity",
                min_value=0.1, max_value=0.8, value=0.4, step=0.1
            )
        
        st.divider()
    
    # Main content - Equal columns with gap
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("#### 📤 Upload X-Ray")
        uploaded_file = st.file_uploader(
            "Choose a panoramic dental X-ray",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Supported formats: JPG, PNG, BMP",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            # Load image
            image = Image.open(uploaded_file)
            image_rgb = np.array(image.convert('RGB'))
            
            # CRITICAL: Convert RGB to BGR for Detectron2
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            st.image(image_rgb, caption="📷 Original X-Ray", use_container_width=True)
        else:
            # Placeholder
            st.markdown("""
            <div style="
                border: 2px dashed #d6d3d1;
                border-radius: 8px;
                padding: 3rem;
                text-align: center;
                color: #6b7280;
                background: #ffffff;
            ">
                <p style="font-size: 1.5rem; margin-bottom: 0.5rem;">📁</p>
                <p style="color: #374151;">Drop your X-ray image here</p>
                <p style="font-size: 0.8rem; color: #9ca3af;">or click to browse</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### 🔬 AI Analysis")
        
        if uploaded_file:
            with st.spinner("🧠 Analyzing X-ray..."):
                # Run inference with primary model
                outputs = run_inference(predictor, image_bgr)
                
                # Process predictions with dynamic zone-based thresholding
                image_width = image_bgr.shape[1]
                raw_teeth, raw_anomalies = process_predictions(
                    outputs,
                    image_width=image_width,
                    teeth_threshold=teeth_conf,
                    anomaly_threshold=anomaly_conf,
                    class_map=model_config.get("class_map")
                )
                
                # HYBRID MODE: If Complete Scan, also run the Pathology model
                # for better anomaly detection and fuse results
                if model_config.get("type") == "hybrid" and model_config.get("pathology_config"):
                    pathology_predictor, _ = load_model(
                        model_config["pathology_config"],
                        model_config["pathology_weights"]
                    )
                    pathology_outputs = run_inference(pathology_predictor, image_bgr)
                    _, pathology_anomalies = process_predictions(
                        pathology_outputs,
                        image_width=image_width,
                        teeth_threshold=0.50,  # High threshold - we don't want teeth from this model
                        anomaly_threshold=anomaly_conf,
                        class_map=PATHOLOGY_CLASS_MAP
                    )
                    # Filter out any "Tooth" class detections from pathology model
                    pathology_anomalies = [a for a in pathology_anomalies if a['class_name'] != "Tooth"]
                    
                    # Merge: add pathology anomalies that don't overlap with existing ones
                    for pa in pathology_anomalies:
                        is_duplicate = False
                        for existing in raw_anomalies:
                            if compute_iou(pa['box'], existing['box']) > 0.3:
                                is_duplicate = True
                                # Keep the higher-confidence one
                                if pa['score'] > existing['score']:
                                    existing['score'] = pa['score']
                                    existing['class_name'] = pa['class_name']
                                    existing['mask'] = pa.get('mask')
                                break
                        if not is_duplicate:
                            raw_anomalies.append(pa)
                
                # Handling for Anomalies-Only Model
                if model_config.get("type") == "anomalies_only":
                    # Discard tooth detections (if any were accidentally found)
                    raw_teeth = []
                    
                    # Ensure "Tooth" class is not in anomalies
                    raw_anomalies = [a for a in raw_anomalies if a['class_name'] != "Tooth"]
                    
                    # Skip geometric processing since we have no teeth
                    all_teeth = []
                    
                else:
                    # NORMAL MODE: Run geometric engine
                    # Apply DIoU-based NMS to teeth (threshold 0.4 for center-distance penalty)
                    teeth_nms = DentalGeometricEngine.apply_nms(raw_teeth, iou_threshold=0.4)
                    
                    # Split jaws using Y-median
                    upper_jaw, lower_jaw = DentalGeometricEngine.split_jaws(teeth_nms)
                    
                    # Assign FDI numbers
                    upper_teeth = DentalGeometricEngine.sort_and_assign_fdi(upper_jaw, 'upper')
                    lower_teeth = DentalGeometricEngine.sort_and_assign_fdi(lower_jaw, 'lower')
                    
                    # Combine all teeth
                    all_teeth = upper_teeth + lower_teeth
                
                # Map anomalies to teeth (will map nothing if all_teeth is empty)
                all_teeth = DentalGeometricEngine.map_anomalies(all_teeth, raw_anomalies)
                
                # Clean hierarchy (remove Filling if Root Canal exists, etc.)
                all_teeth = clean_clinical_findings(all_teeth)
                
                # === ANATOMICAL PLAUSIBILITY ENFORCEMENT ===
                # Validates: count ≤ 32, no duplicate FDIs, metal artifact suppression
                all_teeth, raw_anomalies = validate_anatomy(
                    all_teeth, raw_anomalies, anomaly_threshold=anomaly_conf
                )
                
                # Visualize
                result_image = draw_detections(
                    image_bgr, all_teeth, raw_anomalies,
                    show_masks=show_masks, opacity=mask_opacity
                )
                
                # Convert back to RGB for display
                result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
                st.image(result_rgb, caption="🔍 AI Detection Results", use_container_width=True)
                
                # Metrics row with better spacing
                st.markdown("")  # Spacer
                col_a, col_b, col_c = st.columns(3, gap="small")
                with col_a:
                    st.metric("🦷 Teeth", len(all_teeth))
                with col_b:
                    # Metric: Total findings (Teeth findings + Unmarked anomalies)
                    if all_teeth:
                        teeth_findings_count = sum(len(t.get('findings', [])) for t in all_teeth)
                        unmarked_count = len([a for a in raw_anomalies if a.get('associated_fdi') is None])
                        total_findings = teeth_findings_count + unmarked_count
                    else:
                        total_findings = len(raw_anomalies)
                    st.metric("⚠️ Anomalies", total_findings)
                with col_c:
                    avg_conf = np.mean([t['score'] for t in all_teeth]) if all_teeth else (np.mean([a['score'] for a in raw_anomalies]) if raw_anomalies else 0)
                    st.metric("📊 Confidence", f"{avg_conf:.0%}")
        else:
            # Placeholder when no image uploaded
            st.markdown("""
            <div style="
                border: 1px solid #e7e5e4;
                border-radius: 8px;
                padding: 3rem;
                text-align: center;
                color: #6b7280;
                background: #ffffff;
            ">
                <p style="font-size: 2rem; margin-bottom: 0.5rem;">🔬</p>
                <p style="color: #374151;">Analysis results will appear here</p>
                <p style="font-size: 0.8rem; color: #9ca3af;">Upload an X-ray to get started</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Clinical Report Section
    if uploaded_file and (all_teeth or raw_anomalies):
        st.divider()
        st.markdown("### 📋 Clinical Report")
        
        # Generate appropriate report based on mode
        is_pathology_mode = model_config.get("type") == "anomalies_only"
        
        if is_pathology_mode:
            report = generate_anomaly_report(raw_anomalies, model_name)
        else:
            report = generate_clinical_report(all_teeth, raw_anomalies, model_name)
        
        st.markdown(report)
        
        # Download buttons
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            # PDF Download with both images
            try:
                pdf_bytes = generate_pdf_report(all_teeth, raw_anomalies, image_rgb, result_rgb, model_name)
                st.download_button(
                    "📥 Download PDF Report",
                    pdf_bytes,
                    file_name="dental_report.pdf",
                    mime="application/pdf"
                )
            except ImportError:
                st.warning("Install reportlab for PDF: `pip install reportlab`")
        
        with col_dl2:
            # Markdown Download
            st.download_button(
                "📄 Download Markdown",
                report,
                file_name="clinical_findings.md",
                mime="text/markdown"
            )


if __name__ == "__main__":
    main()
