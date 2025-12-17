#!/usr/bin/env python3
"""
Hybrid inference strategy: automatically selects between 1024×512 and tiled inference
based on image size and model confidence.
"""

import argparse
import os
import cv2
import numpy as np
from typing import Dict, Optional

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog

from tools.tiled_inference import run_tiled_inference


def hybrid_inference(predictor_1024: DefaultPredictor, predictor_tiled: DefaultPredictor,
                    image_path: str, width_threshold: int = 1500,
                    confidence_threshold: float = 0.7,
                    output_path: Optional[str] = None) -> Dict:
    """
    Run hybrid inference: select method based on image size and confidence.
    
    Args:
        predictor_1024: Predictor for 1024×512 inference
        predictor_tiled: Predictor for tiled inference (can be same as 1024)
        image_path: Path to input image
        width_threshold: If image width >= this, use tiled inference
        confidence_threshold: If max confidence < this, use tiled inference
        output_path: Path to save visualization
    
    Returns:
        Dictionary with predictions and method used
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    h, w = image.shape[:2]
    
    # Decision logic
    use_tiled = False
    decision_reason = ""
    
    if w >= width_threshold:
        use_tiled = True
        decision_reason = f"Image width ({w}px) >= threshold ({width_threshold}px)"
    else:
        # Try 1024×512 first and check confidence
        target_w = 1024
        target_h = int(h * target_w / w)
        image_resized = cv2.resize(image, (target_w, target_h))
        
        outputs = predictor_1024(image_resized)
        instances = outputs["instances"]
        
        if len(instances) > 0:
            max_confidence = instances.scores.max().item()
            if max_confidence < confidence_threshold:
                use_tiled = True
                decision_reason = f"Max confidence ({max_confidence:.3f}) < threshold ({confidence_threshold})"
            else:
                decision_reason = f"Using 1024×512 (confidence: {max_confidence:.3f})"
        else:
            # No detections, try tiled
            use_tiled = True
            decision_reason = "No detections with 1024×512, trying tiled"
    
    # Run inference
    if use_tiled:
        print(f"Using tiled inference: {decision_reason}")
        results = run_tiled_inference(
            predictor_tiled,
            image_path,
            tile_size=1024,
            overlap=256,
            output_path=output_path,
            visualize=output_path is not None
        )
        method_used = "tiled"
    else:
        print(f"Using 1024×512 inference: {decision_reason}")
        # Use 1024×512 inference
        target_w = 1024
        target_h = int(h * target_w / w)
        image_resized = cv2.resize(image, (target_w, target_h))
        
        outputs = predictor_1024(image_resized)
        instances = outputs["instances"]
        
        # Convert to full resolution
        pred_masks = []
        pred_boxes = []
        pred_classes = []
        pred_scores = []
        
        for i in range(len(instances)):
            mask = instances.pred_masks[i].cpu().numpy()
            mask_resized = cv2.resize(mask.astype(np.uint8), (w, h), 
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
            pred_masks.append(mask_resized)
            
            box = instances.pred_boxes.tensor[i].cpu().numpy()
            scale_x = w / target_w
            scale_y = h / target_h
            box_scaled = box.copy()
            box_scaled[0] *= scale_x
            box_scaled[2] *= scale_x
            box_scaled[1] *= scale_y
            box_scaled[3] *= scale_y
            pred_boxes.append(box_scaled)
            
            pred_classes.append(instances.pred_classes[i].item())
            pred_scores.append(instances.scores[i].item())
        
        results = {
            "merged_masks": pred_masks,
            "merged_boxes": pred_boxes,
            "merged_FDI_labels": pred_classes,
            "merged_scores": pred_scores
        }
        method_used = "1024x512"
    
    results["method_used"] = method_used
    results["decision_reason"] = decision_reason
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Hybrid inference strategy')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--input', type=str, required=True, help='Input image path')
    parser.add_argument('--output', type=str, required=True, help='Output visualization path')
    parser.add_argument('--width-threshold', type=int, default=1500,
                       help='Width threshold for tiled inference (default: 1500)')
    parser.add_argument('--confidence-threshold', type=float, default=0.7,
                       help='Confidence threshold for tiled inference (default: 0.7)')
    
    args = parser.parse_args()
    
    # Setup predictor
    cfg = get_cfg()
    cfg.merge_from_file(args.config)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    
    predictor = DefaultPredictor(cfg)
    
    # Run hybrid inference
    results = hybrid_inference(
        predictor, predictor,
        args.input,
        width_threshold=args.width_threshold,
        confidence_threshold=args.confidence_threshold,
        output_path=args.output
    )
    
    print(f"\nInference complete!")
    print(f"  Method used: {results['method_used']}")
    print(f"  Reason: {results['decision_reason']}")
    print(f"  Detections: {len(results['merged_masks'])}")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()



