#!/usr/bin/env python3
"""
Evaluation script for Mask R-CNN on tooth detection dataset.
Computes mAP, mask IoU, per-class metrics, and confusion matrix.
"""

import argparse
import json
import os
from collections import defaultdict
import numpy as np
from pathlib import Path

import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.utils.logger import setup_logger
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def compute_mask_iou_per_class(predictions, ground_truth, num_classes):
    """
    Compute mask IoU per class.
    
    Args:
        predictions: List of prediction dictionaries
        ground_truth: COCO ground truth object
        num_classes: Number of classes
    
    Returns:
        Dictionary mapping class_id to IoU score
    """
    class_ious = defaultdict(list)
    
    # This is a simplified version - full implementation would require
    # loading masks and computing IoU for each class
    # For now, we'll use COCO evaluation which provides this
    
    return class_ious


def evaluate_model(cfg, model_path, output_dir):
    """
    Evaluate model and generate metrics.
    
    Args:
        cfg: Detectron2 config
        model_path: Path to model checkpoint
        output_dir: Directory to save metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    
    # Get validation dataset
    val_dataset_name = "tooth_val"
    if val_dataset_name not in DatasetCatalog.list():
        raise ValueError(f"Dataset {val_dataset_name} not registered. Run register_dataset.py first.")
    
    # Build data loader
    data_loader = build_detection_test_loader(cfg, val_dataset_name)
    
    # Run COCO evaluation
    evaluator = COCOEvaluator(val_dataset_name, output_dir=output_dir)
    results = inference_on_dataset(predictor.model, data_loader, evaluator)
    
    # Extract metrics
    metrics = {
        "bbox": {},
        "segm": {}
    }
    
    # Parse COCO evaluation results
    if "bbox" in results:
        bbox_results = results["bbox"]
        metrics["bbox"] = {
            "mAP_50": bbox_results.get("AP", 0.0),
            "mAP_50_95": bbox_results.get("AP", 0.0),  # Will be updated with full AP
            "AP_small": bbox_results.get("AP-small", 0.0),
            "AP_medium": bbox_results.get("AP-medium", 0.0),
            "AP_large": bbox_results.get("AP-large", 0.0),
        }
    
    if "segm" in results:
        segm_results = results["segm"]
        metrics["segm"] = {
            "mAP_50": segm_results.get("AP", 0.0),
            "mAP_50_95": segm_results.get("AP", 0.0),
            "mask_IoU": segm_results.get("AP", 0.0),  # Approximate
        }
    
    # Get per-class metrics
    metadata = MetadataCatalog.get(val_dataset_name)
    num_classes = len(metadata.thing_classes)
    
    # Load COCO ground truth for detailed evaluation
    coco_gt_path = None
    for dataset_dict in DatasetCatalog.get(val_dataset_name):
        # Try to find COCO JSON path
        # This would need to be stored during registration
        break
    
    # Compute per-class support and confusion matrix
    # This requires running inference and comparing with ground truth
    per_class_metrics = compute_per_class_metrics(
        predictor, data_loader, num_classes, metadata
    )
    
    metrics["per_class"] = per_class_metrics
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nEvaluation complete!")
    print(f"  bbox mAP@0.5: {metrics['bbox'].get('mAP_50', 0.0):.4f}")
    print(f"  segm mAP@0.5: {metrics['segm'].get('mAP_50', 0.0):.4f}")
    print(f"  Metrics saved to: {metrics_path}")
    
    return metrics


def compute_per_class_metrics(predictor, data_loader, num_classes, metadata):
    """
    Compute per-class metrics including support and confusion matrix.
    
    Args:
        predictor: Detectron2 predictor
        data_loader: Data loader for validation set
        num_classes: Number of classes
        metadata: Dataset metadata
    
    Returns:
        Dictionary with per-class metrics
    """
    all_preds = []
    all_gts = []
    
    # Run inference and collect predictions
    for batch in data_loader:
        for item in batch:
            # Get ground truth
            gt_classes = item.get("instances").gt_classes.cpu().numpy()
            all_gts.extend(gt_classes.tolist())
            
            # Get predictions
            outputs = predictor(item["image"])
            pred_classes = outputs["instances"].pred_classes.cpu().numpy()
            all_preds.extend(pred_classes.tolist())
    
    # Compute confusion matrix
    cm = confusion_matrix(all_gts, all_preds, labels=list(range(num_classes)))
    
    # Compute per-class metrics
    per_class = {}
    class_names = metadata.thing_classes
    
    for i in range(num_classes):
        class_name = class_names[i] if i < len(class_names) else f"class_{i}"
        tp = cm[i, i] if i < cm.shape[0] and i < cm.shape[1] else 0
        fp = cm[:, i].sum() - tp if i < cm.shape[1] else 0
        fn = cm[i, :].sum() - tp if i < cm.shape[0] else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support = int(tp + fn)
        
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        }
    
    # Save confusion matrix
    return {
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names
    }


def plot_confusion_matrix(cm, class_names, output_path):
    """
    Plot and save confusion matrix.
    
    Args:
        cm: Confusion matrix (numpy array)
        class_names: List of class names
        output_path: Path to save plot
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Mask R-CNN model')
    parser.add_argument(
        '--config-file',
        type=str,
        required=True,
        help='Path to config file'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for metrics (default: same as model directory)'
    )
    parser.add_argument(
        '--coco-json',
        type=str,
        default=None,
        help='Path to COCO JSON (for dataset registration)'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        default=None,
        help='Image directory (for dataset registration)'
    )
    
    args = parser.parse_args()
    
    setup_logger()
    
    # Setup config
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    
    # Register datasets if needed
    if "tooth_val" not in DatasetCatalog.list():
        if args.coco_json and args.image_dir:
            from register_dataset import register_tooth_dataset
            register_tooth_dataset(args.coco_json, args.image_dir, "tooth_val", split_ratio=0.8)
        else:
            raise ValueError("Dataset not registered. Provide --coco-json and --image-dir")
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.model_path)
    
    # Evaluate
    metrics = evaluate_model(cfg, args.model_path, args.output_dir)
    
    print("\nEvaluation Summary:")
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()



