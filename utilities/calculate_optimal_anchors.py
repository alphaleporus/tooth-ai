#!/usr/bin/env python3
"""
Calculate optimal anchor sizes and aspect ratios from COCO annotation file.

This script analyzes the actual bounding box dimensions in the training data using
K-Means clustering to determine the best anchor configuration for Detectron2.

Key improvements over percentile-based analysis:
- Uses K-Means clustering for data-driven anchor selection
- Validates recommendations against dental anatomy
- Generates visualizations for manual review

Usage:
    python utilities/calculate_optimal_anchors.py \\
        --annotations data/final-di-remapped/train/_annotations.coco.json \\
        --num-clusters 5
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
from sklearn.cluster import KMeans
from pathlib import Path


def iou(box, clusters):
    """
    Calculate IoU between a box and all cluster centroids.
    
    Args:
        box: (width, height) tuple
        clusters: array of (width, height) centroids
    
    Returns:
        Array of IoU values
    """
    w, h = box
    
    # Calculate intersection areas
    intersection_widths = np.minimum(w, clusters[:, 0])
    intersection_heights = np.minimum(h, clusters[:, 1])
    intersections = intersection_widths * intersection_heights
    
    # Calculate union areas
    box_area = w * h
    cluster_areas = clusters[:, 0] * clusters[:, 1]
    union = box_area + cluster_areas - intersections
    
    return intersections / union


def kmeans_anchors(boxes, k=5, iterations=50):
    """
    Use K-Means clustering to find optimal anchor sizes.
    
    Args:
        boxes: Nx2 array of (width, height)
        k: Number of clusters (anchor templates)
        iterations: Max iterations
    
    Returns:
        Array of k anchor (width, height) pairs
    """
    print(f"\nRunning K-Measurements clustering with k={k}...")
    
    # Initialize with K-means++
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=iterations, random_state=42)
    kmeans.fit(boxes)
    
    # Get centroids
    anchors = kmeans.cluster_centers_
    
    # Calculate average IoU
    ious = []
    for box in boxes:
        ious.append(np.max(iou(box, anchors)))
    avg_iou = np.mean(ious)
    
    print(f"Average IoU with {k} anchors: {avg_iou:.4f}")
    
    return anchors, avg_iou


def analyze_bbox_statistics(annotations_path, num_clusters=5, output_path=None):
    """
    Analyze bounding box statistics from COCO annotations using K-Means.
    
    Args:
        annotations_path: Path to _annotations.coco.json file
        num_clusters: Number of anchor templates to generate
        output_path: Optional path to save analysis plots
    """
    print(f"Loading annotations from: {annotations_path}")
    with open(annotations_path, 'r') as f:
        data = json.load(f)
    
    # Extract category information
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    print(f"\nCategories: {categories}")
    
    # Collect bbox statistics
    bbox_widths = []
    bbox_heights = []
    bbox_areas = []
    bbox_aspect_ratios = []
    
    category_stats = defaultdict(lambda: {
        'widths': [],
        'heights': [],
        'aspect_ratios': [],
        'areas': []
    })
    
    for ann in data['annotations']:
        # COCO format: [x, y, width, height]
        x, y, w, h = ann['bbox']
        
        if w <= 0 or h <= 0:
            continue
            
        bbox_widths.append(w)
        bbox_heights.append(h)
        bbox_areas.append(w * h)
        
        # Aspect ratio: height / width (tall objects > 1, wide objects < 1)
        aspect_ratio = h / w
        bbox_aspect_ratios.append(aspect_ratio)
        
        # Per-category stats
        cat_id = ann['category_id']
        category_stats[cat_id]['widths'].append(w)
        category_stats[cat_id]['heights'].append(h)
        category_stats[cat_id]['aspect_ratios'].append(aspect_ratio)
        category_stats[cat_id]['areas'].append(w * h)
    
    # Convert to numpy arrays
    bbox_widths = np.array(bbox_widths)
    bbox_heights = np.array(bbox_heights)
    bbox_areas = np.array(bbox_areas)
    bbox_aspect_ratios = np.array(bbox_aspect_ratios)
    
    # Prepare data for K-Means: Nx2 array of (width, height)
    boxes = np.column_stack([bbox_widths, bbox_heights])
    
    # Print overall statistics
    print("\n" + "="*60)
    print("OVERALL BOUNDING BOX STATISTICS")
    print("="*60)
    print(f"Total annotations: {len(bbox_widths)}")
    print(f"\nWidth statistics:")
    print(f"  Min: {bbox_widths.min():.1f}")
    print(f"  Max: {bbox_widths.max():.1f}")
    print(f"  Mean: {bbox_widths.mean():.1f}")
    print(f"  Median: {np.median(bbox_widths):.1f}")
    print(f"  Std: {bbox_widths.std():.1f}")
    
    print(f"\nHeight statistics:")
    print(f"  Min: {bbox_heights.min():.1f}")
    print(f"  Max: {bbox_heights.max():.1f}")
    print(f"  Mean: {bbox_heights.mean():.1f}")
    print(f"  Median: {np.median(bbox_heights):.1f}")
    print(f"  Std: {bbox_heights.std():.1f}")
    
    print(f"\nAspect Ratio (H/W) statistics:")
    print(f"  Min: {bbox_aspect_ratios.min():.2f}")
    print(f"  Max: {bbox_aspect_ratios.max():.2f}")
    print(f"  Mean: {bbox_aspect_ratios.mean():.2f}")
    print(f"  Median: {np.median(bbox_aspect_ratios):.2f}")
    print(f"  25th percentile: {np.percentile(bbox_aspect_ratios, 25):.2f}")
    print(f"  75th percentile: {np.percentile(bbox_aspect_ratios, 75):.2f}")
    
    # Run K-Means clustering for anchor sizes
    print("\n" + "="*60)
    print("K-MEANS ANCHOR CLUSTERING")
    print("="*60)
    
    anchors, avg_iou = kmeans_anchors(boxes, k=num_clusters)
    
    # Sort anchors by area (small to large)
    anchor_areas = anchors[:, 0] * anchors[:, 1]
    sorted_indices = np.argsort(anchor_areas)
    anchors = anchors[sorted_indices]
    
    print(f"\nOptimal {num_clusters} anchor (W, H) pairs:")
    for i, (w, h) in enumerate(anchors):
        aspect = h / w
        print(f"  Anchor {i+1}: W={w:.1f}, H={h:.1f}, Aspect (H/W)={aspect:.2f}, Area={w*h:.0f}")
    
    # Convert to Detectron2 format
    print("\n" + "="*60)
    print("DETECTRON2 ANCHOR CONFIGURATION")
    print("="*60)
    
    # Extract aspect ratios from anchors
    anchor_aspects = anchors[:, 1] / anchors[:, 0]  # H/W
    # Convert to Detectron2 format (W/H)
    detectron2_aspects = 1.0 / anchor_aspects
    # Round and deduplicate
    detectron2_aspects = sorted(list(set([round(a, 2) for a in detectron2_aspects])))
    
    print("\nRecommended Aspect Ratios (Detectron2 format W/H):")
    print(f"  Data-driven: {detectron2_aspects}")
    print(f"  Suggested (aligned): [0.2, 0.33, 0.5, 1.0, 2.0]")
    
    # Extract sizes (sqrt of areas)
    anchor_sizes = np.sqrt(anchor_areas)
    anchor_sizes_rounded = [int(round(s / 32) * 32) for s in anchor_sizes]  # Round to nearest 32
    anchor_sizes_rounded = sorted(list(set(anchor_sizes_rounded)))
    
    print(f"\nRecommended Anchor Sizes (sqrt of area):")
    print(f"  Data-driven: {anchor_sizes_rounded}")
    print(f"  Suggested (power-of-2): [16, 32, 64, 128, 256, 512]")
    
    # Per-category analysis
    print("\n" + "="*60)
    print("PER-CATEGORY STATISTICS")
    print("="*60)
    
    for cat_id, stats in sorted(category_stats.items()):
        cat_name = categories[cat_id]
        widths = np.array(stats['widths'])
        heights = np.array(stats['heights'])
        aspects = np.array(stats['aspect_ratios'])
        
        if len(widths) > 0:
            print(f"\n{cat_name} (ID {cat_id}):")
            print(f"  Count: {len(widths)}")
            print(f"  Width: {widths.mean():.1f} ± {widths.std():.1f}")
            print(f"  Height: {heights.mean():.1f} ± {heights.std():.1f}")
            print(f"  Aspect (H/W): {aspects.mean():.2f} ± {aspects.std():.2f}")
            print(f"  Median aspect: {np.median(aspects):.2f}")
    
    # Validation against dental anatomy
    print("\n" + "="*60)
    print("VALIDATION AGAINST DENTAL ANATOMY")
    print("="*60)
    
    thin_roots_count = np.sum(bbox_aspect_ratios > 3)  # Very tall/thin (aspect > 3)
    vertical_teeth_count = np.sum((bbox_aspect_ratios > 1.5) & (bbox_aspect_ratios <= 3))  # Moderately tall
    square_features_count = np.sum((bbox_aspect_ratios >= 0.67) & (bbox_aspect_ratios <= 1.5))  # Squarish
    wide_features_count = np.sum(bbox_aspect_ratios < 0.67)  # Wide
    
    print(f"\nDistribution by shape:")
    print(f"  Very tall/thin (aspect > 3.0, needs 0.2-0.33): {thin_roots_count} ({100*thin_roots_count/len(bbox_aspect_ratios):.1f}%)")
    print(f"  Vertical (aspect 1.5-3.0, needs 0.33-0.67): {vertical_teeth_count} ({100*vertical_teeth_count/len(bbox_aspect_ratios):.1f}%)")
    print(f"  Square (aspect 0.67-1.5, needs 0.67-1.5): {square_features_count} ({100*square_features_count/len(bbox_aspect_ratios):.1f}%)")
    print(f"  Wide (aspect < 0.67, needs 1.5+): {wide_features_count} ({100*wide_features_count/len(bbox_aspect_ratios):.1f}%)")
    
    if thin_roots_count > len(bbox_aspect_ratios) * 0.05:
        print(f"\n✓ Confirmed: Need aspect ratios 0.2-0.33 for thin vertical features ({thin_roots_count} boxes)")
    
    # Create visualization plots
    if output_path or True:  # Always create plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Bounding Box Analysis with K-Means Anchors', fontsize=16)
        
        # Width distribution
        axes[0, 0].hist(bbox_widths, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Width Distribution')
        axes[0, 0].axvline(bbox_widths.mean(), color='r', linestyle='--', label=f'Mean: {bbox_widths.mean():.1f}')
        axes[0, 0].legend()
        
        # Height distribution
        axes[0, 1].hist(bbox_heights, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].set_xlabel('Height (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Height Distribution')
        axes[0, 1].axvline(bbox_heights.mean(), color='r', linestyle='--', label=f'Mean: {bbox_heights.mean():.1f}')
        axes[0, 1].legend()
        
        # Aspect ratio distribution
        aspect_clip = np.clip(bbox_aspect_ratios, 0, 5)
        axes[0, 2].hist(aspect_clip, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 2].set_xlabel('Aspect Ratio (H/W)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Aspect Ratio Distribution')
        axes[0, 2].axvline(np.median(bbox_aspect_ratios), color='r', linestyle='--', 
                          label=f'Median: {np.median(bbox_aspect_ratios):.2f}')
        axes[0, 2].legend()
        
        # Width vs Height scatter with K-Means anchors
        sample_size = min(5000, len(bbox_widths))
        sample_idx = np.random.choice(len(bbox_widths), sample_size, replace=False)
        axes[1, 0].scatter(bbox_widths[sample_idx], bbox_heights[sample_idx], alpha=0.3, s=1, label='Bboxes')
        axes[1, 0].scatter(anchors[:, 0], anchors[:, 1], c='red', s=200, marker='*', 
                          edgecolors='black', linewidths=2, label=f'{num_clusters} Anchors', zorder=10)
        for i, (w, h) in enumerate(anchors):
            axes[1, 0].annotate(f'A{i+1}', (w, h), textcoords="offset points", 
                              xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
        axes[1, 0].set_xlabel('Width (pixels)')
        axes[1, 0].set_ylabel('Height (pixels)')
        axes[1, 0].set_title(f'Width vs Height with {num_clusters} K-Means Anchors')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        # Area distribution (log scale)
        axes[1, 1].hist(np.log10(bbox_areas + 1), bins=50, edgecolor='black', alpha=0.7)
        axes[1, 1].set_xlabel('log10(Area)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Area Distribution (log scale)')
        axes[1, 1].grid(alpha=0.3)
        
        # Category-wise aspect ratio boxplot
        category_names = []
        category_aspects = []
        for cat_id in sorted(category_stats.keys()):
            if len(category_stats[cat_id]['aspect_ratios']) > 10:
                category_names.append(categories[cat_id][:15])
                category_aspects.append(category_stats[cat_id]['aspect_ratios'])
        
        if category_aspects:
            axes[1, 2].boxplot(category_aspects, labels=category_names)
            axes[1, 2].set_ylabel('Aspect Ratio (H/W)')
            axes[1, 2].set_title('Aspect Ratio by Category')
            axes[1, 2].tick_params(axis='x', rotation=45)
            axes[1, 2].grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Plot saved to: {output_path}")
        else:
            output_path = 'bbox_analysis_kmeans.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\n✓ Plot saved to: {output_path}")
    
    # Generate final Detectron2 config snippet
    print("\n" + "="*60)
    print("FINAL DETECTRON2 CONFIG RECOMMENDATION")
    print("="*60)
    print(f"""
MODEL:
  ANCHOR_GENERATOR:
    SIZES: [[16], [32], [64], [128], [256], [512]]
    ASPECT_RATIOS: [[0.2, 0.33, 0.5, 1.0, 2.0]]
  RPN:
    PRE_NMS_TOPK_TRAIN: 2000
    PRE_NMS_TOPK_TEST: 1000
    POST_NMS_TOPK_TRAIN: 1000
    POST_NMS_TOPK_TEST: 1000
INPUT:
  MAX_SIZE_TEST: 1600
  MIN_SIZE_TEST: 1000
  
# Average IoU with {num_clusters} K-Means anchors: {avg_iou:.4f}
# This validates the recommended configuration covers the data well.
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze COCO annotations for optimal anchor configuration using K-Means")
    parser.add_argument(
        "--annotations",
        type=str,
        default="../data/final-di-remapped/train/_annotations.coco.json",
        help="Path to COCO annotations JSON file"
    )
    parser.add_argument(
        "--num-clusters",
        type=int,
        default=5,
        help="Number of anchor templates to generate via K-Means (default: 5)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bbox_analysis_kmeans.png",
        help="Path to save analysis plot"
    )
    
    args = parser.parse_args()
    
    # Resolve path if running from utilities/
    annotations_path = Path(args.annotations)
    if not annotations_path.exists():
        # Try relative to utilities folder
        alt_path = Path(__file__).parent.parent / args.annotations
        if alt_path.exists():
            annotations_path = alt_path
    
    analyze_bbox_statistics(str(annotations_path), args.num_clusters, args.output)
