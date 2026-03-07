#!/usr/bin/env python3
"""
Calculate optimal anchor sizes and aspect ratios from COCO annotation file.
This analyzes the actual bounding box dimensions in the training data to determine
the best anchor configuration for Detectron2.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse


def analyze_bbox_statistics(annotations_path, output_path=None):
    """
    Analyze bounding box statistics from COCO annotations.
    
    Args:
        annotations_path: Path to _annotations.coco.json file
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
    
    print(f"\nArea statistics:")
    print(f"  Min: {bbox_areas.min():.1f}")
    print(f"  Max: {bbox_areas.max():.1f}")
    print(f"  Mean: {bbox_areas.mean():.1f}")
    print(f"  Median: {np.median(bbox_areas):.1f}")
    
    print(f"\nAspect Ratio (H/W) statistics:")
    print(f"  Min: {bbox_aspect_ratios.min():.2f}")
    print(f"  Max: {bbox_aspect_ratios.max():.2f}")
    print(f"  Mean: {bbox_aspect_ratios.mean():.2f}")
    print(f"  Median: {np.median(bbox_aspect_ratios):.2f}")
    print(f"  25th percentile: {np.percentile(bbox_aspect_ratios, 25):.2f}")
    print(f"  75th percentile: {np.percentile(bbox_aspect_ratios, 75):.2f}")
    
    # Calculate recommended anchor sizes using k-means clustering
    print("\n" + "="*60)
    print("RECOMMENDED ANCHOR CONFIGURATION")
    print("="*60)
    
    # Recommended aspect ratios based on percentiles
    aspect_ratio_percentiles = [5, 15, 35, 50, 75, 90]
    recommended_aspects = [np.percentile(bbox_aspect_ratios, p) for p in aspect_ratio_percentiles]
    
    print("\nRecommended Aspect Ratios (H/W):")
    print("  Based on percentiles:", [f"{ar:.2f}" for ar in recommended_aspects])
    
    # Convert to Detectron2 format (W/H)
    detectron2_aspects = [1.0 / ar if ar > 0 else 1.0 for ar in recommended_aspects]
    # Remove duplicates and sort
    detectron2_aspects = sorted(list(set([round(a, 2) for a in detectron2_aspects])))
    
    print(f"  Detectron2 format (W/H): {detectron2_aspects}")
    print(f"  Suggested: {[0.2, 0.33, 0.5, 1.0, 2.0]}")
    
    # Recommended anchor sizes based on area sqrt
    sqrt_areas = np.sqrt(bbox_areas)
    size_percentiles = [10, 25, 40, 55, 70, 85]
    recommended_sizes = [np.percentile(sqrt_areas, p) for p in size_percentiles]
    recommended_sizes = [int(round(s)) for s in recommended_sizes]
    
    print(f"\nRecommended Anchor Sizes (based on sqrt(area)):")
    print(f"  {recommended_sizes}")
    print(f"  Suggested power-of-2: {[16, 32, 64, 128, 256, 512]}")
    
    # Per-category analysis
    print("\n" + "="*60)
    print("PER-CATEGORY STATISTICS")
    print("="*60)
    
    for cat_id, stats in sorted(category_stats.items()):
        cat_name = categories[cat_id]
        widths = np.array(stats['widths'])
        heights = np.array(stats['heights'])
        aspects = np.array(stats['aspect_ratios'])
        
        print(f"\n{cat_name} (ID {cat_id}):")
        print(f"  Count: {len(widths)}")
        print(f"  Width: {widths.mean():.1f} ± {widths.std():.1f}")
        print(f"  Height: {heights.mean():.1f} ± {heights.std():.1f}")
        print(f"  Aspect (H/W): {aspects.mean():.2f} ± {aspects.std():.2f}")
        print(f"  Median aspect: {np.median(aspects):.2f}")
    
    # Create visualization plots
    if output_path or True:  # Always create plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Bounding Box Analysis', fontsize=16)
        
        # Width distribution
        axes[0, 0].hist(bbox_widths, bins=50, edgecolor='black')
        axes[0, 0].set_xlabel('Width (pixels)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Width Distribution')
        axes[0, 0].axvline(bbox_widths.mean(), color='r', linestyle='--', label=f'Mean: {bbox_widths.mean():.1f}')
        axes[0, 0].legend()
        
        # Height distribution
        axes[0, 1].hist(bbox_heights, bins=50, edgecolor='black')
        axes[0, 1].set_xlabel('Height (pixels)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Height Distribution')
        axes[0, 1].axvline(bbox_heights.mean(), color='r', linestyle='--', label=f'Mean: {bbox_heights.mean():.1f}')
        axes[0, 1].legend()
        
        # Aspect ratio distribution
        # Clip extreme outliers for better visualization
        aspect_clip = np.clip(bbox_aspect_ratios, 0, 5)
        axes[0, 2].hist(aspect_clip, bins=50, edgecolor='black')
        axes[0, 2].set_xlabel('Aspect Ratio (H/W)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Aspect Ratio Distribution')
        axes[0, 2].axvline(np.median(bbox_aspect_ratios), color='r', linestyle='--', 
                          label=f'Median: {np.median(bbox_aspect_ratios):.2f}')
        axes[0, 2].legend()
        
        # Area distribution (log scale)
        axes[1, 0].hist(np.log10(bbox_areas + 1), bins=50, edgecolor='black')
        axes[1, 0].set_xlabel('log10(Area)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Area Distribution (log scale)')
        
        # Width vs Height scatter (sample for performance)
        sample_size = min(5000, len(bbox_widths))
        sample_idx = np.random.choice(len(bbox_widths), sample_size, replace=False)
        axes[1, 1].scatter(bbox_widths[sample_idx], bbox_heights[sample_idx], alpha=0.3, s=1)
        axes[1, 1].set_xlabel('Width (pixels)')
        axes[1, 1].set_ylabel('Height (pixels)')
        axes[1, 1].set_title('Width vs Height')
        axes[1, 1].plot([0, bbox_widths.max()], [0, bbox_widths.max()], 'r--', alpha=0.5, label='y=x')
        axes[1, 1].legend()
        
        # Category-wise aspect ratio boxplot
        category_names = []
        category_aspects = []
        for cat_id in sorted(category_stats.keys()):
            if len(category_stats[cat_id]['aspect_ratios']) > 10:  # Only if enough samples
                category_names.append(categories[cat_id][:15])  # Truncate long names
                category_aspects.append(category_stats[cat_id]['aspect_ratios'])
        
        if category_aspects:
            axes[1, 2].boxplot(category_aspects, labels=category_names)
            axes[1, 2].set_ylabel('Aspect Ratio (H/W)')
            axes[1, 2].set_title('Aspect Ratio by Category')
            axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {output_path}")
        else:
            output_path = 'bbox_analysis.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {output_path}")
    
    # Generate Detectron2 config snippet
    print("\n" + "="*60)
    print("DETECTRON2 CONFIG SNIPPET")
    print("="*60)
    print("""
MODEL:
  ANCHOR_GENERATOR:
    SIZES: [[16], [32], [64], [128], [256], [512]]
    ASPECT_RATIOS: [[0.2, 0.33, 0.5, 1.0, 2.0]]
    ANGLES: [[-90, 0, 90]]  # For rotated teeth
  RPN:
    PRE_NMS_TOPK_TRAIN: 2000
    PRE_NMS_TOPK_TEST: 1000
    POST_NMS_TOPK_TRAIN: 1000
    POST_NMS_TOPK_TEST: 1000
INPUT:
  MAX_SIZE_TEST: 1600
  MIN_SIZE_TEST: 1000
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze COCO annotations for optimal anchor configuration")
    parser.add_argument(
        "--annotations",
        type=str,
        default="data/final-di-remapped/train/_annotations.coco.json",
        help="Path to COCO annotations JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bbox_analysis.png",
        help="Path to save analysis plot"
    )
    
    args = parser.parse_args()
    analyze_bbox_statistics(args.annotations, args.output)
