#!/usr/bin/env python3
"""
Dataset Statistical Comparison
================================
Compare training dataset vs new failing dataset to identify domain shift.

Usage:
    python dataset_comparison.py --train data/train --new data/new_failing
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm


def analyze_image(image_path):
    """Extract statistics from a single image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    
    h, w = img.shape[:2]
    
    # Convert to grayscale for X-ray analysis
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    
    stats = {
        "width": w,
        "height": h,
        "aspect_ratio": w / h,
        "mean_intensity": gray.mean(),
        "std_intensity": gray.std(),
        "min_intensity": gray.min(),
        "max_intensity": gray.max(),
        "median_intensity": np.median(gray),
        "histogram": cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    }
    
    return stats


def analyze_dataset(dataset_path, sample_size=None):
    """Analyze all images in a dataset."""
    image_paths = []
    
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_paths.extend(Path(dataset_path).rglob(ext))
    
    if not image_paths:
        return None
    
    print(f"Found {len(image_paths)} images in {dataset_path}")
    
    if sample_size and len(image_paths) > sample_size:
        import random
        image_paths = random.sample(image_paths, sample_size)
        print(f"Sampling {sample_size} images")
    
    stats_list = []
    
    for img_path in tqdm(image_paths, desc="Analyzing images"):
        stats = analyze_image(img_path)
        if stats:
            stats_list.append(stats)
    
    if not stats_list:
        return None
    
    # Aggregate statistics
    aggregate = {
        "count": len(stats_list),
        "width": {
            "mean": np.mean([s["width"] for s in stats_list]),
            "std": np.std([s["width"] for s in stats_list]),
            "min": np.min([s["width"] for s in stats_list]),
            "max": np.max([s["width"] for s in stats_list])
        },
        "height": {
            "mean": np.mean([s["height"] for s in stats_list]),
            "std": np.std([s["height"] for s in stats_list]),
            "min": np.min([s["height"] for s in stats_list]),
            "max": np.max([s["height"] for s in stats_list])
        },
        "aspect_ratio": {
            "mean": np.mean([s["aspect_ratio"] for s in stats_list]),
            "std": np.std([s["aspect_ratio"] for s in stats_list])
        },
        "intensity": {
            "mean": np.mean([s["mean_intensity"] for s in stats_list]),
            "std": np.mean([s["std_intensity"] for s in stats_list]),
            "global_min": np.min([s["min_intensity"] for s in stats_list]),
            "global_max": np.max([s["max_intensity"] for s in stats_list])
        },
        "histogram_mean": np.mean([s["histogram"] for s in stats_list], axis=0)
    }
    
    return aggregate, stats_list


def compare_datasets(train_path, new_path, sample_size=100):
    """Compare two datasets and visualize differences."""
    print("="*70)
    print("DATASET COMPARISON ANALYSIS")
    print("="*70)
    
    print("\n[1/2] Analyzing training dataset...")
    train_stats, train_raw = analyze_dataset(train_path, sample_size)
    
    if not train_stats:
        print("❌ No images found in training dataset")
        return None
    
    print("\n[2/2] Analyzing new dataset...")
    new_stats, new_raw = analyze_dataset(new_path, sample_size)
    
    if not new_stats:
        print("❌ No images found in new dataset")
        return None
    
    # Comparison report
    print("\n" + "="*70)
    print("COMPARISON REPORT")
    print("="*70)
    
    print(f"\nDataset Sizes:")
    print(f"  Training: {train_stats['count']} images")
    print(f"  New: {new_stats['count']} images")
    
    print(f"\nImage Dimensions:")
    print(f"  Training: {train_stats['width']['mean']:.0f}x{train_stats['height']['mean']:.0f} "
          f"(±{train_stats['width']['std']:.0f}x{train_stats['height']['std']:.0f})")
    print(f"  New:      {new_stats['width']['mean']:.0f}x{new_stats['height']['mean']:.0f} "
          f"(±{new_stats['width']['std']:.0f}x{new_stats['height']['std']:.0f})")
    
    width_diff = abs(train_stats['width']['mean'] - new_stats['width']['mean'])
    if width_diff > 200:
        print(f"  ⚠️  WARNING: Large resolution difference ({width_diff:.0f}px)")
    
    print(f"\nAspect Ratio:")
    print(f"  Training: {train_stats['aspect_ratio']['mean']:.3f}")
    print(f"  New:      {new_stats['aspect_ratio']['mean']:.3f}")
    
    print(f"\nIntensity Statistics:")
    print(f"  Training: mean={train_stats['intensity']['mean']:.1f}, "
          f"std={train_stats['intensity']['std']:.1f}")
    print(f"  New:      mean={new_stats['intensity']['mean']:.1f}, "
          f"std={new_stats['intensity']['std']:.1f}")
    
    intensity_diff = abs(train_stats['intensity']['mean'] - new_stats['intensity']['mean'])
    if intensity_diff > 20:
        print(f"  ⚠️  WARNING: Large intensity difference ({intensity_diff:.1f})")
        if new_stats['intensity']['mean'] > train_stats['intensity']['mean']:
            print(f"     → New images are BRIGHTER")
        else:
            print(f"     → New images are DARKER")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Dataset Comparison", fontsize=16, fontweight='bold')
    
    # 1. Resolution distribution
    ax = axes[0, 0]
    train_widths = [s["width"] for s in train_raw]
    new_widths = [s["width"] for s in new_raw]
    ax.hist(train_widths, bins=30, alpha=0.5, label='Training', color='blue')
    ax.hist(new_widths, bins=30, alpha=0.5, label='New', color='red')
    ax.set_xlabel("Image Width (px)")
    ax.set_ylabel("Frequency")
    ax.set_title("Width Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Intensity distribution
    ax = axes[0, 1]
    train_intensities = [s["mean_intensity"] for s in train_raw]
    new_intensities = [s["mean_intensity"] for s in new_raw]
    ax.hist(train_intensities, bins=30, alpha=0.5, label='Training', color='blue')
    ax.hist(new_intensities, bins=30, alpha=0.5, label='New', color='red')
    ax.set_xlabel("Mean Intensity")
    ax.set_ylabel("Frequency")
    ax.set_title("Intensity Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Histogram comparison
    ax = axes[0, 2]
    ax.plot(train_stats['histogram_mean'], label='Training', color='blue', linewidth=2)
    ax.plot(new_stats['histogram_mean'], label='New', color='red', linewidth=2)
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Average Frequency")
    ax.set_title("Average Histogram")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Aspect ratio
    ax = axes[1, 0]
    train_ratios = [s["aspect_ratio"] for s in train_raw]
    new_ratios = [s["aspect_ratio"] for s in new_raw]
    ax.hist(train_ratios, bins=30, alpha=0.5, label='Training', color='blue')
    ax.hist(new_ratios, bins=30, alpha=0.5, label='New', color='red')
    ax.set_xlabel("Aspect Ratio (W/H)")
    ax.set_ylabel("Frequency")
    ax.set_title("Aspect Ratio Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Contrast (std dev)
    ax = axes[1, 1]
    train_stds = [s["std_intensity"] for s in train_raw]
    new_stds = [s["std_intensity"] for s in new_raw]
    ax.hist(train_stds, bins=30, alpha=0.5, label='Training', color='blue')
    ax.hist(new_stds, bins=30, alpha=0.5, label='New', color='red')
    ax.set_xlabel("Intensity Std Dev (Contrast)")
    ax.set_ylabel("Frequency")
    ax.set_title("Contrast Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')
    
    summary = f"""
    DOMAIN SHIFT ANALYSIS
    {'='*30}
    
    Resolution:
      Δ Width: {width_diff:.0f}px
      Δ Height: {abs(train_stats['height']['mean'] - new_stats['height']['mean']):.0f}px
    
    Intensity:
      Δ Mean: {intensity_diff:.1f}
      Training: {train_stats['intensity']['mean']:.1f}
      New: {new_stats['intensity']['mean']:.1f}
    
    Contrast:
      Δ Std: {abs(train_stats['intensity']['std'] - new_stats['intensity']['std']):.1f}
    
    Assessment:
    """
    
    # Assess domain shift severity
    issues = []
    if width_diff > 200:
        issues.append("• Large resolution mismatch")
    if intensity_diff > 30:
        issues.append("• Significant brightness shift")
    if abs(train_stats['intensity']['std'] - new_stats['intensity']['std']) > 15:
        issues.append("• Contrast mismatch")
    
    if not issues:
        summary += "\n  ✅ Minimal domain shift"
        summary += "\n  → Issue is likely threshold"
    elif len(issues) == 1:
        summary += "\n  ⚠️  Moderate domain shift"
        summary += "\n" + "\n".join(f"  {i}" for i in issues)
        summary += "\n\n  → Try different model"
    else:
        summary += "\n  ❌ Severe domain shift"
        summary += "\n" + "\n".join(f"  {i}" for i in issues)
        summary += "\n\n  → Fine-tuning required"
    
    ax.text(0.1, 0.9, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig("dataset_comparison.png", dpi=150, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: dataset_comparison.png")
    
    return {
        "train": train_stats,
        "new": new_stats,
        "differences": {
            "resolution_diff": float(width_diff),
            "intensity_diff": float(intensity_diff),
            "severity": "severe" if len(issues) > 1 else ("moderate" if issues else "minimal")
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Compare training vs new dataset")
    parser.add_argument("--train", type=str, required=True, 
                       help="Path to training dataset directory")
    parser.add_argument("--new", type=str, required=True,
                       help="Path to new (failing) dataset directory")
    parser.add_argument("--sample", type=int, default=100,
                       help="Number of images to sample from each dataset")
    
    args = parser.parse_args()
    
    result = compare_datasets(args.train, args.new, args.sample)
    
    if result is None:
        sys.exit(1)
    
    # Save report
    import json
    output_path = Path("dataset_comparison.json")
    
    # Remove non-serializable data (histograms)
    result_copy = result.copy()
    if 'histogram_mean' in result_copy.get('train', {}):
        del result_copy['train']['histogram_mean']
    if 'histogram_mean' in result_copy.get('new', {}):
        del result_copy['new']['histogram_mean']
    
    with open(output_path, 'w') as f:
        json.dump(result_copy, f, indent=2)
    
    print(f"📄 Report saved to: {output_path}")


if __name__ == "__main__":
    main()
