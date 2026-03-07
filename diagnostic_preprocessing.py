#!/usr/bin/env python3
"""
Diagnostic Script 3: Preprocessing Validation
==============================================
Validates that image preprocessing matches training pipeline.
Helps identify if BGR conversion, normalization, or resizing is corrupting input.

Usage:
    python diagnostic_preprocessing.py path/to/image.jpg --save-intermediates
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt


def validate_preprocessing(image_path, save_intermediates=False):
    """
    Validate preprocessing pipeline step-by-step.
    
    Args:
        image_path: Path to input image
        save_intermediates: If True, save intermediate processing steps
        
    Returns:
        Report dict with validation results
    """
    print("="*60)
    print("PREPROCESSING VALIDATION")
    print("="*60)
    
    # Step 1: Load image
    print("\n[1/5] Loading image...")
    image_rgb = cv2.imread(str(image_path))
    
    if image_rgb is None:
        print(f"❌ Failed to load image: {image_path}")
        return None
    
    print(f"✅ Image loaded successfully")
    print(f"   Shape: {image_rgb.shape}")
    print(f"   Dtype: {image_rgb.dtype}")
    print(f"   Color space: BGR (OpenCV default)")
    
    # Check if image is already grayscale
    if len(image_rgb.shape) == 2:
        print("⚠️  WARNING: Image is grayscale, not RGB!")
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2BGR)
    
    # Pixel statistics
    print(f"   Pixel range: [{image_rgb.min()}, {image_rgb.max()}]")
    print(f"   Mean: {image_rgb.mean():.2f}")
    print(f"   Std: {image_rgb.std():.2f}")
    
    # Check for suspicious pixel values
    if image_rgb.max() <= 1.0:
        print("❌ ERROR: Pixels in [0, 1] range! Model expects [0, 255]")
        print("   → Convert to uint8: (image * 255).astype(np.uint8)")
    
    if image_rgb.dtype != np.uint8:
        print(f"⚠️  WARNING: Dtype is {image_rgb.dtype}, not uint8")
    
    # Step 2: BGR channel order (Detectron2 expects BGR)
    print("\n[2/5] Verifying BGR channel order...")
    # OpenCV loads as BGR by default, so no conversion needed
    # But let's verify channels are not corrupted
    
    b_mean, g_mean, r_mean = image_rgb[:,:,0].mean(), image_rgb[:,:,1].mean(), image_rgb[:,:,2].mean()
    print(f"   Channel means: B={b_mean:.2f}, G={g_mean:.2f}, R={r_mean:.2f}")
    
    # For X-ray images, channels should be similar (grayscale-like)
    channel_diff = max(abs(b_mean - g_mean), abs(g_mean - r_mean), abs(r_mean - b_mean))
    if channel_diff > 10:
        print(f"⚠️  WARNING: Large channel difference ({channel_diff:.2f})")
        print("   X-rays are typically grayscale, channels should be similar")
    else:
        print(f"✅ Channels are balanced (diff={channel_diff:.2f})")
    
    # Step 3: Check for NaN or Inf
    print("\n[3/5] Checking for invalid values...")
    if np.isnan(image_rgb).any():
        print("❌ ERROR: NaN values detected!")
    elif np.isinf(image_rgb).any():
        print("❌ ERROR: Infinity values detected!")
    else:
        print("✅ No NaN or Inf values")
    
    # Step 4: Simulate Detectron2 resize
    print("\n[4/5] Simulating Detectron2 resize...")
    # Detectron2 uses min dimension scaling
    # For resnet50_9class_20k: min_size=800, max_size=1333
    
    h, w = image_rgb.shape[:2]
    min_size = 800
    max_size = 1333
    
    scale = min_size / min(h, w)
    if max(h, w) * scale > max_size:
        scale = max_size / max(h, w)
    
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    print(f"   Original size: {w}x{h}")
    print(f"   Scale factor: {scale:.3f}")
    print(f"   Resized size: {new_w}x{new_h}")
    
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    print(f"✅ Resize successful")
    
    # Step 5: Normalization (ImageNet stats)
    print("\n[5/5] Simulating ImageNet normalization...")
    # NOTE: Detectron2 subtracts pixel mean WITHOUT dividing by std
    # Training config: PIXEL_MEAN=[103.53, 116.28, 123.675], PIXEL_STD=[1.0, 1.0, 1.0]
    
    pixel_mean = np.array([103.53, 116.28, 123.675])
    pixel_std = np.array([1.0, 1.0, 1.0])
    
    normalized = (resized.astype(np.float32) - pixel_mean) / pixel_std
    
    print(f"   Pixel mean: {pixel_mean}")
    print(f"   Pixel std: {pixel_std}")
    print(f"   Normalized range: [{normalized.min():.2f}, {normalized.max():.2f}]")
    print(f"   Normalized mean: {normalized.mean():.2f}")
    
    # Check if normalization looks reasonable
    if normalized.max() > 200:
        print("⚠️  WARNING: Normalized values very high (> 200)")
    elif normalized.min() < -200:
        print("⚠️  WARNING: Normalized values very low (< -200)")
    else:
        print("✅ Normalization looks reasonable")
    
    # Save intermediate steps for visual inspection
    if save_intermediates:
        print("\n" + "="*60)
        print("SAVING INTERMEDIATE STEPS")
        print("="*60)
        
        output_dir = Path("preprocessing_debug")
        output_dir.mkdir(exist_ok=True)
        
        # 1. Original
        cv2.imwrite(str(output_dir / "01_original.jpg"), image_rgb)
        print(f"✅ Saved: {output_dir / '01_original.jpg'}")
        
        # 2. Channel visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(image_rgb[:,:,0], cmap='gray')
        axes[0].set_title(f"Blue Channel (mean={b_mean:.1f})")
        axes[0].axis('off')
        
        axes[1].imshow(image_rgb[:,:,1], cmap='gray')
        axes[1].set_title(f"Green Channel (mean={g_mean:.1f})")
        axes[1].axis('off')
        
        axes[2].imshow(image_rgb[:,:,2], cmap='gray')
        axes[2].set_title(f"Red Channel (mean={r_mean:.1f})")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "02_channels.png", dpi=100)
        plt.close()
        print(f"✅ Saved: {output_dir / '02_channels.png'}")
        
        # 3. Resized
        cv2.imwrite(str(output_dir / "03_resized.jpg"), resized)
        print(f"✅ Saved: {output_dir / '03_resized.jpg'}")
        
        # 4. Normalized (convert back to uint8 for visualization)
        normalized_vis = ((normalized - normalized.min()) / 
                         (normalized.max() - normalized.min()) * 255).astype(np.uint8)
        cv2.imwrite(str(output_dir / "04_normalized.jpg"), normalized_vis)
        print(f"✅ Saved: {output_dir / '04_normalized.jpg'}")
        
        # 5. Histogram comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original histogram
        ax = axes[0, 0]
        for i, color, name in zip([0, 1, 2], ['b', 'g', 'r'], ['Blue', 'Green', 'Red']):
            hist = cv2.calcHist([image_rgb], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, label=name, alpha=0.7)
        ax.set_title("Original Image Histogram")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Resized histogram
        ax = axes[0, 1]
        for i, color, name in zip([0, 1, 2], ['b', 'g', 'r'], ['Blue', 'Green', 'Red']):
            hist = cv2.calcHist([resized], [i], None, [256], [0, 256])
            ax.plot(hist, color=color, label=name, alpha=0.7)
        ax.set_title("Resized Image Histogram")
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Normalized histogram
        ax = axes[1, 0]
        for i, color, name in zip([0, 1, 2], ['b', 'g', 'r'], ['Blue', 'Green', 'Red']):
            channel_data = normalized[:,:,i].flatten()
            ax.hist(channel_data, bins=100, color=color, alpha=0.5, label=name)
        ax.set_title("Normalized Image Histogram")
        ax.set_xlabel("Normalized Value")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f"""
        PREPROCESSING SUMMARY
        {'='*30}
        
        Original:
          Size: {w}x{h}
          Pixel range: [{image_rgb.min()}, {image_rgb.max()}]
          Mean: {image_rgb.mean():.2f}
          Std: {image_rgb.std():.2f}
        
        Resized:
          Size: {new_w}x{new_h}
          Scale: {scale:.3f}
        
        Normalized:
          Mean: {normalized.mean():.2f}
          Std: {normalized.std():.2f}
          Range: [{normalized.min():.2f}, {normalized.max():.2f}]
        
        Channel Balance:
          B: {b_mean:.2f}
          G: {g_mean:.2f}
          R: {r_mean:.2f}
          Diff: {channel_diff:.2f}
        """
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(output_dir / "05_analysis.png", dpi=100)
        plt.close()
        print(f"✅ Saved: {output_dir / '05_analysis.png'}")
        
        print(f"\n📁 All intermediate files saved to: {output_dir}")
    
    # Build report
    report = {
        "image_path": str(image_path),
        "original_shape": image_rgb.shape,
        "original_dtype": str(image_rgb.dtype),
        "pixel_range": [int(image_rgb.min()), int(image_rgb.max())],
        "pixel_mean": float(image_rgb.mean()),
        "pixel_std": float(image_rgb.std()),
        "channel_means": {
            "B": float(b_mean),
            "G": float(g_mean),
            "R": float(r_mean)
        },
        "channel_balance_diff": float(channel_diff),
        "resize": {
            "original_size": [w, h],
            "scale": float(scale),
            "resized_size": [new_w, new_h]
        },
        "normalized": {
            "range": [float(normalized.min()), float(normalized.max())],
            "mean": float(normalized.mean()),
            "std": float(normalized.std())
        },
        "validation_status": "PASS"
    }
    
    # Determine validation status
    issues = []
    if image_rgb.max() <= 1.0:
        issues.append("Pixel range is [0,1] instead of [0,255]")
    if image_rgb.dtype != np.uint8:
        issues.append(f"Dtype is {image_rgb.dtype} instead of uint8")
    if channel_diff > 20:
        issues.append(f"Channels unbalanced (diff={channel_diff:.2f})")
    if np.isnan(image_rgb).any() or np.isinf(image_rgb).any():
        issues.append("Contains NaN or Inf values")
    
    if issues:
        report["validation_status"] = "FAIL"
        report["issues"] = issues
        
        print("\n" + "="*60)
        print("❌ VALIDATION FAILED")
        print("="*60)
        for issue in issues:
            print(f"  • {issue}")
    else:
        print("\n" + "="*60)
        print("✅ VALIDATION PASSED")
        print("="*60)
        print("Preprocessing pipeline is correct.")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate preprocessing pipeline")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("--save-intermediates", action="store_true",
                       help="Save intermediate processing steps for debugging")
    
    args = parser.parse_args()
    
    report = validate_preprocessing(args.image, args.save_intermediates)
    
    if report is None:
        sys.exit(1)
    
    # Save report
    import json
    output_path = Path("preprocessing_validation.json")
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to: {output_path}")
    
    if report["validation_status"] == "FAIL":
        sys.exit(1)


if __name__ == "__main__":
    main()
