#!/usr/bin/env python3
"""
Stress testing script for tooth detection pipeline.
Applies various distortions and measures performance degradation.
"""

import argparse
import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple
import random
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.engine import ToothDetectionEngine, load_engine
from inference.visualize import visualize_predictions, save_visualization


def apply_gaussian_noise(image: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    """Apply Gaussian noise to image."""
    noise = np.random.normal(0, sigma * 255, image.shape).astype(np.float32)
    noisy = image.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_motion_blur(image: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    """Apply motion blur to image."""
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(image, -1, kernel)


def apply_gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply gamma correction to image."""
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def apply_jpeg_compression(image: np.ndarray, quality: int = 20) -> np.ndarray:
    """Apply JPEG compression to image."""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg


def apply_occlusion(image: np.ndarray, patch_size: Tuple[int, int] = (50, 50)) -> np.ndarray:
    """Apply random occlusion patch."""
    h, w = image.shape[:2]
    x = random.randint(0, w - patch_size[0])
    y = random.randint(0, h - patch_size[1])
    
    occluded = image.copy()
    occluded[y:y+patch_size[1], x:x+patch_size[0]] = 0
    return occluded


def apply_horizontal_flip(image: np.ndarray) -> np.ndarray:
    """Apply horizontal flip."""
    return cv2.flip(image, 1)


def apply_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    """Apply rotation to image."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    return rotated


def compute_baseline_metrics(engine: ToothDetectionEngine, image_path: str) -> Dict:
    """Compute baseline metrics for an image."""
    results = engine.predict(image_path, return_visualization=False)
    
    return {
        'num_detections': results['num_detections'],
        'mean_confidence': np.mean([t['score'] for t in results['teeth']]) if results['teeth'] else 0.0,
        'min_confidence': np.min([t['score'] for t in results['teeth']]) if results['teeth'] else 0.0,
        'max_confidence': np.max([t['score'] for t in results['teeth']]) if results['teeth'] else 0.0
    }


def stress_test_image(engine: ToothDetectionEngine, image_path: str,
                      distortion_name: str, distorted_image: np.ndarray,
                      baseline_metrics: Dict, output_dir: str) -> Dict:
    """
    Run stress test on a distorted image.
    
    Returns:
        Dictionary with stress test results
    """
    # Save distorted image temporarily
    temp_path = os.path.join(output_dir, f'temp_{distortion_name}.png')
    cv2.imwrite(temp_path, distorted_image)
    
    try:
        # Run inference on distorted image
        results = engine.predict(temp_path, return_visualization=False)
        
        # Compute metrics
        num_detections = results['num_detections']
        mean_confidence = np.mean([t['score'] for t in results['teeth']]) if results['teeth'] else 0.0
        
        # Compute degradation
        detection_drop = baseline_metrics['num_detections'] - num_detections
        confidence_drop = baseline_metrics['mean_confidence'] - mean_confidence
        
        # Create visualization
        vis_image = visualize_predictions(distorted_image, [
            {
                'fdi': t['fdi'],
                'bbox': t['bbox'],
                'mask': t.get('mask'),
                'centroid': t['centroid'],
                'final_confidence': t['score'],
                'method_used': t['method_used'],
                'correction_applied': t['correction_applied']
            } for t in results['teeth']
        ])
        
        # Save visualization
        vis_dir = os.path.join(output_dir, distortion_name)
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, os.path.basename(image_path))
        save_visualization(vis_image, vis_path)
        
        return {
            'distortion': distortion_name,
            'baseline_detections': baseline_metrics['num_detections'],
            'distorted_detections': num_detections,
            'detection_drop': int(detection_drop),
            'baseline_confidence': float(baseline_metrics['mean_confidence']),
            'distorted_confidence': float(mean_confidence),
            'confidence_drop': float(confidence_drop),
            'visualization': vis_path
        }
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def run_stress_tests(images_dir: str, engine: ToothDetectionEngine,
                    output_dir: str, num_samples: int = 10, seed: int = 42):
    """
    Run stress tests with various distortions.
    
    Args:
        images_dir: Directory containing images
        engine: Inference engine
        output_dir: Output directory
        num_samples: Number of images to test
        seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Find images
    image_patterns = [
        os.path.join(images_dir, '**', '*.png'),
        os.path.join(images_dir, '**', '*.jpg'),
        os.path.join(images_dir, '**', '*.jpeg')
    ]
    
    all_images = []
    for pattern in image_patterns:
        all_images.extend(glob.glob(pattern, recursive=True))
    
    if len(all_images) > num_samples:
        test_images = random.sample(all_images, num_samples)
    else:
        test_images = all_images
    
    print(f"Running stress tests on {len(test_images)} images...")
    
    # Define distortions
    distortions = {
        'gaussian_noise': lambda img: apply_gaussian_noise(img, sigma=0.03),
        'motion_blur': lambda img: apply_motion_blur(img, kernel_size=7),
        'gamma_low': lambda img: apply_gamma_correction(img, gamma=0.6),
        'gamma_high': lambda img: apply_gamma_correction(img, gamma=1.6),
        'jpeg_compression': lambda img: apply_jpeg_compression(img, quality=20),
        'occlusion': lambda img: apply_occlusion(img, patch_size=(50, 50)),
        'horizontal_flip': lambda img: apply_horizontal_flip(img),
        'rotation_neg7': lambda img: apply_rotation(img, angle=-7),
        'rotation_pos7': lambda img: apply_rotation(img, angle=7)
    }
    
    all_results = []
    
    for i, image_path in enumerate(test_images):
        print(f"\nProcessing image {i+1}/{len(test_images)}: {os.path.basename(image_path)}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        # Compute baseline
        baseline_metrics = compute_baseline_metrics(engine, image_path)
        
        image_results = {
            'image_path': image_path,
            'baseline': baseline_metrics,
            'distortions': {}
        }
        
        # Test each distortion
        for dist_name, dist_func in distortions.items():
            print(f"  Testing {dist_name}...")
            
            try:
                distorted = dist_func(image.copy())
                result = stress_test_image(
                    engine, image_path, dist_name, distorted,
                    baseline_metrics, output_dir
                )
                image_results['distortions'][dist_name] = result
            except Exception as e:
                print(f"    Error: {e}")
                continue
        
        all_results.append(image_results)
    
    # Aggregate results
    aggregated = {}
    for dist_name in distortions.keys():
        dist_results = []
        for img_result in all_results:
            if dist_name in img_result['distortions']:
                dist_results.append(img_result['distortions'][dist_name])
        
        if dist_results:
            aggregated[dist_name] = {
                'mean_detection_drop': float(np.mean([r['detection_drop'] for r in dist_results])),
                'std_detection_drop': float(np.std([r['detection_drop'] for r in dist_results])),
                'mean_confidence_drop': float(np.mean([r['confidence_drop'] for r in dist_results])),
                'std_confidence_drop': float(np.std([r['confidence_drop'] for r in dist_results])),
                'num_tests': len(dist_results)
            }
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'stress_report.json')
    with open(report_path, 'w') as f:
        json.dump({
            'summary': aggregated,
            'per_image': all_results
        }, f, indent=2)
    
    print(f"\nStress testing complete!")
    print(f"  Report saved to: {report_path}")
    print(f"\nSummary of degradation:")
    for dist_name, stats in aggregated.items():
        print(f"  {dist_name}:")
        print(f"    Detection drop: {stats['mean_detection_drop']:.2f} ± {stats['std_detection_drop']:.2f}")
        print(f"    Confidence drop: {stats['mean_confidence_drop']:.4f} ± {stats['std_confidence_drop']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Stress testing for tooth detection pipeline')
    parser.add_argument('--images', type=str, required=True, help='Image directory')
    parser.add_argument('--engine', type=str, default=None, help='Engine path (or use model-dir)')
    parser.add_argument('--model-dir', type=str, default=None, help='Model directory')
    parser.add_argument('--out', type=str, required=True, help='Output directory')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of images to test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Load engine
    print("Loading inference engine...")
    if args.model_dir:
        engine = load_engine(args.model_dir)
    else:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'tooth-ai', 'models')
        engine = load_engine(model_dir)
    
    run_stress_tests(args.images, engine, args.out, args.num_samples, args.seed)


if __name__ == '__main__':
    main()



