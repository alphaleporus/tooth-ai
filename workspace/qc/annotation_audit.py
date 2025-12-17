#!/usr/bin/env python3
"""
Annotation Quality Control (QC) Audit Script.
Checks COCO annotations for common issues and generates an audit report.
"""

import argparse
import json
import os
import random
import glob
from typing import Dict, List, Tuple
import cv2
import numpy as np


def load_coco_json(coco_path: str) -> Dict:
    """Load COCO format JSON file."""
    with open(coco_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def find_image_file(filename: str, search_dir: str) -> str:
    """
    Find image file by filename in search directory (recursive).
    
    Returns:
        Full path to image file or None if not found
    """
    # Try different extensions
    base_name = os.path.splitext(filename)[0]
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        # Direct match
        potential_path = os.path.join(search_dir, filename)
        if os.path.exists(potential_path):
            return potential_path
        
        # Try with base name
        potential_path = os.path.join(search_dir, base_name + ext)
        if os.path.exists(potential_path):
            return potential_path
        
        # Recursive search
        pattern = os.path.join(search_dir, '**', filename)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
        
        pattern = os.path.join(search_dir, '**', base_name + ext)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    
    return None


def check_bbox_in_image(bbox: List[float], img_width: int, img_height: int) -> Tuple[bool, List[str]]:
    """
    Check if bounding box is within image bounds.
    
    Args:
        bbox: [x, y, width, height] in COCO format
        img_width: Image width
        img_height: Image height
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    x, y, w, h = bbox
    
    # Check for negative values
    if x < 0 or y < 0:
        issues.append(f"bbox has negative coordinates (x={x:.2f}, y={y:.2f})")
    
    if w <= 0 or h <= 0:
        issues.append(f"bbox has invalid dimensions (w={w:.2f}, h={h:.2f})")
    
    # Check if bbox extends outside image
    if x + w > img_width:
        issues.append(f"bbox extends beyond image width (x+w={x+w:.2f} > {img_width})")
    
    if y + h > img_height:
        issues.append(f"bbox extends beyond image height (y+h={y+h:.2f} > {img_height})")
    
    return len(issues) == 0, issues


def check_polygon_valid(segmentation: List[List[float]], img_width: int, img_height: int) -> Tuple[bool, List[str]]:
    """
    Check if polygon segmentation is valid.
    
    Args:
        segmentation: List of polygons, each is a flattened list [x1, y1, x2, y2, ...]
        img_width: Image width
        img_height: Image height
    
    Returns:
        (is_valid, list_of_issues)
    """
    issues = []
    
    for poly_idx, poly in enumerate(segmentation):
        if len(poly) < 6:  # Need at least 3 points (6 values)
            issues.append(f"polygon {poly_idx} has insufficient points ({len(poly)//2} points)")
            continue
        
        # Extract x and y coordinates
        x_coords = poly[::2]  # Every even index
        y_coords = poly[1::2]  # Every odd index
        
        # Check for points outside image
        out_of_bounds_x = [x for x in x_coords if x < 0 or x > img_width]
        out_of_bounds_y = [y for y in y_coords if y < 0 or y > img_height]
        
        if out_of_bounds_x:
            issues.append(f"polygon {poly_idx} has x coordinates outside image bounds")
        
        if out_of_bounds_y:
            issues.append(f"polygon {poly_idx} has y coordinates outside image bounds")
    
    return len(issues) == 0, issues


def calculate_mask_area(segmentation: List[List[float]], img_width: int, img_height: int) -> float:
    """
    Calculate mask area from polygon segmentation.
    Uses shoelace formula for polygon area.
    
    Args:
        segmentation: List of polygons
        img_width: Image width (for validation)
        img_height: Image height (for validation)
    
    Returns:
        Total area of all polygons
    """
    total_area = 0.0
    
    for poly in segmentation:
        if len(poly) < 6:  # Need at least 3 points
            continue
        
        # Extract points
        points = []
        for i in range(0, len(poly), 2):
            if i + 1 < len(poly):
                points.append([poly[i], poly[i + 1]])
        
        if len(points) < 3:
            continue
        
        # Calculate area using shoelace formula
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        
        total_area += abs(area) / 2.0
    
    return total_area


def audit_annotation(annotation: Dict, img_width: int, img_height: int) -> Tuple[bool, List[str], float]:
    """
    Audit a single annotation for issues.
    
    Args:
        annotation: COCO annotation dictionary
        img_width: Image width
        img_height: Image height
    
    Returns:
        (is_valid, list_of_issues, mask_area)
    """
    issues = []
    
    # Check area
    area = annotation.get('area', 0)
    if area <= 0:
        issues.append("mask area is zero or negative")
    
    # Check bounding box
    bbox = annotation.get('bbox', [])
    if len(bbox) != 4:
        issues.append("bbox has invalid format (expected 4 values)")
    else:
        bbox_valid, bbox_issues = check_bbox_in_image(bbox, img_width, img_height)
        if not bbox_valid:
            issues.extend(bbox_issues)
    
    # Check segmentation
    segmentation = annotation.get('segmentation', [])
    if not segmentation:
        issues.append("empty segmentation (no polygons)")
    else:
        # Check polygon validity
        poly_valid, poly_issues = check_polygon_valid(segmentation, img_width, img_height)
        if not poly_valid:
            issues.extend(poly_issues)
    
    # Calculate actual mask area from segmentation
    mask_area = calculate_mask_area(segmentation, img_width, img_height)
    
    # Compare with reported area (allow small tolerance)
    if area > 0 and abs(mask_area - area) / area > 0.1:  # 10% tolerance
        issues.append(f"mask area mismatch (reported={area:.2f}, calculated={mask_area:.2f})")
    
    is_valid = len(issues) == 0
    return is_valid, issues, mask_area


def audit_annotations(coco_path: str, img_dir: str, output_path: str, num_samples: int = 50):
    """
    Perform QC audit on COCO annotations.
    
    Args:
        coco_path: Path to COCO JSON file
        img_dir: Directory containing images (searched recursively)
        output_path: Path to save audit report JSON
        num_samples: Number of random images to sample
    """
    # Load COCO data
    print(f"Loading COCO annotations from {coco_path}...")
    coco_data = load_coco_json(coco_path)
    
    images = coco_data.get('images', [])
    annotations = coco_data.get('annotations', [])
    categories = coco_data.get('categories', [])
    
    print(f"Found {len(images)} images and {len(annotations)} annotations")
    
    # Create mapping from image_id to annotations
    img_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Sample random images
    if len(images) < num_samples:
        print(f"Warning: Only {len(images)} images available, sampling all")
        sampled_images = images
    else:
        sampled_images = random.sample(images, num_samples)
    
    print(f"Auditing {len(sampled_images)} random images...")
    
    # Audit results
    audit_results = []
    total_issues = 0
    suspicious_count = 0
    
    for img_info in sampled_images:
        img_id = img_info['id']
        filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Find image file
        image_path = find_image_file(filename, img_dir)
        
        if image_path is None:
            audit_results.append({
                "filename": filename,
                "issues": [f"Image file not found: {filename}"],
                "mask_count": 0,
                "avg_mask_area": 0.0,
                "suspicious": True
            })
            suspicious_count += 1
            total_issues += 1
            continue
        
        # Load image to verify dimensions
        try:
            img = cv2.imread(image_path)
            if img is None:
                audit_results.append({
                    "filename": filename,
                    "issues": [f"Could not load image: {image_path}"],
                    "mask_count": 0,
                    "avg_mask_area": 0.0,
                    "suspicious": True
                })
                suspicious_count += 1
                total_issues += 1
                continue
            
            actual_height, actual_width = img.shape[:2]
            
            # Check dimension mismatch
            dimension_issues = []
            if actual_width != img_width:
                dimension_issues.append(f"Width mismatch: COCO={img_width}, actual={actual_width}")
            if actual_height != img_height:
                dimension_issues.append(f"Height mismatch: COCO={img_height}, actual={actual_height}")
            
            # Use actual dimensions for validation
            validation_width = actual_width
            validation_height = actual_height
        
        except Exception as e:
            audit_results.append({
                "filename": filename,
                "issues": [f"Error loading image: {str(e)}"],
                "mask_count": 0,
                "avg_mask_area": 0.0,
                "suspicious": True
            })
            suspicious_count += 1
            total_issues += 1
            continue
        
        # Get annotations for this image
        img_annotations = img_to_anns.get(img_id, [])
        
        # Audit each annotation
        image_issues = list(dimension_issues)
        mask_areas = []
        
        for ann in img_annotations:
            is_valid, issues, mask_area = audit_annotation(ann, validation_width, validation_height)
            if not is_valid:
                image_issues.extend(issues)
            if mask_area > 0:
                mask_areas.append(mask_area)
        
        # Calculate statistics
        mask_count = len(img_annotations)
        avg_mask_area = np.mean(mask_areas) if mask_areas else 0.0
        
        # Determine if suspicious
        suspicious = len(image_issues) > 0
        if suspicious:
            suspicious_count += 1
            total_issues += len(image_issues)
        
        audit_results.append({
            "filename": filename,
            "issues": image_issues,
            "mask_count": mask_count,
            "avg_mask_area": avg_mask_area,
            "suspicious": suspicious
        })
    
    # Create audit report
    report = {
        "summary": {
            "total_samples": len(sampled_images),
            "suspicious_samples": suspicious_count,
            "suspicious_percentage": (suspicious_count / len(sampled_images) * 100) if sampled_images else 0,
            "total_issues": total_issues
        },
        "results": audit_results
    }
    
    # Save report
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\nAudit complete!")
    print(f"  Samples audited: {len(sampled_images)}")
    print(f"  Suspicious samples: {suspicious_count} ({report['summary']['suspicious_percentage']:.2f}%)")
    print(f"  Total issues found: {total_issues}")
    print(f"  Report saved to: {output_path}")
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Perform QC audit on COCO annotations'
    )
    parser.add_argument(
        '--coco',
        type=str,
        required=True,
        help='Path to COCO JSON annotations file'
    )
    parser.add_argument(
        '--img-dir',
        type=str,
        required=True,
        help='Directory containing images (searched recursively)'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output path for audit report JSON'
    )
    parser.add_argument(
        '--sample',
        type=int,
        default=50,
        help='Number of random samples to audit (default: 50)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Validate inputs
    if not os.path.isfile(args.coco):
        print(f"Error: COCO file does not exist: {args.coco}")
        return
    
    if not os.path.isdir(args.img_dir):
        print(f"Error: Images directory does not exist: {args.img_dir}")
        return
    
    report = audit_annotations(args.coco, args.img_dir, args.out, args.sample)
    
    # Check critical conditions
    print("\n" + "="*60)
    print("CRITICAL CHECKS:")
    print("="*60)
    
    suspicious_pct = report['summary']['suspicious_percentage']
    
    # Check 1: JSON parse errors (handled during loading)
    print("✓ JSON parsing: OK (no parse errors)")
    
    # Check 2: Zero area polygons
    zero_area_count = sum(1 for r in report['results'] if any('area is zero' in issue for issue in r['issues']))
    if zero_area_count == 0:
        print("✓ Zero area polygons: OK (none found)")
    else:
        print(f"✗ Zero area polygons: FAIL ({zero_area_count} found)")
    
    # Check 3: Bounding box outside image
    bbox_outside_count = sum(1 for r in report['results'] if any('extends beyond' in issue for issue in r['issues']))
    if bbox_outside_count == 0:
        print("✓ Bounding boxes: OK (all within image bounds)")
    else:
        print(f"✗ Bounding boxes: FAIL ({bbox_outside_count} with boxes outside image)")
    
    # Check 4: Suspicious percentage
    if suspicious_pct < 15.0:
        print(f"✓ Suspicious samples: OK ({suspicious_pct:.2f}% < 15%)")
    else:
        print(f"✗ Suspicious samples: FAIL ({suspicious_pct:.2f}% >= 15%)")
    
    print("="*60)
    
    # Overall status
    all_passed = (zero_area_count == 0 and bbox_outside_count == 0 and suspicious_pct < 15.0)
    if all_passed:
        print("STATUS: ✓ ALL CHECKS PASSED - Ready for training")
    else:
        print("STATUS: ✗ CHECKS FAILED - Fix issues before training")


if __name__ == '__main__':
    main()



