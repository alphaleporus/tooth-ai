#!/usr/bin/env python3
"""
Convert LabelMe JSON annotations to COCO format.
Handles polygon annotations and creates COCO instance segmentation format.
"""

import argparse
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


def load_labelme_json(json_path: str) -> Dict:
    """Load and parse a LabelMe JSON file."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON {json_path}: {e}")
        raise
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        raise


def find_image_path(json_path: str, img_dir: str, labelme_data: Dict) -> Tuple[str, int, int]:
    """
    Find the corresponding image path and get dimensions.
    
    Returns:
        (image_path, width, height) or (None, width, height) if image not found
    """
    # Get dimensions from JSON if available
    width = labelme_data.get('imageWidth', 0)
    height = labelme_data.get('imageHeight', 0)
    
    # Strategy 1: Use imagePath from JSON
    if 'imagePath' in labelme_data:
        image_path_from_json = labelme_data['imagePath']
        # Handle Windows path separators
        image_path_from_json = image_path_from_json.replace('\\', '/')
        
        # Try relative to JSON file location
        json_dir = os.path.dirname(json_path)
        potential_path = os.path.join(json_dir, image_path_from_json)
        if os.path.exists(potential_path):
            return potential_path, width, height
        
        # Try normalized path
        potential_path = os.path.normpath(os.path.join(json_dir, image_path_from_json))
        if os.path.exists(potential_path):
            return potential_path, width, height
    
    # Strategy 2: Match by filename
    json_basename = os.path.splitext(os.path.basename(json_path))[0]
    
    # Search in img_dir recursively
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        # Try direct match
        potential_path = os.path.join(img_dir, json_basename + ext)
        if os.path.exists(potential_path):
            return potential_path, width, height
        
        # Try recursive search
        pattern = os.path.join(img_dir, '**', json_basename + ext)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0], width, height
    
    return None, width, height


def polygon_to_bbox(points: List[List[float]]) -> List[float]:
    """
    Convert polygon points to COCO bbox format [x, y, width, height].
    
    Args:
        points: List of [x, y] coordinates
    
    Returns:
        [x_min, y_min, width, height]
    """
    if not points or len(points) < 3:
        return [0, 0, 0, 0]
    
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    
    width = x_max - x_min
    height = y_max - y_min
    
    return [x_min, y_min, width, height]


def polygon_area(points: List[List[float]]) -> float:
    """
    Calculate polygon area using shoelace formula.
    
    Args:
        points: List of [x, y] coordinates
    
    Returns:
        Area of the polygon
    """
    if len(points) < 3:
        return 0.0
    
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    
    return abs(area) / 2.0


def polygon_to_segmentation(points: List[List[float]]) -> List[float]:
    """
    Convert polygon points to COCO segmentation format (flattened list).
    
    Args:
        points: List of [x, y] coordinates
    
    Returns:
        Flattened list [x1, y1, x2, y2, ...]
    """
    segmentation = []
    for point in points:
        segmentation.extend([float(point[0]), float(point[1])])
    return segmentation


def labelme2coco(labelme_dir: str, img_dir: str, output_path: str):
    """
    Convert LabelMe JSON annotations to COCO format.
    
    Args:
        labelme_dir: Directory containing LabelMe JSON files (searched recursively)
        img_dir: Directory containing images (searched recursively)
        output_path: Path to save COCO JSON file
    """
    # Find all JSON files
    json_pattern = os.path.join(labelme_dir, '**', '*.json')
    json_files = glob.glob(json_pattern, recursive=True)
    
    if not json_files:
        raise ValueError(f"No JSON files found in {labelme_dir}")
    
    print(f"Found {len(json_files)} LabelMe JSON files")
    
    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "Tooth detection dataset converted from LabelMe",
            "version": "1.0",
            "year": 2024
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # Category mapping
    category_map = {}  # label -> category_id
    category_id_counter = 1
    
    # Image and annotation counters
    image_id_counter = 1
    annotation_id_counter = 1
    
    # Process each JSON file
    processed_images = {}  # image_path -> image_id
    errors = []
    
    for json_path in json_files:
        try:
            # Load LabelMe JSON
            labelme_data = load_labelme_json(json_path)
            
            # Find corresponding image
            image_path, width, height = find_image_path(json_path, img_dir, labelme_data)
            
            # If image not found, try to get dimensions from JSON
            if image_path is None:
                if width == 0 or height == 0:
                    print(f"Warning: Could not find image for {json_path} and no dimensions in JSON")
                    errors.append(f"Image not found: {json_path}")
                    continue
                # Use JSON basename as filename
                json_basename = os.path.splitext(os.path.basename(json_path))[0]
                image_path = f"{json_basename}.png"  # Placeholder
            
            # Get or create image entry
            if image_path not in processed_images:
                # Add image to COCO
                image_entry = {
                    "id": image_id_counter,
                    "file_name": os.path.basename(image_path),
                    "width": width,
                    "height": height
                }
                coco_data["images"].append(image_entry)
                processed_images[image_path] = image_id_counter
                image_id = image_id_counter
                image_id_counter += 1
            else:
                image_id = processed_images[image_path]
            
            # Process shapes (annotations)
            shapes = labelme_data.get('shapes', [])
            
            for shape in shapes:
                label = shape.get('label', 'unknown')
                points = shape.get('points', [])
                shape_type = shape.get('shape_type', 'polygon')
                
                # Skip if not a polygon or insufficient points
                if shape_type != 'polygon' or len(points) < 3:
                    continue
                
                # Get or create category
                if label not in category_map:
                    category_entry = {
                        "id": category_id_counter,
                        "name": label,
                        "supercategory": "tooth"
                    }
                    coco_data["categories"].append(category_entry)
                    category_map[label] = category_id_counter
                    category_id_counter += 1
                
                category_id = category_map[label]
                
                # Convert polygon to COCO format
                segmentation = polygon_to_segmentation(points)
                bbox = polygon_to_bbox(points)
                area = polygon_area(points)
                
                # Skip if area is zero or negative
                if area <= 0:
                    print(f"Warning: Zero or negative area polygon in {json_path}, label: {label}")
                    continue
                
                # Create annotation
                annotation = {
                    "id": annotation_id_counter,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [segmentation],  # COCO expects list of polygons
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0
                }
                
                coco_data["annotations"].append(annotation)
                annotation_id_counter += 1
        
        except Exception as e:
            error_msg = f"Error processing {json_path}: {str(e)}"
            print(f"Error: {error_msg}")
            errors.append(error_msg)
            continue
    
    # Save COCO JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    print(f"\nConversion complete!")
    print(f"  Images: {len(coco_data['images'])}")
    print(f"  Annotations: {len(coco_data['annotations'])}")
    print(f"  Categories: {len(coco_data['categories'])}")
    print(f"  Saved to: {output_path}")
    
    if errors:
        print(f"\nErrors encountered ({len(errors)}):")
        for error in errors[:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")


def main():
    parser = argparse.ArgumentParser(
        description='Convert LabelMe JSON annotations to COCO format'
    )
    parser.add_argument(
        '--labelme-dir',
        type=str,
        required=True,
        help='Directory containing LabelMe JSON files (searched recursively)'
    )
    parser.add_argument(
        '--img-dir',
        type=str,
        required=True,
        help='Directory containing images (searched recursively)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for COCO JSON file'
    )
    
    args = parser.parse_args()
    
    # Validate input directories
    if not os.path.isdir(args.labelme_dir):
        print(f"Error: LabelMe directory does not exist: {args.labelme_dir}")
        return
    
    if not os.path.isdir(args.img_dir):
        print(f"Error: Images directory does not exist: {args.img_dir}")
        return
    
    labelme2coco(args.labelme_dir, args.img_dir, args.output)


if __name__ == '__main__':
    main()

