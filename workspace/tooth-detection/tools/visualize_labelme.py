#!/usr/bin/env python3
"""
Visualize LabelMe annotations on images.
Displays polygon overlays from LabelMe JSON files on corresponding images.
"""

import argparse
import json
import os
import glob
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Dict


def load_labelme_json(json_path: str) -> Dict:
    """Load and parse a LabelMe JSON file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def find_image_path(json_path: str, images_dir: str) -> str:
    """
    Find the corresponding image path for a LabelMe JSON file.
    Tries multiple strategies:
    1. Use imagePath from JSON (relative to JSON location)
    2. Match by filename (without extension)
    """
    json_data = load_labelme_json(json_path)
    
    # Strategy 1: Use imagePath from JSON if available
    if 'imagePath' in json_data:
        image_path_from_json = json_data['imagePath']
        # Handle Windows path separators
        image_path_from_json = image_path_from_json.replace('\\', '/')
        
        # Try relative to JSON file location
        json_dir = os.path.dirname(json_path)
        potential_path = os.path.join(json_dir, image_path_from_json)
        if os.path.exists(potential_path):
            return potential_path
        
        # Try with normalized path
        potential_path = os.path.normpath(os.path.join(json_dir, image_path_from_json))
        if os.path.exists(potential_path):
            return potential_path
    
    # Strategy 2: Match by filename
    json_basename = os.path.splitext(os.path.basename(json_path))[0]
    
    # Search in images_dir recursively
    for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
        # Try direct match
        potential_path = os.path.join(images_dir, json_basename + ext)
        if os.path.exists(potential_path):
            return potential_path
        
        # Try recursive search
        pattern = os.path.join(images_dir, '**', json_basename + ext)
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
    
    return None


def draw_polygons(image: np.ndarray, shapes: List[Dict], alpha: float = 0.5) -> np.ndarray:
    """
    Draw polygons from LabelMe shapes on an image.
    
    Args:
        image: Input image (BGR format)
        shapes: List of shape dictionaries from LabelMe JSON
        alpha: Transparency factor for overlay
    
    Returns:
        Image with polygons drawn
    """
    overlay = image.copy()
    
    # Color map for different labels
    colors = {
        'molar': (0, 255, 0),      # Green
        'premolar': (255, 0, 0),   # Blue
        'canine': (0, 0, 255),     # Red
        'incisor': (255, 255, 0),  # Cyan
        'tooth': (255, 0, 255),    # Magenta
    }
    
    for shape in shapes:
        label = shape.get('label', 'unknown')
        points = shape.get('points', [])
        
        if len(points) < 3:  # Need at least 3 points for a polygon
            continue
        
        # Convert points to numpy array
        pts = np.array(points, dtype=np.int32)
        
        # Get color for label
        color = colors.get(label.lower(), (128, 128, 128))  # Gray for unknown
        
        # Draw filled polygon
        cv2.fillPoly(overlay, [pts], color)
        
        # Draw polygon outline
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
        
        # Add label text
        if len(pts) > 0:
            centroid = pts.mean(axis=0).astype(int)
            cv2.putText(overlay, label, tuple(centroid), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Blend overlay with original image
    result = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    
    # Draw outlines more prominently on blended result
    for shape in shapes:
        label = shape.get('label', 'unknown')
        points = shape.get('points', [])
        
        if len(points) < 3:
            continue
        
        pts = np.array(points, dtype=np.int32)
        color = colors.get(label.lower(), (128, 128, 128))
        cv2.polylines(result, [pts], isClosed=True, color=color, thickness=2)
    
    return result


def visualize_labelme(images_dir: str, ann_dir: str, output_dir: str, num_samples: int = 5):
    """
    Visualize LabelMe annotations on images.
    
    Args:
        images_dir: Directory containing images (searched recursively)
        ann_dir: Directory containing LabelMe JSON annotations (searched recursively)
        output_dir: Directory to save visualization outputs
        num_samples: Number of samples to visualize
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON annotation files
    json_pattern = os.path.join(ann_dir, '**', '*.json')
    json_files = glob.glob(json_pattern, recursive=True)
    
    if not json_files:
        print(f"Warning: No JSON files found in {ann_dir}")
        return
    
    print(f"Found {len(json_files)} annotation files")
    
    # Process up to num_samples files
    processed = 0
    errors = []
    
    for json_path in json_files[:num_samples]:
        try:
            # Load JSON
            json_data = load_labelme_json(json_path)
            
            # Find corresponding image
            image_path = find_image_path(json_path, images_dir)
            
            if image_path is None:
                error_msg = f"Could not find image for {json_path}"
                print(f"Warning: {error_msg}")
                errors.append(error_msg)
                continue
            
            if not os.path.exists(image_path):
                error_msg = f"Image file does not exist: {image_path}"
                print(f"Warning: {error_msg}")
                errors.append(error_msg)
                continue
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                error_msg = f"Could not load image: {image_path}"
                print(f"Warning: {error_msg}")
                errors.append(error_msg)
                continue
            
            # Draw polygons
            shapes = json_data.get('shapes', [])
            if not shapes:
                print(f"Warning: No shapes found in {json_path}")
                continue
            
            visualized = draw_polygons(image, shapes)
            
            # Save output
            output_filename = os.path.splitext(os.path.basename(json_path))[0] + '_vis.png'
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, visualized)
            
            print(f"Saved: {output_path} ({len(shapes)} shapes)")
            processed += 1
            
        except Exception as e:
            error_msg = f"Error processing {json_path}: {str(e)}"
            print(f"Error: {error_msg}")
            errors.append(error_msg)
    
    print(f"\nProcessed {processed} images successfully")
    if errors:
        print(f"\nErrors encountered ({len(errors)}):")
        for error in errors:
            print(f"  - {error}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize LabelMe annotations on images'
    )
    parser.add_argument(
        '--images',
        type=str,
        required=True,
        help='Directory containing images (searched recursively)'
    )
    parser.add_argument(
        '--ann',
        type=str,
        required=True,
        help='Directory containing LabelMe JSON annotations (searched recursively)'
    )
    parser.add_argument(
        '--out',
        type=str,
        required=True,
        help='Output directory for visualization images'
    )
    parser.add_argument(
        '--num',
        type=int,
        default=5,
        help='Number of samples to visualize (default: 5)'
    )
    
    args = parser.parse_args()
    
    # Validate input directories
    if not os.path.isdir(args.images):
        print(f"Error: Images directory does not exist: {args.images}")
        return
    
    if not os.path.isdir(args.ann):
        print(f"Error: Annotations directory does not exist: {args.ann}")
        return
    
    visualize_labelme(args.images, args.ann, args.out, args.num)


if __name__ == '__main__':
    main()

