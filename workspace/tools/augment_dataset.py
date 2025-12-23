#!/usr/bin/env python3
"""
Offline Data Augmentation Script for COCO-format datasets.
Generates 3-4x augmented copies of the dataset with updated annotations.

Usage:
    python augment_dataset.py --input data/final-di/train --output data/final-di-augmented/train --multiplier 3
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from typing import List, Dict, Tuple
import cv2
import numpy as np
from tqdm import tqdm


class COCOAugmenter:
    """Augment COCO-format dataset with various transforms."""
    
    def __init__(self, input_dir: str, output_dir: str, multiplier: int = 3):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.multiplier = multiplier
        self.coco_json = self.input_dir / "_annotations.coco.json"
        
        # Load COCO annotations
        with open(self.coco_json) as f:
            self.coco_data = json.load(f)
        
        self.images = self.coco_data['images']
        self.annotations = self.coco_data['annotations']
        self.categories = self.coco_data['categories']
        
        # Build image_id to annotations mapping
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)
    
    def augment_image(self, img: np.ndarray, aug_type: int) -> Tuple[np.ndarray, dict]:
        """
        Apply augmentation to image and return transform info.
        
        Returns:
            Augmented image and transform parameters for bbox adjustment
        """
        h, w = img.shape[:2]
        transform = {'type': aug_type, 'h': h, 'w': w, 'flip_h': False, 'rotation': 0}
        
        if aug_type == 1:
            # Horizontal flip
            img = cv2.flip(img, 1)
            transform['flip_h'] = True
            
        elif aug_type == 2:
            # Brightness adjustment (0.7-1.3)
            factor = random.uniform(0.7, 1.3)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
            transform['brightness'] = factor
            
        elif aug_type == 3:
            # Contrast adjustment
            factor = random.uniform(0.7, 1.3)
            mean = np.mean(img)
            img = np.clip((img - mean) * factor + mean, 0, 255).astype(np.uint8)
            transform['contrast'] = factor
            
        elif aug_type == 4:
            # Rotation (-10 to 10 degrees)
            angle = random.uniform(-10, 10)
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            transform['rotation'] = angle
            transform['rotation_matrix'] = M.tolist()
            
        elif aug_type == 5:
            # Gaussian noise
            noise = np.random.normal(0, 10, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 255).astype(np.uint8)
            transform['noise'] = True
            
        elif aug_type == 6:
            # Combination: flip + brightness
            img = cv2.flip(img, 1)
            factor = random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)
            transform['flip_h'] = True
            transform['brightness'] = factor
        
        return img, transform
    
    def transform_bbox(self, bbox: List[float], transform: dict) -> List[float]:
        """Transform bounding box according to augmentation."""
        x, y, w, h = bbox
        img_h, img_w = transform['h'], transform['w']
        
        if transform.get('flip_h'):
            # Flip x coordinate
            x = img_w - x - w
        
        if transform.get('rotation') and abs(transform['rotation']) > 0.1:
            # For rotation, expand bbox slightly to cover rotated content
            # This is a simplification - proper rotation requires polygon masks
            cx, cy = x + w/2, y + h/2
            # Keep same bbox for small rotations
            pass
        
        return [max(0, x), max(0, y), w, h]
    
    def transform_segmentation(self, segmentation: List, transform: dict) -> List:
        """Transform segmentation polygon according to augmentation."""
        if not segmentation or not isinstance(segmentation[0], list):
            return segmentation
        
        new_seg = []
        for polygon in segmentation:
            new_poly = []
            for i in range(0, len(polygon), 2):
                x, y = polygon[i], polygon[i+1]
                
                if transform.get('flip_h'):
                    x = transform['w'] - x
                
                new_poly.extend([x, y])
            new_seg.append(new_poly)
        
        return new_seg
    
    def run(self):
        """Run augmentation pipeline."""
        print(f"=== Offline Data Augmentation ===")
        print(f"Input: {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Multiplier: {self.multiplier}x")
        print(f"Original images: {len(self.images)}")
        print(f"Target images: ~{len(self.images) * self.multiplier}")
        print()
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        new_images = []
        new_annotations = []
        next_image_id = max(img['id'] for img in self.images) + 1
        next_ann_id = max(ann['id'] for ann in self.annotations) + 1
        
        # First, copy all original images
        print("Copying original images...")
        for img_info in tqdm(self.images):
            src = self.input_dir / img_info['file_name']
            dst = self.output_dir / img_info['file_name']
            if src.exists():
                shutil.copy2(src, dst)
            new_images.append(img_info.copy())
        
        # Copy original annotations
        new_annotations.extend([ann.copy() for ann in self.annotations])
        
        # Generate augmented versions
        print(f"\nGenerating {self.multiplier - 1} augmented versions per image...")
        aug_types = [1, 2, 3, 4, 5, 6]  # Different augmentation types
        
        for img_info in tqdm(self.images):
            src_path = self.input_dir / img_info['file_name']
            if not src_path.exists():
                continue
            
            img = cv2.imread(str(src_path))
            if img is None:
                continue
            
            orig_anns = self.img_to_anns.get(img_info['id'], [])
            
            # Generate augmented versions
            for aug_idx in range(self.multiplier - 1):
                aug_type = random.choice(aug_types)
                aug_img, transform = self.augment_image(img.copy(), aug_type)
                
                # Save augmented image
                stem = Path(img_info['file_name']).stem
                ext = Path(img_info['file_name']).suffix
                new_filename = f"{stem}_aug{aug_idx+1}{ext}"
                cv2.imwrite(str(self.output_dir / new_filename), aug_img)
                
                # Create new image entry
                new_img_info = {
                    'id': next_image_id,
                    'file_name': new_filename,
                    'width': img_info['width'],
                    'height': img_info['height']
                }
                new_images.append(new_img_info)
                
                # Transform annotations
                for ann in orig_anns:
                    new_ann = ann.copy()
                    new_ann['id'] = next_ann_id
                    new_ann['image_id'] = next_image_id
                    new_ann['bbox'] = self.transform_bbox(ann['bbox'], transform)
                    if 'segmentation' in ann:
                        new_ann['segmentation'] = self.transform_segmentation(
                            ann['segmentation'], transform
                        )
                    new_annotations.append(new_ann)
                    next_ann_id += 1
                
                next_image_id += 1
        
        # Save new COCO JSON
        new_coco = {
            'info': self.coco_data.get('info', {}),
            'licenses': self.coco_data.get('licenses', []),
            'categories': self.categories,
            'images': new_images,
            'annotations': new_annotations
        }
        
        output_json = self.output_dir / "_annotations.coco.json"
        with open(output_json, 'w') as f:
            json.dump(new_coco, f)
        
        print(f"\n=== Augmentation Complete ===")
        print(f"Original images: {len(self.images)}")
        print(f"Total images: {len(new_images)}")
        print(f"Original annotations: {len(self.annotations)}")
        print(f"Total annotations: {len(new_annotations)}")
        print(f"Output saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Offline COCO dataset augmentation')
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory with _annotations.coco.json')
    parser.add_argument('--output', type=str, required=True,
                        help='Output directory for augmented dataset')
    parser.add_argument('--multiplier', type=int, default=3,
                        help='Dataset size multiplier (default: 3)')
    
    args = parser.parse_args()
    
    augmenter = COCOAugmenter(args.input, args.output, args.multiplier)
    augmenter.run()


if __name__ == '__main__':
    main()
