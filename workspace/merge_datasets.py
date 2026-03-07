#!/usr/bin/env python3
"""
Dataset Fusion Script: Merge Final-DI-Remapped + NIIHHAA-Remapped
Creates unified merged-9class dataset for training.

Features:
- ID collision prevention (NIIHHAA IDs += 1,000,000)
- Unified train/valid/test splits
- Verification output
"""

import json
import os
import shutil
from pathlib import Path
from collections import Counter

# 9-Class Schema (must match both source datasets)
NINE_CLASS_CATEGORIES = [
    {"id": 0, "name": "Tooth", "supercategory": "dental"},
    {"id": 1, "name": "Caries", "supercategory": "anomaly"},
    {"id": 2, "name": "Crown", "supercategory": "anomaly"},
    {"id": 3, "name": "Filling", "supercategory": "anomaly"},
    {"id": 4, "name": "Implant", "supercategory": "anomaly"},
    {"id": 5, "name": "Prefabricated metal post", "supercategory": "anomaly"},
    {"id": 6, "name": "Retained root", "supercategory": "anomaly"},
    {"id": 7, "name": "Root canal filling", "supercategory": "anomaly"},
    {"id": 8, "name": "Root canal obturation", "supercategory": "anomaly"},
]

ID_OFFSET = 1_000_000  # Offset for NIIHHAA IDs to prevent collisions


def load_coco(json_path):
    """Load COCO format JSON."""
    with open(json_path) as f:
        return json.load(f)


def merge_datasets():
    print("="*70)
    print("DATASET FUSION: Final-DI-Remapped + NIIHHAA-Remapped")
    print("="*70)
    
    final_di_dir = Path("data/final-di-remapped")
    niihhaa_dir = Path("data/niihhaa-remapped")
    output_dir = Path("data/merged-9class")
    
    # Create output directories
    for split in ['train', 'valid', 'test']:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'valid', 'test']:
        print(f"\n{'='*50}")
        print(f"Processing: {split.upper()}")
        print("-"*50)
        
        merged_images = []
        merged_annotations = []
        
        # 1. Load Final-DI-Remapped
        final_di_json = final_di_dir / split / "_annotations.coco.json"
        if final_di_json.exists():
            final_data = load_coco(final_di_json)
            print(f"  Final-DI: {len(final_data['images'])} images, {len(final_data['annotations'])} annotations")
            
            # Add to merged (IDs unchanged)
            merged_images.extend(final_data['images'])
            merged_annotations.extend(final_data['annotations'])
            
            # Copy images
            src_dir = final_di_dir / split
            dst_dir = output_dir / split
            for img in final_data['images']:
                src = src_dir / img['file_name']
                dst = dst_dir / img['file_name']
                if src.exists() and not dst.exists():
                    shutil.copy2(src, dst)
        
        # 2. Load NIIHHAA-Remapped (train only - no valid/test splits)
        if split == 'train':
            niihhaa_json = niihhaa_dir / "train" / "_annotations.coco.json"
            if niihhaa_json.exists():
                niihhaa_data = load_coco(niihhaa_json)
                print(f"  NIIHHAA:  {len(niihhaa_data['images'])} images, {len(niihhaa_data['annotations'])} annotations")
                
                # Offset IDs to prevent collisions
                for img in niihhaa_data['images']:
                    img['id'] += ID_OFFSET
                    # Prefix filename to avoid collisions
                    img['file_name'] = f"niihhaa_{img['file_name']}"
                
                for ann in niihhaa_data['annotations']:
                    ann['id'] += ID_OFFSET
                    ann['image_id'] += ID_OFFSET
                
                merged_images.extend(niihhaa_data['images'])
                merged_annotations.extend(niihhaa_data['annotations'])
                
                # Copy images with prefix
                src_dir = niihhaa_dir / "train"
                dst_dir = output_dir / split
                for img in niihhaa_data['images']:
                    # Remove prefix for source lookup
                    src_name = img['file_name'].replace("niihhaa_", "")
                    src = src_dir / src_name
                    dst = dst_dir / img['file_name']
                    if src.exists() and not dst.exists():
                        shutil.copy2(src, dst)
        
        # 3. Create merged COCO file
        merged_data = {
            "images": merged_images,
            "annotations": merged_annotations,
            "categories": NINE_CLASS_CATEGORIES
        }
        
        output_json = output_dir / split / "_annotations.coco.json"
        with open(output_json, 'w') as f:
            json.dump(merged_data, f, indent=2)
        
        print(f"  MERGED:   {len(merged_images)} images, {len(merged_annotations)} annotations")
        
        # Category distribution
        cat_counts = Counter(a['category_id'] for a in merged_annotations)
        print(f"\n  Category distribution:")
        for cat in NINE_CLASS_CATEGORIES:
            count = cat_counts.get(cat['id'], 0)
            if count > 0:
                print(f"    {cat['name']:30s}: {count:6,}")
    
    # Final verification
    print("\n" + "="*70)
    print("VERIFICATION")
    print("="*70)
    
    total_images = 0
    total_annotations = 0
    
    for split in ['train', 'valid', 'test']:
        json_path = output_dir / split / "_annotations.coco.json"
        data = load_coco(json_path)
        img_count = len(data['images'])
        ann_count = len(data['annotations'])
        total_images += img_count
        total_annotations += ann_count
        
        # Count actual image files
        img_dir = output_dir / split
        actual_files = len([f for f in img_dir.iterdir() if f.suffix in ['.jpg', '.png']])
        
        status = "✓" if actual_files == img_count else f"⚠️ ({actual_files} files)"
        print(f"  {split:6s}: {img_count:5,} images {status}, {ann_count:6,} annotations")
    
    print(f"\n  TOTAL: {total_images:,} images, {total_annotations:,} annotations")
    print(f"  Output: {output_dir}")
    
    print("\n" + "="*70)
    print("DATASET FUSION COMPLETE")
    print("="*70)
    
    return output_dir


if __name__ == "__main__":
    merge_datasets()
