#!/usr/bin/env python3
"""
NIIHHAA Dataset Normalization to 9-Class Schema
Converts NIIHHAA tooth-type annotations to unified 9-class format.

Mapping:
- molar, premolar, canine, lateral incisor, central incisor → Class 0 (Tooth)
- implant → Class 4 (Implant)
"""

import json
import os
import shutil
from pathlib import Path

# 9-Class Schema
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

# NIIHHAA to 9-Class mapping
NIIHHAA_MAPPING = {
    "molar": 0,           # → Tooth
    "premolar": 0,        # → Tooth
    "canine": 0,          # → Tooth
    "lateral incisor": 0, # → Tooth
    "central incisor": 0, # → Tooth
    "implant": 4,         # → Implant (anomaly class)
}


def remap_niihhaa():
    input_dir = Path("data/niihhaa")
    output_dir = Path("data/niihhaa-remapped")
    
    print("="*60)
    print("NIIHHAA NORMALIZATION TO 9-CLASS")
    print("="*60)
    
    # Load original annotations
    input_json = input_dir / "coco_annotations.json"
    with open(input_json) as f:
        data = json.load(f)
    
    print(f"\nInput: {input_json}")
    print(f"  Images: {len(data['images'])}")
    print(f"  Annotations: {len(data['annotations'])}")
    print(f"  Categories: {len(data['categories'])}")
    
    # Build category name lookup
    cat_id_to_name = {c['id']: c['name'] for c in data['categories']}
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dir = output_dir / "train"
    train_dir.mkdir(exist_ok=True)
    
    # Remap annotations
    new_annotations = []
    skipped = 0
    
    for ann in data['annotations']:
        old_cat_id = ann['category_id']
        old_cat_name = cat_id_to_name.get(old_cat_id, "unknown")
        
        if old_cat_name in NIIHHAA_MAPPING:
            new_cat_id = NIIHHAA_MAPPING[old_cat_name]
            new_ann = ann.copy()
            new_ann['category_id'] = new_cat_id
            new_annotations.append(new_ann)
        else:
            skipped += 1
    
    print(f"\n  Remapped: {len(new_annotations)} annotations")
    print(f"  Skipped: {skipped} annotations")
    
    # Count by new class
    from collections import Counter
    new_counts = Counter(a['category_id'] for a in new_annotations)
    print("\n  New class distribution:")
    for cat in NINE_CLASS_CATEGORIES:
        count = new_counts.get(cat['id'], 0)
        if count > 0:
            print(f"    {cat['name']:30s}: {count:,}")
    
    # Create new COCO file
    new_data = {
        "images": data['images'],
        "annotations": new_annotations,
        "categories": NINE_CLASS_CATEGORIES
    }
    
    # Save annotations
    output_json = train_dir / "_annotations.coco.json"
    with open(output_json, 'w') as f:
        json.dump(new_data, f, indent=2)
    print(f"\n  Saved: {output_json}")
    
    # Copy images
    src_img_dir = input_dir / "dataset"
    print(f"\n  Copying images from {src_img_dir}...")
    
    copied = 0
    for img_info in data['images']:
        src_path = src_img_dir / img_info['file_name']
        dst_path = train_dir / img_info['file_name']
        
        if src_path.exists() and not dst_path.exists():
            shutil.copy2(src_path, dst_path)
            copied += 1
    
    print(f"  Copied {copied} images to {train_dir}")
    
    print("\n" + "="*60)
    print("NIIHHAA NORMALIZATION COMPLETE")
    print("="*60)
    
    return output_dir


if __name__ == "__main__":
    remap_niihhaa()
