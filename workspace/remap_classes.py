#!/usr/bin/env python3
"""
Remap COCO annotations: Merge all 33 tooth classes into a single "Tooth" class.
This simplifies the classification problem from 41 classes to 9 classes:
  0: Tooth (merged from classes 0-32)
  1-8: Anomaly classes (Caries, Crown, Filling, etc.)
"""

import json
import os
import shutil
from pathlib import Path

# Input/Output directories
INPUT_DIR = Path("data/final-di-stratified")
OUTPUT_DIR = Path("data/final-di-remapped")

# Original class IDs (from annotations)
# IDs 0-32: Teeth (t, 1-32)
# IDs 33-40: Anomalies

# New class mapping
NEW_CATEGORIES = [
    {"id": 1, "name": "Tooth", "supercategory": "tooth"},
    {"id": 2, "name": "Caries", "supercategory": "anomaly"},
    {"id": 3, "name": "Crown", "supercategory": "anomaly"},
    {"id": 4, "name": "Filling", "supercategory": "anomaly"},
    {"id": 5, "name": "Implant", "supercategory": "anomaly"},
    {"id": 6, "name": "Prefabricated metal post", "supercategory": "anomaly"},
    {"id": 7, "name": "Retained root", "supercategory": "anomaly"},
    {"id": 8, "name": "Root canal filling", "supercategory": "anomaly"},
    {"id": 9, "name": "Root canal obturation", "supercategory": "anomaly"},
]

# Mapping from old IDs to new IDs
def get_new_category_id(old_id: int) -> int:
    """Map old category ID to new category ID."""
    if old_id <= 32:  # All teeth (IDs 0-32) -> Tooth (ID 1)
        return 1
    else:
        # Anomalies: 33->2, 34->3, 35->4, etc.
        return old_id - 31


def remap_annotations(input_json: Path, output_json: Path):
    """Remap annotations in a COCO JSON file."""
    print(f"Processing: {input_json}")
    
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    # Replace categories
    data['categories'] = NEW_CATEGORIES
    
    # Remap annotation category IDs
    stats = {"teeth": 0, "anomalies": 0}
    
    for ann in data['annotations']:
        old_id = ann['category_id']
        new_id = get_new_category_id(old_id)
        ann['category_id'] = new_id
        
        if new_id == 1:
            stats["teeth"] += 1
        else:
            stats["anomalies"] += 1
    
    # Save remapped annotations
    with open(output_json, 'w') as f:
        json.dump(data, f)
    
    print(f"  Images: {len(data['images'])}")
    print(f"  Teeth annotations: {stats['teeth']}")
    print(f"  Anomaly annotations: {stats['anomalies']}")
    print(f"  Saved to: {output_json}")
    

def main():
    print("="*60)
    print("TOOTH CLASS REMAPPING")
    print("Merging 33 tooth classes (t, 1-32) into single 'Tooth' class")
    print("="*60)
    
    # Create output directory structure
    for split in ["train", "valid", "test"]:
        output_split_dir = OUTPUT_DIR / split
        output_split_dir.mkdir(parents=True, exist_ok=True)
        
        input_split_dir = INPUT_DIR / split
        
        # Copy images (symlink or copy)
        print(f"\n--- {split.upper()} ---")
        
        # Copy/link images
        for img_file in input_split_dir.glob("*.jpg"):
            dest = output_split_dir / img_file.name
            if not dest.exists():
                shutil.copy2(img_file, dest)
        
        for img_file in input_split_dir.glob("*.png"):
            dest = output_split_dir / img_file.name
            if not dest.exists():
                shutil.copy2(img_file, dest)
        
        # Remap annotations
        input_json = input_split_dir / "_annotations.coco.json"
        output_json = output_split_dir / "_annotations.coco.json"
        
        if input_json.exists():
            remap_annotations(input_json, output_json)
        else:
            print(f"  Warning: {input_json} not found!")
    
    print("\n" + "="*60)
    print("REMAPPING COMPLETE!")
    print(f"New dataset: {OUTPUT_DIR}")
    print("\nNew class structure:")
    for cat in NEW_CATEGORIES:
        print(f"  ID {cat['id']}: {cat['name']}")
    print("="*60)


if __name__ == "__main__":
    main()
