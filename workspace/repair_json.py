#!/usr/bin/env python3
"""
Repair COCO JSON by removing annotations without valid segmentation.
"""

import json
import os
import shutil

def repair_json(file_path):
    """Remove annotations missing 'segmentation' key or with empty segmentation."""
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    
    # Backup original file
    backup_path = file_path + ".bak"
    shutil.copy2(file_path, backup_path)
    print(f"[OK] Backup created: {backup_path}")
    
    # Load JSON
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    original_count = len(data['annotations'])
    print(f"Original annotations: {original_count}")
    
    # Filter out bad annotations
    clean_annotations = []
    removed_count = 0
    
    for ann in data['annotations']:
        # Check if segmentation key exists and is not empty
        if 'segmentation' not in ann:
            removed_count += 1
            continue
        if ann['segmentation'] is None or len(ann['segmentation']) == 0:
            removed_count += 1
            continue
        # Keep valid annotation
        clean_annotations.append(ann)
    
    # Update data
    data['annotations'] = clean_annotations
    
    # Save cleaned JSON
    with open(file_path, 'w') as f:
        json.dump(data, f)
    
    print(f"[OK] Removed {removed_count} bad annotations")
    print(f"[OK] Remaining annotations: {len(clean_annotations)}")
    print(f"[OK] File saved: {file_path}")


def main():
    # Process all splits
    splits = ['train', 'valid', 'test']
    base_path = 'data/final-di'
    
    for split in splits:
        json_path = os.path.join(base_path, split, '_annotations.coco.json')
        print(f"\n{'='*50}")
        print(f"Processing: {split}")
        print(f"{'='*50}")
        repair_json(json_path)


if __name__ == '__main__':
    main()
