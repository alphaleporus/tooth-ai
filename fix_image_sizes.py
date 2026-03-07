#!/usr/bin/env python3
"""Fix image size metadata in merged dataset annotations"""

import json
from pathlib import Path
from PIL import Image

print("="*60)
print("FIXING IMAGE SIZE METADATA")
print("="*60)

for split in ['train', 'valid', 'test']:
    json_path = Path(f'data/merged-9class/{split}/_annotations.coco.json')
    img_dir = Path(f'data/merged-9class/{split}')
    
    if not json_path.exists():
        continue
    
    print(f"\n{split.upper()}")
    print("-"*40)
    
    with open(json_path) as f:
        data = json.load(f)
    
    fixed = 0
    missing = 0
    
    for img_info in data['images']:
        img_path = img_dir / img_info['file_name']
        
        if not img_path.exists():
            missing += 1
            continue
        
        # Get actual image size
        try:
            with Image.open(img_path) as im:
                actual_w, actual_h = im.size
        except Exception as e:
            print(f"  Error loading {img_info['file_name']}: {e}")
            continue
        
        # Check if metadata is wrong
        if img_info.get('width') != actual_w or img_info.get('height') != actual_h:
            img_info['width'] = actual_w
            img_info['height'] = actual_h
            fixed += 1
    
    # Save
    with open(json_path, 'w') as f:
        json.dump(data, f)
    
    print(f"  Fixed: {fixed} image metadata entries")
    print(f"  Missing images: {missing}")

print("\nDone!")
