#!/usr/bin/env python3
"""Fix category_id=9 → 8 in all merged dataset splits"""

import json
from pathlib import Path

print("="*60)
print("FIXING CATEGORY ID MISMATCH")
print("="*60)

# Category 9 → 8 (both are Root canal obturation - off by one error)
for split in ['train', 'valid', 'test']:
    json_path = Path(f'data/merged-9class/{split}/_annotations.coco.json')
    
    if not json_path.exists():
        continue
    
    with open(json_path) as f:
        data = json.load(f)
    
    # Fix annotations with category_id = 9
    fixed = 0
    for ann in data['annotations']:
        if ann['category_id'] == 9:
            ann['category_id'] = 8  # Root canal obturation
            fixed += 1
    
    # Save
    with open(json_path, 'w') as f:
        json.dump(data, f)
    
    print(f"{split}: Fixed {fixed} annotations (9 → 8)")

print("\nDone! All category_id=9 remapped to 8.")
