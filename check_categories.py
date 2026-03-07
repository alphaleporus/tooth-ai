#!/usr/bin/env python3
"""Fix category ID mismatch in merged dataset"""

import json
from pathlib import Path
from collections import Counter

# Check and fix all merged dataset splits
for split in ['train', 'valid', 'test']:
    json_path = Path(f'data/merged-9class/{split}/_annotations.coco.json')
    
    if not json_path.exists():
        print(f"Skipping {split}: not found")
        continue
    
    print(f"\n{'='*50}")
    print(f"Checking {split}")
    print("="*50)
    
    with open(json_path) as f:
        data = json.load(f)
    
    print("Categories defined:")
    cat_ids = set()
    for c in data['categories']:
        cat_ids.add(c['id'])
        print(f"  ID {c['id']}: {c['name']}")
    
    print("\nAnnotation category_id distribution:")
    cat_counts = Counter(a['category_id'] for a in data['annotations'])
    
    invalid_ids = []
    for cat_id, count in sorted(cat_counts.items()):
        status = "✓" if cat_id in cat_ids else "✗ INVALID"
        print(f"  category_id {cat_id}: {count} annotations {status}")
        if cat_id not in cat_ids:
            invalid_ids.append(cat_id)
    
    if invalid_ids:
        print(f"\n⚠️ Found invalid category IDs: {invalid_ids}")
        print("Need to remap annotations to valid category IDs")

print("\n\nDone")
