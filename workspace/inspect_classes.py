#!/usr/bin/env python3
"""
Inspect Classes Utility
Extracts and prints the class list from COCO annotations in the correct order.
"""

import json
import os

# Dataset paths to check
DATASET_PATHS = [
    "data/final-di-stratified/train/_annotations.coco.json",
    "data/final-di/train/_annotations.coco.json",
]

def extract_classes(json_path):
    """Extract categories from COCO JSON, sorted by ID."""
    
    print(f"\n{'='*60}")
    print(f"Loading: {json_path}")
    print('='*60)
    
    if not os.path.exists(json_path):
        print(f"ERROR: File not found!")
        return None
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    categories = data.get('categories', [])
    
    if not categories:
        print("ERROR: No categories found in JSON!")
        return None
    
    # Sort by ID (crucial for Detectron2 mapping)
    categories_sorted = sorted(categories, key=lambda x: x['id'])
    
    print(f"\nFound {len(categories_sorted)} categories")
    print(f"ID Range: {categories_sorted[0]['id']} to {categories_sorted[-1]['id']}")
    
    # Check if IDs are contiguous
    ids = [c['id'] for c in categories_sorted]
    expected_ids = list(range(min(ids), max(ids) + 1))
    if ids != expected_ids:
        print("\n⚠️  WARNING: Category IDs are NOT contiguous!")
        print(f"   Missing IDs: {set(expected_ids) - set(ids)}")
    
    return categories_sorted


def print_python_format(categories):
    """Print in copy-paste Python format."""
    
    print("\n" + "="*60)
    print("COPY-PASTE THIS INTO app.py:")
    print("="*60)
    
    print("\nALL_CLASSES = [")
    for cat in categories:
        print(f'    "{cat["name"]}",  # ID {cat["id"]}')
    print("]")
    
    print("\n" + "="*60)
    print("ANOMALY CLASSES (non-teeth):")
    print("="*60)
    
    anomalies = []
    teeth = []
    
    for cat in categories:
        name = cat['name']
        # Check if it's a tooth number or anomaly
        if name.isdigit() or name == 't':
            teeth.append(name)
        else:
            anomalies.append(name)
    
    print("\nANOMALY_CLASSES = [")
    for name in anomalies:
        print(f'    "{name}",')
    print("]")
    
    print(f"\n# Summary:")
    print(f"#   Total classes: {len(categories)}")
    print(f"#   Teeth: {len(teeth)}")
    print(f"#   Anomalies: {len(anomalies)}")


def main():
    print("\n" + "#"*60)
    print("# TOOTH-AI CLASS INSPECTOR")
    print("#"*60)
    
    categories = None
    
    for path in DATASET_PATHS:
        result = extract_classes(path)
        if result:
            categories = result
            break
    
    if categories:
        print_python_format(categories)
        
        # Also show ID mapping table
        print("\n" + "="*60)
        print("FULL ID MAPPING TABLE:")
        print("="*60)
        print(f"{'ID':<5} {'Name':<30} {'Type':<15}")
        print("-"*50)
        
        for cat in categories:
            name = cat['name']
            cat_type = "Tooth" if (name.isdigit() or name == 't') else "Anomaly"
            print(f"{cat['id']:<5} {name:<30} {cat_type:<15}")
    else:
        print("\nERROR: Could not load any dataset!")


if __name__ == "__main__":
    main()
