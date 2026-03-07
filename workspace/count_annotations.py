#!/usr/bin/env python3
"""
Verify if teeth are annotated in the dataset.
"""

import json
from collections import Counter

# Load annotations
with open('data/final-di-stratified/train/_annotations.coco.json', 'r') as f:
    data = json.load(f)

# Get category mapping
cat_map = {c['id']: c['name'] for c in data['categories']}

# Count annotations per category
counts = Counter()
for ann in data['annotations']:
    cat_name = cat_map.get(ann['category_id'], 'Unknown')
    counts[cat_name] += 1

# Print sorted by count
print('='*60)
print('ANNOTATION COUNTS PER CLASS')
print('='*60)
print(f"{'Class':<30} {'Count':<10} {'Type':<10}")
print('-'*60)

for name, count in sorted(counts.items(), key=lambda x: -x[1]):
    cat_type = 'Anomaly' if not name.isdigit() and name != 't' else 'Tooth'
    print(f"{name:<30} {count:<10} {cat_type:<10}")

# Summary
teeth_count = sum(c for n, c in counts.items() if n.isdigit() or n == 't')
anomaly_count = sum(c for n, c in counts.items() if not n.isdigit() and n != 't')
print('-'*60)
print(f'TEETH ANNOTATIONS TOTAL: {teeth_count}')
print(f'ANOMALY ANNOTATIONS TOTAL: {anomaly_count}')
print(f'TEETH PERCENTAGE: {teeth_count/(teeth_count+anomaly_count)*100:.1f}%')
