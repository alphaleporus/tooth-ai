#!/usr/bin/env python3
"""Analyze RepeatFactor sampling behavior."""

import json
from collections import Counter, defaultdict

with open('data/final-di-stratified/train/_annotations.coco.json', 'r') as f:
    data = json.load(f)

# Count annotations per category
cat_map = {c['id']: c['name'] for c in data['categories']}
counts = Counter()
for ann in data['annotations']:
    counts[ann['category_id']] += 1

# How many images contain each class
images_per_class = defaultdict(set)
for ann in data['annotations']:
    images_per_class[ann['category_id']].add(ann['image_id'])

# Print
print('='*70)
print('REPEAT FACTOR ANALYSIS')
print('='*70)
print(f"{'ID':<5} {'Class':<25} {'Annotations':<12} {'Images':<10} {'Freq':<10}")
print('-'*70)

total_images = len(data['images'])
teeth_freq = []
anomaly_freq = []

for cat_id in sorted(counts.keys()):
    name = cat_map[cat_id]
    ann_count = counts[cat_id]
    img_count = len(images_per_class[cat_id])
    freq = img_count / total_images
    
    if name.isdigit() or name == 't':
        teeth_freq.append(freq)
    else:
        anomaly_freq.append(freq)
    
    print(f"{cat_id:<5} {name:<25} {ann_count:<12} {img_count:<10} {freq:.4f}")

print('-'*70)
print(f"Teeth avg frequency: {sum(teeth_freq)/len(teeth_freq):.4f}")
print(f"Anomaly avg frequency: {sum(anomaly_freq)/len(anomaly_freq):.4f}")
print(f"\nTotal images: {total_images}")
