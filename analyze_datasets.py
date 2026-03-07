#!/usr/bin/env python3
"""Analyze datasets for merging potential"""

import json
import os

print("="*70)
print("DATASET ANALYSIS")
print("="*70)

# Final-DI dataset
print("\n[1] FINAL-DI-STRATIFIED")
print("-"*50)

for split in ['train', 'valid', 'test']:
    path = f'data/final-di-stratified/{split}/_annotations.coco.json'
    with open(path) as f:
        d = json.load(f)
    print(f"{split:6s}: {len(d['images']):5d} images, {len(d['annotations']):6d} annotations")

# Show categories
print("\nCategories (41 total):")
with open('data/final-di-stratified/train/_annotations.coco.json') as f:
    d = json.load(f)
for c in d['categories'][:5]:
    print(f"  ID {c['id']:2d}: {c['name']}")
print("  ...")
for c in d['categories'][-5:]:
    print(f"  ID {c['id']:2d}: {c['name']}")

# NIIHHAA dataset
print("\n" + "="*70)
print("[2] NIIHHAA DATASET")
print("-"*50)

nihaa_json = 'data/niihhaa/coco_annotations.json'
with open(nihaa_json) as f:
    nd = json.load(f)

print(f"Images: {len(nd['images'])}")
print(f"Annotations: {len(nd['annotations'])}")
print(f"Categories: {len(nd['categories'])}")

print("\nCategories:")
for c in nd['categories']:
    print(f"  ID {c['id']:2d}: {c['name']}")

# Check image paths
print("\nSample image paths:")
for img in nd['images'][:3]:
    print(f"  {img['file_name']}")

# Merging analysis
print("\n" + "="*70)
print("[3] MERGING ANALYSIS")
print("-"*50)

final_cats = {c['name']: c['id'] for c in d['categories']}
nihaa_cats = {c['name']: c['id'] for c in nd['categories']}

common = set(final_cats.keys()) & set(nihaa_cats.keys())
only_final = set(final_cats.keys()) - set(nihaa_cats.keys())
only_nihaa = set(nihaa_cats.keys()) - set(final_cats.keys())

print(f"Common categories: {len(common)}")
print(f"Only in Final-DI: {len(only_final)}")
print(f"Only in NIIHHAA: {len(only_nihaa)}")

if only_nihaa:
    print("\nCategories only in NIIHHAA:")
    for cat in only_nihaa:
        print(f"  - {cat}")

print("\nMERGE FEASIBILITY: ", end="")
if len(common) > 0:
    print("YES - if category IDs are remapped")
else:
    print("DIFFICULT - no common categories")
