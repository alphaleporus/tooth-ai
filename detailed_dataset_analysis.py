#!/usr/bin/env python3
"""Detailed dataset analysis for merge planning"""

import json
import os
from collections import Counter

print("="*80)
print("DETAILED DATASET ANALYSIS REPORT")
print("="*80)

# ============================================================
# FINAL-DI DATASET
# ============================================================
print("\n" + "="*80)
print("PART 1: FINAL-DI-STRATIFIED DATASET")
print("="*80)

# Load all splits
final_data = {}
for split in ['train', 'valid', 'test']:
    path = f'data/final-di-stratified/{split}/_annotations.coco.json'
    with open(path) as f:
        final_data[split] = json.load(f)

# Basic stats
print("\n[1.1] DATASET SIZE")
print("-"*50)
total_images = 0
total_annotations = 0
for split, data in final_data.items():
    imgs = len(data['images'])
    anns = len(data['annotations'])
    total_images += imgs
    total_annotations += anns
    print(f"  {split:6s}: {imgs:5,} images, {anns:6,} annotations")
print(f"  {'TOTAL':6s}: {total_images:5,} images, {total_annotations:6,} annotations")

# Category analysis
print("\n[1.2] CATEGORY BREAKDOWN")
print("-"*50)
categories = final_data['train']['categories']
print(f"Total categories: {len(categories)}")

# Group by type
teeth_cats = [c for c in categories if c['name'].isdigit() or c['name'] == 't']
anomaly_cats = [c for c in categories if c['name'] not in ['t'] and not c['name'].isdigit()]

print(f"\n  Teeth classes (IDs 0-32): {len(teeth_cats)}")
print(f"  Anomaly classes (IDs 33-40): {len(anomaly_cats)}")
print("\n  Anomaly categories:")
for c in anomaly_cats:
    print(f"    ID {c['id']:2d}: {c['name']}")

# Annotation distribution per category
print("\n[1.3] ANNOTATIONS PER CATEGORY (Training Set)")
print("-"*50)
train_anns = final_data['train']['annotations']
cat_counts = Counter(a['category_id'] for a in train_anns)
cat_names = {c['id']: c['name'] for c in categories}

# Sort by count
sorted_cats = sorted(cat_counts.items(), key=lambda x: -x[1])
print("  Top 10 categories:")
for cat_id, count in sorted_cats[:10]:
    name = cat_names.get(cat_id, 'Unknown')
    print(f"    {name:30s}: {count:5,} annotations")

print("\n  Bottom 5 categories:")
for cat_id, count in sorted_cats[-5:]:
    name = cat_names.get(cat_id, 'Unknown')
    print(f"    {name:30s}: {count:5,} annotations")

# Image source analysis
print("\n[1.4] IMAGE SOURCE ANALYSIS")
print("-"*50)
train_images = final_data['train']['images']
prefixes = Counter()
for img in train_images:
    fname = img['file_name']
    if fname.startswith('cate'):
        prefixes['cate*'] += 1
    elif fname[0].isdigit():
        prefixes['numeric*'] += 1
    else:
        prefixes['other'] += 1

print("  Image filename patterns (training set):")
for prefix, count in prefixes.most_common():
    pct = count / len(train_images) * 100
    print(f"    {prefix:15s}: {count:5,} images ({pct:.1f}%)")

# ============================================================
# NIIHHAA DATASET
# ============================================================
print("\n" + "="*80)
print("PART 2: NIIHHAA DATASET")
print("="*80)

nihaa_path = 'data/niihhaa/coco_annotations.json'
with open(nihaa_path) as f:
    nihaa_data = json.load(f)

print("\n[2.1] DATASET SIZE")
print("-"*50)
print(f"  Images: {len(nihaa_data['images']):,}")
print(f"  Annotations: {len(nihaa_data['annotations']):,}")
print(f"  Categories: {len(nihaa_data['categories'])}")
print(f"  Annotations per image: {len(nihaa_data['annotations'])/len(nihaa_data['images']):.1f}")

print("\n[2.2] CATEGORIES")
print("-"*50)
for c in nihaa_data['categories']:
    print(f"  ID {c['id']}: {c['name']}")

print("\n[2.3] ANNOTATIONS PER CATEGORY")
print("-"*50)
nihaa_cat_counts = Counter(a['category_id'] for a in nihaa_data['annotations'])
nihaa_cat_names = {c['id']: c['name'] for c in nihaa_data['categories']}
for cat_id, count in sorted(nihaa_cat_counts.items(), key=lambda x: -x[1]):
    name = nihaa_cat_names.get(cat_id, 'Unknown')
    print(f"  {name:20s}: {count:5,} annotations")

print("\n[2.4] IMAGE DIMENSIONS")
print("-"*50)
widths = [img.get('width', 0) for img in nihaa_data['images']]
heights = [img.get('height', 0) for img in nihaa_data['images']]
if widths[0] > 0:
    print(f"  Width range: {min(widths)} - {max(widths)}")
    print(f"  Height range: {min(heights)} - {max(heights)}")
else:
    print("  Dimensions not specified in metadata")

# ============================================================
# MERGE ANALYSIS
# ============================================================
print("\n" + "="*80)
print("PART 3: MERGE FEASIBILITY ANALYSIS")
print("="*80)

print("\n[3.1] CATEGORY MAPPING OPTIONS")
print("-"*50)

print("""
OPTION A: Direct Merge (41 classes)
  Problem: NIIHHAA uses tooth TYPE names, not tooth NUMBERS
  - 'molar' ≠ specific tooth number
  - Would need manual re-annotation
  Feasibility: ❌ NOT RECOMMENDED

OPTION B: 9-Class Merge (Recommended)
  Map ALL tooth types to single 'Tooth' class:
  - NIIHHAA 'molar' → Tooth
  - NIIHHAA 'premolar' → Tooth
  - NIIHHAA 'canine' → Tooth
  - NIIHHAA 'lateral incisor' → Tooth
  - NIIHHAA 'central incisor' → Tooth
  - NIIHHAA 'implant' → Implant (direct match!)
  Feasibility: ✅ RECOMMENDED

OPTION C: Tooth-Type Merge (6 classes)
  Train on tooth TYPES instead of numbers:
  - Molar, Premolar, Canine, Incisor, etc.
  - Would require re-annotating Final-DI
  Feasibility: ⚠️ REQUIRES SIGNIFICANT WORK
""")

print("\n[3.2] MERGED DATASET STATISTICS (Option B)")
print("-"*50)
print("  After 9-class merge:")
print(f"    Final-DI images: {total_images:,}")
print(f"    NIIHHAA images:  {len(nihaa_data['images']):,}")
print(f"    TOTAL:           {total_images + len(nihaa_data['images']):,} images")
print()
print(f"    Final-DI annotations: {total_annotations:,}")
print(f"    NIIHHAA annotations:  {len(nihaa_data['annotations']):,}")
print(f"    TOTAL:                {total_annotations + len(nihaa_data['annotations']):,} annotations")

increase_pct = len(nihaa_data['images']) / total_images * 100
print(f"\n  Dataset size increase: +{increase_pct:.1f}%")

print("\n[3.3] MERGE BENEFITS")
print("-"*50)
print("""
✅ More diverse training data
✅ Different X-ray sources → better generalization
✅ More tooth annotations → stronger tooth detection
✅ NIIHHAA has clean tooth-type labels
""")

print("\n[3.4] MERGE CHALLENGES")
print("-"*50)
print("""
⚠️ Image path handling (different directory structures)
⚠️ Potential duplicate images (need deduplication check)
⚠️ Different annotation quality/style
⚠️ Category ID remapping required
""")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
print("""
PROCEED WITH 9-CLASS MERGE:
1. Remap Final-DI to 9 classes (already done: data/final-di-remapped)
2. Remap NIIHHAA to 9 classes (new script needed)
3. Combine datasets
4. Train ResNet-50 for ~20K iterations

Expected improvement: Better generalization on diverse X-ray sources
""")
