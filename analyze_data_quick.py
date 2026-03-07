import json
import sys

# Load annotations
with open('data/final-di-remapped/train/_annotations.coco.json', 'r') as f:
    data = json.load(f)

print("Dataset Statistics:")
print("=" * 60)
print(f"Total images: {len(data['images'])}")
print(f"Total annotations: {len(data['annotations'])}")
print(f"\nCategories ({len(data['categories'])}):")
print("-" * 60)

for cat in sorted(data['categories'], key=lambda x: x['id']):
    print(f"  ID {cat['id']}: {cat['name']}")

# Count annotations per category
cat_counts = {}
for ann in data['annotations']:
    cat_id = ann['category_id']
    cat_counts[cat_id] = cat_counts.get(cat_id, 0) + 1

print(f"\nAnnotation Counts:")
print("-" * 60)
for cat in sorted(data['categories'], key=lambda x: x['id']):
    count = cat_counts.get(cat['id'], 0)
    print(f"  {cat['name']}: {count}")

# Calculate avg annotations per image
avg_per_image = len(data['annotations']) / len(data['images']) if len(data['images']) > 0 else 0
print(f"\nAverage annotations per image: {avg_per_image:.2f}")
