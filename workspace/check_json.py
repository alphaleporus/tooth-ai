import json
import os

# Path to your training annotations
file_path = "data/final-di/train/_annotations.coco.json"

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit()

with open(file_path, 'r') as f:
    data = json.load(f)
    
print(f"Dataset Stats:")
print(f" - Images: {len(data['images'])}")
print(f" - Annotations: {len(data['annotations'])}")

# Check the first 5 annotations
print("\nInspecting first 5 annotations for 'segmentation' key:")
for i in range(5):
    ann = data['annotations'][i]
    has_seg = 'segmentation' in ann
    print(f" - ID {ann['id']}: Has Segmentation? {has_seg}")
    if has_seg:
        # Check if it's not empty
        print(f"   -> Data length: {len(ann['segmentation'])}")