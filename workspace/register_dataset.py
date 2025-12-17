#!/usr/bin/env python3
"""
Register COCO format dataset with Detectron2.
Handles train/val split and category mapping for FDI tooth notation.
"""

import json
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger

# FDI tooth notation mapping (32 teeth: 11-48)
# For now, we'll map generic labels to FDI numbers
# This should be updated based on actual annotation mapping
FDI_MAPPING = {
    "molar": [16, 17, 18, 26, 27, 28, 36, 37, 38, 46, 47, 48],  # 12 molars
    "premolar": [14, 15, 24, 25, 34, 35, 44, 45],  # 8 premolars
    "canine": [13, 23, 33, 43],  # 4 canines
    "lateral incisor": [12, 22, 32, 42],  # 4 lateral incisors
    "central incisor": [11, 21, 31, 41],  # 4 central incisors
    "implant": []  # Implants don't map to specific FDI numbers
}

# For 32-class setup, we can map to 1-32 (simplified)
# Or use actual FDI numbers 11-48 (38 classes, but user wants 32)
# Let's create a 32-class mapping: map to 1-32 sequential
TOOTH_CLASSES = [
    "tooth_1", "tooth_2", "tooth_3", "tooth_4", "tooth_5", "tooth_6", "tooth_7", "tooth_8",
    "tooth_9", "tooth_10", "tooth_11", "tooth_12", "tooth_13", "tooth_14", "tooth_15", "tooth_16",
    "tooth_17", "tooth_18", "tooth_19", "tooth_20", "tooth_21", "tooth_22", "tooth_23", "tooth_24",
    "tooth_25", "tooth_26", "tooth_27", "tooth_28", "tooth_29", "tooth_30", "tooth_31", "tooth_32"
]


def register_tooth_dataset(coco_json_path: str, image_dir: str, dataset_name: str, split_ratio: float = 0.8):
    """
    Register COCO dataset with Detectron2.
    
    Args:
        coco_json_path: Path to COCO format JSON file
        image_dir: Directory containing images
        dataset_name: Name for the dataset (e.g., "tooth_train", "tooth_val")
        split_ratio: Ratio for train/val split (default: 0.8 for 80% train)
    """
    # Load COCO JSON
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Split into train/val
    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data.get('categories', [])
    
    # Create mapping from image_id to annotations
    img_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Split images
    import random
    random.seed(42)
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    
    if 'train' in dataset_name.lower():
        split_images = images[:split_idx]
    elif 'val' in dataset_name.lower() or 'test' in dataset_name.lower():
        split_images = images[split_idx:]
    else:
        # Use all images if no split specified
        split_images = images
    
    # Create temporary COCO JSON for this split
    split_anns = []
    split_img_ids = {img['id'] for img in split_images}
    
    for ann in annotations:
        if ann['image_id'] in split_img_ids:
            split_anns.append(ann)
    
    split_coco_data = {
        "info": coco_data.get('info', {}),
        "licenses": coco_data.get('licenses', []),
        "images": split_images,
        "annotations": split_anns,
        "categories": categories
    }
    
    # Save temporary JSON
    temp_json = f"/tmp/{dataset_name}_coco.json"
    os.makedirs(os.path.dirname(temp_json), exist_ok=True)
    with open(temp_json, 'w') as f:
        json.dump(split_coco_data, f)
    
    # Register with Detectron2
    register_coco_instances(
        dataset_name,
        {},
        temp_json,
        image_dir
    )
    
    # Set metadata
    metadata = MetadataCatalog.get(dataset_name)
    metadata.thing_classes = TOOTH_CLASSES
    metadata.thing_dataset_id_to_contiguous_id = {i+1: i for i in range(32)}
    metadata.contiguous_id_to_thing_dataset_id = {i: i+1 for i in range(32)}
    
    print(f"Registered dataset: {dataset_name}")
    print(f"  Images: {len(split_images)}")
    print(f"  Annotations: {len(split_anns)}")
    print(f"  Categories: {len(categories)}")
    
    return temp_json


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Register COCO dataset with Detectron2')
    parser.add_argument('--coco-json', type=str, required=True, help='Path to COCO JSON file')
    parser.add_argument('--image-dir', type=str, required=True, help='Directory containing images')
    parser.add_argument('--split', type=float, default=0.8, help='Train/val split ratio (default: 0.8)')
    
    args = parser.parse_args()
    
    setup_logger()
    
    # Register train and val datasets
    register_tooth_dataset(args.coco_json, args.image_dir, "tooth_train", args.split)
    register_tooth_dataset(args.coco_json, args.image_dir, "tooth_val", args.split)
    
    print("\nDatasets registered successfully!")
    print("Use 'tooth_train' and 'tooth_val' in your config file.")


if __name__ == '__main__':
    main()



