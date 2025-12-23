#!/usr/bin/env python3
"""
Register COCO format dataset with Detectron2.
Handles the final-di dataset with 41 classes.
Includes data augmentation support.
"""

import json
import os
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger

# Dataset paths
DATASET_ROOT = "data/final-di"
TRAIN_JSON = os.path.join(DATASET_ROOT, "train", "_annotations.coco.json")
TRAIN_IMAGES = os.path.join(DATASET_ROOT, "train")
VAL_JSON = os.path.join(DATASET_ROOT, "valid", "_annotations.coco.json")
VAL_IMAGES = os.path.join(DATASET_ROOT, "valid")
TEST_JSON = os.path.join(DATASET_ROOT, "test", "_annotations.coco.json")
TEST_IMAGES = os.path.join(DATASET_ROOT, "test")

# 41 classes from final-di dataset (from actual JSON)
FINAL_DI_CLASSES = [
    "t",  # Generic tooth marker (id 0)
    "1", "2", "3", "4", "5", "6", "7", "8",
    "9", "10", "11", "12", "13", "14", "15", "16",
    "17", "18", "19", "20", "21", "22", "23", "24",
    "25", "26", "27", "28", "29", "30", "31", "32",
    "Caries",
    "Crown",
    "Filling",
    "Implant",
    "Prefabricated metal post",
    "Retained root",
    "Root canal filling",
    "Root canal obturation"
]

NUM_CLASSES = 41


def get_augmentation_config():
    """Return augmentation transforms for training."""
    from detectron2.data import transforms as T
    
    return [
        T.ResizeShortestEdge(
            short_edge_length=(480, 512, 544, 576, 608),
            max_size=1024,
            sample_style="choice"
        ),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomBrightness(0.8, 1.2),
        T.RandomContrast(0.8, 1.2),
        T.RandomSaturation(0.8, 1.2),
        T.RandomRotation(angle=[-10, 10]),
    ]


def register_final_di_datasets(base_path: str = None):
    """Register the final-di dataset with Detectron2."""
    train_json = TRAIN_JSON
    train_images = TRAIN_IMAGES
    val_json = VAL_JSON
    val_images = VAL_IMAGES
    test_json = TEST_JSON
    test_images = TEST_IMAGES
    
    if base_path:
        train_json = os.path.join(base_path, TRAIN_JSON)
        train_images = os.path.join(base_path, TRAIN_IMAGES)
        val_json = os.path.join(base_path, VAL_JSON)
        val_images = os.path.join(base_path, VAL_IMAGES)
        test_json = os.path.join(base_path, TEST_JSON)
        test_images = os.path.join(base_path, TEST_IMAGES)
    
    registered = DatasetCatalog.list()
    
    datasets = [
        ("tooth_train", train_json, train_images),
        ("tooth_val", val_json, val_images),
        ("tooth_test", test_json, test_images),
    ]
    
    for name, json_path, img_dir in datasets:
        if name in registered:
            print(f"'{name}' already registered")
            continue
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found")
            continue
        
        register_coco_instances(name, {}, json_path, img_dir)
        
        metadata = MetadataCatalog.get(name)
        metadata.thing_classes = FINAL_DI_CLASSES
        
        with open(json_path) as f:
            coco = json.load(f)
        
        metadata.thing_dataset_id_to_contiguous_id = {
            cat['id']: idx for idx, cat in enumerate(coco['categories'])
        }
        
        print(f"Registered: {name} ({len(coco['images'])} images, {len(coco['annotations'])} annotations)")


def verify_dataset():
    """Verify datasets are properly registered."""
    print("\n=== Verification ===")
    for name in ["tooth_train", "tooth_val", "tooth_test"]:
        if name not in DatasetCatalog.list():
            print(f"✗ {name}: Not registered")
            continue
        dicts = DatasetCatalog.get(name)
        meta = MetadataCatalog.get(name)
        print(f"✓ {name}: {len(dicts)} samples, {len(meta.thing_classes)} classes")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', default=None)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    
    setup_logger()
    print(f"Final-DI Dataset ({NUM_CLASSES} classes)")
    register_final_di_datasets(args.base_path)
    if args.verify:
        verify_dataset()
    print("\nDone! Use 'tooth_train', 'tooth_val', 'tooth_test' in config.")


if __name__ == '__main__':
    main()
