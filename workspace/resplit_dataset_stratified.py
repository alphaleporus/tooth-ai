#!/usr/bin/env python3
"""
Priority-Based Stratified Dataset Re-Splitter
Ensures rare classes (Implant, Caries, Crown) appear in all splits.
"""

import json
import os
import shutil
import random
from collections import defaultdict
from pathlib import Path

# Configuration
DATASET_ROOT = "data/final-di"
OUTPUT_ROOT = "data/final-di-stratified"
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10
RANDOM_SEED = 42

# Define class rarity (lower = rarer, higher priority for stratification)
# Classes with fewer instances get higher priority
CLASS_PRIORITY = {
    "Implant": 1,
    "Retained root": 2,
    "Prefabricated metal post": 3,
    "Crown": 4,
    "Root canal filling": 5,
    "Caries": 6,
    "Root canal obturation": 7,
    # Tooth numbers are common
}


def load_all_annotations():
    """Load and merge annotations from all splits."""
    all_images = []
    all_annotations = []
    all_categories = None
    
    image_id_offset = 0
    ann_id_offset = 0
    
    for split in ["train", "valid", "test"]:
        json_path = os.path.join(DATASET_ROOT, split, "_annotations.coco.json")
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found, skipping...")
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Store categories (should be same across all)
        if all_categories is None:
            all_categories = data['categories']
        
        # Remap image IDs and track original file paths
        id_map = {}
        for img in data['images']:
            old_id = img['id']
            new_id = image_id_offset + old_id
            id_map[old_id] = new_id
            
            img['id'] = new_id
            img['original_split'] = split
            img['original_path'] = os.path.join(DATASET_ROOT, split, img['file_name'])
            all_images.append(img)
        
        # Remap annotation IDs and image references
        for ann in data['annotations']:
            ann['id'] = ann_id_offset + ann['id']
            ann['image_id'] = id_map[ann['image_id']]
            all_annotations.append(ann)
            ann_id_offset += 1
        
        image_id_offset += len(data['images']) + 1
        
    print(f"Loaded {len(all_images)} images with {len(all_annotations)} annotations")
    return all_images, all_annotations, all_categories


def get_category_name_map(categories):
    """Create mapping from category ID to name."""
    return {cat['id']: cat['name'] for cat in categories}


def get_image_priority(image_id, annotations, cat_map):
    """
    Get priority tag for an image based on its rarest class.
    Returns (priority_score, category_name)
    """
    image_anns = [a for a in annotations if a['image_id'] == image_id]
    
    if not image_anns:
        return (999, "no_annotations")
    
    best_priority = 999
    best_category = "common"
    
    for ann in image_anns:
        cat_name = cat_map.get(ann['category_id'], "unknown")
        priority = CLASS_PRIORITY.get(cat_name, 100)  # Common classes get 100
        
        if priority < best_priority:
            best_priority = priority
            best_category = cat_name
    
    return (best_priority, best_category)


def stratified_split(images, annotations, categories):
    """
    Split images ensuring rare classes appear in all splits.
    """
    random.seed(RANDOM_SEED)
    cat_map = get_category_name_map(categories)
    
    # Tag each image with its priority
    image_priorities = {}
    for img in images:
        priority, category = get_image_priority(img['id'], annotations, cat_map)
        image_priorities[img['id']] = (priority, category)
    
    # Group images by their priority category
    category_groups = defaultdict(list)
    for img in images:
        _, category = image_priorities[img['id']]
        category_groups[category].append(img)
    
    # Print distribution before split
    print("\n=== Class Distribution (Pre-Split) ===")
    for cat, imgs in sorted(category_groups.items(), key=lambda x: len(x[1])):
        print(f"  {cat}: {len(imgs)} images")
    
    # Stratified split within each category group
    train_images = []
    val_images = []
    test_images = []
    
    for category, cat_images in category_groups.items():
        random.shuffle(cat_images)
        n = len(cat_images)
        
        n_train = max(1, int(n * TRAIN_RATIO))
        n_val = max(1, int(n * VAL_RATIO)) if n > 2 else 0
        n_test = n - n_train - n_val
        
        # Ensure at least 1 in test for rare classes
        if category in CLASS_PRIORITY and n >= 3:
            n_test = max(1, n_test)
            n_train = n - n_val - n_test
        
        train_images.extend(cat_images[:n_train])
        val_images.extend(cat_images[n_train:n_train + n_val])
        test_images.extend(cat_images[n_train + n_val:])
    
    # Shuffle within splits
    random.shuffle(train_images)
    random.shuffle(val_images)
    random.shuffle(test_images)
    
    return train_images, val_images, test_images


def create_split_annotations(images, all_annotations):
    """Create annotation list for a specific image split."""
    image_ids = {img['id'] for img in images}
    return [ann for ann in all_annotations if ann['image_id'] in image_ids]


def save_split(images, annotations, categories, split_name, output_dir):
    """Save a split to disk."""
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    # Remap IDs for clean output
    new_images = []
    new_annotations = []
    old_to_new_img_id = {}
    
    for new_id, img in enumerate(images, 1):
        old_to_new_img_id[img['id']] = new_id
        new_img = {
            'id': new_id,
            'file_name': os.path.basename(img['file_name']),
            'width': img['width'],
            'height': img['height']
        }
        new_images.append(new_img)
        
        # Copy image file
        src_path = img['original_path']
        dst_path = os.path.join(split_dir, new_img['file_name'])
        if os.path.exists(src_path) and not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
    
    for new_id, ann in enumerate(annotations, 1):
        if ann['image_id'] in old_to_new_img_id:
            new_ann = ann.copy()
            new_ann['id'] = new_id
            new_ann['image_id'] = old_to_new_img_id[ann['image_id']]
            new_annotations.append(new_ann)
    
    # Save COCO JSON
    coco_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': categories
    }
    
    json_path = os.path.join(split_dir, "_annotations.coco.json")
    with open(json_path, 'w') as f:
        json.dump(coco_data, f)
    
    print(f"  {split_name}: {len(new_images)} images, {len(new_annotations)} annotations")


def verify_rare_classes(output_dir, categories):
    """Verify rare classes exist in all splits."""
    cat_map = {cat['id']: cat['name'] for cat in categories}
    rare_classes = set(CLASS_PRIORITY.keys())
    
    print("\n=== Rare Class Verification ===")
    
    for split in ["train", "valid", "test"]:
        json_path = os.path.join(output_dir, split, "_annotations.coco.json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        class_counts = defaultdict(int)
        for ann in data['annotations']:
            cat_name = cat_map.get(ann['category_id'], "unknown")
            if cat_name in rare_classes:
                class_counts[cat_name] += 1
        
        print(f"\n  {split}:")
        for cls in sorted(rare_classes):
            count = class_counts.get(cls, 0)
            status = "OK" if count > 0 else "MISSING"
            print(f"    {cls}: {count} [{status}]")


def main():
    print("=" * 50)
    print("Priority-Based Stratified Dataset Re-Splitter")
    print("=" * 50)
    
    # Load all data
    print("\n[1/4] Loading all annotations...")
    all_images, all_annotations, categories = load_all_annotations()
    
    # Perform stratified split
    print("\n[2/4] Performing stratified split...")
    train_imgs, val_imgs, test_imgs = stratified_split(
        all_images, all_annotations, categories
    )
    
    print(f"\n=== Split Sizes ===")
    print(f"  Train: {len(train_imgs)} ({len(train_imgs)/len(all_images)*100:.1f}%)")
    print(f"  Val:   {len(val_imgs)} ({len(val_imgs)/len(all_images)*100:.1f}%)")
    print(f"  Test:  {len(test_imgs)} ({len(test_imgs)/len(all_images)*100:.1f}%)")
    
    # Create annotations for each split
    train_anns = create_split_annotations(train_imgs, all_annotations)
    val_anns = create_split_annotations(val_imgs, all_annotations)
    test_anns = create_split_annotations(test_imgs, all_annotations)
    
    # Save splits
    print(f"\n[3/4] Saving to {OUTPUT_ROOT}...")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    save_split(train_imgs, train_anns, categories, "train", OUTPUT_ROOT)
    save_split(val_imgs, val_anns, categories, "valid", OUTPUT_ROOT)
    save_split(test_imgs, test_anns, categories, "test", OUTPUT_ROOT)
    
    # Verify rare classes
    print("\n[4/4] Verifying rare class distribution...")
    verify_rare_classes(OUTPUT_ROOT, categories)
    
    print("\n" + "=" * 50)
    print("DONE! New dataset saved to:", OUTPUT_ROOT)
    print("=" * 50)
    print("\nNext steps:")
    print("  1. Update register_dataset.py to use 'data/final-di-stratified'")
    print("  2. Run training with the new config")


if __name__ == "__main__":
    main()
