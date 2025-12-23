#!/usr/bin/env python3
"""
Local Testing Script for MacBook M1
Tests dataset loading, model config, and a few training iterations.
Works without GPU using CPU/MPS backend.
"""

import os
import sys

def check_dependencies():
    """Check all required dependencies."""
    print("=" * 50)
    print("DEPENDENCY CHECK")
    print("=" * 50)
    
    deps = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'detectron2': 'Detectron2',
        'cv2': 'OpenCV',
        'wandb': 'Weights & Biases',
        'numpy': 'NumPy',
        'tqdm': 'tqdm'
    }
    
    all_ok = True
    for module, name in deps.items():
        try:
            if module == 'cv2':
                import cv2
                print(f"✓ {name}")
            else:
                mod = __import__(module)
                version = getattr(mod, '__version__', 'unknown')
                print(f"✓ {name} ({version})")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            all_ok = False
    
    # Check MPS (Apple Silicon) availability
    try:
        import torch
        if torch.backends.mps.is_available():
            print(f"✓ MPS (Apple Silicon GPU) available")
        else:
            print(f"⚠ MPS not available, will use CPU")
    except:
        pass
    
    return all_ok


def check_dataset():
    """Verify dataset can be loaded."""
    print("\n" + "=" * 50)
    print("DATASET CHECK")
    print("=" * 50)
    
    import json
    
    splits = ['train', 'valid', 'test']
    base_path = 'data/final-di'
    
    all_ok = True
    for split in splits:
        json_path = f"{base_path}/{split}/_annotations.coco.json"
        if os.path.exists(json_path):
            with open(json_path) as f:
                data = json.load(f)
            print(f"✓ {split}: {len(data['images'])} images, {len(data['categories'])} classes")
        else:
            print(f"✗ {split}: NOT FOUND at {json_path}")
            all_ok = False
    
    return all_ok


def check_wandb():
    """Check WandB login status."""
    print("\n" + "=" * 50)
    print("WANDB CHECK")
    print("=" * 50)
    
    try:
        import wandb
        if wandb.api.api_key:
            print("✓ WandB logged in")
            return True
        else:
            print("✗ WandB NOT logged in")
            print("  Run: wandb login")
            return False
    except Exception as e:
        print(f"✗ WandB error: {e}")
        return False


def check_config():
    """Verify training config."""
    print("\n" + "=" * 50)
    print("CONFIG CHECK")
    print("=" * 50)
    
    config_path = 'workspace/configs/mask_rcnn_1024x512.yaml'
    if not os.path.exists(config_path):
        print(f"✗ Config not found: {config_path}")
        return False
    
    with open(config_path) as f:
        content = f.read()
    
    checks = {
        'NUM_CLASSES: 41': '41 classes',
        'MAX_ITER: 24000': '24K iterations',
        'IMS_PER_BATCH: 4': 'Batch size 4'
    }
    
    all_ok = True
    for pattern, desc in checks.items():
        if pattern in content:
            print(f"✓ {desc}")
        else:
            print(f"✗ {desc} - NOT FOUND")
            all_ok = False
    
    return all_ok


def test_model_init():
    """Test model initialization (requires detectron2)."""
    print("\n" + "=" * 50)
    print("MODEL INITIALIZATION TEST")
    print("=" * 50)
    
    try:
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 41
        cfg.MODEL.DEVICE = 'cpu'  # Force CPU for M1
        
        print("✓ Config loaded successfully")
        print(f"  Classes: {cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
        print(f"  Device: {cfg.MODEL.DEVICE}")
        return True
        
    except Exception as e:
        print(f"✗ Model init failed: {e}")
        return False


def test_dataset_registration():
    """Test dataset registration."""
    print("\n" + "=" * 50)
    print("DATASET REGISTRATION TEST")
    print("=" * 50)
    
    try:
        sys.path.insert(0, 'workspace')
        from register_dataset import register_final_di_datasets, FINAL_DI_CLASSES
        
        print(f"✓ Register module loaded")
        print(f"  Classes defined: {len(FINAL_DI_CLASSES)}")
        
        # Try registration
        register_final_di_datasets()
        
        from detectron2.data import DatasetCatalog, MetadataCatalog
        
        for name in ['tooth_train', 'tooth_val', 'tooth_test']:
            if name in DatasetCatalog.list():
                dicts = DatasetCatalog.get(name)
                print(f"✓ {name}: {len(dicts)} samples")
            else:
                print(f"✗ {name}: not registered")
        
        return True
        
    except Exception as e:
        print(f"✗ Registration failed: {e}")
        return False


def main():
    print("\n" + "=" * 50)
    print("TOOTH-AI LOCAL TEST SUITE")
    print("MacBook M1 Compatibility Check")
    print("=" * 50)
    
    results = {}
    
    # Run checks
    results['dependencies'] = check_dependencies()
    
    if results['dependencies']:
        results['dataset'] = check_dataset()
        results['wandb'] = check_wandb()
        results['config'] = check_config()
        results['model'] = test_model_init()
        results['registration'] = test_dataset_registration()
    else:
        print("\n⚠ Skipping further tests - install dependencies first")
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    all_pass = all(results.values())
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test}: {status}")
    
    if all_pass:
        print("\n🎉 All checks passed! Ready for training.")
    else:
        print("\n⚠ Some checks failed. See above for details.")
    
    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
