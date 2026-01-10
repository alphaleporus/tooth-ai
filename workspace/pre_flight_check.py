#!/usr/bin/env python3
"""
Pre-Flight Health Check for Tooth-AI Training
Verifies hardware, dataset, config, and environment before training.
"""

import os
import sys
import tempfile
from pathlib import Path

# ASCII-safe symbols for Windows console
PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"

def print_header(title):
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print(f"{'=' * 50}")


def check_hardware():
    """Check CUDA availability and VRAM."""
    print_header("1. HARDWARE & VRAM CHECK")
    
    all_ok = True
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"{PASS} CUDA is available")
            gpu_name = torch.cuda.get_device_name(0)
            print(f"      GPU: {gpu_name}")
            
            # Check VRAM
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            free_vram = (torch.cuda.get_device_properties(0).total_memory - 
                        torch.cuda.memory_allocated(0)) / (1024**3)
            
            print(f"      Total VRAM: {total_vram:.1f} GB")
            print(f"      Free VRAM:  {free_vram:.1f} GB")
            
            if free_vram < 6.0:
                print(f"{WARN} Free VRAM < 6GB - training may OOM")
                all_ok = False
            else:
                print(f"{PASS} Sufficient VRAM available")
        else:
            print(f"{FAIL} CUDA is NOT available")
            all_ok = False
            
    except ImportError:
        print(f"{FAIL} PyTorch not installed")
        all_ok = False
    
    return all_ok


def check_dataset():
    """Verify dataset directory and registration."""
    print_header("2. DATASET INTEGRITY CHECK")
    
    all_ok = True
    
    # Check directory exists
    dataset_path = "data/final-di"
    if os.path.exists(dataset_path):
        print(f"{PASS} Dataset directory exists: {dataset_path}")
    else:
        print(f"{FAIL} Dataset directory NOT found: {dataset_path}")
        return False
    
    # Try registration
    try:
        sys.path.insert(0, 'workspace')
        from register_dataset import register_final_di_datasets
        register_final_di_datasets()
        print(f"{PASS} Dataset registration successful")
        
        # Check image count
        from detectron2.data import DatasetCatalog
        train_data = DatasetCatalog.get("tooth_train")
        num_images = len(train_data)
        
        if num_images > 0:
            print(f"{PASS} tooth_train has {num_images} images")
        else:
            print(f"{FAIL} tooth_train has 0 images")
            all_ok = False
            
    except Exception as e:
        print(f"{FAIL} Dataset registration failed: {e}")
        all_ok = False
    
    return all_ok


def check_config():
    """Load config and verify RTX 4060 safety limits."""
    print_header("3. CONFIGURATION SAFETY CHECK")
    
    all_ok = True
    config_path = "workspace/configs/mask_rcnn_1024x512.yaml"
    
    if not os.path.exists(config_path):
        print(f"{FAIL} Config file not found: {config_path}")
        return False
    
    print(f"{PASS} Config file exists")
    
    try:
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
        cfg.merge_from_file(config_path)
        
        # Check batch size
        batch_size = cfg.SOLVER.IMS_PER_BATCH
        if batch_size > 2:
            print(f"{FAIL} SOLVER.IMS_PER_BATCH = {batch_size} (RTX 4060 cannot handle >2)")
            all_ok = False
        else:
            print(f"{PASS} SOLVER.IMS_PER_BATCH = {batch_size} (safe for RTX 4060)")
        
        # Check workers
        num_workers = cfg.DATALOADER.NUM_WORKERS
        if num_workers > 2:
            print(f"{FAIL} DATALOADER.NUM_WORKERS = {num_workers} (Windows stability risk)")
            all_ok = False
        else:
            print(f"{PASS} DATALOADER.NUM_WORKERS = {num_workers} (safe for Windows)")
        
        # Print other key settings
        print(f"      MAX_ITER: {cfg.SOLVER.MAX_ITER}")
        print(f"      BASE_LR: {cfg.SOLVER.BASE_LR}")
        print(f"      CHECKPOINT_PERIOD: {cfg.SOLVER.CHECKPOINT_PERIOD}")
        
    except Exception as e:
        print(f"{FAIL} Config loading failed: {e}")
        all_ok = False
    
    return all_ok


def check_output_dir():
    """Check/create output directory and verify write permissions."""
    print_header("4. OUTPUT DIRECTORY CHECK")
    
    all_ok = True
    output_dir = "output"
    
    # Create if not exists
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"{PASS} Created output directory: {output_dir}")
        except Exception as e:
            print(f"{FAIL} Could not create output directory: {e}")
            return False
    else:
        print(f"{PASS} Output directory exists: {output_dir}")
    
    # Test write permissions
    test_file = os.path.join(output_dir, ".write_test")
    try:
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        print(f"{PASS} Write permissions verified")
    except Exception as e:
        print(f"{FAIL} No write permissions: {e}")
        all_ok = False
    
    return all_ok


def check_wandb():
    """Check WandB authentication."""
    print_header("5. WANDB AUTHENTICATION CHECK")
    
    all_ok = True
    
    try:
        import wandb
        
        # Check for API key
        if wandb.api.api_key:
            print(f"{PASS} WandB is logged in")
        else:
            # Check for .netrc file
            netrc_path = Path.home() / ".netrc"
            if netrc_path.exists():
                print(f"{PASS} WandB credentials found in .netrc")
            else:
                print(f"{FAIL} WandB NOT logged in")
                print("      Run: wandb login")
                all_ok = False
                
    except ImportError:
        print(f"{FAIL} WandB not installed")
        all_ok = False
    except Exception as e:
        print(f"{WARN} WandB check failed: {e}")
        all_ok = False
    
    return all_ok


def main():
    print("\n" + "=" * 50)
    print("  TOOTH-AI PRE-FLIGHT HEALTH CHECK")
    print("  RTX 4060 Training Environment")
    print("=" * 50)
    
    results = {}
    
    results['hardware'] = check_hardware()
    results['dataset'] = check_dataset()
    results['config'] = check_config()
    results['output'] = check_output_dir()
    results['wandb'] = check_wandb()
    
    # Summary
    print_header("SUMMARY")
    
    all_pass = True
    for check, passed in results.items():
        status = f"{PASS}" if passed else f"{FAIL}"
        print(f"  {check}: {status}")
        if not passed:
            all_pass = False
    
    print()
    if all_pass:
        print("*** ALL CHECKS PASSED - Ready for training! ***")
        return 0
    else:
        print(f"{WARN} Some checks failed. Fix issues before training.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
