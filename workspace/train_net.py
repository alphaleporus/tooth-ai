#!/usr/bin/env python3
"""
Training script for Mask R-CNN on tooth detection dataset.
Uses Detectron2 with wandb integration and custom augmentations.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import wandb
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T

# Import dataset registration
from register_dataset import register_final_di_datasets, get_augmentation_config


class AugmentedTrainer(DefaultTrainer):
    """Custom trainer with data augmentation and optional wandb logging."""
    
    def __init__(self, cfg, wandb_project=None, wandb_name=None):
        super().__init__(cfg)
        self.wandb_initialized = False
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
    
    @classmethod
    def build_train_loader(cls, cfg):
        """Build train loader with custom augmentations."""
        # Get augmentations
        augmentations = get_augmentation_config()
        
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=augmentations
        )
        
        return build_detection_train_loader(cfg, mapper=mapper)
    
    def run_step(self):
        """Override to add wandb logging."""
        loss_dict = super().run_step()
        
        if self.wandb_project and not self.wandb_initialized:
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_name or self.cfg.OUTPUT_DIR.split('/')[-1],
                config=dict(self.cfg),
                reinit=True
            )
            self.wandb_initialized = True
        
        # Log losses every 20 iterations
        if self.wandb_initialized and self.iter % 20 == 0:
            log_dict = {}
            for key, value in loss_dict.items():
                log_dict[f"loss/{key}"] = value.item() if torch.is_tensor(value) else value
            log_dict["iter"] = self.iter
            log_dict["lr"] = self.optimizer.param_groups[0]['lr']
            wandb.log(log_dict)
        
        return loss_dict
    
    def after_train(self):
        """Finalize wandb run after training."""
        super().after_train()
        if self.wandb_initialized:
            wandb.finish()


def setup(args):
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    
    # Load base config from model zoo
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    
    # Override with custom config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    
    # Override with command line options
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    # Register datasets
    try:
        from detectron2.data import DatasetCatalog
        if "tooth_train" not in DatasetCatalog.list():
            register_final_di_datasets(args.base_path)
    except Exception as e:
        print(f"Warning: Could not register datasets: {e}")
        print("Make sure data/final-di exists with train/valid/test splits")
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    """Main training function."""
    cfg = setup(args)
    
    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Create trainer with optional wandb
    wandb_project = args.wandb_project if args.use_wandb else None
    wandb_name = args.wandb_name
    
    trainer = AugmentedTrainer(cfg, wandb_project=wandb_project, wandb_name=wandb_name)
    
    # Resume from checkpoint if provided
    trainer.resume_or_load(resume=args.resume)
    
    # Log augmentation info
    print("\n" + "="*50)
    print("TRAINING WITH AUGMENTATIONS:")
    print("  - Multi-scale resize (480-608)")
    print("  - Random horizontal flip (p=0.5)")
    print("  - Random brightness (0.8-1.2)")
    print("  - Random contrast (0.8-1.2)")
    print("  - Random saturation (0.8-1.2)")
    print("  - Random rotation (±10°)")
    print("="*50 + "\n")
    
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="Base path to project root"
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable wandb logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="tooth-ai",
        help="wandb project name"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="wandb run name"
    )
    
    args = parser.parse_args()
    
    setup_logger()
    
    # Launch training
    if args.num_gpus == 1:
        main(args)
    else:
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
