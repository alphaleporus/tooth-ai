#!/usr/bin/env python3
"""
Training script for Mask R-CNN on tooth detection dataset.
Uses Detectron2 with wandb integration for experiment tracking.
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

# Import dataset registration
from register_dataset import register_tooth_dataset


class WandbTrainer(DefaultTrainer):
    """Custom trainer with wandb logging."""
    
    def __init__(self, cfg, wandb_project="tooth-poc", wandb_name=None):
        super().__init__(cfg)
        self.wandb_initialized = False
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
    
    def run_step(self):
        """
        Override to add wandb logging.
        """
        loss_dict = super().run_step()
        
        if not self.wandb_initialized:
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_name or self.cfg.OUTPUT_DIR.split('/')[-1],
                config=dict(self.cfg),
                reinit=True
            )
            self.wandb_initialized = True
        
        # Log losses
        if self.iter % 20 == 0:  # Log every 20 iterations
            log_dict = {}
            for key, value in loss_dict.items():
                log_dict[f"loss/{key}"] = value.item() if torch.is_tensor(value) else value
            log_dict["iter"] = self.iter
            log_dict["lr"] = self.optimizer.param_groups[0]['lr']
            wandb.log(log_dict)
        
        return loss_dict
    
    def after_step(self):
        """
        Log metrics after each step.
        """
        super().after_step()
        
        if self.iter % self.cfg.SOLVER.CHECKPOINT_PERIOD == 0:
            # Log checkpoint info
            wandb.log({"checkpoint/iter": self.iter})
    
    def after_train(self):
        """
        Finalize wandb run after training.
        """
        super().after_train()
        if self.wandb_initialized:
            wandb.finish()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    
    # Load base config from model zoo first
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    
    # Then override with custom config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    
    # Override with custom settings
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    # Register datasets if not already registered
    try:
        from detectron2.data import DatasetCatalog
        if "tooth_train" not in DatasetCatalog.list():
            # Register datasets
            coco_json = args.coco_json or "/data/niihhaa/coco_annotations.json"
            image_dir = args.image_dir or "/data/niihhaa/dataset"
            
            register_tooth_dataset(coco_json, image_dir, "tooth_train", split_ratio=0.8)
            register_tooth_dataset(coco_json, image_dir, "tooth_val", split_ratio=0.8)
    except Exception as e:
        print(f"Warning: Could not register datasets: {e}")
        print("Make sure to run register_dataset.py first or provide --coco-json and --image-dir")
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    """
    Main training function.
    """
    cfg = setup(args)
    
    # Create output directory
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Initialize wandb
    wandb_name = args.wandb_name or f"maskrcnn_r50_nh_{cfg.SOLVER.MAX_ITER}iter"
    
    # Create trainer
    trainer = WandbTrainer(cfg, wandb_project=args.wandb_project, wandb_name=wandb_name)
    
    # Resume from checkpoint if provided
    if args.resume:
        trainer.resume_or_load(resume=True)
    else:
        trainer.resume_or_load(resume=False)
    
    # Start training
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument(
        "--coco-json",
        type=str,
        default=None,
        help="Path to COCO JSON annotations file"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Directory containing images"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="tooth-poc",
        help="wandb project name"
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="wandb run name"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use"
    )
    
    args = parser.parse_args()
    
    # Setup logger
    setup_logger()
    
    # Launch training (single GPU for now)
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

