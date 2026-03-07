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
from detectron2.evaluation import COCOEvaluator

# Import dataset registration
from register_dataset import register_final_di_datasets, register_remapped_datasets, register_merged_datasets, get_augmentation_config


class AugmentedTrainer(DefaultTrainer):
    """Custom trainer with data augmentation and optional wandb logging."""
    
    def __init__(self, cfg, wandb_project=None, wandb_name=None):
        super().__init__(cfg)
        self.wandb_initialized = False
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
    
    @classmethod
    def build_train_loader(cls, cfg):
        """
        Build train loader with custom augmentations and oversampling for rare classes.
        
        Implements Phase 2 fix for class imbalance:
        - Oversamples images containing rare classes (implant, caries, root_canal_filling, etc.)
        - Uses RepeatFactorTrainingSampler to ensure model sees minority classes frequently
        """
        from detectron2.data import DatasetCatalog, MetadataCatalog
        from detectron2.data.samplers import RepeatFactorTrainingSampler
        from detectron2.data.build import get_detection_dataset_dicts
        
        # Get dataset dicts
        dataset_name = cfg.DATASETS.TRAIN[0]
        dataset_dicts = get_detection_dataset_dicts(
            cfg.DATASETS.TRAIN,
            filter_empty=cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS,
            min_keypoints=cfg.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE
            if cfg.MODEL.KEYPOINT_ON else 0,
            proposal_files=cfg.DATASETS.PROPOSAL_FILES_TRAIN if cfg.MODEL.LOAD_PROPOSALS else None,
        )
        
        # Get augmentations
        augmentations = get_augmentation_config()
        
        mapper = DatasetMapper(
            cfg,
            is_train=True,
            augmentations=augmentations
        )
        
        # ============================================
        # CRITICAL FIX: RepeatFactorTrainingSampler
        # ============================================
        # Oversample images containing rare classes
        # This ensures the model actually sees minority classes during training
        # rather than just having higher loss weights
        
        if cfg.DATALOADER.SAMPLER_TRAIN == "RepeatFactorTrainingSampler":
            # Calculate repeat factors based on category frequency
            # repeat_thresh=0.001 means categories with < 0.1% representation
            # will be repeated more frequently
            repeat_factors = RepeatFactorTrainingSampler.repeat_factors_from_category_frequency(
                dataset_dicts, 
                repeat_thresh=0.001  # Heavily oversample rare classes
            )
            sampler = RepeatFactorTrainingSampler(repeat_factors)
            
            print("\n" + "="*60)
            print("CLASS IMBALANCE HANDLING ENABLED")
            print("="*60)
            print(f"Using RepeatFactorTrainingSampler with thresh=0.001")
            print(f"Rare classes (implant, caries, posts) will be oversampled")
            print(f"Total repeat factor sum: {sum(repeat_factors):.1f}")
            print(f"Max repeat factor: {max(repeat_factors):.2f}")
            print(f"Min repeat factor: {min(repeat_factors):.2f}")
            print("="*60 + "\n")
        else:
            sampler = None  # Use default sampler
        
        return build_detection_train_loader(
            cfg, 
            mapper=mapper,
            sampler=sampler,
            total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
            aspect_ratio_grouping=cfg.DATALOADER.ASPECT_RATIO_GROUPING,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
        )
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Build COCO evaluator for validation during training."""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        os.makedirs(output_folder, exist_ok=True)
        return COCOEvaluator(dataset_name, output_dir=output_folder)
    
    def run_step(self):
        """Override to add wandb logging."""
        super().run_step()
        
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
            # Get losses from storage (not from return value)
            storage = self.storage
            log_dict = {
                "iter": self.iter,
                "lr": self.optimizer.param_groups[0]['lr']
            }
            # Access the latest scalars from storage
            for key in storage.latest().keys():
                if "loss" in key:
                    log_dict[f"loss/{key}"] = storage.latest()[key][0]
            wandb.log(log_dict)
    
    def after_train(self):
        """Finalize wandb run after training."""
        super().after_train()
        if self.wandb_initialized:
            wandb.finish()


def setup(args):
    """Create configs and perform basic setups."""
    cfg = get_cfg()
    
    # Check if config is standalone (has full MODEL definition) or needs base
    config_content = ""
    if args.config_file:
        with open(args.config_file, 'r') as f:
            config_content = f.read()
    
    # Only load model zoo base if config doesn't define META_ARCHITECTURE
    if "META_ARCHITECTURE" not in config_content:
        # Load base config from model zoo (for original R50 config)
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
            register_remapped_datasets()  # Also register 9-class remapped dataset
            register_merged_datasets()    # Register merged 9-class dataset
    except Exception as e:
        print(f"Warning: Could not register datasets: {e}")
        print("Make sure data/final-di-stratified exists with train/valid/test splits")
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg
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
