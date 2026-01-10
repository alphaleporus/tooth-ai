import argparse
import os
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from register_dataset import register_final_di_datasets

def setup_cfg(args):
    cfg = get_cfg()
    # 1. Load the config
    cfg.merge_from_file(args.config_file)
    # 2. Force weights
    cfg.MODEL.WEIGHTS = args.model_path
    # 3. Force device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # 4. Standardize thresholds
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    return cfg

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="tooth_test")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    # 1. Register Data
    try:
        register_final_di_datasets("data/final-di")
    except AssertionError:
        pass

    # 2. Setup
    cfg = setup_cfg(args)
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join("output", "inference", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # 3. Create Evaluator
    print(f"Evaluating {args.dataset}...")
    evaluator = COCOEvaluator(args.dataset, output_dir=output_dir)
    
    # 4. Create Loader
    val_loader = build_detection_test_loader(cfg, args.dataset)
    
    # 5. Load Model
    predictor = DefaultPredictor(cfg)
    
    # 6. Run Inference
    print("Running inference... (This may take a few minutes)")
    results = inference_on_dataset(predictor.model, val_loader, evaluator)
    
    # 7. Print summary
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)
    if "bbox" in results:
        print(f"  Bbox mAP@[.5:.95]: {results['bbox']['AP']:.2f}%")
        print(f"  Bbox mAP@0.50:     {results['bbox']['AP50']:.2f}%")
    if "segm" in results:
        print(f"  Segm mAP@[.5:.95]: {results['segm']['AP']:.2f}%")
        print(f"  Segm mAP@0.50:     {results['segm']['AP50']:.2f}%")
    print(f"  Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
