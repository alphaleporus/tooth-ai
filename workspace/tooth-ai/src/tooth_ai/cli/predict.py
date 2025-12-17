#!/usr/bin/env python3
"""
CLI tool for tooth detection inference.
"""

import argparse
import os
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.engine import ToothDetectionEngine, load_engine
from inference.visualize import visualize_predictions, save_visualization


def main():
    parser = argparse.ArgumentParser(
        description='Tooth detection inference CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic inference
  python cli/predict.py --image data/test.png --output result.json

  # With visualization
  python cli/predict.py --image data/test.png --output result.json --vis result.png

  # Custom model directory
  python cli/predict.py --image data/test.png --model-dir /path/to/models --output result.json
        """
    )
    
    parser.add_argument(
        '--image',
        type=str,
        required=True,
        help='Path to input image'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output JSON file'
    )
    
    parser.add_argument(
        '--vis',
        type=str,
        default=None,
        help='Path to save visualization image (optional)'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default=None,
        help='Path to model directory (default: workspace/tooth-ai/models)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device for inference'
    )
    
    parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.85,
        help='Confidence threshold for using classifier (default: 0.85)'
    )
    
    args = parser.parse_args()
    
    # Determine model directory
    if args.model_dir:
        model_dir = args.model_dir
    else:
        # Default to workspace/tooth-ai/models
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(script_dir, 'models')
    
    # Check if models exist
    maskrcnn_path = os.path.join(model_dir, 'maskrcnn_final.pth')
    effnet_path = os.path.join(model_dir, 'effnet_fdi_final.pth')
    config_path = os.path.join(model_dir, 'config.yaml')
    
    if not os.path.exists(maskrcnn_path):
        print(f"Error: Mask R-CNN model not found at {maskrcnn_path}")
        print("Please ensure models are in the model directory or specify --model-dir")
        return 1
    
    if not os.path.exists(effnet_path):
        print(f"Error: EfficientNet model not found at {effnet_path}")
        print("Please ensure models are in the model directory or specify --model-dir")
        return 1
    
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found at {config_path}")
        print("Using default config path...")
        # Try to find config in workspace/configs
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        workspace_dir = os.path.dirname(os.path.dirname(script_dir))
        config_path = os.path.join(workspace_dir, 'configs', 'mask_rcnn_1024x512.yaml')
        if not os.path.exists(config_path):
            print(f"Error: Config file not found")
            return 1
    
    # Check input image
    if not os.path.exists(args.image):
        print(f"Error: Input image not found: {args.image}")
        return 1
    
    # Load engine
    print("Loading inference engine...")
    try:
        engine = ToothDetectionEngine(
            maskrcnn_path,
            effnet_path,
            config_path,
            confidence_threshold=args.confidence_threshold,
            device=args.device
        )
        print("✓ Engine loaded successfully")
    except Exception as e:
        print(f"Error loading engine: {e}")
        return 1
    
    # Run inference
    print(f"\nRunning inference on {args.image}...")
    try:
        results = engine.predict(args.image, return_visualization=args.vis is not None)
        print(f"✓ Inference complete: {results['num_detections']} teeth detected")
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Save JSON output
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✓ Results saved to {args.output}")
    
    # Save visualization if requested
    if args.vis:
        if 'visualization' in results:
            import base64
            vis_bytes = base64.b64decode(results['visualization'])
            vis_dir = os.path.dirname(args.vis)
            if vis_dir:
                os.makedirs(vis_dir, exist_ok=True)
            
            with open(args.vis, 'wb') as f:
                f.write(vis_bytes)
            print(f"✓ Visualization saved to {args.vis}")
        else:
            # Generate visualization manually
            import cv2
            from inference.visualize import visualize_predictions
            
            image = cv2.imread(args.image)
            predictions = []
            for tooth in results['teeth']:
                # Convert RLE mask back to array if needed
                mask = tooth.get('mask')
                if isinstance(mask, dict):
                    from pycocotools import mask as mask_utils
                    mask = mask_utils.decode(mask)
                
                predictions.append({
                    'fdi': tooth['fdi'],
                    'bbox': tooth['bbox'],
                    'mask': mask,
                    'centroid': tooth['centroid'],
                    'final_confidence': tooth['score'],
                    'method_used': tooth['method_used'],
                    'correction_applied': tooth['correction_applied']
                })
            
            vis_image = visualize_predictions(image, predictions)
            save_visualization(vis_image, args.vis)
            print(f"✓ Visualization saved to {args.vis}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"  Teeth detected: {results['num_detections']}")
    print(f"  Mask R-CNN used: {results['metadata']['maskrcnn_used']}")
    print(f"  EfficientNet used: {results['metadata']['effnet_used']}")
    print(f"  Corrections applied: {results['metadata']['corrections_applied']}")
    print("\nDetected teeth:")
    for tooth in results['teeth']:
        print(f"  FDI {tooth['fdi']}: confidence={tooth['score']:.3f}, method={tooth['method_used']}")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

