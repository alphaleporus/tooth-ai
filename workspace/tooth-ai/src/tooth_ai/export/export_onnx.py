"""
Export models to ONNX format for deployment.
"""

import argparse
import os
import torch
import torch.onnx
import numpy as np
import cv2
from pathlib import Path

import timm
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


def export_effnet_onnx(model_path: str, output_path: str, num_classes: int = 32):
    """Export EfficientNet to ONNX."""
    print(f"Exporting EfficientNet to ONNX...")
    
    # Load model
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 128, 128)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=11,
        do_constant_folding=True
    )
    
    print(f"✓ EfficientNet exported to {output_path}")


def export_maskrcnn_onnx(model_path: str, config_path: str, output_path: str):
    """Export Mask R-CNN to ONNX (simplified - backbone only)."""
    print(f"Exporting Mask R-CNN backbone to ONNX...")
    
    # Load config and model
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_path
    
    # Build model
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()
    
    # Export backbone only (full Mask R-CNN is complex for ONNX)
    backbone = model.backbone
    backbone.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 800, 1024)
    
    # Export backbone
    torch.onnx.export(
        backbone,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['features'],
        opset_version=11,
        do_constant_folding=True
    )
    
    print(f"✓ Mask R-CNN backbone exported to {output_path}")
    print("  Note: Full Mask R-CNN export requires custom operators.")


def validate_onnx(onnx_path: str, num_samples: int = 3):
    """Validate ONNX model by running inference."""
    try:
        import onnx
        import onnxruntime as ort
        
        print(f"\nValidating ONNX model: {onnx_path}")
        
        # Load model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Create inference session
        ort_session = ort.InferenceSession(onnx_path)
        
        # Get input shape
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        
        print(f"  Input shape: {input_shape}")
        print(f"  Running {num_samples} test inferences...")
        
        for i in range(num_samples):
            # Create dummy input
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # Run inference
            outputs = ort_session.run(None, {input_name: dummy_input})
            
            print(f"  Test {i+1}: Output shape = {outputs[0].shape}")
        
        print(f"✓ ONNX model validation passed")
        return True
        
    except ImportError:
        print("  Warning: onnx or onnxruntime not installed, skipping validation")
        return False
    except Exception as e:
        print(f"  Error during validation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export models to ONNX')
    parser.add_argument('--effnet', type=str, required=True, help='Path to EfficientNet model')
    parser.add_argument('--maskrcnn', type=str, required=True, help='Path to Mask R-CNN model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--num-classes', type=int, default=32, help='Number of classes')
    parser.add_argument('--validate', action='store_true', help='Validate exported models')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Export EfficientNet
    effnet_output = os.path.join(args.output_dir, 'effnet.onnx')
    export_effnet_onnx(args.effnet, effnet_output, args.num_classes)
    
    if args.validate:
        validate_onnx(effnet_output)
    
    # Export Mask R-CNN backbone
    maskrcnn_output = os.path.join(args.output_dir, 'maskrcnn_backbone.onnx')
    export_maskrcnn_onnx(args.maskrcnn, args.config, maskrcnn_output)
    
    if args.validate:
        validate_onnx(maskrcnn_output)
    
    print(f"\nExport complete! Models saved to {args.output_dir}")


if __name__ == '__main__':
    main()

