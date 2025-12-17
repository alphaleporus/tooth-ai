#!/usr/bin/env python3
"""
Comprehensive ONNX export script for Mask R-CNN and EfficientNet.
Includes validation with numerical comparison.
"""

import argparse
import os
import sys
import torch
import torch.onnx
import numpy as np
import cv2
from pathlib import Path
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import timm
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultPredictor
from detectron2.structures import Instances


def export_effnet_onnx(model_path: str, output_path: str, num_classes: int = 32):
    """Export EfficientNet to ONNX with validation."""
    print(f"\n{'='*60}")
    print(f"Exporting EfficientNet to ONNX")
    print(f"{'='*60}")
    
    # Load model
    print("Loading EfficientNet model...")
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    # Create dummy input (128x128 ROI)
    dummy_input = torch.randn(1, 3, 128, 128)
    
    print(f"Exporting to {output_path}...")
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
        opset_version=13,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"✓ EfficientNet exported to {output_path}")
    return model, dummy_input


def export_maskrcnn_onnx(model_path: str, config_path: str, output_path: str):
    """
    Export Mask R-CNN to ONNX.
    Note: Full Mask R-CNN is complex, we export the backbone + simplified forward.
    """
    print(f"\n{'='*60}")
    print(f"Exporting Mask R-CNN to ONNX")
    print(f"{'='*60}")
    
    # Load config
    print("Loading Detectron2 config...")
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = 'cpu'  # Export on CPU
    
    # Build model
    print("Building Mask R-CNN model...")
    model = build_model(cfg)
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    model.eval()
    
    # For ONNX export, we'll export the backbone
    # Full Mask R-CNN requires custom operators
    backbone = model.backbone
    backbone.eval()
    
    # Create dummy input (typical OPG size)
    dummy_input = torch.randn(1, 3, 512, 1024)
    
    print(f"Exporting backbone to {output_path}...")
    # Export backbone
    torch.onnx.export(
        backbone,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['p2', 'p3', 'p4', 'p5', 'p6'],
        opset_version=13,
        do_constant_folding=True,
        export_params=True
    )
    
    print(f"✓ Mask R-CNN backbone exported to {output_path}")
    print("  Note: Full Mask R-CNN requires Detectron2 custom ops.")
    return model, dummy_input


def validate_onnx_vs_pytorch(onnx_path: str, pytorch_model, dummy_input: torch.Tensor,
                             tolerance: float = 1e-2, num_tests: int = 3):
    """
    Validate ONNX model by comparing outputs with PyTorch.
    
    Returns:
        bool: True if validation passes
    """
    try:
        import onnx
        import onnxruntime as ort
        
        print(f"\nValidating ONNX model: {onnx_path}")
        
        # Load and check ONNX model
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("  ✓ ONNX model structure is valid")
        
        # Create ONNX Runtime session
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        
        # Get PyTorch output
        with torch.no_grad():
            pytorch_output = pytorch_model(dummy_input)
        
        # Convert to numpy
        if isinstance(pytorch_output, (list, tuple)):
            pytorch_output = pytorch_output[0]
        pytorch_np = pytorch_output.cpu().numpy()
        
        # Run ONNX inference
        onnx_input = dummy_input.cpu().numpy()
        onnx_outputs = ort_session.run(None, {input_name: onnx_input})
        onnx_output = onnx_outputs[0]
        
        # Compare outputs
        max_diff = np.abs(pytorch_np - onnx_output).max()
        mean_diff = np.abs(pytorch_np - onnx_output).mean()
        
        print(f"  Max difference: {max_diff:.6f}")
        print(f"  Mean difference: {mean_diff:.6f}")
        print(f"  Tolerance: {tolerance}")
        
        if max_diff < tolerance:
            print(f"  ✓ Validation PASSED (max diff < tolerance)")
            return True
        else:
            print(f"  ✗ Validation FAILED (max diff >= tolerance)")
            return False
            
    except ImportError:
        print("  Warning: onnx or onnxruntime not installed")
        print("  Install with: pip install onnx onnxruntime")
        return False
    except Exception as e:
        print(f"  Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_onnx_inference(onnx_path: str, test_images: List[str], num_samples: int = 3):
    """Test ONNX model on real images."""
    try:
        import onnxruntime as ort
        
        print(f"\nTesting ONNX inference on {min(num_samples, len(test_images))} images...")
        
        ort_session = ort.InferenceSession(onnx_path)
        input_name = ort_session.get_inputs()[0].name
        input_shape = ort_session.get_inputs()[0].shape
        
        # Handle dynamic batch size
        if input_shape[0] == 'batch_size' or input_shape[0] is None:
            batch_size = 1
        else:
            batch_size = input_shape[0]
        
        h, w = input_shape[2], input_shape[3] if len(input_shape) > 3 else 128
        
        for i, img_path in enumerate(test_images[:num_samples]):
            if not os.path.exists(img_path):
                continue
                
            # Load and preprocess image
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Resize and normalize
            img_resized = cv2.resize(img, (w, h))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype(np.float32) / 255.0
            img_tensor = np.transpose(img_norm, (2, 0, 1))
            img_batch = np.expand_dims(img_tensor, 0)
            
            # Run inference
            outputs = ort_session.run(None, {input_name: img_batch})
            
            print(f"  Image {i+1}: {os.path.basename(img_path)}")
            print(f"    Output shape: {outputs[0].shape}")
            print(f"    Output range: [{outputs[0].min():.4f}, {outputs[0].max():.4f}]")
        
        print("  ✓ ONNX inference test completed")
        return True
        
    except Exception as e:
        print(f"  Error during inference test: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export all models to ONNX')
    parser.add_argument('--maskrcnn', type=str, required=True,
                       help='Path to Mask R-CNN model (.pth)')
    parser.add_argument('--effnet', type=str, required=True,
                       help='Path to EfficientNet model (.pth)')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to Detectron2 config (.yaml)')
    parser.add_argument('--out', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--num-classes', type=int, default=32,
                       help='Number of FDI classes')
    parser.add_argument('--test-images', type=str, nargs='+', default=[],
                       help='Test images for validation')
    parser.add_argument('--tolerance', type=float, default=1e-2,
                       help='Numerical tolerance for validation')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip numerical validation')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.out, exist_ok=True)
    
    # Export EfficientNet
    effnet_output = os.path.join(args.out, 'effnet.onnx')
    effnet_model, effnet_dummy = export_effnet_onnx(
        args.effnet, effnet_output, args.num_classes
    )
    
    # Export Mask R-CNN
    maskrcnn_output = os.path.join(args.out, 'maskrcnn.onnx')
    maskrcnn_model, maskrcnn_dummy = export_maskrcnn_onnx(
        args.maskrcnn, args.config, maskrcnn_output
    )
    
    # Validate models
    if not args.skip_validation:
        print(f"\n{'='*60}")
        print("Validating Exported Models")
        print(f"{'='*60}")
        
        # Validate EfficientNet
        effnet_valid = validate_onnx_vs_pytorch(
            effnet_output, effnet_model, effnet_dummy, args.tolerance
        )
        
        # Validate Mask R-CNN (backbone)
        maskrcnn_valid = validate_onnx_vs_pytorch(
            maskrcnn_output, maskrcnn_model.backbone, maskrcnn_dummy, args.tolerance
        )
        
        # Test on real images if provided
        if args.test_images:
            test_onnx_inference(effnet_output, args.test_images, num_samples=3)
            test_onnx_inference(maskrcnn_output, args.test_images, num_samples=3)
    
    # Save export metadata
    metadata = {
        'effnet_onnx': effnet_output,
        'maskrcnn_onnx': maskrcnn_output,
        'num_classes': args.num_classes,
        'opset_version': 13,
        'validation_passed': effnet_valid and maskrcnn_valid if not args.skip_validation else None
    }
    
    metadata_path = os.path.join(args.out, 'export_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Export Complete!")
    print(f"{'='*60}")
    print(f"EfficientNet ONNX: {effnet_output}")
    print(f"Mask R-CNN ONNX: {maskrcnn_output}")
    print(f"Metadata: {metadata_path}")
    print(f"\nNext step: Convert to TensorRT using convert_to_tensorrt.sh")


if __name__ == '__main__':
    main()

