"""
Export models to TorchScript format for deployment.
"""

import argparse
import os
import torch
import numpy as np
from pathlib import Path

import timm
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


def export_effnet_torchscript(model_path: str, output_path: str, num_classes: int = 32):
    """Export EfficientNet to TorchScript."""
    print(f"Exporting EfficientNet to TorchScript...")
    
    # Load model
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, 128, 128)
    
    # Export to TorchScript
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(output_path)
    
    print(f"✓ EfficientNet exported to {output_path}")


def export_combined_torchscript(maskrcnn_model_path: str, effnet_model_path: str,
                                config_path: str, output_path: str, num_classes: int = 32):
    """
    Export combined pipeline to TorchScript.
    Note: This is a simplified version - full pipeline export is complex.
    """
    print(f"Exporting combined pipeline to TorchScript...")
    print("  Note: Full pipeline export requires custom wrapper.")
    
    # For now, export EfficientNet only
    # Full Mask R-CNN + EfficientNet pipeline would need custom wrapper class
    export_effnet_torchscript(effnet_model_path, output_path, num_classes)
    
    print(f"✓ Combined export (EfficientNet only) saved to {output_path}")


def validate_torchscript(ts_path: str, num_samples: int = 3):
    """Validate TorchScript model by running inference."""
    print(f"\nValidating TorchScript model: {ts_path}")
    
    try:
        # Load model
        model = torch.jit.load(ts_path)
        model.eval()
        
        # Get input shape from model
        # For EfficientNet, it's 1x3x128x128
        dummy_input = torch.randn(1, 3, 128, 128)
        
        print(f"  Running {num_samples} test inferences...")
        
        with torch.no_grad():
            for i in range(num_samples):
                output = model(dummy_input)
                print(f"  Test {i+1}: Output shape = {output.shape}")
        
        print(f"✓ TorchScript model validation passed")
        return True
        
    except Exception as e:
        print(f"  Error during validation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Export models to TorchScript')
    parser.add_argument('--effnet', type=str, required=True, help='Path to EfficientNet model')
    parser.add_argument('--maskrcnn', type=str, default=None, help='Path to Mask R-CNN model (optional)')
    parser.add_argument('--config', type=str, default=None, help='Path to config file (optional)')
    parser.add_argument('--output', type=str, required=True, help='Output path for combined model')
    parser.add_argument('--num-classes', type=int, default=32, help='Number of classes')
    parser.add_argument('--validate', action='store_true', help='Validate exported models')
    
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    if args.maskrcnn and args.config:
        # Export combined
        export_combined_torchscript(args.maskrcnn, args.effnet, args.config, args.output, args.num_classes)
    else:
        # Export EfficientNet only
        export_effnet_torchscript(args.effnet, args.output, args.num_classes)
    
    if args.validate:
        validate_torchscript(args.output)
    
    print(f"\nExport complete! Model saved to {args.output}")


if __name__ == '__main__':
    main()

