#!/bin/bash
# Reproducibility script to regenerate metrics on a small test subset

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/results"
TEST_SUBSET="${SCRIPT_DIR}/small_subset"

echo "=========================================="
echo "Tooth-AI Metrics Reproduction Script"
echo "=========================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""

# Create directories
mkdir -p "${OUTPUT_DIR}" "${TEST_SUBSET}"

# Check if test subset exists
if [ ! -d "${TEST_SUBSET}" ] || [ -z "$(ls -A ${TEST_SUBSET} 2>/dev/null)" ]; then
    echo "Warning: Test subset directory is empty."
    echo "Please place test images in: ${TEST_SUBSET}"
    echo "Skipping metrics reproduction."
    exit 0
fi

# Check if inference engine exists
ENGINE_SCRIPT="${PROJECT_ROOT}/inference/engine.py"
if [ ! -f "${ENGINE_SCRIPT}" ]; then
    echo "Error: Inference engine not found: ${ENGINE_SCRIPT}"
    exit 1
fi

# Run inference on test subset
echo "Running inference on test subset..."
cd "${PROJECT_ROOT}"

python3 << EOF
import os
import sys
import json
import glob
from pathlib import Path

sys.path.insert(0, '.')

try:
    from inference.engine import load_engine
    
    # Load engine
    model_dir = os.path.join('.', 'models')
    engine = load_engine(model_dir)
    
    # Find test images
    test_images = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        test_images.extend(glob.glob(os.path.join('${TEST_SUBSET}', ext)))
        test_images.extend(glob.glob(os.path.join('${TEST_SUBSET}', '**', ext), recursive=True))
    
    if not test_images:
        print("No test images found in ${TEST_SUBSET}")
        sys.exit(0)
    
    print(f"Found {len(test_images)} test images")
    
    # Run inference
    results = []
    for img_path in test_images:
        try:
            result = engine.predict(img_path, return_visualization=False)
            results.append({
                'image': os.path.basename(img_path),
                'num_detections': result.get('num_detections', 0),
                'metadata': result.get('metadata', {})
            })
            print(f"  Processed: {os.path.basename(img_path)}")
        except Exception as e:
            print(f"  Error processing {img_path}: {e}")
            continue
    
    # Save results
    output_file = os.path.join('${OUTPUT_DIR}', 'results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'num_images': len(results),
            'results': results
        }, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {output_file}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

echo ""
echo "Reproduction complete!"
echo "Results saved to: ${OUTPUT_DIR}/results.json"



