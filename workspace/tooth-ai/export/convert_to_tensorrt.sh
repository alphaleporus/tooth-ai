#!/bin/bash
# TensorRT conversion script for ONNX models
# Converts ONNX models to TensorRT engines (FP16 and optionally INT8)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}"
LOG_FILE="${OUTPUT_DIR}/trt_conversion_log.txt"

echo "=========================================="
echo "TensorRT Conversion Script"
echo "=========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo "Log file: ${LOG_FILE}"
echo ""

# Check if TensorRT is available
if ! command -v trtexec &> /dev/null; then
    echo "ERROR: trtexec not found. Please install TensorRT."
    echo "TensorRT is typically installed with:"
    echo "  - NVIDIA TensorRT SDK"
    echo "  - Or via pip: pip install nvidia-tensorrt"
    exit 1
fi

# Check for ONNX models
EFFNET_ONNX="${OUTPUT_DIR}/effnet.onnx"
MASKRCNN_ONNX="${OUTPUT_DIR}/maskrcnn.onnx"

if [ ! -f "${EFFNET_ONNX}" ]; then
    echo "ERROR: EfficientNet ONNX not found: ${EFFNET_ONNX}"
    echo "Please run export_all_to_onnx.py first."
    exit 1
fi

if [ ! -f "${MASKRCNN_ONNX}" ]; then
    echo "ERROR: Mask R-CNN ONNX not found: ${MASKRCNN_ONNX}"
    echo "Please run export_all_to_onnx.py first."
    exit 1
fi

# Function to convert ONNX to TensorRT
convert_to_tensorrt() {
    local onnx_path=$1
    local engine_name=$2
    local precision=$3
    local calib_cache=$4
    
    echo ""
    echo "Converting: $(basename ${onnx_path})"
    echo "  Engine: ${engine_name}"
    echo "  Precision: ${precision}"
    
    local cmd="trtexec --onnx=${onnx_path} --saveEngine=${engine_name}"
    
    if [ "${precision}" == "fp16" ]; then
        cmd="${cmd} --fp16"
    elif [ "${precision}" == "int8" ]; then
        cmd="${cmd} --int8"
        if [ -n "${calib_cache}" ]; then
            cmd="${cmd} --calib=${calib_cache}"
        fi
    fi
    
    # Add common options
    cmd="${cmd} --workspace=4096 --verbose"
    
    echo "  Command: ${cmd}"
    echo ""
    
    # Run conversion
    if eval "${cmd}" >> "${LOG_FILE}" 2>&1; then
        echo "  ✓ Conversion successful"
        if [ -f "${engine_name}" ]; then
            local size=$(du -h "${engine_name}" | cut -f1)
            echo "  Engine size: ${size}"
        fi
    else
        echo "  ✗ Conversion failed. Check ${LOG_FILE} for details."
        return 1
    fi
}

# Start logging
echo "TensorRT Conversion Log" > "${LOG_FILE}"
echo "Started: $(date)" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

# Convert EfficientNet to FP16
EFFNET_FP16="${OUTPUT_DIR}/effnet_trt_fp16.plan"
echo "=========================================="
echo "Converting EfficientNet to TensorRT FP16"
echo "=========================================="
convert_to_tensorrt "${EFFNET_ONNX}" "${EFFNET_FP16}" "fp16" ""

# Convert Mask R-CNN to FP16
MASKRCNN_FP16="${OUTPUT_DIR}/maskrcnn_trt_fp16.plan"
echo ""
echo "=========================================="
echo "Converting Mask R-CNN to TensorRT FP16"
echo "=========================================="
convert_to_tensorrt "${MASKRCNN_ONNX}" "${MASKRCNN_FP16}" "fp16" ""

# Optional: INT8 conversion (requires calibration)
if [ "${ENABLE_INT8:-0}" == "1" ]; then
    echo ""
    echo "=========================================="
    echo "INT8 Conversion (Optional)"
    echo "=========================================="
    echo "Note: INT8 requires calibration data."
    echo "Skipping INT8 conversion by default."
    echo "To enable, set ENABLE_INT8=1 and provide calibration cache."
    
    # Uncomment to enable INT8:
    # CALIB_CACHE="${OUTPUT_DIR}/int8_calib.cache"
    # EFFNET_INT8="${OUTPUT_DIR}/effnet_trt_int8.plan"
    # convert_to_tensorrt "${EFFNET_ONNX}" "${EFFNET_INT8}" "int8" "${CALIB_CACHE}"
fi

# Summary
echo ""
echo "=========================================="
echo "Conversion Summary"
echo "=========================================="
echo "EfficientNet FP16: ${EFFNET_FP16}"
echo "Mask R-CNN FP16: ${MASKRCNN_FP16}"
echo "Log file: ${LOG_FILE}"
echo ""
echo "✓ TensorRT conversion complete!"
echo ""
echo "Next steps:"
echo "  1. Benchmark TensorRT engines: python export/benchmark_inference.py"
echo "  2. Integrate TensorRT engines into inference pipeline"



