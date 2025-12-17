#!/usr/bin/env python3
"""
Performance benchmarking script for inference engines.
Compares PyTorch, ONNX Runtime, and TensorRT performance.
"""

import argparse
import os
import sys
import json
import time
import torch
import numpy as np
import cv2
import glob
from pathlib import Path
from typing import Dict, List, Tuple
import psutil
import gc

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def measure_memory():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB


def benchmark_pytorch_maskrcnn(model_path: str, config_path: str,
                               test_images: List[str], num_warmup: int = 3,
                               num_runs: int = 20) -> Dict:
    """Benchmark PyTorch Mask R-CNN."""
    print("\n" + "="*60)
    print("Benchmarking PyTorch Mask R-CNN")
    print("="*60)
    
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    
    # Load model
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    predictor = DefaultPredictor(cfg)
    
    latencies = []
    memory_usage = []
    
    # Warmup
    print(f"Warming up ({num_warmup} runs)...")
    for i in range(num_warmup):
        img = cv2.imread(test_images[0])
        _ = predictor(img)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    for i, img_path in enumerate(test_images[:num_runs]):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Measure memory before
        mem_before = measure_memory()
        
        # Time inference
        start = time.time()
        outputs = predictor(img)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        
        # Measure memory after
        mem_after = measure_memory()
        
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        memory_usage.append(mem_after - mem_before)
        
        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/{num_runs}")
    
    return {
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_std_ms': float(np.std(latencies)),
        'latency_min_ms': float(np.min(latencies)),
        'latency_max_ms': float(np.max(latencies)),
        'throughput_imgs_per_sec': float(1000.0 / np.mean(latencies)),
        'memory_usage_mb': float(np.mean(memory_usage)),
        'num_runs': len(latencies)
    }


def benchmark_pytorch_effnet(model_path: str, test_images: List[str],
                             num_classes: int = 32, num_warmup: int = 3,
                             num_runs: int = 20) -> Dict:
    """Benchmark PyTorch EfficientNet."""
    print("\n" + "="*60)
    print("Benchmarking PyTorch EfficientNet")
    print("="*60)
    
    import timm
    
    # Load model
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    latencies = []
    memory_usage = []
    
    # Warmup
    print(f"Warming up ({num_warmup} runs)...")
    dummy_input = torch.randn(1, 3, 128, 128).to(device)
    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    
    # Benchmark
    print(f"Benchmarking ({num_runs} runs)...")
    for i, img_path in enumerate(test_images[:num_runs]):
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Preprocess ROI (128x128)
        roi = cv2.resize(img, (128, 128))
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_norm = roi_rgb.astype(np.float32) / 255.0
        roi_tensor = torch.from_numpy(roi_norm).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Measure memory
        mem_before = measure_memory()
        
        # Time inference
        start = time.time()
        with torch.no_grad():
            _ = model(roi_tensor)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.time()
        
        mem_after = measure_memory()
        
        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)
        memory_usage.append(mem_after - mem_before)
        
        if (i + 1) % 5 == 0:
            print(f"  Completed {i+1}/{num_runs}")
    
    return {
        'latency_mean_ms': float(np.mean(latencies)),
        'latency_std_ms': float(np.std(latencies)),
        'latency_min_ms': float(np.min(latencies)),
        'latency_max_ms': float(np.max(latencies)),
        'throughput_imgs_per_sec': float(1000.0 / np.mean(latencies)),
        'memory_usage_mb': float(np.mean(memory_usage)),
        'num_runs': len(latencies)
    }


def benchmark_onnx(onnx_path: str, test_images: List[str],
                   input_size: Tuple[int, int] = (128, 128),
                   num_warmup: int = 3, num_runs: int = 20) -> Dict:
    """Benchmark ONNX Runtime."""
    try:
        import onnxruntime as ort
        
        print("\n" + "="*60)
        print(f"Benchmarking ONNX Runtime: {os.path.basename(onnx_path)}")
        print("="*60)
        
        # Create session
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        session = ort.InferenceSession(onnx_path, providers=providers)
        input_name = session.get_inputs()[0].name
        
        latencies = []
        memory_usage = []
        
        # Warmup
        print(f"Warming up ({num_warmup} runs)...")
        dummy_input = np.random.randn(1, 3, input_size[1], input_size[0]).astype(np.float32)
        for _ in range(num_warmup):
            _ = session.run(None, {input_name: dummy_input})
        
        # Benchmark
        print(f"Benchmarking ({num_runs} runs)...")
        for i, img_path in enumerate(test_images[:num_runs]):
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Preprocess
            roi = cv2.resize(img, input_size)
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_norm = roi_rgb.astype(np.float32) / 255.0
            roi_tensor = np.transpose(roi_norm, (2, 0, 1))
            roi_batch = np.expand_dims(roi_tensor, 0)
            
            mem_before = measure_memory()
            
            start = time.time()
            _ = session.run(None, {input_name: roi_batch})
            end = time.time()
            
            mem_after = measure_memory()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            memory_usage.append(mem_after - mem_before)
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i+1}/{num_runs}")
        
        return {
            'latency_mean_ms': float(np.mean(latencies)),
            'latency_std_ms': float(np.std(latencies)),
            'latency_min_ms': float(np.min(latencies)),
            'latency_max_ms': float(np.max(latencies)),
            'throughput_imgs_per_sec': float(1000.0 / np.mean(latencies)),
            'memory_usage_mb': float(np.mean(memory_usage)),
            'num_runs': len(latencies)
        }
    except ImportError:
        print("  ONNX Runtime not available, skipping...")
        return None


def benchmark_tensorrt(engine_path: str, test_images: List[str],
                      input_size: Tuple[int, int] = (128, 128),
                      num_warmup: int = 3, num_runs: int = 20) -> Dict:
    """Benchmark TensorRT engine."""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        print("\n" + "="*60)
        print(f"Benchmarking TensorRT: {os.path.basename(engine_path)}")
        print("="*60)
        
        # Load engine
        with open(engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        # Allocate buffers
        inputs, outputs, bindings, stream = allocate_buffers(engine)
        
        latencies = []
        memory_usage = []
        
        # Warmup
        print(f"Warming up ({num_warmup} runs)...")
        dummy_input = np.random.randn(1, 3, input_size[1], input_size[0]).astype(np.float32)
        for _ in range(num_warmup):
            inputs[0].host = dummy_input
            do_inference(context, bindings, inputs, outputs, stream)
        
        # Benchmark
        print(f"Benchmarking ({num_runs} runs)...")
        for i, img_path in enumerate(test_images[:num_runs]):
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            # Preprocess
            roi = cv2.resize(img, input_size)
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_norm = roi_rgb.astype(np.float32) / 255.0
            roi_tensor = np.transpose(roi_norm, (2, 0, 1))
            roi_batch = np.expand_dims(roi_tensor, 0)
            
            mem_before = measure_memory()
            
            inputs[0].host = roi_batch
            start = time.time()
            do_inference(context, bindings, inputs, outputs, stream)
            cuda.Context.synchronize()
            end = time.time()
            
            mem_after = measure_memory()
            
            latency_ms = (end - start) * 1000
            latencies.append(latency_ms)
            memory_usage.append(mem_after - mem_before)
            
            if (i + 1) % 5 == 0:
                print(f"  Completed {i+1}/{num_runs}")
        
        return {
            'latency_mean_ms': float(np.mean(latencies)),
            'latency_std_ms': float(np.std(latencies)),
            'latency_min_ms': float(np.min(latencies)),
            'latency_max_ms': float(np.max(latencies)),
            'throughput_imgs_per_sec': float(1000.0 / np.mean(latencies)),
            'memory_usage_mb': float(np.mean(memory_usage)),
            'num_runs': len(latencies)
        }
    except ImportError:
        print("  TensorRT not available, skipping...")
        return None
    except Exception as e:
        print(f"  TensorRT error: {e}")
        return None


def allocate_buffers(engine):
    """Allocate CUDA buffers for TensorRT."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    """Run TensorRT inference."""
    [cuda.memcpy_htod_async(inp['device'], inp['host'], stream) for inp in inputs]
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    [cuda.memcpy_dtoh_async(out['host'], out['device'], stream) for out in outputs]
    stream.synchronize()


def main():
    parser = argparse.ArgumentParser(description='Benchmark inference engines')
    parser.add_argument('--images', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--maskrcnn', type=str, required=True,
                       help='Path to Mask R-CNN model')
    parser.add_argument('--effnet', type=str, required=True,
                       help='Path to EfficientNet model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to Detectron2 config')
    parser.add_argument('--onnx-effnet', type=str, default=None,
                       help='Path to EfficientNet ONNX')
    parser.add_argument('--onnx-maskrcnn', type=str, default=None,
                       help='Path to Mask R-CNN ONNX')
    parser.add_argument('--trt-effnet', type=str, default=None,
                       help='Path to EfficientNet TensorRT engine')
    parser.add_argument('--trt-maskrcnn', type=str, default=None,
                       help='Path to Mask R-CNN TensorRT engine')
    parser.add_argument('--num-runs', type=int, default=20,
                       help='Number of benchmark runs')
    parser.add_argument('--num-classes', type=int, default=32,
                       help='Number of classes')
    parser.add_argument('--out', type=str, required=True,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Find test images
    image_patterns = [
        os.path.join(args.images, '**', '*.png'),
        os.path.join(args.images, '**', '*.jpg'),
        os.path.join(args.images, '**', '*.jpeg')
    ]
    
    test_images = []
    for pattern in image_patterns:
        test_images.extend(glob.glob(pattern, recursive=True))
    
    if len(test_images) < args.num_runs:
        print(f"Warning: Only {len(test_images)} images found, using all available")
        args.num_runs = len(test_images)
    
    print(f"Found {len(test_images)} test images")
    print(f"Will benchmark {args.num_runs} images")
    
    results = {}
    
    # Benchmark PyTorch models
    results['pytorch_maskrcnn'] = benchmark_pytorch_maskrcnn(
        args.maskrcnn, args.config, test_images, num_runs=args.num_runs
    )
    
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    results['pytorch_effnet'] = benchmark_pytorch_effnet(
        args.effnet, test_images, args.num_classes, num_runs=args.num_runs
    )
    
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Benchmark ONNX
    if args.onnx_effnet and os.path.exists(args.onnx_effnet):
        results['onnx_effnet'] = benchmark_onnx(
            args.onnx_effnet, test_images, (128, 128), num_runs=args.num_runs
        )
    
    if args.onnx_maskrcnn and os.path.exists(args.onnx_maskrcnn):
        results['onnx_maskrcnn'] = benchmark_onnx(
            args.onnx_maskrcnn, test_images, (1024, 512), num_runs=args.num_runs
        )
    
    # Benchmark TensorRT
    if args.trt_effnet and os.path.exists(args.trt_effnet):
        results['tensorrt_effnet'] = benchmark_tensorrt(
            args.trt_effnet, test_images, (128, 128), num_runs=args.num_runs
        )
    
    if args.trt_maskrcnn and os.path.exists(args.trt_maskrcnn):
        results['tensorrt_maskrcnn'] = benchmark_tensorrt(
            args.trt_maskrcnn, test_images, (1024, 512), num_runs=args.num_runs
        )
    
    # Save results
    os.makedirs(args.out, exist_ok=True)
    
    latency_path = os.path.join(args.out, 'latency.json')
    throughput_path = os.path.join(args.out, 'throughput.json')
    
    # Extract latency and throughput
    latency_data = {k: {
        'mean_ms': v['latency_mean_ms'],
        'std_ms': v['latency_std_ms'],
        'min_ms': v['latency_min_ms'],
        'max_ms': v['latency_max_ms']
    } for k, v in results.items() if v is not None}
    
    throughput_data = {k: {
        'imgs_per_sec': v['throughput_imgs_per_sec'],
        'memory_mb': v['memory_usage_mb']
    } for k, v in results.items() if v is not None}
    
    with open(latency_path, 'w') as f:
        json.dump(latency_data, f, indent=2)
    
    with open(throughput_path, 'w') as f:
        json.dump(throughput_data, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("Benchmark Summary")
    print("="*60)
    print("\nLatency (ms):")
    for name, data in latency_data.items():
        print(f"  {name}: {data['mean_ms']:.2f} ± {data['std_ms']:.2f}")
    
    print("\nThroughput (imgs/sec):")
    for name, data in throughput_data.items():
        print(f"  {name}: {data['imgs_per_sec']:.2f}")
    
    print(f"\nResults saved to:")
    print(f"  {latency_path}")
    print(f"  {throughput_path}")


if __name__ == '__main__':
    main()



