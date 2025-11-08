#!/usr/bin/env python3
"""
Peak Performance Benchmark
==========================
Measures peak hardware performance metrics:
- HBM memory bandwidth
- FP4 compute TFLOPS (if available)
- FP6 compute TFLOPS (if available)
- FP8 compute TFLOPS (if available)
- FP16 compute TFLOPS
- BF16 compute TFLOPS
- L2 cache bandwidth
- Shared memory (L1-equivalent) characteristics
- GPU hardware information (SMs, cache sizes, etc.)
- NVLink bandwidth (if multi-GPU available)
- torch.compile speedup

This script captures the actual peak performance of the hardware, which is then used
as baseline targets for validation in performance_targets.py.
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

# Suppress CUDA capability warnings for GB10 (12.1) - PyTorch supports up to 12.0
warnings.filterwarnings("ignore", message=".*Found GPU.*which is of cuda capability.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*Minimum and Maximum cuda capability supported.*", category=UserWarning)
# Suppress TF32 API deprecation warnings
warnings.filterwarnings("ignore", message=".*Please use the new API settings to control TF32.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*TensorFloat32 tensor cores.*available but not enabled.*", category=UserWarning)

import torch

from common.python.compile_utils import enable_tf32

# Configure TF32 using new API (PyTorch 2.9+)
# Enable TF32 for optimal performance on Ampere+ GPUs using the shared helper
if torch.cuda.is_available():
    enable_tf32()

try:
    import transformer_engine.pytorch as te
    import transformer_engine.pytorch.constants as te_constants
    TE_AVAILABLE = True
    FP8_AVAILABLE = hasattr(te, 'fp8_autocast')
    FP4_AVAILABLE = hasattr(te_constants, 'NVFP4_BLOCK_SCALING_SIZE')
    FP6_AVAILABLE = hasattr(te_constants, 'NVFP6_BLOCK_SCALING_SIZE')
except ImportError:
    te = None
    te_constants = None
    TE_AVAILABLE = False
    FP8_AVAILABLE = False
    FP4_AVAILABLE = False
    FP6_AVAILABLE = False


def measure_hbm_bandwidth(device: torch.device = None, size_gb: float = 4.0, iterations: int = 20) -> dict:
    """Measure HBM memory bandwidth."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        return {"peak_bandwidth_tbs": None, "peak_utilization_percent": None, "error": "CUDA not available"}
    
    print(f"\nMeasuring HBM memory bandwidth...")
    print(f"  Test size: {size_gb} GB")
    print(f"  Iterations: {iterations}")
    
    try:
        size_bytes = int(size_gb * 1024**3)
        size_elements = size_bytes // 4  # float32 = 4 bytes
        
        # Allocate tensors
        x = torch.randn(size_elements, device=device, dtype=torch.float32)
        y = torch.empty_like(x)
        
        # Warmup
        for _ in range(10):
            y.copy_(x)
        torch.cuda.synchronize(device)
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iterations):
            y.copy_(x)
        end_event.record()
        torch.cuda.synchronize(device)
        
        elapsed_ms = start_event.elapsed_time(end_event)
        if elapsed_ms <= 0:
            raise RuntimeError("HBM bandwidth measurement returned non-positive elapsed time")
        elapsed_s = elapsed_ms / 1000.0
        bandwidth_gbs = (size_bytes * iterations / elapsed_s) / 1e9
        bandwidth_tbs = bandwidth_gbs / 1000.0
        
        # Theoretical peak for B200 is ~8.0 TB/s
        theoretical_tbs = 8.0
        utilization = (bandwidth_tbs / theoretical_tbs * 100) if theoretical_tbs > 0 else 0.0
        
        print(f"  Result: {bandwidth_tbs:.3f} TB/s ({bandwidth_gbs:.1f} GB/s)")
        print(f"  Utilization: {utilization:.1f}%")
        
        return {
            "peak_bandwidth_tbs": bandwidth_tbs,
            "peak_bandwidth_gbs": bandwidth_gbs,
            "peak_utilization_percent": utilization,
        }
    except Exception as e:
        return {"peak_bandwidth_tbs": None, "peak_utilization_percent": None, "error": str(e)}


def measure_fp16_compute(device: torch.device = None, matrix_size: int = 8192, iterations: int = 20) -> dict:
    """Measure FP16 compute TFLOPS."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        return {"peak_tflops": None, "error": "CUDA not available"}
    
    print(f"\nMeasuring FP16 compute performance...")
    print(f"  Matrix size: {matrix_size}x{matrix_size}")
    print(f"  Iterations: {iterations}")
    
    try:
        # Create large matrices for tensor core utilization
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float16)
        c = torch.empty(matrix_size, matrix_size, device=device, dtype=torch.float16)
        
        # Warmup
        for _ in range(10):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize(device)
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iterations):
            torch.mm(a, b, out=c)
        end_event.record()
        torch.cuda.synchronize(device)
        
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_s = elapsed_ms / 1000.0
        
        # FLOPs = 2 * M * N * K
        flops_per_iteration = 2 * matrix_size * matrix_size * matrix_size
        total_flops = flops_per_iteration * iterations
        tflops = total_flops / (elapsed_s * 1e12)
        
        print(f"  Result: {tflops:.2f} TFLOPS")
        
        return {
            "peak_tflops": tflops,
            "matrix_size": matrix_size,
        }
    except Exception as e:
        return {"peak_tflops": None, "error": str(e)}


def measure_bf16_compute(device: torch.device = None, matrix_size: int = 8192, iterations: int = 20) -> dict:
    """Measure BF16 compute TFLOPS."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        return {"peak_tflops": None, "error": "CUDA not available"}
    
    is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)
    if not is_bf16_supported():
        return {"peak_tflops": None, "error": "BF16 not supported on this GPU"}
    
    print(f"\nMeasuring BF16 compute performance...")
    print(f"  Matrix size: {matrix_size}x{matrix_size}")
    print(f"  Iterations: {iterations}")
    
    try:
        dtype = torch.bfloat16
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
        c = torch.empty(matrix_size, matrix_size, device=device, dtype=dtype)
        
        for _ in range(10):
            torch.mm(a, b, out=c)
        torch.cuda.synchronize(device)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iterations):
            torch.mm(a, b, out=c)
        end_event.record()
        torch.cuda.synchronize(device)
        
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_s = elapsed_ms / 1000.0
        
        flops_per_iteration = 2 * matrix_size * matrix_size * matrix_size
        total_flops = flops_per_iteration * iterations
        tflops = total_flops / (elapsed_s * 1e12)
        
        print(f"  Result: {tflops:.2f} TFLOPS")
        
        return {
            "peak_tflops": tflops,
            "matrix_size": matrix_size,
        }
    except Exception as e:
        return {"peak_tflops": None, "error": str(e)}


def measure_fp4_compute(device: torch.device = None, matrix_size: int = 8192, iterations: int = 20) -> dict:
    """Measure FP4 compute TFLOPS using Transformer Engine NVFP4 (if available)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        return {"peak_tflops": None, "error": "CUDA not available"}
    
    if not FP4_AVAILABLE:
        return {"peak_tflops": None, "error": "Transformer Engine FP4 (NVFP4) not available"}
    
    print(f"\nMeasuring FP4 compute performance...")
    print(f"  Matrix size: {matrix_size}x{matrix_size}")
    print(f"  Iterations: {iterations}")
    
    try:
        # Use Transformer Engine Linear layer (supports FP4 via fp8_autocast with FP4 recipe)
        from transformer_engine.pytorch import Linear as TELinear
        
        in_features = matrix_size
        out_features = matrix_size
        
        # Input tensor - create outside autocast
        x = torch.randn(1024, in_features, device=device, dtype=torch.float16)
        
        # Create TE linear layer - initialize on device first
        fp4_linear = TELinear(
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.float16,
        ).to(device)
        
        # Create and initialize FP4 recipe if available
        fp4_recipe = None
        if hasattr(te, 'fp8') and hasattr(te.fp8, 'Recipe'):
            try:
                fp4_recipe = te.fp8.Recipe.override(FP4=True)
            except (AttributeError, TypeError):
                fp4_recipe = None
        
        # Initialize the layer INSIDE autocast context with FP4 recipe
        # This ensures parameters are properly initialized for FP4
        if fp4_recipe is not None:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe):
                # Do initial forward pass to initialize FP4 weights properly
                _ = fp4_linear(x)
        else:
            with te.fp8_autocast():
                # Fallback if Recipe API not available
                _ = fp4_linear(x)
        
        torch.cuda.synchronize(device)
        
        # Warmup - FP4 uses fp8_autocast API (name is historical, supports FP4/FP8)
        # Use the same recipe for consistency
        if fp4_recipe is not None:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe):
                for _ in range(10):
                    _ = fp4_linear(x)
        else:
            with te.fp8_autocast():
                for _ in range(10):
                    _ = fp4_linear(x)
        torch.cuda.synchronize(device)
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        if fp4_recipe is not None:
            with te.fp8_autocast(enabled=True, fp8_recipe=fp4_recipe):
                for _ in range(iterations):
                    _ = fp4_linear(x)
        else:
            with te.fp8_autocast():
                for _ in range(iterations):
                    _ = fp4_linear(x)
        end_event.record()
        torch.cuda.synchronize(device)
        
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_s = elapsed_ms / 1000.0
        
        # FLOPs = 2 * batch_size * M * N
        batch_size = x.shape[0]
        flops_per_iteration = 2 * batch_size * in_features * out_features
        total_flops = flops_per_iteration * iterations
        tflops = total_flops / (elapsed_s * 1e12)
        
        print(f"  Result: {tflops:.2f} TFLOPS")
        
        return {
            "peak_tflops": tflops,
            "matrix_size": matrix_size,
        }
    except Exception as e:
        return {"peak_tflops": None, "error": f"FP4 measurement failed: {str(e)}"}


def measure_fp6_compute(device: torch.device = None, matrix_size: int = 8192, iterations: int = 20) -> dict:
    """Measure FP6 compute TFLOPS using Transformer Engine NVFP6 (if available)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        return {"peak_tflops": None, "error": "CUDA not available"}
    
    if not FP6_AVAILABLE:
        return {"peak_tflops": None, "error": "Transformer Engine FP6 (NVFP6) not available"}
    
    print(f"\nMeasuring FP6 compute performance...")
    print(f"  Matrix size: {matrix_size}x{matrix_size}")
    print(f"  Iterations: {iterations}")
    
    try:
        from transformer_engine.pytorch import Linear as TELinear
        
        in_features = matrix_size
        out_features = matrix_size
        x = torch.randn(1024, in_features, device=device, dtype=torch.float16)
        fp6_linear = TELinear(
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.float16,
        ).to(device)
        
        fp6_recipe = None
        if hasattr(te, "fp8") and hasattr(te.fp8, "Recipe"):
            try:
                fp6_recipe = te.fp8.Recipe.override(FP6=True)
            except (AttributeError, TypeError):
                fp6_recipe = None
        
        def _run(iterations_to_run: int) -> None:
            if fp6_recipe is not None:
                with te.fp8_autocast(enabled=True, fp8_recipe=fp6_recipe):
                    for _ in range(iterations_to_run):
                        _ = fp6_linear(x)
            else:
                with te.fp8_autocast():
                    for _ in range(iterations_to_run):
                        _ = fp6_linear(x)
        
        _run(10)
        torch.cuda.synchronize(device)
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        _run(iterations)
        end_event.record()
        torch.cuda.synchronize(device)
        
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_s = elapsed_ms / 1000.0
        
        batch_size = x.shape[0]
        flops_per_iteration = 2 * batch_size * in_features * out_features
        total_flops = flops_per_iteration * iterations
        tflops = total_flops / (elapsed_s * 1e12)
        
        print(f"  Result: {tflops:.2f} TFLOPS")
        
        return {
            "peak_tflops": tflops,
            "matrix_size": matrix_size,
        }
    except Exception as e:
        return {"peak_tflops": None, "error": f"FP6 measurement failed: {str(e)}"}


def measure_fp8_compute(device: torch.device = None, matrix_size: int = 8192, iterations: int = 20) -> dict:
    """Measure FP8 compute TFLOPS using Transformer Engine (if available)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        return {"peak_tflops": None, "error": "CUDA not available"}
    
    if not FP8_AVAILABLE:
        return {"peak_tflops": None, "error": "Transformer Engine FP8 not available"}
    
    print(f"\nMeasuring FP8 compute performance...")
    print(f"  Matrix size: {matrix_size}x{matrix_size}")
    print(f"  Iterations: {iterations}")
    
    try:
        # Use Transformer Engine FP8 linear layer
        from transformer_engine.pytorch import Linear as Fp8Linear
        
        in_features = matrix_size
        out_features = matrix_size
        
        # Create FP8 linear layer
        fp8_linear = Fp8Linear(
            in_features,
            out_features,
            bias=False,
            params_dtype=torch.float16,
        ).to(device)
        
        # Input tensor
        x = torch.randn(1024, in_features, device=device, dtype=torch.float16)
        
        # Warmup
        with te.fp8_autocast():
            for _ in range(10):
                _ = fp8_linear(x)
        torch.cuda.synchronize(device)
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with te.fp8_autocast():
            for _ in range(iterations):
                _ = fp8_linear(x)
        end_event.record()
        torch.cuda.synchronize(device)
        
        elapsed_ms = start_event.elapsed_time(end_event)
        elapsed_s = elapsed_ms / 1000.0
        
        # FLOPs = 2 * batch_size * M * N
        batch_size = x.shape[0]
        flops_per_iteration = 2 * batch_size * in_features * out_features
        total_flops = flops_per_iteration * iterations
        tflops = total_flops / (elapsed_s * 1e12)
        
        print(f"  Result: {tflops:.2f} TFLOPS")
        
        return {
            "peak_tflops": tflops,
            "matrix_size": matrix_size,
        }
    except Exception as e:
        return {"peak_tflops": None, "error": str(e)}


def measure_l2_cache_bandwidth(device: torch.device = None, size_mb: float = 50.0, iterations: int = 20) -> dict:
    """Measure L2 cache bandwidth by using data that fits in L2 cache."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        return {"peak_bandwidth_gbs": None, "error": "CUDA not available"}
    
    # Get L2 cache size from device properties
    props = torch.cuda.get_device_properties(device)
    l2_cache_size_bytes = getattr(props, 'l2_cache_size', 0)
    l2_cache_size_mb = l2_cache_size_bytes / (1024**2)
    
    # Use a size that fits comfortably in L2 cache (75% of L2 size)
    if l2_cache_size_bytes > 0:
        test_size_mb = min(size_mb, l2_cache_size_mb * 0.75)
    else:
        test_size_mb = size_mb  # Fallback if L2 size unknown
    
    print(f"\nMeasuring L2 cache bandwidth...")
    print(f"  L2 cache size: {l2_cache_size_mb:.1f} MB" if l2_cache_size_bytes > 0 else "  L2 cache size: Unknown")
    print(f"  Test size: {test_size_mb:.1f} MB")
    print(f"  Iterations: {iterations}")
    
    try:
        size_bytes = int(test_size_mb * 1024**2)
        size_elements = size_bytes // 4  # float32 = 4 bytes
        
        # Allocate tensors that should fit in L2 cache
        x = torch.randn(size_elements, device=device, dtype=torch.float32)
        y = torch.empty_like(x)
        
        # Warmup to ensure data is in cache
        for _ in range(20):
            y.copy_(x)
        torch.cuda.synchronize(device)
        
        # Benchmark - small operations should hit L2 cache
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iterations):
            y.copy_(x)
        end_event.record()
        torch.cuda.synchronize(device)
        
        elapsed_ms = start_event.elapsed_time(end_event)
        if elapsed_ms <= 0:
            raise RuntimeError("L2 cache bandwidth measurement returned non-positive elapsed time")
        elapsed_s = elapsed_ms / 1000.0
        bandwidth_gbs = (size_bytes * iterations / elapsed_s) / 1e9
        
        print(f"  Result: {bandwidth_gbs:.1f} GB/s")
        
        return {
            "peak_bandwidth_gbs": bandwidth_gbs,
            "l2_cache_size_mb": l2_cache_size_mb,
            "test_size_mb": test_size_mb,
        }
    except Exception as e:
        return {"peak_bandwidth_gbs": None, "error": str(e)}


def measure_shared_memory_info(device: torch.device = None) -> dict:
    """Capture shared memory characteristics (acts as L1 cache equivalent)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    print(f"\nCapturing shared memory (L1-equivalent) information...")
    
    try:
        props = torch.cuda.get_device_properties(device)
        
        shared_mem_per_block = getattr(props, 'shared_memory_per_block', 0) / 1024  # KB
        shared_mem_per_sm = getattr(props, 'shared_memory_per_multiprocessor', 0) / 1024  # KB
        num_sms = props.multi_processor_count
        
        print(f"  Shared memory per block: {shared_mem_per_block:.1f} KB")
        print(f"  Shared memory per SM: {shared_mem_per_sm:.1f} KB")
        print(f"  Total SMs: {num_sms}")
        print(f"  Total shared memory: {shared_mem_per_sm * num_sms / 1024:.1f} MB")
        
        return {
            "shared_memory_per_block_kb": shared_mem_per_block,
            "shared_memory_per_sm_kb": shared_mem_per_sm,
            "num_sms": num_sms,
            "total_shared_memory_mb": shared_mem_per_sm * num_sms / 1024,
        }
    except Exception as e:
        return {"error": str(e)}


def measure_nvlink_bandwidth(device: torch.device = None, iterations: int = 20) -> dict:
    """Measure NVLink bandwidth between GPUs (if multi-GPU available)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        return {"peak_bandwidth_gbs": None, "error": "CUDA not available"}
    
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2:
        return {"peak_bandwidth_gbs": None, "error": "Multi-GPU not available (requires 2+ GPUs)"}
    
    print(f"\nMeasuring NVLink bandwidth...")
    print(f"  GPU count: {gpu_count}")
    print(f"  Testing GPU 0 -> GPU 1")
    print(f"  Iterations: {iterations}")
    
    try:
        # Use ~1GB for bandwidth test
        size_mb = 1024
        size = size_mb * 1024 * 1024 // 4  # float32
        
        # Create tensors on different GPUs
        with torch.cuda.device(0):
            src = torch.randn(size, device='cuda:0')
        
        with torch.cuda.device(1):
            dst = torch.empty(size, device='cuda:1')
        
        # Warmup
        for _ in range(10):
            dst.copy_(src)
        torch.cuda.synchronize()
        
        # Benchmark
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iterations):
            dst.copy_(src)
        end_event.record()
        torch.cuda.synchronize()
        
        elapsed_ms = start_event.elapsed_time(end_event)
        bandwidth_gbs = (size_mb / 1024) / (elapsed_ms / 1000) * iterations
        
        print(f"  Result: {bandwidth_gbs:.2f} GB/s")
        
        # Clean up
        del src, dst
        torch.cuda.empty_cache()
        
        return {
            "peak_bandwidth_gbs": bandwidth_gbs,
            "gpu_count": gpu_count,
        }
    except Exception as e:
        return {"peak_bandwidth_gbs": None, "error": str(e)}


def capture_gpu_hardware_info(device: torch.device = None) -> dict:
    """Capture comprehensive GPU hardware information."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}
    
    print(f"\nCapturing GPU hardware information...")
    
    try:
        props = torch.cuda.get_device_properties(device)
        
        info = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_gb": props.total_memory / (1024**3),
            "num_sms": props.multi_processor_count,
            "max_threads_per_block": getattr(props, 'max_threads_per_block', 1024),
            "max_threads_per_sm": props.max_threads_per_multi_processor,
            "warp_size": props.warp_size,
            "l2_cache_size_kb": getattr(props, 'l2_cache_size', 0) / 1024,
            "shared_memory_per_block_kb": getattr(props, 'shared_memory_per_block', 0) / 1024,
            "shared_memory_per_sm_kb": getattr(props, 'shared_memory_per_multiprocessor', 0) / 1024,
            "registers_per_block": getattr(props, 'registers_per_block', 0),
            "registers_per_sm": getattr(props, 'registers_per_multiprocessor', 0),
        }
        
        print(f"  GPU: {info['name']}")
        print(f"  Compute Capability: {info['compute_capability']}")
        print(f"  Total Memory: {info['total_memory_gb']:.2f} GB")
        print(f"  SMs: {info['num_sms']}")
        print(f"  L2 Cache: {info['l2_cache_size_kb']:.1f} KB")
        
        return info
    except Exception as e:
        return {"error": str(e)}


def measure_torch_compile_speedup(device: torch.device = None, matrix_size: int = 4096, iterations: int = 20) -> dict:
    """Measure torch.compile speedup."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not torch.cuda.is_available():
        return {"speedup": None, "error": "CUDA not available"}
    
    print(f"\nMeasuring torch.compile speedup...")
    print(f"  Matrix size: {matrix_size}x{matrix_size}")
    print(f"  Iterations: {iterations}")
    
    try:
        # Define a function that will benefit from compilation (more complex)
        def compute_fn(x, y):
            # Chain operations to benefit from fusion
            z = torch.mm(x, y)
            z = torch.relu(z)
            z = torch.mm(z, y.t())
            return z
        
        # Create test tensors
        a = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        b = torch.randn(matrix_size, matrix_size, device=device, dtype=torch.float32)
        
        # Warmup eager (more iterations for stable timing)
        for _ in range(20):
            _ = compute_fn(a, b)
        torch.cuda.synchronize(device)
        
        # Benchmark eager
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iterations):
            _ = compute_fn(a, b)
        end_event.record()
        torch.cuda.synchronize(device)
        eager_time_ms = start_event.elapsed_time(end_event)
        
        # Compile the function - use "default" mode for better optimization
        compiled_fn = torch.compile(compute_fn, mode="default")
        
        # More extensive warmup for compiled (compilation happens here)
        # Do enough iterations to trigger compilation and stabilize
        for _ in range(30):
            _ = compiled_fn(a, b)
        torch.cuda.synchronize(device)
        
        # Ensure compilation is complete by doing a few more runs
        for _ in range(10):
            _ = compiled_fn(a, b)
        torch.cuda.synchronize(device)
        
        # Benchmark compiled
        start_event.record()
        for _ in range(iterations):
            _ = compiled_fn(a, b)
        end_event.record()
        torch.cuda.synchronize(device)
        compiled_time_ms = start_event.elapsed_time(end_event)
        
        speedup = eager_time_ms / compiled_time_ms if compiled_time_ms > 0 else 1.0
        
        print(f"  Eager time: {eager_time_ms/iterations:.3f} ms/iter")
        print(f"  Compiled time: {compiled_time_ms/iterations:.3f} ms/iter")
        print(f"  Speedup: {speedup:.2f}x")
        
        return {
            "speedup": speedup,
            "eager_time_ms": eager_time_ms / iterations,
            "compiled_time_ms": compiled_time_ms / iterations,
        }
    except Exception as e:
        return {"speedup": None, "error": str(e)}


def run_all_benchmarks(output_dir: Path = None) -> dict:
    """Run all peak performance benchmarks."""
    if output_dir is None:
        output_dir = Path.cwd()
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Cannot run peak performance benchmarks.")
        return {"error": "CUDA not available"}
    
    device = torch.device("cuda")
    
    print("="*70)
    print("Peak Performance Benchmark")
    print("="*70)
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Transformer Engine Available: {TE_AVAILABLE}")
    print(f"FP4 Available: {FP4_AVAILABLE}")
    print(f"FP6 Available: {FP6_AVAILABLE}")
    print(f"FP8 Available: {FP8_AVAILABLE}")
    bf16_supported = getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    print(f"BF16 Supported: {bf16_supported}")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "gpu_name": torch.cuda.get_device_name(0),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "te_available": TE_AVAILABLE,
        "fp4_available": FP4_AVAILABLE,
        "fp6_available": FP6_AVAILABLE,
        "fp8_available": FP8_AVAILABLE,
        "bf16_supported": bf16_supported,
    }
    
    # Use 20 iterations for all measurements
    iterations = 20
    
    # Measure HBM memory bandwidth
    results["hbm"] = measure_hbm_bandwidth(device, iterations=iterations)
    
    # Measure FP4 compute (if available)
    results["fp4_compute"] = measure_fp4_compute(device, iterations=iterations)
    
    # Measure FP6 compute (if available)
    results["fp6_compute"] = measure_fp6_compute(device, iterations=iterations)
    
    # Measure FP8 compute (if available)
    results["fp8_compute"] = measure_fp8_compute(device, iterations=iterations)
    
    # Measure FP16 compute
    results["fp16_compute"] = measure_fp16_compute(device, iterations=iterations)
    
    # Measure BF16 compute (if supported)
    results["bf16_compute"] = measure_bf16_compute(device, iterations=iterations)
    
    # Measure L2 cache bandwidth
    results["l2_cache"] = measure_l2_cache_bandwidth(device, iterations=iterations)
    
    # Capture shared memory info (L1-equivalent)
    results["shared_memory"] = measure_shared_memory_info(device)
    
    # Capture comprehensive GPU hardware info
    results["gpu_hardware"] = capture_gpu_hardware_info(device)
    
    # Measure NVLink bandwidth (if multi-GPU)
    results["nvlink"] = measure_nvlink_bandwidth(device, iterations=iterations)
    
    # Measure torch.compile speedup
    results["torch_compile"] = measure_torch_compile_speedup(device, iterations=iterations)
    
    # Print summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    if results["hbm"].get("peak_bandwidth_tbs"):
        print(f"HBM Memory Bandwidth: {results['hbm']['peak_bandwidth_tbs']:.3f} TB/s")
    
    if results["fp4_compute"].get("peak_tflops"):
        print(f"FP4 Compute: {results['fp4_compute']['peak_tflops']:.2f} TFLOPS")
    elif results["fp4_compute"].get("error"):
        print(f"FP4 Compute: Not available ({results['fp4_compute']['error']})")
    
    if results["fp6_compute"].get("peak_tflops"):
        print(f"FP6 Compute: {results['fp6_compute']['peak_tflops']:.2f} TFLOPS")
    elif results["fp6_compute"].get("error"):
        print(f"FP6 Compute: Not available ({results['fp6_compute']['error']})")
    
    if results["fp8_compute"].get("peak_tflops"):
        print(f"FP8 Compute: {results['fp8_compute']['peak_tflops']:.2f} TFLOPS")
    elif results["fp8_compute"].get("error"):
        print(f"FP8 Compute: Not available ({results['fp8_compute']['error']})")
    
    if results["fp16_compute"].get("peak_tflops"):
        print(f"FP16 Compute: {results['fp16_compute']['peak_tflops']:.2f} TFLOPS")
    
    if results["bf16_compute"].get("peak_tflops"):
        print(f"BF16 Compute: {results['bf16_compute']['peak_tflops']:.2f} TFLOPS")
    elif results["bf16_compute"].get("error"):
        print(f"BF16 Compute: Not available ({results['bf16_compute']['error']})")
    
    if results["l2_cache"].get("peak_bandwidth_gbs"):
        print(f"L2 Cache Bandwidth: {results['l2_cache']['peak_bandwidth_gbs']:.1f} GB/s")
    
    if results["nvlink"].get("peak_bandwidth_gbs"):
        print(f"NVLink Bandwidth (GPU 0->1): {results['nvlink']['peak_bandwidth_gbs']:.2f} GB/s")
    elif results["nvlink"].get("error"):
        print(f"NVLink: Not available ({results['nvlink']['error']})")
    
    if results["torch_compile"].get("speedup"):
        print(f"torch.compile Speedup: {results['torch_compile']['speedup']:.2f}x")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_peak_results_{timestamp}.json"
    output_path = output_dir / filename
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Measure peak hardware performance")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: current directory)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        results = run_all_benchmarks(output_dir)
        if "error" in results and results["error"] == "CUDA not available":
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
