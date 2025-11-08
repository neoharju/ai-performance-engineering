#!/usr/bin/env python3
"""Dump comprehensive hardware capabilities for the current system.

This script outputs all hardware capabilities including:
- GPU architecture and compute capability
- CUDA version and features
- CUTLASS availability
- Triton support
- CUDA extension compilation support
- Memory and bandwidth specifications
"""

from common.python import compile_utils as _compile_utils_patch  # noqa: F401
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import importlib_metadata


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def dump_gpu_info():
    """Dump GPU hardware information."""
    print_section("GPU Hardware")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    
    print(f"GPU Name: {props.name}")
    print(f"Compute Capability: {props.major}.{props.minor}")
    print(f"SM Version: sm_{props.major}{props.minor}")
    print(f"Total Memory: {props.total_memory / (1024**3):.2f} GB")
    print(f"Number of SMs: {props.multi_processor_count}")
    print(f"Max Threads per Block: {getattr(props, 'max_threads_per_block', 1024)}")
    print(f"Max Threads per SM: {props.max_threads_per_multi_processor}")
    print(f"Warp Size: {props.warp_size}")
    print(f"Shared Memory per Block: {props.shared_memory_per_block / 1024:.1f} KB")
    print(f"Shared Memory per SM: {props.shared_memory_per_multiprocessor / 1024:.1f} KB")
    print(f"L2 Cache Size: {props.L2_cache_size / 1024:.1f} KB")
    
    # Architecture-specific features
    if props.major == 12:
        print("\nArchitecture: Grace-Blackwell GB10")
        print("Features: TMA, NVLink-C2C, Stream-ordered memory APIs")
    elif props.major == 10:
        print("\nArchitecture: Blackwell B200/B300")
        print("Features: HBM3e, TMA, NVLink-C2C, 5th Gen Tensor Cores")
        print("Memory Bandwidth: ~8 TB/s")
    elif props.major >= 9:
        print("\nArchitecture: Hopper or newer")
        print("Features: TMA support")


def dump_cuda_info():
    """Dump CUDA version and toolkit information."""
    print_section("CUDA Toolkit")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        return
    
    print(f"PyTorch CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print(f"CUDA Available: [OK]")
    
    # Check for CUDA toolkit components
    import subprocess
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\nNVCC Compiler:")
            for line in result.stdout.split('\n')[:3]:
                if line.strip():
                    print(f"  {line}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("WARNING: NVCC not found (CUDA extensions may not compile)")


def dump_cutlass_info():
    """Dump CUTLASS backend information."""
    print_section("CUTLASS Backend")
    
    try:
        import cutlass
        print(f"[OK] CUTLASS module found: {cutlass.__file__}")
    except ImportError:
        print("ERROR: CUTLASS module not found")
        return
    
    try:
        version = importlib_metadata.version("nvidia-cutlass-dsl")
        print(f"[OK] nvidia-cutlass-dsl version: {version}")
        
        # Check version
        parts = [int(x) for x in version.split('.')]
        if len(parts) >= 2 and (parts[0] > 4 or (parts[0] == 4 and parts[1] >= 2)):
            print("[OK] CUTLASS 4.2+ detected (Blackwell support)")
        else:
            print("WARNING: CUTLASS < 4.2 (upgrade recommended for Blackwell)")
    except importlib_metadata.PackageNotFoundError:
        print("WARNING: nvidia-cutlass-dsl package not found")
    
    # Check torch.compile CUTLASS backend configuration
    try:
        import torch._inductor.config as config
        print(f"\ntorch.compile CUTLASS Configuration:")
        print(f"  max_autotune_gemm_backends: {getattr(config, 'max_autotune_gemm_backends', 'N/A')}")
        print(f"  cutlass_enabled_ops: {getattr(config.cuda, 'cutlass_enabled_ops', 'N/A')}")
        
        if hasattr(config.cuda, 'cutlass_dir'):
            cutlass_dir = getattr(config.cuda, 'cutlass_dir', None)
            if cutlass_dir:
                print(f"  cutlass_dir: {cutlass_dir}")
            else:
                print(f"  cutlass_dir: Not configured")
    except Exception as e:
        print(f"WARNING: Could not check torch.compile CUTLASS config: {e}")


def dump_triton_info():
    """Dump Triton compiler information."""
    print_section("Triton Compiler")
    
    try:
        import triton
        print(f"[OK] Triton version: {triton.__version__}")
    except ImportError:
        print("ERROR: Triton not available")
        return
    
    # Check for SM architecture patching
    try:
        arch_cfg = arch_config.ArchitectureConfig()
        print(f"[OK] Architecture detected: {arch_cfg.arch}")
        
        # Check if SM 12.1 patching is enabled
        patch_enabled = arch_config.ENABLE_TRITON_PATCH
        if patch_enabled:
            print("[OK] Triton SM architecture patching: ENABLED")
            print("   (Fixes sm_121a → sm_121 for GB10)")
        else:
            print("WARNING: Triton SM architecture patching: DISABLED")
    except Exception as e:
        print(f"WARNING: Could not check arch_config: {e}")


def dump_cuda_extensions_info():
    """Dump CUDA extension compilation support."""
    print_section("CUDA Extensions")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available - extensions cannot be compiled")
        return
    
    # Check if extensions can be compiled
    print("Testing CUDA extension compilation support...")
    
    try:
        from ch12.cuda_extensions import load_graph_bandwidth_extension
        try:
            ext = load_graph_bandwidth_extension()
            print("[OK] graph_bandwidth extension: Can compile and load")
            print(f"   Module: {ext}")
        except Exception as e:
            print(f"ERROR: graph_bandwidth extension: Compilation failed")
            print(f"   Error: {e}")
    except ImportError as e:
        print(f"WARNING: Could not import extension loader: {e}")


def dump_torch_compile_info():
    """Dump torch.compile capabilities."""
    print_section("torch.compile Support")
    
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available - torch.compile requires GPU")
        return
    
    print(f"PyTorch Version: {torch.__version__}")
    
    # Test basic compilation
    print("\nTesting basic torch.compile...")
    try:
        device = torch.device('cuda')
        x = torch.randn(100, 100, device=device)
        
        def test_fn(x):
            return torch.mm(x, x.t())
        
        compiled_fn = torch.compile(test_fn, mode="reduce-overhead")
        result = compiled_fn(x)
        
        print("[OK] torch.compile: Working")
        print(f"   Test result shape: {result.shape}")
    except Exception as e:
        print(f"ERROR: torch.compile: Failed")
        print(f"   Error: {e}")


def dump_summary():
    """Print summary of capabilities."""
    print_section("Capability Summary")
    
    capabilities = []
    
    if torch.cuda.is_available():
        capabilities.append("[OK] CUDA Available")
        props = torch.cuda.get_device_properties(0)
        capabilities.append(f"[OK] GPU: {props.name} (SM {props.major}.{props.minor})")
    else:
        capabilities.append("ERROR: CUDA Not Available")
    
    try:
        import cutlass
        capabilities.append("[OK] CUTLASS Backend Available")
    except ImportError:
        capabilities.append("ERROR: CUTLASS Backend Not Available")
    
    try:
        import triton
        capabilities.append(f"[OK] Triton Compiler (v{triton.__version__})")
    except ImportError:
        capabilities.append("ERROR: Triton Compiler Not Available")
    
    # Check CUDA extensions
    try:
        from ch12.cuda_extensions import load_graph_bandwidth_extension
        try:
            load_graph_bandwidth_extension()
            capabilities.append("[OK] CUDA Extensions: Can Compile")
        except:
            capabilities.append("WARNING: CUDA Extensions: Compilation Issues")
    except ImportError:
        capabilities.append("WARNING: CUDA Extensions: Not Tested")
    
    for cap in capabilities:
        print(cap)


def main():
    """Main function to dump all hardware capabilities."""
    print("=" * 80)
    print("  Hardware Capabilities Report")
    print("=" * 80)
    
    dump_gpu_info()
    dump_cuda_info()
    dump_cutlass_info()
    dump_triton_info()
    dump_cuda_extensions_info()
    dump_torch_compile_info()
    dump_summary()
    
    print("\n" + "=" * 80)
    print("  End of Hardware Capabilities Report")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

