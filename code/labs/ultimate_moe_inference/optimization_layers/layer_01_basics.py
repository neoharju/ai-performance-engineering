"""Layer 01: Basics (Ch1-6).

Foundational optimizations that should always be applied:
- NVTX profiling markers for visibility
- NUMA binding to GPU's local memory
- TF32 enable for Tensor Core fallback
- cuDNN benchmark mode for algorithm selection
- Proper warmup patterns
"""

from __future__ import annotations

import os
from typing import Any, Optional

import torch

try:
    import torch.cuda.nvtx as nvtx
    NVTX_AVAILABLE = True
except ImportError:
    NVTX_AVAILABLE = False


class Layer01Basics:
    """Layer 1: Basic optimizations from Chapters 1-6.
    
    These are foundational optimizations that have minimal overhead
    and should almost always be applied.
    
    Techniques:
    - Ch1: NVTX markers for profiling
    - Ch2: Device selection and initialization
    - Ch3: NUMA awareness for CPU-GPU topology
    - Ch6: TF32 and cuDNN configuration
    """
    
    name = "Layer 01: Basics (Ch1-6)"
    chapters = [1, 2, 3, 4, 5, 6]
    
    def __init__(self):
        """Initialize layer."""
        self._numa_bound = False
        self._tf32_enabled = False
        self._cudnn_benchmark = False
    
    def apply(self, benchmark: Any) -> None:
        """Apply all basic optimizations.
        
        Args:
            benchmark: Benchmark instance to configure
        """
        device = getattr(benchmark, 'device', torch.device('cuda:0'))
        
        # Apply all optimizations
        self.enable_tf32()
        self.enable_cudnn_benchmark()
        self.bind_numa(device)
        self.configure_memory_allocator()
    
    def enable_tf32(self) -> None:
        """Enable TF32 for Tensor Core operations (Ch6).
        
        TF32 provides ~8x throughput improvement over FP32 on Tensor Cores
        with minimal precision loss for most inference workloads.
        """
        if not torch.cuda.is_available():
            return
        
        # Enable TF32 for matmul
        torch.backends.cuda.matmul.allow_tf32 = True
        
        # Enable TF32 for cuDNN
        torch.backends.cudnn.allow_tf32 = True
        
        self._tf32_enabled = True
        print("  [Layer 01] TF32 enabled for Tensor Cores")
    
    def enable_cudnn_benchmark(self) -> None:
        """Enable cuDNN benchmark mode (Ch6).
        
        cuDNN will run multiple algorithm implementations and select
        the fastest one for the given input shapes. This adds overhead
        on first run but improves subsequent performance.
        """
        if not torch.cuda.is_available():
            return
        
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        self._cudnn_benchmark = True
        print("  [Layer 01] cuDNN benchmark mode enabled")
    
    def bind_numa(self, device: torch.device) -> None:
        """Bind to NUMA node closest to GPU (Ch3).
        
        On multi-socket systems, memory access from the wrong NUMA node
        can significantly impact performance. This binds the process to
        the CPU/memory closest to the target GPU.
        
        Args:
            device: Target CUDA device
        """
        if not torch.cuda.is_available():
            return
        
        try:
            # Get GPU device index
            device_idx = device.index if device.index is not None else 0
            
            # Try to read NUMA node from sysfs
            numa_path = f"/sys/bus/pci/devices/{self._get_pci_bus(device_idx)}/numa_node"
            if os.path.exists(numa_path):
                with open(numa_path) as f:
                    numa_node = int(f.read().strip())
                
                if numa_node >= 0:
                    # Set CPU affinity to NUMA node
                    try:
                        os.sched_setaffinity(0, self._get_numa_cpus(numa_node))
                        self._numa_bound = True
                        print(f"  [Layer 01] Bound to NUMA node {numa_node} for GPU {device_idx}")
                    except Exception:
                        pass  # CPU affinity setting is best-effort
        except Exception:
            # NUMA binding is best-effort
            pass
    
    def _get_pci_bus(self, device_idx: int) -> str:
        """Get PCI bus address for GPU."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
            pci_info = pynvml.nvmlDeviceGetPciInfo(handle)
            return f"0000:{pci_info.bus:02x}:{pci_info.device:02x}.{pci_info.function}"
        except Exception:
            return ""
    
    def _get_numa_cpus(self, numa_node: int) -> set:
        """Get CPU IDs for a NUMA node."""
        cpus_path = f"/sys/devices/system/node/node{numa_node}/cpulist"
        if os.path.exists(cpus_path):
            with open(cpus_path) as f:
                cpulist = f.read().strip()
            
            cpus = set()
            for part in cpulist.split(','):
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    cpus.update(range(start, end + 1))
                else:
                    cpus.add(int(part))
            return cpus
        
        # Fallback: use all CPUs
        return set(range(os.cpu_count() or 1))
    
    def configure_memory_allocator(self) -> None:
        """Configure PyTorch memory allocator (Ch2/Ch6).
        
        Enable expandable segments to reduce memory fragmentation
        for variable-length sequences.
        """
        if not torch.cuda.is_available():
            return
        
        # Set allocator config via environment
        # This should be set before CUDA initialization, so it's best-effort here
        alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" not in alloc_conf:
            print("  [Layer 01] Note: Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before launch")
    
    def warmup(self, model: Any, sample_input: torch.Tensor, iterations: int = 3) -> None:
        """Run warmup iterations (Ch1).
        
        Warmup ensures:
        - CUDA context is initialized
        - cuDNN autotuning completes
        - JIT compilation finishes
        - Memory pools are primed
        
        Args:
            model: Model to warm up
            sample_input: Sample input tensor
            iterations: Number of warmup iterations
        """
        print(f"  [Layer 01] Running {iterations} warmup iterations")
        
        model.eval()
        with torch.no_grad():
            for i in range(iterations):
                with self.nvtx_range(f"warmup_{i}"):
                    _ = model(sample_input)
        
        torch.cuda.synchronize()
        print("  [Layer 01] Warmup complete")
    
    @staticmethod
    def nvtx_range(name: str, color: str = "blue"):
        """Create an NVTX range for profiling (Ch1).
        
        Args:
            name: Range name (visible in Nsight Systems)
            color: Range color
            
        Returns:
            Context manager for the range
        """
        if NVTX_AVAILABLE:
            return nvtx.range(name)
        else:
            # Fallback to no-op context manager
            from contextlib import nullcontext
            return nullcontext()
    
    def get_status(self) -> dict:
        """Get status of applied optimizations.
        
        Returns:
            Dictionary with optimization status
        """
        return {
            "tf32_enabled": self._tf32_enabled,
            "cudnn_benchmark": self._cudnn_benchmark,
            "numa_bound": self._numa_bound,
        }
    
    def __str__(self) -> str:
        """Return string representation."""
        return f"{self.name}: TF32={self._tf32_enabled}, cuDNN={self._cudnn_benchmark}, NUMA={self._numa_bound}"

