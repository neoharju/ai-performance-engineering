"""optimized_moe.py - Optimized Mixture of Experts in AI optimization context.

Demonstrates MoE architecture with sparse activation.
MoE: Uses Mixture of Experts where only subset of experts activate per input.
Sparse activation reduces computation while maintaining model capacity.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

from common.python import compile_utils as _compile_utils_patch  # noqa: F401
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from ch20.inductor_guard import (
    disable_inductor_cudagraph_features,
    restore_inductor_cudagraph_features,
    InductorCudagraphState,
)

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")

class MoELayer(nn.Module):
    """Mixture of Experts layer with sparse activation."""
    
    def __init__(self, hidden_dim: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router scores which experts should activate
        self.router = nn.Linear(hidden_dim, num_experts)
        
        # Expert MLPs operate at a smaller width; only top-k run per token
        expert_inner = hidden_dim * 2
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, expert_inner),
                nn.ReLU(),
                nn.Linear(expert_inner, hidden_dim),
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparse expert activation."""
        batch_size, seq_len, hidden_dim = x.shape
        tokens = x.view(-1, hidden_dim)
        
        router_logits = self.router(tokens)
        top_k_weights, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        flat_experts = top_k_indices.reshape(-1)
        flat_tokens = torch.arange(tokens.size(0), device=tokens.device).repeat_interleave(self.top_k)
        flat_weights = top_k_weights.reshape(-1)
        
        sorted_experts, sort_idx = torch.sort(flat_experts)
        flat_tokens = flat_tokens[sort_idx]
        flat_weights = flat_weights[sort_idx]
        expert_inputs = tokens[flat_tokens]
        
        counts = torch.bincount(sorted_experts, minlength=self.num_experts)
        output = torch.zeros_like(tokens)
        offset = 0
        for expert_idx in range(self.num_experts):
            count = int(counts[expert_idx].item())
            if count == 0:
                continue
            chunk_tokens = expert_inputs[offset:offset + count]
            chunk_outputs = self.experts[expert_idx](chunk_tokens)
            weights = flat_weights[offset:offset + count].unsqueeze(1).to(chunk_outputs.dtype)
            token_ids = flat_tokens[offset:offset + count]
            output.index_add_(0, token_ids, chunk_outputs * weights)
            offset += count
        
        return output.view(batch_size, seq_len, hidden_dim)

class OptimizedMoeBenchmark(Benchmark):
    """Optimized: Mixture of Experts with sparse activation.
    
    MoE: Uses Mixture of Experts architecture where only subset of experts activate per input.
    Sparse activation reduces computation while maintaining model capacity.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization

        # Optimization: Compile model for kernel fusion and optimization

        self.input = None
        self.hidden_dim = 1024
        self.batch_size = 8
        self.seq_len = 64
        self.num_experts = 8
        self.top_k = 2
        self._inductor_cfg_state: InductorCudagraphState = None
    
    def setup(self) -> None:
        """Setup: Initialize MoE model."""
        
        self._inductor_cfg_state = disable_inductor_cudagraph_features()
        try:
            # Optimization: Enable cuDNN benchmarking for optimal kernel selection
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            torch.manual_seed(42)
            # Optimization: MoE architecture
            # Only subset of experts activate per input (sparse activation)
            # Reduces computation while maintaining model capacity
            
            self.model = nn.Sequential(
                MoELayer(self.hidden_dim, num_experts=self.num_experts, top_k=self.top_k),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            
            self.model = self.model.to(self.device)
            # Optimization: Use FP16 for faster computation - FAIL FAST if not supported
            if self.device.type != "cuda":
                raise RuntimeError("CUDA required for optimized_moe benchmark")
            self.model = self.model.half()
            
            # Compile model for kernel fusion and optimization
            self.model = torch.compile(self.model, mode="reduce-overhead")
            # Warmup to trigger compilation and catch errors early
            test_input = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device, dtype=torch.float16)
            for _ in range(3):
                with torch.no_grad():
                    _ = self.model(test_input)
            torch.cuda.synchronize()
            self.model.eval()
            
            # Get model dtype - FAIL FAST if model has no parameters
            params = list(self.model.parameters())
            if not params:
                raise RuntimeError("Model has no parameters - cannot determine dtype")
            input_dtype = params[0].dtype
            self.input = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device, dtype=input_dtype)
            torch.cuda.synchronize()
        except Exception:
            restore_inductor_cudagraph_features(self._inductor_cfg_state)
            self._inductor_cfg_state = None
            raise
    
    def benchmark_fn(self) -> None:
        """Benchmark: MoE model inference."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_moe", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: MoE with sparse activation
                # Only top-k experts activate per input (e.g., top-2 of 4 experts)
                # Reduces computation while maintaining model capacity
                output = self.model(self.input)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        torch.cuda.empty_cache()
        restore_inductor_cudagraph_features(self._inductor_cfg_state)
        self._inductor_cfg_state = None
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
            use_subprocess=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedMoeBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedMoeBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: moe")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
