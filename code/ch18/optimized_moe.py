"""optimized_moe.py - Optimized Mixture of Experts (MoE) model.

Demonstrates MoE with expert routing and selective activation.
MoE: Uses expert routing to activate only a subset of experts per token.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch18")
    return torch.device("cuda")


class ExpertLayer(nn.Module):
    """Single expert in MoE model."""
    
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.expert = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expert(x)


class OptimizedMoEBenchmark(Benchmark):
    """Optimized: MoE model with expert routing and selective activation."""
    
    def __init__(self):
        self.device = resolve_device()
        self.experts = None
        self.router = None
        self.input_data = None
        self.num_experts = 8
        self.top_k = 2  # Top-k experts per token
    
    def setup(self) -> None:
        """Setup: Initialize MoE model with expert routing."""
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        # Optimization: MoE uses expert routing to activate only a subset of experts
        # This reduces computation by only processing tokens through selected experts
        # MoE enables scaling model capacity without proportional increase in computation
        self.experts = nn.ModuleList([
            ExpertLayer(256).to(self.device) for _ in range(self.num_experts)
        ])
        
        # Router (gating network) to select top-k experts per token
        self.router = nn.Linear(256, self.num_experts).to(self.device)
        
        self.input_data = torch.randn(32, 256, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: MoE inference with expert routing."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        # Optimization: MoE routes tokens to top-k experts
        # Only selected experts process each token, reducing computation
        with nvtx_range("optimized_moe", enable=enable_nvtx):
            with torch.no_grad():
                router_logits = self.router(self.input_data)  # [batch, num_experts]
                top_k_weights, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
                top_k_weights = torch.softmax(top_k_weights, dim=-1)
                
                # Process tokens through selected experts
                output = torch.zeros_like(self.input_data)
                for expert_id in range(self.num_experts):
                    expert_mask = (top_k_indices == expert_id).any(dim=-1)
                    if expert_mask.any():
                        expert_input = self.input_data[expert_mask]
                        expert_output = self.experts[expert_id](expert_input)
                        # Weighted aggregation
                        weights = top_k_weights[expert_mask, (top_k_indices[expert_mask] == expert_id).nonzero(as_tuple=True)[1]]
                        output[expert_mask] += expert_output * weights.unsqueeze(-1)
        
        torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Cleanup: Clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.experts is None or len(self.experts) == 0:
            return "Experts not initialized"
        if self.router is None:
            return "Router not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedMoEBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)

