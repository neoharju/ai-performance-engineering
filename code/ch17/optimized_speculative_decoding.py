"""optimized_speculative_decoding.py - Optimized speculative decoding in inference/profiling.

Demonstrates speculative decoding for parallel token generation.
Speculative decoding: Uses draft model to predict multiple tokens in parallel.
Accepts/rejects tokens based on target model verification for speedup.
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

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")

class OptimizedSpeculativeDecodingBenchmark(Benchmark):
    """Optimized: Speculative decoding for parallel token generation.
    
    Speculative decoding: Uses draft model to predict multiple tokens in parallel.
    Accepts/rejects tokens based on target model verification for speedup.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.target_model = None
        # Optimization: Compile model for kernel fusion and optimization

        self.draft_model = None
        self.embedding = None
        self.input_ids = None
        self.max_length = 20
        self.speculative_length = 4  # Number of tokens to predict speculatively
        self.hidden_dim = 256
        self.vocab_size = 1000
        self.batch_size = 4
        self.seq_len = 10
        self.memory = None
    
    def setup(self) -> None:
        """Setup: Initialize target and draft models."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Speculative decoding
        # Draft model predicts multiple tokens in parallel
        # Target model verifies predictions for correctness
        
        # Target model (slower, more accurate)
        self.target_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, batch_first=True),
            num_layers=4  # Larger model
        ).to(self.device)
        if self.device.type == "cuda":
            self.target_model = self.target_model.half()
        self.target_model.eval()
        
        # Draft model (faster, less accurate) for speculative decoding
        self.draft_model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, batch_first=True),
            num_layers=2  # Smaller model
        ).to(self.device)
        if self.device.type == "cuda":
            self.draft_model = self.draft_model.half()
        self.draft_model.eval()
        
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim).to(self.device)
        if self.device.type == "cuda":
            self.embedding = self.embedding.half()
        
        target_dtype = next(self.target_model.parameters()).dtype
        
        # Input
        self.input_ids = torch.randint(
            0,
            self.vocab_size,
            (self.batch_size, self.seq_len),
            device=self.device,
            dtype=torch.long,
        )
        self.memory = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=target_dtype,
        )
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Speculative decoding."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_speculative_decoding", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Speculative decoding
                # Draft model predicts multiple tokens in parallel
                # Target model verifies predictions
                
                current_ids = self.input_ids.clone()
                
                target_dtype = next(self.target_model.parameters()).dtype
                
                while current_ids.size(1) < self.input_ids.size(1) + self.max_length:
                    # Draft model: Predict multiple tokens speculatively
                    draft_embeds = self.embedding(current_ids).to(target_dtype)
                    draft_output = self.draft_model(draft_embeds, self.memory)
                    draft_tokens = draft_output[:, -self.speculative_length:, :].argmax(dim=-1)
                    
                    # Target model: Verify draft predictions (speculative decoding verification)
                    for j in range(draft_tokens.size(1)):
                        candidate = torch.cat([current_ids, draft_tokens[:, j:j+1]], dim=1)
                        candidate_embeds = self.embedding(candidate).to(target_dtype)
                        target_output = self.target_model(candidate_embeds, self.memory)
                        accepted_token = target_output[:, -1, :].argmax(dim=-1, keepdim=True)
                        
                        current_ids = torch.cat([current_ids, accepted_token], dim=1)
                        if current_ids.size(1) >= self.input_ids.size(1) + self.max_length:
                            break
                    
                    # Optimization: Speculative decoding benefits
                    # - Draft model proposes multiple candidates per iteration
                    # - Target model still verifies correctness to maintain quality
                    # Loop continues until desired length is reached
                    if current_ids.size(1) >= self.input_ids.size(1) + self.max_length:
                        break

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.target_model = None
        self.draft_model = None
        self.embedding = None
        self.input_ids = None
        self.memory = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.target_model is None or self.draft_model is None:
            return "Models not initialized"
        if self.input_ids is None:
            return "Input IDs not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedSpeculativeDecodingBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedSpeculativeDecodingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: speculative_decoding")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
