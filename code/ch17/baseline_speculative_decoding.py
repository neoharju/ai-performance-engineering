"""baseline_speculative_decoding.py - Baseline decoding without speculative execution in inference/profiling.

Demonstrates standard autoregressive decoding without speculative decoding optimization.
Speculative decoding: This baseline does not use speculative decoding.
Generates tokens one at a time, sequential and slow.
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


class BaselineSpeculativeDecodingBenchmark(Benchmark):
    """Baseline: Standard autoregressive decoding (no speculative execution).
    
    Speculative decoding: This baseline does not use speculative decoding.
    Generates tokens one at a time, sequential and slow.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.embedding = None
        self.input_ids = None
        self.memory = None
        self.max_length = 20
        self.hidden_dim = 256
        self.vocab_size = 1000
        self.batch_size = 4
        self.seq_len = 10
    
    def setup(self) -> None:
        """Setup: Initialize model and input."""
        torch.manual_seed(42)
        # Baseline: Standard decoding - generate tokens one at a time
        # Speculative decoding predicts multiple tokens in parallel
        # This baseline does not use speculative decoding
        
        self.model = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.hidden_dim, nhead=8, batch_first=True),
            num_layers=2
        )
        self.model = self.model.to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.half()
        self.model.eval()
        
        self.embedding = nn.Embedding(self.vocab_size, self.hidden_dim).to(self.device)
        if self.device.type == "cuda":
            self.embedding = self.embedding.half()
        
        decoder_dtype = next(self.model.parameters()).dtype
        
        # Baseline: Standard decoding - sequential token generation
        self.input_ids = torch.randint(
            0,
            self.vocab_size,
            (self.batch_size, self.seq_len),
            device=self.device,
            dtype=torch.long,
        )
        # Create dummy memory tensor for TransformerDecoder (encoder output)
        self.memory = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=decoder_dtype,
        )
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Standard autoregressive decoding."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_speculative_decoding", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Standard autoregressive decoding
                # Generate tokens one at a time (sequential)
                # No speculative decoding - cannot predict multiple tokens in parallel
                
                current_ids = self.input_ids.clone()
                for _ in range(self.max_length):
                    # Generate next token (sequential - no speculative decoding)
                    tgt_embeddings = self.embedding(current_ids).to(self.memory.dtype)
                    # TransformerDecoder requires both tgt and memory arguments
                    output = self.model(tgt_embeddings, self.memory)
                    next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                    current_ids = torch.cat([current_ids, next_token], dim=1)
                
                # Baseline: No speculative decoding
                # Sequential token generation (slow)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.embedding = None
        self.input_ids = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input_ids is None:
            return "Input IDs not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineSpeculativeDecodingBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineSpeculativeDecodingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: speculative_decoding")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
