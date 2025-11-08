"""optimized_precision_fp4.py - FP4 precision training with Transformer Engine.

Uses Transformer Engine FP4 quantization for maximum memory savings and speed.
Optimized for Blackwell B200/GB10 with FP4 (NVFP4) support.
"""

from __future__ import annotations

import sys
from pathlib import Path
from contextlib import contextmanager

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

# Check if Transformer Engine is available
try:
    from transformer_engine.pytorch import Linear as TELinear
    from transformer_engine.pytorch import fp8_autocast
    import transformer_engine.pytorch.constants as te_constants
    from transformer_engine.common import recipe as te_recipe
    TE_AVAILABLE = True
    # Check if FP4 is supported (NVFP4)
    FP4_AVAILABLE = hasattr(te_constants, 'NVFP4_BLOCK_SCALING_SIZE')
except ImportError:
    TE_AVAILABLE = False
    FP4_AVAILABLE = False
    TELinear = nn.Linear  # Fallback
    te_recipe = None

from typing import Optional, Any

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")


class ProductionTransformer(nn.Module):
    """Production-scale transformer with optional FP4 support."""
    
    def __init__(self, use_te=False, hidden_dim=2048, num_layers=20):
        super().__init__()
        self.use_te = use_te and TE_AVAILABLE
        self.num_layers = num_layers
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # TE Linear layers - these handle FP4 internally when fp8_autocast is active
            if self.use_te:
                layer = nn.ModuleDict({
                    'attn_qkv': TELinear(hidden_dim, hidden_dim * 3, bias=True),
                    'attn_proj': TELinear(hidden_dim, hidden_dim, bias=True),
                    'ffn_fc1': TELinear(hidden_dim, hidden_dim * 4, bias=True),
                    'ffn_fc2': TELinear(hidden_dim * 4, hidden_dim, bias=True),
                })
            else:
                # Fallback to regular Linear (will be converted to bfloat16 in setup)
                layer = nn.ModuleDict({
                    'attn_qkv': nn.Linear(hidden_dim, hidden_dim * 3),
                    'attn_proj': nn.Linear(hidden_dim, hidden_dim),
                    'ffn_fc1': nn.Linear(hidden_dim, hidden_dim * 4),
                    'ffn_fc2': nn.Linear(hidden_dim * 4, hidden_dim),
                })
            self.layers.append(layer)
    
    def forward(self, x):
        # Note: fp8_autocast should be wrapped around the call to forward(), not inside
        # This allows the caller to control the autocast context
        # FP4 uses fp8_autocast API (name is historical, supports FP4)
        for layer in self.layers:
            qkv = layer['attn_qkv'](x)
            q, k, v = qkv.chunk(3, dim=-1)
            attn_out = layer['attn_proj'](q)
            x = x + attn_out
            ffn_out = layer['ffn_fc2'](torch.relu(layer['ffn_fc1'](x)))
            x = x + ffn_out
        return x


class OptimizedFP4Benchmark(Benchmark):
    """Optimized: FP4 precision training with Transformer Engine.
    
    Uses Transformer Engine FP4 quantization for maximum memory savings and speed.
    Optimized for Blackwell B200/GB10 with FP4 (NVFP4) support.
    FP4 support is mandatory—missing hardware or software dependencies should fail immediately.
    """
    
    def __init__(self):
        self.device = resolve_device()
        if not TE_AVAILABLE:
            raise RuntimeError("Transformer Engine with NVFP4 support is required for optimized_precision_fp4 benchmark")
        if not FP4_AVAILABLE:
            raise RuntimeError("This build of Transformer Engine lacks NVFP4 support")
        if not self._device_supports_fp4():
            raise RuntimeError("FP4 kernels require compute capability >= 12.0")
        
        self.model = None
        self.x = None
        self.optimizer = None
        # Larger model to better amortize FP4 conversion overhead
        self.batch_size = 8
        self.seq_len = 2048
        self.hidden_dim = 2048
        self.num_layers = 20
        self.fp4_recipe: Optional[Any] = self._build_fp4_recipe()
        if self.fp4_recipe is None:
            raise RuntimeError("Transformer Engine NVFP4 recipe unavailable")
        self.use_te_layers = True
        self._configure_workload()

    def _configure_workload(self) -> None:
        """Scale workload based on available GPU memory to avoid timeouts."""
        if not torch.cuda.is_available():
            return

        device_index = (
            self.device.index
            if self.device.index is not None
            else torch.cuda.current_device()
        )
        total_memory_gb = (
            torch.cuda.get_device_properties(device_index).total_memory / (1024 ** 3)
        )

        # Tiered configs roughly matching A100/B200 (>80GB) down to laptop GPUs
        tiers = [
            (80, dict(batch_size=8, seq_len=2048, hidden_dim=2048, num_layers=20)),
            (48, dict(batch_size=4, seq_len=2048, hidden_dim=1536, num_layers=18)),
            (32, dict(batch_size=4, seq_len=1536, hidden_dim=1536, num_layers=16)),
            (24, dict(batch_size=4, seq_len=1024, hidden_dim=1024, num_layers=12)),
            (16, dict(batch_size=2, seq_len=1024, hidden_dim=768, num_layers=10)),
            (12, dict(batch_size=2, seq_len=768, hidden_dim=768, num_layers=8)),
        ]
        fallback = dict(batch_size=1, seq_len=512, hidden_dim=512, num_layers=6)

        selected = fallback
        for min_mem_gb, config in tiers:
            if total_memory_gb >= min_mem_gb:
                selected = config
                break

        self.batch_size = selected["batch_size"]
        self.seq_len = selected["seq_len"]
        self.hidden_dim = selected["hidden_dim"]
        self.num_layers = selected["num_layers"]

    def _device_supports_fp4(self) -> bool:
        """Return True if the current GPU exposes NVFP4 instructions."""
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return False
        device_index = self.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device_index)
        # FP4 kernels are supported on Blackwell (SM 12.x) and beyond.
        return major >= 12

    def _build_fp4_recipe(self) -> Optional[Any]:
        """Instantiate the NVFP4 recipe if Transformer Engine exposes it."""
        if te_recipe is None or not hasattr(te_recipe, "NVFP4BlockScaling"):
            return None
        try:
            return te_recipe.NVFP4BlockScaling()
        except Exception:
            return None

    @contextmanager
    def _precision_context(self):
        """Combine PyTorch autocast with Transformer Engine fp8_autocast for FP4."""
        autocast_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        with torch.autocast(device_type=self.device.type, dtype=autocast_dtype):
            with fp8_autocast(enabled=True, fp8_recipe=self.fp4_recipe):
                yield
    
    def setup(self) -> None:
        """Setup: initialize model and data."""
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        self.model = ProductionTransformer(
            use_te=self.use_te_layers,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
        )
        self.model = self.model.to(self.device)
        self.model.train()
        
        tensor_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.x = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=tensor_dtype,
        )

        with torch.no_grad():
            with self._precision_context():
                _ = self.model(self.x)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: FP4 precision training step with forward and backward pass.
        
        Performs a single training iteration using Transformer Engine's FP4 quantization
        (or FP8 fallback) to reduce memory usage and improve training throughput.
        """
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_precision_fp4", enable=enable_nvtx):
            self.optimizer.zero_grad()
            # TE Linear requires fp8_autocast to be active when forward() is called
            # Wrap entire forward+backward in fp8_autocast for TE
            # FP4 uses the same API (fp8_autocast) but with FP4 format internally
            with self._precision_context():
                output = self.model(self.x)  # TE Linear layers called here
                loss = output.mean()
                loss.backward()  # Backward also needs autocast for TE
            self.optimizer.step()

    
    def teardown(self) -> None:
        """Cleanup."""
        for attr in ("model", "x", "optimizer"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=3,
            warmup=1,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.x is None:
            return "Input tensor not initialized"
        try:
            with torch.no_grad():
                with self._precision_context():
                    test_output = self.model(self.x)
                if test_output.shape[0] != self.batch_size:
                    return f"Output shape mismatch: expected batch_size={self.batch_size}, got {test_output.shape[0]}"
                if test_output.shape[1] != self.seq_len:
                    return f"Output shape mismatch: expected seq_len={self.seq_len}, got {test_output.shape[1]}"
                if test_output.shape[2] != self.hidden_dim:
                    return f"Output shape mismatch: expected hidden_dim={self.hidden_dim}, got {test_output.shape[2]}"
                if not torch.isfinite(test_output).all():
                    return "Output contains non-finite values"
            return None
        except Exception as e:
            return f"Validation failed: {e}"

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery.
    
    NOTE: FP4 support is experimental and may hang.
    If FP4 is not working, this will gracefully fall back to FP8 behavior.
    """
    return OptimizedFP4Benchmark()

def main() -> None:
    """Standalone execution with timing."""
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=10, warmup=5)
    )
    benchmark = OptimizedFP4Benchmark()
    result = harness.benchmark(benchmark)

    print("=" * 70)
    print("Optimized: FP4 Precision (NVFP4)")
    print("=" * 70)
    print(f"FP4 Enabled: {benchmark.use_te_layers}")
    if benchmark.fp4_skip_reason:
        print(f"FP4 status: {benchmark.fp4_skip_reason}")
    print(f"Average time per iteration: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median time: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std deviation: {result.timing.std_ms if result.timing else 0.0:.3f} ms")
    print(f"Min: {result.timing.min_ms if result.timing else 0.0:.3f} ms, Max: {result.timing.max_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
