"""Tests for optimization layers.

Verifies that each optimization layer can be applied without errors
and produces expected configurations.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


class TestLayer01Basics:
    """Tests for Layer 01: Basics."""
    
    def test_import(self):
        """Test layer can be imported."""
        from optimization_layers import Layer01Basics
        layer = Layer01Basics()
        assert layer.name == "Layer 01: Basics (Ch1-6)"
    
    def test_enable_tf32(self):
        """Test TF32 enablement."""
        from optimization_layers import Layer01Basics
        layer = Layer01Basics()
        
        layer.enable_tf32()
        
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch.backends.cudnn.allow_tf32 is True
    
    def test_enable_cudnn_benchmark(self):
        """Test cuDNN benchmark enablement."""
        from optimization_layers import Layer01Basics
        layer = Layer01Basics()
        
        layer.enable_cudnn_benchmark()
        
        assert torch.backends.cudnn.benchmark is True


class TestLayer03Pipelining:
    """Tests for Layer 03: Pipelining."""
    
    def test_import(self):
        """Test layer can be imported."""
        from optimization_layers import Layer03Pipelining
        layer = Layer03Pipelining()
        assert layer.name == "Layer 03: Pipelining (Ch9-10)"
    
    def test_flash_attention_context(self):
        """Test FlashAttention context manager."""
        from optimization_layers import Layer03Pipelining
        layer = Layer03Pipelining()
        
        # Should not raise
        with layer.flash_attention_backend():
            pass
    
    def test_double_buffer_setup(self):
        """Test double buffer stream creation."""
        from optimization_layers import Layer03Pipelining
        layer = Layer03Pipelining()
        
        layer.setup_double_buffering()
        
        if torch.cuda.is_available():
            assert "load" in layer._double_buffer_streams
            assert "compute" in layer._double_buffer_streams


class TestLayer04Concurrency:
    """Tests for Layer 04: Concurrency."""
    
    def test_import(self):
        """Test layer can be imported."""
        from optimization_layers import Layer04Concurrency
        layer = Layer04Concurrency()
        assert layer.name == "Layer 04: Concurrency (Ch11-12)"
    
    def test_stream_setup(self):
        """Test CUDA stream setup."""
        from optimization_layers import Layer04Concurrency
        layer = Layer04Concurrency()
        
        layer.setup_streams()
        
        if torch.cuda.is_available():
            assert layer._prefill_stream is not None
            assert layer._decode_stream is not None


class TestLayer05PyTorch:
    """Tests for Layer 05: PyTorch."""
    
    def test_import(self):
        """Test layer can be imported."""
        from optimization_layers import Layer05PyTorch
        layer = Layer05PyTorch()
        assert layer.name == "Layer 05: PyTorch (Ch13-14)"
    
    def test_inductor_config(self):
        """Test TorchInductor configuration."""
        from optimization_layers import Layer05PyTorch
        layer = Layer05PyTorch()
        
        # Should not raise
        layer.configure_inductor()


class TestLayerOrdering:
    """Tests for layer ordering and composition."""
    
    def test_get_all_layers(self):
        """Test getting all layers."""
        from optimization_layers import get_all_layers
        
        layers = get_all_layers()
        
        assert len(layers) == 6
        assert layers[0].name.startswith("Layer 01")
        assert layers[-1].name.startswith("Layer 06")
    
    def test_get_layers_up_to(self):
        """Test getting layers up to a specific number."""
        from optimization_layers import get_layers_up_to
        
        layers = get_layers_up_to(3)
        
        assert len(layers) == 3
        assert layers[0].name.startswith("Layer 01")
        assert layers[2].name.startswith("Layer 03")


class TestMetricsCollector:
    """Tests for MetricsCollector."""
    
    def test_request_tracking(self):
        """Test request start/end tracking."""
        from components.monitoring import MetricsCollector
        
        collector = MetricsCollector(track_power=False)
        
        collector.start_request("req_1")
        collector.record_first_token("req_1")
        collector.end_request("req_1", prompt_tokens=100, output_tokens=50)
        
        metrics = collector.compute_metrics()
        
        assert metrics.prompt_tokens == 100
        assert metrics.output_tokens == 50
        assert metrics.batch_size == 1
    
    def test_metrics_reset(self):
        """Test metrics reset."""
        from components.monitoring import MetricsCollector
        
        collector = MetricsCollector(track_power=False)
        collector.start_request("req_1")
        collector.end_request("req_1", 100, 50)
        
        collector.reset()
        
        metrics = collector.compute_metrics()
        assert metrics.batch_size == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
