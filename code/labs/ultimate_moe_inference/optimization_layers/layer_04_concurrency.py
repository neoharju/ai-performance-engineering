"""Layer 04: Concurrency Optimizations (Ch11-12).

Inter-kernel pipelining and execution optimizations:
- CUDA streams for concurrent kernel execution
- CUDA graphs for reduced launch overhead
- Graph bucketing for variable sequence AND batch sizes
- Graph pool management for memory efficiency

BEFORE (no graphs):
    Each decode step: ~200µs CPU overhead for kernel launches
    For 1000 tokens: 200ms pure launch overhead!

AFTER (with CUDA graphs):
    Graph replay: ~2µs per step
    For 1000 tokens: 2ms total (100x reduction!)

GRAPH BUCKETING (from PERFORMANCE_OPTIMIZATION_ANALYSIS.md):
    Problem: CUDA graphs require fixed tensor shapes
    Solution: Pre-capture graphs for common size "buckets"
    
    Example buckets: [1, 2, 4, 8, 16, 32] batch × [64, 128, 256, 512, 1024] seq_len
    Runtime: Select smallest bucket that fits, pad if needed
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class GraphBucket:
    """Represents a CUDA graph bucket for specific tensor dimensions."""
    batch_size: int
    seq_len: int
    graph: Optional[torch.cuda.CUDAGraph] = None
    input_buffer: Optional[torch.Tensor] = None
    output_buffer: Optional[torch.Tensor] = None
    hit_count: int = 0  # For profiling bucket usage


class Layer04Concurrency:
    """Layer 4: Concurrency optimizations from Chapters 11-12.
    
    These optimizations reduce kernel launch overhead and enable
    concurrent execution of independent operations.
    
    Techniques:
    - Ch11: CUDA streams, events, stream-ordered memory
    - Ch12: CUDA graphs, graph capture, graph replay
    """
    
    name = "Layer 04: Concurrency (Ch11-12)"
    chapters = [11, 12]
    
    def __init__(self):
        """Initialize layer."""
        self._streams: Dict[str, torch.cuda.Stream] = {}
        self._graphs: Dict[str, torch.cuda.CUDAGraph] = {}
        self._graph_inputs: Dict[str, torch.Tensor] = {}
        self._graph_outputs: Dict[str, torch.Tensor] = {}
        self._graphs_enabled = False
        
        # Enhanced: 2D bucketing (batch × seq_len)
        self._graph_buckets: Dict[Tuple[int, int], GraphBucket] = {}
        self._bucket_batch_sizes = [1, 2, 4, 8, 16, 32]
        self._bucket_seq_lens = [64, 128, 256, 512, 1024, 2048]
        
        # Graph pool for memory management
        self._graph_pool = None
        if torch.cuda.is_available():
            try:
                # Use memory pool to reduce graph memory overhead
                self._graph_pool = torch.cuda.graph_pool_handle()
            except AttributeError:
                pass  # Older PyTorch version
    
    def apply(self, benchmark: Any) -> None:
        """Apply concurrency optimizations.
        
        Args:
            benchmark: Benchmark instance to configure
        """
        config = getattr(benchmark, 'config', None)
        
        self.setup_streams()
        
        if config and getattr(config, 'use_cuda_graphs', True):
            self._graphs_enabled = True
    
    def setup_streams(self) -> None:
        """Setup CUDA streams for concurrent execution (Ch11).
        
        Creates dedicated streams for different phases of inference:
        - prefill: For prompt processing
        - decode: For token generation
        - transfer: For async data transfers
        """
        if not torch.cuda.is_available():
            return
        
        self._streams = {
            "prefill": torch.cuda.Stream(),
            "decode": torch.cuda.Stream(),
            "transfer": torch.cuda.Stream(),
        }
        
        print("  [Layer 04] Created CUDA streams for prefill/decode/transfer")
    
    def get_stream(self, name: str) -> Optional[torch.cuda.Stream]:
        """Get a named stream.
        
        Args:
            name: Stream name
            
        Returns:
            CUDA stream or None
        """
        return self._streams.get(name)
    
    def capture_graph(
        self,
        name: str,
        model: nn.Module,
        static_input: torch.Tensor,
        warmup_runs: int = 3,
    ) -> torch.cuda.CUDAGraph:
        """Capture a CUDA graph for repeated execution (Ch12).
        
        CUDA graphs capture a sequence of kernel launches and replay
        them with minimal CPU overhead. This is especially beneficial
        for the decode phase where the same operations repeat.
        
        Args:
            name: Graph name for later retrieval
            model: Model to capture
            static_input: Static input tensor (will be copied to before replay)
            warmup_runs: Number of warmup runs before capture
            
        Returns:
            Captured CUDA graph
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        
        print(f"  [Layer 04] Capturing CUDA graph '{name}'")
        
        # Store static input
        self._graph_inputs[name] = static_input.clone()
        
        # Warmup runs (required before capture)
        stream = self._streams.get("decode") or torch.cuda.current_stream()
        with torch.cuda.stream(stream):
            for _ in range(warmup_runs):
                with torch.no_grad():
                    _ = model(self._graph_inputs[name])
        
        stream.synchronize()
        
        # Capture graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            with torch.no_grad():
                self._graph_outputs[name] = model(self._graph_inputs[name])
        
        self._graphs[name] = graph
        print(f"  [Layer 04] Graph '{name}' captured successfully")
        
        return graph
    
    def replay_graph(
        self,
        name: str,
        input_tensor: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Replay a captured CUDA graph (Ch12).
        
        Args:
            name: Graph name
            input_tensor: Optional input to copy before replay
            
        Returns:
            Graph output tensor
        """
        if name not in self._graphs:
            raise ValueError(f"Graph '{name}' not found")
        
        # Copy new input to static buffer
        if input_tensor is not None:
            self._graph_inputs[name].copy_(input_tensor)
        
        # Replay graph
        self._graphs[name].replay()
        
        return self._graph_outputs[name]
    
    def capture_decode_graph(
        self,
        model: nn.Module,
        batch_size: int,
        seq_len: int,
        hidden_dim: int,
        num_buckets: int = 4,
    ) -> Dict[int, torch.cuda.CUDAGraph]:
        """Capture decode graphs for multiple sequence length buckets (Ch12).
        
        Different sequence lengths require different graph instances
        because CUDA graphs require fixed tensor shapes.
        
        Args:
            model: Model to capture
            batch_size: Batch size
            seq_len: Maximum sequence length
            hidden_dim: Hidden dimension
            num_buckets: Number of length buckets
            
        Returns:
            Dictionary mapping bucket index to graph
        """
        graphs = {}
        
        # Create buckets with exponentially increasing sizes
        bucket_sizes = []
        size = 64  # Minimum size
        while size <= seq_len and len(bucket_sizes) < num_buckets:
            bucket_sizes.append(size)
            size *= 2
        
        print(f"  [Layer 04] Capturing graphs for buckets: {bucket_sizes}")
        
        for bucket_idx, bucket_size in enumerate(bucket_sizes):
            # Create static input for this bucket
            static_input = torch.zeros(
                batch_size, bucket_size, hidden_dim,
                device="cuda", dtype=torch.bfloat16
            )
            
            try:
                graph = self.capture_graph(
                    f"decode_bucket_{bucket_idx}",
                    model,
                    static_input,
                )
                graphs[bucket_idx] = graph
            except Exception as e:
                print(f"  [Layer 04] Warning: Could not capture graph for bucket {bucket_idx}: {e}")
        
        return graphs
    
    def select_graph_bucket(self, seq_len: int) -> int:
        """Select appropriate graph bucket for sequence length.
        
        Args:
            seq_len: Actual sequence length
            
        Returns:
            Bucket index
        """
        # Find smallest bucket that fits
        for bucket_idx in range(len(self._graphs)):
            name = f"decode_bucket_{bucket_idx}"
            if name in self._graph_inputs:
                bucket_size = self._graph_inputs[name].shape[1]
                if bucket_size >= seq_len:
                    return bucket_idx
        
        # Fall back to largest bucket
        return len(self._graphs) - 1
    
    def synchronize_streams(self) -> None:
        """Synchronize all streams."""
        for stream in self._streams.values():
            stream.synchronize()
    
    def record_event(self, stream_name: str) -> Optional[torch.cuda.Event]:
        """Record event on a stream for synchronization (Ch11).
        
        Args:
            stream_name: Name of stream to record on
            
        Returns:
            Recorded event or None
        """
        stream = self._streams.get(stream_name)
        if stream is None:
            return None
        
        event = torch.cuda.Event()
        event.record(stream)
        return event
    
    def wait_event(self, stream_name: str, event: torch.cuda.Event) -> None:
        """Make stream wait for an event (Ch11).
        
        Args:
            stream_name: Name of stream that should wait
            event: Event to wait for
        """
        stream = self._streams.get(stream_name)
        if stream is not None:
            stream.wait_event(event)
    
    def capture_2d_buckets(
        self,
        model: nn.Module,
        hidden_dim: int,
        batch_sizes: Optional[List[int]] = None,
        seq_lens: Optional[List[int]] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> int:
        """Capture graphs for 2D bucket grid (batch × seq_len).
        
        From PERFORMANCE_OPTIMIZATION_ANALYSIS.md:
        Pre-capture graphs for common dimension combinations to handle
        variable batch sizes AND sequence lengths efficiently.
        
        Args:
            model: Model to capture
            hidden_dim: Hidden dimension
            batch_sizes: Batch size buckets (default: [1,2,4,8,16,32])
            seq_lens: Sequence length buckets (default: [64,128,256,512,1024,2048])
            dtype: Tensor dtype
            
        Returns:
            Number of successfully captured graphs
        """
        batch_sizes = batch_sizes or self._bucket_batch_sizes
        seq_lens = seq_lens or self._bucket_seq_lens
        
        captured = 0
        total = len(batch_sizes) * len(seq_lens)
        
        print(f"  [Layer 04] Capturing {total} graphs for 2D bucket grid...")
        print(f"             Batch sizes: {batch_sizes}")
        print(f"             Seq lengths: {seq_lens}")
        
        stream = self._streams.get("decode") or torch.cuda.current_stream()
        
        for bs in batch_sizes:
            for sl in seq_lens:
                bucket_key = (bs, sl)
                
                try:
                    # Create static buffers
                    input_buffer = torch.zeros(
                        bs, sl, hidden_dim,
                        device="cuda", dtype=dtype
                    )
                    
                    # Warmup
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            with torch.no_grad():
                                _ = model(input_buffer)
                    stream.synchronize()
                    
                    # Capture
                    graph = torch.cuda.CUDAGraph()
                    capture_kwargs = {"stream": stream}
                    if self._graph_pool is not None:
                        capture_kwargs["pool"] = self._graph_pool
                    
                    with torch.cuda.graph(graph, **capture_kwargs):
                        with torch.no_grad():
                            output_buffer = model(input_buffer)
                    
                    self._graph_buckets[bucket_key] = GraphBucket(
                        batch_size=bs,
                        seq_len=sl,
                        graph=graph,
                        input_buffer=input_buffer,
                        output_buffer=output_buffer,
                    )
                    captured += 1
                    
                except Exception as e:
                    # Skip buckets that fail (e.g., OOM for large sizes)
                    pass
        
        print(f"  [Layer 04] Captured {captured}/{total} graphs successfully")
        return captured
    
    def replay_bucketed(
        self,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Replay graph using optimal bucket selection.
        
        Selects the smallest bucket that can fit the input,
        copies input with padding, replays graph, returns output.
        
        Args:
            input_tensor: Input tensor [batch, seq_len, hidden]
            
        Returns:
            Output tensor
        """
        bs, sl, _ = input_tensor.shape
        
        # Find smallest bucket that fits
        bucket = None
        for (bucket_bs, bucket_sl), candidate in sorted(self._graph_buckets.items()):
            if bucket_bs >= bs and bucket_sl >= sl:
                bucket = candidate
                break
        
        if bucket is None:
            # No bucket fits - fall back to eager execution
            raise ValueError(f"No bucket fits input shape ({bs}, {sl})")
        
        # Update stats
        bucket.hit_count += 1
        
        # Copy input to buffer (with padding if needed)
        bucket.input_buffer[:bs, :sl, :] = input_tensor
        if bs < bucket.batch_size or sl < bucket.seq_len:
            # Zero out padding
            bucket.input_buffer[bs:, :, :] = 0
            bucket.input_buffer[:, sl:, :] = 0
        
        # Replay
        bucket.graph.replay()
        
        # Return only valid portion
        return bucket.output_buffer[:bs, :sl, :]
    
    def get_bucket_stats(self) -> Dict[str, Any]:
        """Get bucket usage statistics for profiling."""
        stats = {}
        for (bs, sl), bucket in self._graph_buckets.items():
            stats[f"bucket_{bs}x{sl}"] = bucket.hit_count
        return stats
    
    def get_status(self) -> dict:
        """Get status of applied optimizations."""
        return {
            "num_streams": len(self._streams),
            "num_graphs": len(self._graphs),
            "num_2d_buckets": len(self._graph_buckets),
            "graphs_enabled": self._graphs_enabled,
            "bucket_stats": self.get_bucket_stats(),
        }
    
    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"{self.name}: streams={len(self._streams)}, "
            f"graphs={len(self._graphs)}, "
            f"2d_buckets={len(self._graph_buckets)}"
        )

