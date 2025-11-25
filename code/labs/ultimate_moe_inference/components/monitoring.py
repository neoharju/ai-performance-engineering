"""Monitoring and metrics collection for inference benchmarks.

Provides comprehensive metrics tracking:
- TTFT (Time to First Token)
- TPOT (Time Per Output Token)
- TPS (Tokens Per Second)
- Latency percentiles (p50, p90, p95, p99)
- Memory usage
- Power/energy consumption
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean, median, stdev
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False


@dataclass
class InferenceMetrics:
    """Comprehensive inference metrics.
    
    Latency metrics are in milliseconds.
    Throughput metrics are in tokens per second.
    Memory metrics are in gigabytes.
    Energy metrics are in joules.
    """
    
    # Latency (milliseconds)
    ttft_ms: float = 0.0  # Time to First Token (prefill latency)
    tpot_ms: float = 0.0  # Time Per Output Token (decode latency, average)
    e2e_latency_ms: float = 0.0  # End-to-end request latency
    
    # Throughput
    tokens_per_sec: float = 0.0  # Output tokens per second
    prefill_tokens_per_sec: float = 0.0  # Input tokens per second (prefill phase)
    
    # Batch info
    batch_size: int = 1
    prompt_tokens: int = 0
    output_tokens: int = 0
    
    # Memory (GB)
    peak_memory_gb: float = 0.0
    kv_cache_memory_gb: float = 0.0
    
    # Energy (optional)
    energy_joules: Optional[float] = None
    avg_power_watts: Optional[float] = None
    tokens_per_joule: Optional[float] = None
    
    # Percentiles (for multiple requests)
    ttft_p50_ms: Optional[float] = None
    ttft_p90_ms: Optional[float] = None
    ttft_p95_ms: Optional[float] = None
    ttft_p99_ms: Optional[float] = None
    
    tpot_p50_ms: Optional[float] = None
    tpot_p90_ms: Optional[float] = None
    tpot_p95_ms: Optional[float] = None
    tpot_p99_ms: Optional[float] = None
    
    e2e_p50_ms: Optional[float] = None
    e2e_p90_ms: Optional[float] = None
    e2e_p95_ms: Optional[float] = None
    e2e_p99_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "latency": {
                "ttft_ms": self.ttft_ms,
                "tpot_ms": self.tpot_ms,
                "e2e_latency_ms": self.e2e_latency_ms,
            },
            "throughput": {
                "tokens_per_sec": self.tokens_per_sec,
                "prefill_tokens_per_sec": self.prefill_tokens_per_sec,
            },
            "batch": {
                "batch_size": self.batch_size,
                "prompt_tokens": self.prompt_tokens,
                "output_tokens": self.output_tokens,
            },
            "memory": {
                "peak_memory_gb": self.peak_memory_gb,
                "kv_cache_memory_gb": self.kv_cache_memory_gb,
            },
            "energy": {
                "energy_joules": self.energy_joules,
                "avg_power_watts": self.avg_power_watts,
                "tokens_per_joule": self.tokens_per_joule,
            },
            "percentiles": {
                "ttft": {
                    "p50": self.ttft_p50_ms,
                    "p90": self.ttft_p90_ms,
                    "p95": self.ttft_p95_ms,
                    "p99": self.ttft_p99_ms,
                },
                "tpot": {
                    "p50": self.tpot_p50_ms,
                    "p90": self.tpot_p90_ms,
                    "p95": self.tpot_p95_ms,
                    "p99": self.tpot_p99_ms,
                },
                "e2e": {
                    "p50": self.e2e_p50_ms,
                    "p90": self.e2e_p90_ms,
                    "p95": self.e2e_p95_ms,
                    "p99": self.e2e_p99_ms,
                },
            },
        }
    
    def __str__(self) -> str:
        """Return human-readable summary."""
        lines = [
            "Inference Metrics:",
            f"  TTFT: {self.ttft_ms:.2f} ms",
            f"  TPOT: {self.tpot_ms:.2f} ms",
            f"  E2E Latency: {self.e2e_latency_ms:.2f} ms",
            f"  Throughput: {self.tokens_per_sec:.1f} tok/s",
            f"  Peak Memory: {self.peak_memory_gb:.2f} GB",
        ]
        
        if self.tokens_per_joule:
            lines.append(f"  Efficiency: {self.tokens_per_joule:.2f} tok/J")
        
        return "\n".join(lines)


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    
    request_id: str = ""
    
    # Timestamps (seconds)
    start_time: float = 0.0
    first_token_time: float = 0.0
    end_time: float = 0.0
    
    # Token counts
    prompt_tokens: int = 0
    output_tokens: int = 0
    
    # Per-token decode times (milliseconds)
    decode_times_ms: List[float] = field(default_factory=list)
    
    @property
    def ttft_ms(self) -> float:
        """Time to first token in milliseconds."""
        if self.first_token_time <= 0:
            return 0.0
        return (self.first_token_time - self.start_time) * 1000
    
    @property
    def tpot_ms(self) -> float:
        """Average time per output token in milliseconds."""
        if not self.decode_times_ms:
            return 0.0
        return mean(self.decode_times_ms)
    
    @property
    def e2e_latency_ms(self) -> float:
        """End-to-end latency in milliseconds."""
        if self.end_time <= 0:
            return 0.0
        return (self.end_time - self.start_time) * 1000


class MetricsCollector:
    """Collect and aggregate inference metrics.
    
    Usage:
        collector = MetricsCollector()
        
        # For each request
        collector.start_request("req_1")
        collector.record_first_token("req_1")
        for token in generated_tokens:
            collector.record_decode_step("req_1", step_time_ms)
        collector.end_request("req_1", prompt_tokens, output_tokens)
        
        # Get aggregated metrics
        metrics = collector.compute_metrics()
    """
    
    def __init__(
        self,
        track_power: bool = True,
        power_sample_interval: float = 0.1,
    ):
        """Initialize metrics collector.
        
        Args:
            track_power: Whether to track power consumption
            power_sample_interval: Power sampling interval in seconds
        """
        self.track_power = track_power and PYNVML_AVAILABLE
        self.power_sample_interval = power_sample_interval
        
        self._requests: Dict[str, RequestMetrics] = {}
        self._completed_requests: List[RequestMetrics] = []
        
        # Power tracking
        self._power_samples: List[Tuple[float, float]] = []  # (timestamp, watts)
        self._power_start_time: float = 0.0
        
        # Memory tracking
        self._peak_memory_bytes: int = 0
        
        # Initialize NVML if available
        self._nvml_initialized = False
        if self.track_power:
            self._init_nvml()
    
    def _init_nvml(self) -> None:
        """Initialize NVML for power monitoring."""
        try:
            pynvml.nvmlInit()
            self._nvml_initialized = True
        except Exception:
            self.track_power = False
    
    def _shutdown_nvml(self) -> None:
        """Shutdown NVML."""
        if self._nvml_initialized:
            pynvml.nvmlShutdown()
            self._nvml_initialized = False
    
    def start_request(self, request_id: str) -> None:
        """Mark start of a request.
        
        Args:
            request_id: Unique request identifier
        """
        self._requests[request_id] = RequestMetrics(
            request_id=request_id,
            start_time=time.perf_counter(),
        )
    
    def record_first_token(self, request_id: str) -> None:
        """Record time of first token generation.
        
        Args:
            request_id: Request identifier
        """
        if request_id in self._requests:
            self._requests[request_id].first_token_time = time.perf_counter()
    
    def record_decode_step(self, request_id: str, step_time_ms: float) -> None:
        """Record a single decode step.
        
        Args:
            request_id: Request identifier
            step_time_ms: Time for this decode step in milliseconds
        """
        if request_id in self._requests:
            self._requests[request_id].decode_times_ms.append(step_time_ms)
    
    def end_request(
        self,
        request_id: str,
        prompt_tokens: int,
        output_tokens: int,
    ) -> None:
        """Mark end of a request.
        
        Args:
            request_id: Request identifier
            prompt_tokens: Number of prompt tokens
            output_tokens: Number of generated tokens
        """
        if request_id in self._requests:
            req = self._requests[request_id]
            req.end_time = time.perf_counter()
            req.prompt_tokens = prompt_tokens
            req.output_tokens = output_tokens
            
            self._completed_requests.append(req)
            del self._requests[request_id]
    
    def start_power_monitoring(self) -> None:
        """Start power monitoring."""
        if not self.track_power:
            return
        
        self._power_samples = []
        self._power_start_time = time.perf_counter()
    
    def sample_power(self) -> Optional[float]:
        """Sample current power consumption.
        
        Returns:
            Power in watts, or None if unavailable
        """
        if not self.track_power or not self._nvml_initialized:
            return None
        
        try:
            total_power = 0.0
            device_count = pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                total_power += power_mw / 1000.0  # Convert to watts
            
            timestamp = time.perf_counter()
            self._power_samples.append((timestamp, total_power))
            return total_power
        except Exception:
            return None
    
    def stop_power_monitoring(self) -> Tuple[float, float]:
        """Stop power monitoring and compute energy.
        
        Returns:
            Tuple of (energy_joules, avg_power_watts)
        """
        if not self._power_samples:
            return 0.0, 0.0
        
        # Compute energy using trapezoidal integration
        energy_joules = 0.0
        for i in range(1, len(self._power_samples)):
            t0, p0 = self._power_samples[i - 1]
            t1, p1 = self._power_samples[i]
            dt = t1 - t0
            avg_power = (p0 + p1) / 2
            energy_joules += avg_power * dt
        
        # Average power
        powers = [p for _, p in self._power_samples]
        avg_power = mean(powers) if powers else 0.0
        
        return energy_joules, avg_power
    
    def record_memory(self) -> None:
        """Record current memory usage."""
        if torch.cuda.is_available():
            current_bytes = torch.cuda.max_memory_allocated()
            self._peak_memory_bytes = max(self._peak_memory_bytes, current_bytes)
    
    def reset_memory_tracking(self) -> None:
        """Reset memory tracking."""
        self._peak_memory_bytes = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    
    def compute_metrics(self) -> InferenceMetrics:
        """Compute aggregated metrics from all completed requests.
        
        Returns:
            Aggregated InferenceMetrics
        """
        if not self._completed_requests:
            return InferenceMetrics()
        
        # Collect individual metrics
        ttft_values = [r.ttft_ms for r in self._completed_requests]
        tpot_values = [r.tpot_ms for r in self._completed_requests if r.tpot_ms > 0]
        e2e_values = [r.e2e_latency_ms for r in self._completed_requests]
        
        total_prompt_tokens = sum(r.prompt_tokens for r in self._completed_requests)
        total_output_tokens = sum(r.output_tokens for r in self._completed_requests)
        total_time_s = sum(r.e2e_latency_ms for r in self._completed_requests) / 1000
        
        # Compute averages
        avg_ttft = mean(ttft_values) if ttft_values else 0.0
        avg_tpot = mean(tpot_values) if tpot_values else 0.0
        avg_e2e = mean(e2e_values) if e2e_values else 0.0
        
        # Compute throughput
        tokens_per_sec = total_output_tokens / total_time_s if total_time_s > 0 else 0.0
        prefill_tps = total_prompt_tokens / (sum(ttft_values) / 1000) if ttft_values else 0.0
        
        # Memory
        peak_memory_gb = self._peak_memory_bytes / 1e9
        
        # Power/energy
        energy_joules, avg_power = self.stop_power_monitoring()
        tokens_per_joule = total_output_tokens / energy_joules if energy_joules > 0 else None
        
        # Create metrics object
        metrics = InferenceMetrics(
            ttft_ms=avg_ttft,
            tpot_ms=avg_tpot,
            e2e_latency_ms=avg_e2e,
            tokens_per_sec=tokens_per_sec,
            prefill_tokens_per_sec=prefill_tps,
            batch_size=len(self._completed_requests),
            prompt_tokens=total_prompt_tokens,
            output_tokens=total_output_tokens,
            peak_memory_gb=peak_memory_gb,
            energy_joules=energy_joules if energy_joules > 0 else None,
            avg_power_watts=avg_power if avg_power > 0 else None,
            tokens_per_joule=tokens_per_joule,
        )
        
        # Compute percentiles
        if len(ttft_values) >= 10:
            metrics.ttft_p50_ms = self._percentile(ttft_values, 50)
            metrics.ttft_p90_ms = self._percentile(ttft_values, 90)
            metrics.ttft_p95_ms = self._percentile(ttft_values, 95)
            metrics.ttft_p99_ms = self._percentile(ttft_values, 99)
        
        if len(tpot_values) >= 10:
            metrics.tpot_p50_ms = self._percentile(tpot_values, 50)
            metrics.tpot_p90_ms = self._percentile(tpot_values, 90)
            metrics.tpot_p95_ms = self._percentile(tpot_values, 95)
            metrics.tpot_p99_ms = self._percentile(tpot_values, 99)
        
        if len(e2e_values) >= 10:
            metrics.e2e_p50_ms = self._percentile(e2e_values, 50)
            metrics.e2e_p90_ms = self._percentile(e2e_values, 90)
            metrics.e2e_p95_ms = self._percentile(e2e_values, 95)
            metrics.e2e_p99_ms = self._percentile(e2e_values, 99)
        
        return metrics
    
    @staticmethod
    def _percentile(values: List[float], p: int) -> float:
        """Compute percentile.
        
        Args:
            values: List of values
            p: Percentile (0-100)
            
        Returns:
            Percentile value
        """
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        idx = int(len(sorted_values) * p / 100)
        idx = min(idx, len(sorted_values) - 1)
        return sorted_values[idx]
    
    def reset(self) -> None:
        """Reset all collected metrics."""
        self._requests.clear()
        self._completed_requests.clear()
        self._power_samples.clear()
        self._peak_memory_bytes = 0
    
    def __del__(self) -> None:
        """Cleanup."""
        self._shutdown_nvml()


def track_inference(
    generate_fn: Callable[..., Any],
    prompt_tokens: int,
    expected_output_tokens: int,
    request_id: str = "default",
) -> Tuple[Any, RequestMetrics]:
    """Convenience function to track a single inference call.
    
    Args:
        generate_fn: Function that generates tokens (returns output)
        prompt_tokens: Number of input tokens
        expected_output_tokens: Expected number of output tokens
        request_id: Request identifier
        
    Returns:
        Tuple of (generation output, RequestMetrics)
    """
    collector = MetricsCollector(track_power=False)
    
    collector.start_request(request_id)
    output = generate_fn()
    
    # Estimate first token time (rough approximation)
    collector.record_first_token(request_id)
    collector.end_request(request_id, prompt_tokens, expected_output_tokens)
    
    metrics = collector.compute_metrics()
    return output, collector._completed_requests[0] if collector._completed_requests else RequestMetrics()

