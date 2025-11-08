"""Centralized profiler configuration.

Single source of truth for profiler command generation and metric sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


# Metric sets for different profiling scenarios
ROOFLINE_METRICS = [
    # Kernel timing
    "gpu__time_duration.avg",
    # Compute throughput (SM)
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    # Memory throughput - DRAM (HBM)
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    # Memory throughput - L2 cache
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    # Occupancy - active warps
    "sm__warps_active.avg.pct_of_peak_sustained_active",
]

DEEP_DIVE_METRICS = [
    # All roofline metrics
    *ROOFLINE_METRICS,
    # Compute proxy - FP32 instructions
    "sm__sass_thread_inst_executed_op_fp32_pred_on.sum",
    # Memory efficiency metrics
    "dram__sectors_read.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    # Memory load efficiency
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    # Tensor Core utilization (if applicable)
    "sm__inst_executed_pipe_tensor.sum",
]

MINIMAL_METRICS = [
    "gpu__time_duration.avg",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
]


@dataclass
class ProfilerConfig:
    """Configuration for profiler execution."""
    
    metric_set: str = "deep_dive"  # "roofline", "deep_dive", or "minimal"
    nsys_trace_types: str = "cuda,nvtx,osrt,cublas,cudnn"
    nsys_sample_cpu: bool = True
    nsys_python_sampling: bool = True
    nsys_python_sampling_frequency: int = 1000
    nsys_cudabacktrace: bool = True
    nsys_stats: bool = True
    ncu_replay_mode: str = "kernel"  # "kernel" or "application"
    
    def get_nsys_command(
        self, 
        output_path: str, 
        python_script: str,
        python_executable: Optional[str] = None
    ) -> List[str]:
        """Generate nsys command with configured flags.
        
        Args:
            output_path: Path for nsys output (without .nsys-rep extension)
            python_script: Path to Python script to profile
            python_executable: Python executable to use (defaults to sys.executable)
        
        Returns:
            List of command arguments for subprocess.run
        """
        import sys
        
        cmd = [
            "nsys",
            "profile",
            "--force-overwrite=true",
            "-o",
            output_path,
            "-t", self.nsys_trace_types,
        ]
        
        if self.nsys_sample_cpu:
            cmd.extend(["-s", "cpu"])
        
        if self.nsys_python_sampling:
            cmd.extend([
                "--python-sampling=true",
                f"--python-sampling-frequency={self.nsys_python_sampling_frequency}",
            ])
        
        if self.nsys_cudabacktrace:
            cmd.append("--cudabacktrace=true")
        
        if self.nsys_stats:
            cmd.append("--stats=true")
        
        cmd.append(python_executable or sys.executable)
        cmd.append(python_script)
        
        return cmd
    
    def get_ncu_command(
        self,
        output_path: str,
        python_script: str,
        python_executable: Optional[str] = None,
        metrics: Optional[List[str]] = None
    ) -> List[str]:
        """Generate ncu command with configured metrics.
        
        Args:
            output_path: Path for ncu output (without .ncu-rep extension)
            python_script: Path to Python script to profile
            python_executable: Python executable to use (defaults to sys.executable)
            metrics: Optional list of metrics (defaults to metric_set)
        
        Returns:
            List of command arguments for subprocess.run
        """
        import sys
        
        if metrics is None:
            if self.metric_set == "roofline":
                metrics = ROOFLINE_METRICS
            elif self.metric_set == "deep_dive":
                metrics = DEEP_DIVE_METRICS
            elif self.metric_set == "minimal":
                metrics = MINIMAL_METRICS
            else:
                metrics = DEEP_DIVE_METRICS
        
        cmd = [
            "ncu",
            "--set", "full",
            "--metrics", ",".join(metrics),
            "--replay-mode", self.ncu_replay_mode,
            "-o", output_path,
            python_executable or sys.executable,
            python_script,
        ]
        
        return cmd
    
    def get_torch_profiler_config(self) -> dict:
        """Get PyTorch profiler configuration.
        
        Returns:
            Dictionary of kwargs for torch.profiler.profile()
        """
        return {
            "activities": [
                "torch.profiler.ProfilerActivity.CPU",
                "torch.profiler.ProfilerActivity.CUDA",
            ],
            "record_shapes": True,
            "profile_memory": True,
            "with_stack": True,
            "with_flops": True,
        }


# Default profiler config instance
DEFAULT_PROFILER_CONFIG = ProfilerConfig()

