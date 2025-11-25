#!/usr/bin/env python3
"""Ultimate MoE Inference Lab - Full Analysis Pipeline.

One command to run everything and generate a comprehensive report showing
the incremental performance improvement from each chapter's techniques.

Usage:
    python run_full_analysis.py                    # Full analysis
    python run_full_analysis.py --quick            # Quick mode (fewer iterations)
    python run_full_analysis.py --profile nsys     # Include Nsight Systems
    python run_full_analysis.py --profile all      # Full profiling (nsys + ncu + HTA)
    python run_full_analysis.py --report-only      # Generate report from existing data

Output:
    artifacts/ultimate_moe_inference/
    ├── results.json                  # Raw benchmark results
    ├── layer_comparison.png          # Speedup visualization
    ├── roofline_analysis.json        # Roofline analysis
    ├── profiling/
    │   ├── *.nsys-rep               # Nsight Systems traces
    │   ├── *.ncu-rep                # Nsight Compute reports
    │   └── *.pt.trace.json          # PyTorch profiler traces (for HTA)
    └── REPORT.md                     # Human-readable summary
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add paths
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(Path(__file__).parent))

import torch


@dataclass
class LayerResult:
    """Result for a single optimization layer."""
    
    name: str
    chapters: List[int]
    techniques: List[str]
    mean_ms: float
    std_ms: float
    tokens_per_sec: float
    speedup_vs_baseline: float
    incremental_speedup: float
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    memory_gb: Optional[float] = None


@dataclass
class AnalysisResult:
    """Complete analysis result."""
    
    timestamp: str
    hardware: str
    model: str
    layers: List[LayerResult]
    total_speedup: float
    profiling_artifacts: List[str] = field(default_factory=list)
    roofline: Optional[Dict[str, Any]] = None


# Layer configurations - Educational progression through the book
LAYERS = [
    {
        "name": "00_baseline",
        "module": "baseline_ultimate_inference",
        "chapters": [],
        "techniques": ["No optimizations (eager mode)"],
        "description": "Measure first! Identify bottlenecks.",
    },
    {
        "name": "01_basics",
        "module": "01_basics",
        "chapters": [1, 2, 3, 4, 5, 6],
        "techniques": ["TF32", "cuDNN", "NUMA", "NVTX", "Tensor Cores"],
        "description": "Foundation: GPU config, profiling setup",
    },
    {
        "name": "02_memory",
        "module": "02_memory_bottleneck",
        "chapters": [7, 8],
        "techniques": ["Coalescing", "Vectorization", "Occupancy", "ILP", "L2 cache"],
        "description": "Understand WHY memory is the bottleneck",
    },
    {
        "name": "03_flash",
        "module": "03_flash_attention",
        "chapters": [9, 10],
        "techniques": ["Tiling", "Double buffering", "TMA", "Online softmax", "FlashAttention"],
        "description": "THE BIG WIN! Tiled attention solves O(n²)",
    },
    {
        "name": "04_graphs",
        "module": "04_cuda_graphs",
        "chapters": [11, 12],
        "techniques": ["CUDA streams", "CUDA graphs", "Cooperative Groups", "DSMEM"],
        "description": "Eliminate kernel launch overhead",
    },
    {
        "name": "05_compile",
        "module": "05_torch_compile",
        "chapters": [13, 14],
        "techniques": ["FP8", "torch.compile", "TorchInductor", "Triton", "Autotuning"],
        "description": "Kernel fusion, reduced precision",
    },
    {
        "name": "06_ultimate",
        "module": "06_ultimate",
        "chapters": list(range(1, 21)),
        "techniques": ["ALL above", "MoE", "PagedAttention", "Speculative decode", "NVFP4"],
        "description": "Production inference - everything combined!",
    },
]


def get_hardware_info() -> str:
    """Get GPU hardware info."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"{name} (SM {cc[0]}.{cc[1]}, {mem:.0f}GB)"
    return "No GPU"


def run_benchmark(
    module_name: str,
    iterations: int = 10,
    warmup: int = 3,
    enable_profiler: bool = False,
    profiler_output: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a single benchmark and return results."""
    print(f"\n{'='*60}")
    print(f"Running: {module_name}")
    print(f"{'='*60}")
    
    try:
        # Import benchmark module
        module = __import__(module_name)
        benchmark = module.get_benchmark()
        
        # Import harness
        from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode, BenchmarkConfig
        
        config = BenchmarkConfig(iterations=iterations, warmup=warmup)
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        # Run with optional torch profiler
        if enable_profiler and profiler_output:
            result = _run_with_profiler(benchmark, harness, profiler_output)
        else:
            result = harness.benchmark(benchmark)
        
        # Extract metrics
        metrics = {
            "mean_ms": result.timing.mean_ms if result.timing else 0,
            "std_ms": result.timing.std_ms if result.timing else 0,
            "median_ms": result.timing.median_ms if result.timing else 0,
        }
        
        # Get additional metrics from benchmark
        if hasattr(benchmark, 'last_metrics') and benchmark.last_metrics:
            m = benchmark.last_metrics
            metrics["tokens_per_sec"] = m.tokens_per_sec
            metrics["ttft_ms"] = m.ttft_ms
            metrics["tpot_ms"] = m.tpot_ms
            metrics["memory_gb"] = m.peak_memory_gb
        
        # Cleanup
        benchmark.teardown()
        torch.cuda.empty_cache()
        
        return metrics
        
    except Exception as e:
        print(f"  Error: {e}")
        return {"mean_ms": 0, "std_ms": 0, "error": str(e)}


def _run_with_profiler(benchmark, harness, output_path: Path):
    """Run benchmark with torch.profiler for HTA analysis."""
    from torch.profiler import profile, ProfilerActivity, schedule
    
    # Setup benchmark
    benchmark.setup()
    
    # Profile configuration
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    
    with profile(
        activities=activities,
        schedule=schedule(wait=1, warmup=2, active=5, repeat=1),
        on_trace_ready=lambda p: p.export_chrome_trace(str(output_path)),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(8):  # wait + warmup + active
            benchmark.benchmark_fn()
            prof.step()
    
    print(f"  Profiler trace saved to: {output_path}")
    
    # Run actual benchmark
    result = harness.benchmark(benchmark)
    return result


def run_nsys_profiling(module_name: str, output_dir: Path) -> Optional[Path]:
    """Run Nsight Systems profiling."""
    output_file = output_dir / f"{module_name}.nsys-rep"
    
    cmd = [
        "nsys", "profile",
        "-o", str(output_file.with_suffix("")),
        "--trace=cuda,nvtx",
        "--cuda-memory-usage=true",
        sys.executable, "-c",
        f"import {module_name}; b = {module_name}.get_benchmark(); "
        f"b.setup(); [b.benchmark_fn() for _ in range(5)]; b.teardown()"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=300)
        if result.returncode == 0 and output_file.exists():
            print(f"  Nsight Systems trace: {output_file}")
            return output_file
    except Exception as e:
        print(f"  Nsight Systems failed: {e}")
    
    return None


def run_ncu_profiling(module_name: str, output_dir: Path) -> Optional[Path]:
    """Run Nsight Compute profiling."""
    output_file = output_dir / f"{module_name}.ncu-rep"
    
    cmd = [
        "ncu",
        "-o", str(output_file.with_suffix("")),
        "--set", "full",
        "--target-processes", "all",
        sys.executable, "-c",
        f"import {module_name}; b = {module_name}.get_benchmark(); "
        f"b.setup(); b.benchmark_fn(); b.teardown()"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode == 0 and output_file.exists():
            print(f"  Nsight Compute report: {output_file}")
            return output_file
    except Exception as e:
        print(f"  Nsight Compute failed: {e}")
    
    return None


def generate_comparison_plot(results: List[LayerResult], output_path: Path) -> None:
    """Generate speedup comparison visualization."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Prepare data
        names = [r.name for r in results]
        speedups = [r.speedup_vs_baseline for r in results]
        incremental = [r.incremental_speedup for r in results]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar colors
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
        
        # Plot 1: Cumulative speedup
        bars1 = ax1.bar(names, speedups, color=colors)
        ax1.set_ylabel('Speedup vs Baseline', fontsize=12)
        ax1.set_title('Cumulative Optimization Impact', fontsize=14, fontweight='bold')
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Baseline')
        
        # Add value labels
        for bar, val in zip(bars1, speedups):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=10)
        
        # Plot 2: Incremental contribution
        bars2 = ax2.bar(names[1:], incremental[1:], color=colors[1:])
        ax2.set_ylabel('Incremental Speedup', fontsize=12)
        ax2.set_title('Per-Layer Contribution', fontsize=14, fontweight='bold')
        ax2.set_xticklabels(names[1:], rotation=45, ha='right')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='No improvement')
        
        for bar, val in zip(bars2, incremental[1:]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2f}x', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\nComparison plot saved to: {output_path}")
        
    except ImportError:
        print("  Warning: matplotlib not available, skipping plot")


def generate_report(analysis: AnalysisResult, output_path: Path) -> None:
    """Generate markdown report."""
    lines = [
        "# Ultimate MoE Inference Lab - Analysis Report",
        "",
        f"**Generated:** {analysis.timestamp}",
        f"**Hardware:** {analysis.hardware}",
        f"**Model:** {analysis.model}",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        f"**Total Speedup: {analysis.total_speedup:.2f}x** over baseline",
        "",
        "This report demonstrates the cumulative impact of optimization techniques",
        "from Chapters 1-20 of the AI Performance Engineering book.",
        "",
        "---",
        "",
        "## Layer-by-Layer Breakdown",
        "",
        "| Layer | Chapters | Speedup | Incremental | Latency (ms) | Techniques |",
        "|-------|----------|---------|-------------|--------------|------------|",
    ]
    
    for r in analysis.layers:
        chapters = f"Ch{min(r.chapters)}-{max(r.chapters)}" if r.chapters else "N/A"
        techniques = ", ".join(r.techniques[:2]) + ("..." if len(r.techniques) > 2 else "")
        lines.append(
            f"| {r.name} | {chapters} | {r.speedup_vs_baseline:.2f}x | "
            f"{r.incremental_speedup:.2f}x | {r.mean_ms:.1f} | {techniques} |"
        )
    
    lines.extend([
        "",
        "---",
        "",
        "## Detailed Technique Breakdown",
        "",
    ])
    
    for r in analysis.layers:
        if r.name == "baseline":
            continue
        lines.extend([
            f"### {r.name.replace('_', ' ').title()}",
            "",
            f"**Chapters:** {', '.join(map(str, r.chapters))}",
            f"**Speedup:** {r.speedup_vs_baseline:.2f}x cumulative, {r.incremental_speedup:.2f}x incremental",
            "",
            "**Techniques applied:**",
        ])
        for tech in r.techniques:
            lines.append(f"- {tech}")
        
        if r.tokens_per_sec:
            lines.append(f"\n**Throughput:** {r.tokens_per_sec:.1f} tokens/sec")
        if r.ttft_ms:
            lines.append(f"**TTFT:** {r.ttft_ms:.2f} ms")
        lines.append("")
    
    # Roofline section
    if analysis.roofline:
        lines.extend([
            "---",
            "",
            "## Roofline Analysis",
            "",
            f"**Workload Classification:** {analysis.roofline.get('bound', 'unknown').upper()}-BOUND",
            "",
            f"- Arithmetic Intensity: {analysis.roofline.get('arithmetic_intensity', 0):.2f} FLOP/Byte",
            f"- Compute Efficiency: {analysis.roofline.get('achieved_compute_pct', 0):.1f}%",
            f"- Bandwidth Efficiency: {analysis.roofline.get('achieved_bandwidth_pct', 0):.1f}%",
            "",
        ])
    
    # Profiling artifacts
    if analysis.profiling_artifacts:
        lines.extend([
            "---",
            "",
            "## Profiling Artifacts",
            "",
            "The following profiling artifacts were generated:",
            "",
        ])
        for artifact in analysis.profiling_artifacts:
            lines.append(f"- `{artifact}`")
        
        lines.extend([
            "",
            "### Viewing Profiling Data",
            "",
            "**Nsight Systems:**",
            "```bash",
            "nsys-ui artifacts/ultimate_moe_inference/profiling/<name>.nsys-rep",
            "```",
            "",
            "**Nsight Compute:**",
            "```bash",
            "ncu-ui artifacts/ultimate_moe_inference/profiling/<name>.ncu-rep",
            "```",
            "",
            "**HTA (Holistic Trace Analysis):**",
            "```bash",
            "# Install HTA: pip install HolisticTraceAnalysis",
            "from hta.trace_analysis import TraceAnalysis",
            "analyzer = TraceAnalysis(trace_dir='artifacts/ultimate_moe_inference/profiling/')",
            "analyzer.get_gpu_kernel_breakdown()",
            "```",
            "",
        ])
    
    # How to reproduce
    lines.extend([
        "---",
        "",
        "## How to Reproduce",
        "",
        "```bash",
        "# Full analysis (recommended)",
        "python run_full_analysis.py",
        "",
        "# Quick mode (fewer iterations)",
        "python run_full_analysis.py --quick",
        "",
        "# With full profiling",
        "python run_full_analysis.py --profile all",
        "",
        "# Using benchmark_cli harness",
        "python tools/cli/benchmark_cli.py run --targets labs/ultimate_moe_inference --profile deep_dive",
        "```",
        "",
        "---",
        "",
        "*Generated by Ultimate MoE Inference Lab*",
    ])
    
    output_path.write_text("\n".join(lines))
    print(f"\nReport saved to: {output_path}")


def run_full_analysis(
    quick: bool = False,
    profile: str = "none",
    report_only: bool = False,
    output_dir: Optional[Path] = None,
) -> AnalysisResult:
    """Run complete analysis pipeline.
    
    Args:
        quick: Use fewer iterations for faster results
        profile: Profiling mode: "none", "nsys", "ncu", "hta", "all"
        report_only: Only generate report from existing data
        output_dir: Output directory for artifacts
        
    Returns:
        AnalysisResult with all data
    """
    # Setup output directory
    if output_dir is None:
        output_dir = Path("artifacts/ultimate_moe_inference")
    output_dir.mkdir(parents=True, exist_ok=True)
    profiling_dir = output_dir / "profiling"
    profiling_dir.mkdir(exist_ok=True)
    
    # Check for existing results
    results_file = output_dir / "results.json"
    if report_only and results_file.exists():
        print("Loading existing results...")
        with open(results_file) as f:
            data = json.load(f)
        # Reconstruct AnalysisResult
        analysis = AnalysisResult(
            timestamp=data["timestamp"],
            hardware=data["hardware"],
            model=data["model"],
            layers=[LayerResult(**l) for l in data["layers"]],
            total_speedup=data["total_speedup"],
            profiling_artifacts=data.get("profiling_artifacts", []),
            roofline=data.get("roofline"),
        )
    else:
        # Run benchmarks
        iterations = 5 if quick else 10
        warmup = 2 if quick else 3
        
        print("\n" + "=" * 70)
        print("ULTIMATE MOE INFERENCE LAB - FULL ANALYSIS")
        print("=" * 70)
        print(f"\nHardware: {get_hardware_info()}")
        print(f"Mode: {'Quick' if quick else 'Full'} ({iterations} iterations)")
        print(f"Profiling: {profile}")
        print()
        
        layer_results = []
        baseline_ms = None
        prev_ms = None
        profiling_artifacts = []
        
        for layer_config in LAYERS:
            module_name = layer_config["module"]
            
            # Enable torch profiler for HTA traces
            enable_profiler = profile in ["hta", "all"]
            profiler_output = profiling_dir / f"{module_name}.pt.trace.json" if enable_profiler else None
            
            # Run benchmark
            metrics = run_benchmark(
                module_name,
                iterations=iterations,
                warmup=warmup,
                enable_profiler=enable_profiler,
                profiler_output=profiler_output,
            )
            
            mean_ms = metrics.get("mean_ms", 0)
            
            # Calculate speedups
            if baseline_ms is None:
                baseline_ms = mean_ms
                speedup = 1.0
            else:
                speedup = baseline_ms / mean_ms if mean_ms > 0 else 0
            
            if prev_ms is None:
                incremental = 1.0
            else:
                incremental = prev_ms / mean_ms if mean_ms > 0 else 1.0
            
            prev_ms = mean_ms
            
            result = LayerResult(
                name=layer_config["name"],
                chapters=layer_config["chapters"],
                techniques=layer_config["techniques"],
                mean_ms=mean_ms,
                std_ms=metrics.get("std_ms", 0),
                tokens_per_sec=metrics.get("tokens_per_sec", 0),
                speedup_vs_baseline=speedup,
                incremental_speedup=incremental,
                ttft_ms=metrics.get("ttft_ms"),
                tpot_ms=metrics.get("tpot_ms"),
                memory_gb=metrics.get("memory_gb"),
            )
            layer_results.append(result)
            
            # Track profiler artifacts
            if profiler_output and profiler_output.exists():
                profiling_artifacts.append(str(profiler_output.relative_to(output_dir)))
            
            # Run nsys/ncu profiling
            if profile in ["nsys", "all"]:
                nsys_file = run_nsys_profiling(module_name, profiling_dir)
                if nsys_file:
                    profiling_artifacts.append(str(nsys_file.relative_to(output_dir)))
            
            if profile in ["ncu", "all"] and layer_config["name"] in ["baseline", "ultimate"]:
                ncu_file = run_ncu_profiling(module_name, profiling_dir)
                if ncu_file:
                    profiling_artifacts.append(str(ncu_file.relative_to(output_dir)))
        
        # Run roofline analysis
        roofline = None
        try:
            from scaling_study.roofline_analyzer import RooflineAnalyzer
            analyzer = RooflineAnalyzer()
            # Use placeholder metrics for now
            roofline_metrics = analyzer.analyze_from_metrics(
                flops=1e15,
                dram_bytes=1e12,
                duration_s=layer_results[-1].mean_ms / 1000,
            )
            roofline = roofline_metrics.to_dict()
        except Exception as e:
            print(f"Roofline analysis skipped: {e}")
        
        # Create analysis result
        analysis = AnalysisResult(
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            hardware=get_hardware_info(),
            model="openai/gpt-oss-20b",
            layers=layer_results,
            total_speedup=layer_results[-1].speedup_vs_baseline if layer_results else 1.0,
            profiling_artifacts=profiling_artifacts,
            roofline=roofline,
        )
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": analysis.timestamp,
                "hardware": analysis.hardware,
                "model": analysis.model,
                "layers": [asdict(l) for l in analysis.layers],
                "total_speedup": analysis.total_speedup,
                "profiling_artifacts": analysis.profiling_artifacts,
                "roofline": analysis.roofline,
            }, f, indent=2)
        print(f"\nResults saved to: {results_file}")
    
    # Generate visualizations
    generate_comparison_plot(analysis.layers, output_dir / "layer_comparison.png")
    
    # Generate report
    generate_report(analysis, output_dir / "REPORT.md")
    
    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nTotal Speedup: {analysis.total_speedup:.2f}x")
    print(f"\nOutputs:")
    print(f"  Results: {output_dir / 'results.json'}")
    print(f"  Report:  {output_dir / 'REPORT.md'}")
    print(f"  Plot:    {output_dir / 'layer_comparison.png'}")
    if analysis.profiling_artifacts:
        print(f"  Profiles: {len(analysis.profiling_artifacts)} files in {profiling_dir}")
    print()
    
    return analysis


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Ultimate MoE Inference Lab - Full Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_full_analysis.py                    # Full analysis
  python run_full_analysis.py --quick            # Quick mode
  python run_full_analysis.py --profile all      # With full profiling
  python run_full_analysis.py --report-only      # Regenerate report
        """,
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode with fewer iterations"
    )
    parser.add_argument(
        "--profile", choices=["none", "nsys", "ncu", "hta", "all"],
        default="none",
        help="Profiling mode (default: none)"
    )
    parser.add_argument(
        "--report-only", action="store_true",
        help="Only generate report from existing data"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory (default: artifacts/ultimate_moe_inference)"
    )
    
    args = parser.parse_args()
    
    run_full_analysis(
        quick=args.quick,
        profile=args.profile,
        report_only=args.report_only,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

