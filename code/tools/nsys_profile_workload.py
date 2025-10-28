"""Automated Nsight Systems profiling wrapper for specific workloads."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional


WORKLOAD_PROFILES = {
    "fp8_benchmark": {
        "script": "ch16/test_gpt_large_optimized.py",
        "args": ["--fp8-mode", "transformer-engine", "--batch-size", "1", "--seq-len", "2048"],
        "description": "FP8 quantization with transformer_engine"
    },
    "flex_attention": {
        "script": "ch18/flex_attention_native.py",
        "args": [],
        "description": "FlexAttention workload"
    },
    "moe_inference": {
        "script": "ch16/synthetic_moe_inference_benchmark.py",
        "args": [],
        "env": {"MOE_BENCH_QUICK": "1"},
        "description": "MoE inference benchmark"
    },
    "torch_compile": {
        "script": "ch14/torch_compiler_examples.py",
        "args": [],
        "description": "torch.compile examples"
    },
}


def check_nsys_available() -> bool:
    """Check if nsys is available on the system."""
    try:
        result = subprocess.run(
            ["nsys", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def run_nsys_profile(
    workload_name: str,
    script: str,
    script_args: List[str],
    output_path: Path,
    duration: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
    capture_cuda_api: bool = True,
    capture_nvtx: bool = True,
    capture_cudnn: bool = True,
) -> bool:
    """Run nsys profiling on a workload.
    
    Args:
        workload_name: Name of the workload
        script: Python script to profile
        script_args: Arguments for the script
        output_path: Path to save .nsys-rep file
        duration: Optional duration limit in seconds
        env: Optional environment variables
        capture_cuda_api: Capture CUDA API calls
        capture_nvtx: Capture NVTX markers
        capture_cudnn: Capture cuDNN calls
    
    Returns:
        True if profiling succeeded
    """
    
    # Build nsys command
    nsys_cmd = [
        "nsys", "profile",
        "--output", str(output_path),
        "--force-overwrite", "true",
        "--stats", "true",
        "--capture-range", "cudaProfilerApi",
        "--stop-on-range-end", "true",
    ]
    
    # Add trace options
    trace_opts = ["cuda"]
    if capture_cuda_api:
        trace_opts.append("cuda-api")
    if capture_nvtx:
        trace_opts.append("nvtx")
    if capture_cudnn:
        trace_opts.append("cudnn")
    
    nsys_cmd.extend(["--trace", ",".join(trace_opts)])
    
    # Add duration if specified
    if duration:
        nsys_cmd.extend(["--duration", str(duration)])
    
    # Add the Python command
    nsys_cmd.extend([sys.executable, script] + script_args)
    
    print(f"Profiling {workload_name}...")
    print(f"  Script: {script}")
    print(f"  Output: {output_path}")
    print(f"  Command: {' '.join(nsys_cmd)}")
    
    try:
        # Merge environment
        import os
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        
        result = subprocess.run(
            nsys_cmd,
            env=merged_env,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        
        if result.returncode != 0:
            print(f"  Error: Profiling failed with exit code {result.returncode}")
            print(f"  stderr: {result.stderr}")
            return False
        
        print(f"  ✓ Profiling complete")
        return True
    
    except subprocess.TimeoutExpired:
        print(f"  Error: Profiling timed out")
        return False
    except Exception as e:
        print(f"  Error: {e}")
        return False


def generate_profile_report(
    profile_path: Path,
    output_dir: Path,
    kernel_regex: Optional[str] = None,
    top_k: int = 20
) -> None:
    """Generate a summary report from the nsys profile."""
    
    summary_path = output_dir / f"{profile_path.stem}_summary.txt"
    
    print(f"  Generating summary report...")
    
    try:
        cmd = [
            sys.executable,
            "tools/nsys_summary.py",
            "--report", str(profile_path),
            "--top-k", str(top_k),
            "--output", str(summary_path)
        ]
        
        if kernel_regex:
            cmd.extend(["--kernel-regex", kernel_regex])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode == 0:
            print(f"  ✓ Summary saved to: {summary_path}")
        else:
            print(f"  Warning: Summary generation had issues")
    
    except Exception as e:
        print(f"  Warning: Could not generate summary: {e}")


def profile_all_workloads(
    output_dir: Path,
    workloads: Optional[List[str]] = None,
    kernel_regex: Optional[str] = None
) -> Dict[str, bool]:
    """Profile multiple workloads.
    
    Args:
        output_dir: Directory to save profiles
        workloads: List of workload names to profile (None = all)
        kernel_regex: Optional regex filter for kernel summary
    
    Returns:
        Dict mapping workload name to success status
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if workloads is None:
        workloads = list(WORKLOAD_PROFILES.keys())
    
    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for workload_name in workloads:
        if workload_name not in WORKLOAD_PROFILES:
            print(f"Warning: Unknown workload '{workload_name}', skipping")
            results[workload_name] = False
            continue
        
        profile = WORKLOAD_PROFILES[workload_name]
        
        print(f"\n{'='*80}")
        print(f"Workload: {workload_name}")
        print(f"Description: {profile['description']}")
        print(f"{'='*80}\n")
        
        output_path = output_dir / f"{workload_name}_{timestamp}.nsys-rep"
        
        success = run_nsys_profile(
            workload_name=workload_name,
            script=profile["script"],
            script_args=profile.get("args", []),
            output_path=output_path,
            env=profile.get("env"),
        )
        
        results[workload_name] = success
        
        # Generate summary if successful
        if success and output_path.exists():
            generate_profile_report(output_path, output_dir, kernel_regex)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Automated Nsight Systems profiling for key workloads"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("nsys_profiles"),
        help="Directory to save profile outputs"
    )
    parser.add_argument(
        "--workloads",
        nargs="+",
        choices=list(WORKLOAD_PROFILES.keys()) + ["all"],
        default=["all"],
        help="Workloads to profile"
    )
    parser.add_argument(
        "--kernel-regex",
        default="attn|mma|nvjet|cublas|gemm|fp8",
        help="Regex filter for kernel summary"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available workload profiles"
    )
    args = parser.parse_args()
    
    if args.list:
        print("Available workload profiles:\n")
        for name, profile in WORKLOAD_PROFILES.items():
            print(f"  {name}")
            print(f"    Description: {profile['description']}")
            print(f"    Script: {profile['script']}")
            print()
        return 0
    
    # Check if nsys is available
    if not check_nsys_available():
        print("Error: Nsight Systems (nsys) not found on system")
        print("Install it from: https://developer.nvidia.com/nsight-systems")
        return 1
    
    # Determine which workloads to profile
    if "all" in args.workloads:
        workloads = list(WORKLOAD_PROFILES.keys())
    else:
        workloads = args.workloads
    
    print("="*80)
    print("Automated Nsight Systems Profiling")
    print("="*80)
    print(f"\nWorkloads: {', '.join(workloads)}")
    print(f"Output directory: {args.output_dir}")
    print(f"Kernel filter: {args.kernel_regex}")
    print()
    
    # Run profiling
    results = profile_all_workloads(
        args.output_dir,
        workloads,
        args.kernel_regex
    )
    
    # Print summary
    print("\n" + "="*80)
    print("Profiling Summary")
    print("="*80 + "\n")
    
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    
    for workload, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {workload}: {status}")
    
    print(f"\nTotal: {success_count}/{total_count} succeeded")
    print(f"\nProfiles saved to: {args.output_dir}/")
    
    # Generate index file
    index_path = args.output_dir / "INDEX.md"
    index_lines = [
        "# Nsight Systems Profiling Results\n",
        f"\n**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "\n## Profiles\n",
        "\n| Workload | Status | Files |\n",
        "|----------|--------|-------|\n"
    ]
    
    for workload, success in results.items():
        status = "✓" if success else "✗"
        # Find profile files
        profile_files = list(args.output_dir.glob(f"{workload}_*.nsys-rep"))
        summary_files = list(args.output_dir.glob(f"{workload}_*_summary.txt"))
        
        files_str = ""
        if profile_files:
            files_str += f"[.nsys-rep]({profile_files[0].name})"
        if summary_files:
            files_str += f" [summary]({summary_files[0].name})"
        
        index_lines.append(f"| {workload} | {status} | {files_str} |\n")
    
    index_lines.extend([
        "\n## How to View\n",
        "\n### Option 1: Nsight Systems GUI\n",
        "```bash\n",
        "nsys-ui <profile_name>.nsys-rep\n",
        "```\n",
        "\n### Option 2: Command-line stats\n",
        "```bash\n",
        "nsys stats <profile_name>.nsys-rep\n",
        "```\n",
        "\n### Option 3: Summary reports\n",
        "View the `*_summary.txt` files for quick kernel analysis.\n",
    ])
    
    index_path.write_text("".join(index_lines))
    print(f"\nIndex created: {index_path}")
    
    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())


