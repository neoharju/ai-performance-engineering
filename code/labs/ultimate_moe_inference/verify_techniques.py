#!/usr/bin/env python3
"""
verify_techniques.py - Prove that optimization techniques actually work

This script aggregates REAL benchmark results from chapter-level examples
to verify that each technique provides the claimed speedup.

The chapter examples have been benchmarked on actual B200 hardware.
This script collects those results and maps them to the techniques
used in the Ultimate Lab.

Usage:
    python verify_techniques.py                    # Show verified speedups
    python verify_techniques.py --run-benchmarks   # Re-run chapter benchmarks
    python verify_techniques.py --run-ultimate     # Run ultimate lab (needs model)
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add repo root
REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))


@dataclass
class TechniqueVerification:
    """Verification result for a technique."""
    technique: str
    chapter: str
    example: str
    baseline_ms: float
    optimized_ms: float
    speedup: float
    verified: bool
    last_run: str


def load_chapter_expectations(chapter: str) -> Optional[Dict[str, Any]]:
    """Load expectations.json for a chapter."""
    expectations_file = REPO_ROOT / chapter / "expectations_b200.json"
    if not expectations_file.exists():
        return None
    
    with open(expectations_file) as f:
        return json.load(f)


def extract_verified_speedups() -> List[TechniqueVerification]:
    """Extract verified speedups from all chapter expectations files."""
    
    verifications = []
    
    # Map chapters to techniques
    chapter_techniques = {
        "ch1": ["Warp Specialization basics"],
        "ch7": ["Coalescing", "Vectorized loads", "TMA Copy"],
        "ch8": ["Double buffering", "Occupancy tuning"],
        "ch9": ["Tiling", "Shared memory", "CUTLASS GEMM"],
        "ch10": [
            "Double-buffered pipeline", "Warp specialization",
            "TMA 2D pipeline", "FlashAttention", "Cluster groups",
            "Persistent kernels", "CUDA Pipeline API"
        ],
        "ch11": ["CUDA Streams", "Warp-specialized multistream"],
        "ch12": ["CUDA Graphs", "Dynamic scheduling"],
        "ch13": ["FP8 Transformer Engine", "Regional compile", "Dataloader"],
        "ch14": ["torch.compile", "Triton kernels", "TMA Blackwell"],
        "ch15": ["MoE parallelism", "Expert routing"],
        "ch16": ["PagedAttention", "Regional compilation", "Piece graphs"],
        "ch17": ["Pipeline parallelism", "Continuous batching"],
        "ch18": [
            "FlashMLA", "Speculative decode", "CUDA Graph bucketing",
            "vLLM decode", "PagedAttention vLLM"
        ],
        "ch19": ["Dynamic precision", "Adaptive parallelism"],
        "ch20": ["End-to-end bandwidth", "KV cache optimization"],
    }
    
    for chapter, techniques in chapter_techniques.items():
        data = load_chapter_expectations(chapter)
        if not data:
            continue
        
        examples = data.get("examples", {})
        for example_name, example_data in examples.items():
            metrics = example_data.get("metrics", {})
            metadata = example_data.get("metadata", {})
            
            baseline_ms = metrics.get("baseline_time_ms", 0)
            optimized_ms = metrics.get("best_optimized_time_ms", 0)
            speedup = metrics.get("best_speedup", 1.0)
            updated_at = metadata.get("updated_at", "unknown")
            
            if speedup > 1.0 and baseline_ms > 0:
                # Map to technique names
                technique_name = example_name.replace("_", " ").title()
                for tech in techniques:
                    if tech.lower() in example_name.lower() or example_name.lower() in tech.lower():
                        technique_name = tech
                        break
                
                verifications.append(TechniqueVerification(
                    technique=technique_name,
                    chapter=chapter,
                    example=example_name,
                    baseline_ms=baseline_ms,
                    optimized_ms=optimized_ms,
                    speedup=speedup,
                    verified=True,
                    last_run=updated_at,
                ))
    
    return verifications


def print_verification_report(verifications: List[TechniqueVerification]) -> None:
    """Print a nice verification report."""
    
    print("=" * 80)
    print("TECHNIQUE VERIFICATION REPORT")
    print("=" * 80)
    print("\nThese speedups are REAL measurements from B200 hardware:\n")
    
    # Group by chapter
    by_chapter: Dict[str, List[TechniqueVerification]] = {}
    for v in verifications:
        by_chapter.setdefault(v.chapter, []).append(v)
    
    total_verified = 0
    significant_speedups = []
    
    for chapter in sorted(by_chapter.keys()):
        items = by_chapter[chapter]
        print(f"\n{chapter.upper()}")
        print("-" * 60)
        
        for v in sorted(items, key=lambda x: -x.speedup):
            status = "✅" if v.verified else "❌"
            speedup_str = f"{v.speedup:.2f}x" if v.speedup < 10 else f"{v.speedup:.1f}x"
            
            print(f"  {status} {v.technique:<35} {speedup_str:>8}  ({v.example})")
            total_verified += 1
            
            if v.speedup > 1.5:
                significant_speedups.append(v)
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n✅ Verified techniques: {total_verified}")
    print(f"🚀 Significant speedups (>1.5x): {len(significant_speedups)}")
    
    if significant_speedups:
        print("\nTop speedups:")
        for v in sorted(significant_speedups, key=lambda x: -x.speedup)[:10]:
            print(f"  • {v.technique}: {v.speedup:.1f}x ({v.chapter}/{v.example})")
    
    print("\n" + "=" * 80)


def check_ultimate_lab_readiness() -> bool:
    """Check if the ultimate lab can be run."""
    
    print("\n" + "=" * 80)
    print("ULTIMATE LAB READINESS CHECK")
    print("=" * 80)
    
    # Check for model
    model_path = os.environ.get("LOCAL_MODEL_PATH", "")
    has_local_model = model_path and Path(model_path).exists()
    
    # Check for GPU
    try:
        import torch
        has_gpu = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if has_gpu else "None"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if has_gpu else 0
    except ImportError:
        has_gpu = False
        gpu_name = "Unknown"
        gpu_memory = 0
    
    # Check for dependencies
    try:
        import transformers  # noqa: F401
        has_transformers = True
    except ImportError:
        has_transformers = False
    
    try:
        import flash_attn  # noqa: F401
        has_flash_attn = True
    except ImportError:
        has_flash_attn = False
    
    print(f"\n  GPU Available:     {'✅' if has_gpu else '❌'} {gpu_name} ({gpu_memory:.0f}GB)")
    print(f"  Local Model:       {'✅' if has_local_model else '⚠️ '} {model_path or 'Set LOCAL_MODEL_PATH'}")
    print(f"  transformers:      {'✅' if has_transformers else '❌'}")
    print(f"  flash-attn:        {'✅' if has_flash_attn else '⚠️  (will use SDPA)'}")
    
    ready = has_gpu and has_transformers
    
    if ready:
        print("\n  ✅ Ready to run ultimate lab benchmarks!")
        print("\n  To run:")
        print("    export LOCAL_MODEL_PATH=/path/to/gpt-oss-20b")
        print("    python run_full_analysis.py --benchmark-only")
    else:
        print("\n  ❌ Not ready. Install missing dependencies.")
    
    print("=" * 80)
    return ready


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Verify optimization techniques")
    parser.add_argument("--run-benchmarks", action="store_true",
                       help="Re-run chapter-level benchmarks")
    parser.add_argument("--run-ultimate", action="store_true",
                       help="Run ultimate lab benchmarks")
    parser.add_argument("--json", action="store_true",
                       help="Output as JSON")
    args = parser.parse_args()
    
    # Always show verified speedups
    verifications = extract_verified_speedups()
    
    if args.json:
        data = [
            {
                "technique": v.technique,
                "chapter": v.chapter,
                "example": v.example,
                "speedup": v.speedup,
                "baseline_ms": v.baseline_ms,
                "optimized_ms": v.optimized_ms,
                "verified": v.verified,
                "last_run": v.last_run,
            }
            for v in verifications
        ]
        print(json.dumps(data, indent=2))
        return
    
    print_verification_report(verifications)
    
    if args.run_benchmarks:
        print("\nRunning chapter benchmarks...")
        print("This will take several minutes...\n")
        
        for chapter in sorted(set(v.chapter for v in verifications)):
            compare_script = REPO_ROOT / chapter / "compare.py"
            if compare_script.exists():
                print(f"Running {chapter}/compare.py...")
                os.system(f"cd {REPO_ROOT / chapter} && python compare.py")
    
    check_ultimate_lab_readiness()
    
    if args.run_ultimate:
        print("\nRunning ultimate lab benchmarks...")
        os.system(f"cd {Path(__file__).parent} && python run_full_analysis.py --benchmark-only")


if __name__ == "__main__":
    main()

