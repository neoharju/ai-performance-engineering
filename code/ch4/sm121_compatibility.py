"""
Helper module to handle sm_121 (GB10) compatibility issues.

Some features have known limitations on sm_121 due to toolchain support.
This module provides transparent messaging for unsupported features.
"""
from common.python import compile_utils as _compile_utils_patch  # noqa: F401
import pathlib
import sys

_EXTRAS_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_EXTRAS_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_EXTRAS_REPO_ROOT))

from pathlib import Path


import torch


def get_compute_capability() -> tuple[int, int]:
    """Get the compute capability of the first GPU."""
    if not torch.cuda.is_available():
        return (0, 0)
    
    props = torch.cuda.get_device_properties(0)
    return (props.major, props.minor)


def is_sm121() -> bool:
    """Check if running on sm_121 (GB10) hardware."""
    major, minor = get_compute_capability()
    return major == 12 and minor == 1


def skip_if_sm121_triton_issue(script_name: str) -> None:
    """
    Skip execution on sm_121 with a clear message about Triton compiler limitations.
    
    Some PyTorch features using torch.compile() have Triton compiler issues on sm_121
    due to incomplete support for the 'sm_121a' PTX target in the Triton toolchain.
    
    Args:
        script_name: Name of the script being skipped
    """
    if not is_sm121():
        return  # Not sm_121, proceed normally
    
    major, minor = get_compute_capability()
    
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║                    FEATURE NOT SUPPORTED ON SM_121                          ║")
    print("╠══════════════════════════════════════════════════════════════════════════════╣")
    print(f"║ Script: {script_name:<68} ║")
    print(f"║ Hardware: sm_{major}{minor} (GB10)                                                      ║")
    print("║                                                                              ║")
    print("║ This script uses torch.compile() which has a known Triton compiler issue    ║")
    print("║ on sm_121 hardware:                                                          ║")
    print("║                                                                              ║")
    print("║   Error: ptxas fatal: Value 'sm_121a' is not defined                         ║")
    print("║                                                                              ║")
    print("║ This is a PyTorch/Triton toolchain limitation, not a code bug.              ║")
    print("║                                                                              ║")
    print("║ Status: ⏸️  SKIPPED (waiting for toolchain update)                           ║")
    print("║                                                                              ║")
    print("║ This feature will work on:                                                  ║")
    print("║   • sm_100 (B200/B300) [OK]                                                     ║")
    print("║   • sm_90 (H100/H200) [OK]                                                      ║")
    print("║   • Future PyTorch versions with sm_121 support [OK]                           ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()
    
    sys.exit(0)  # Exit successfully (not an error)


def warn_if_sm121_experimental(feature_name: str) -> None:
    """
    Print a warning if running experimental features on sm_121.
    
    Args:
        feature_name: Name of the experimental feature
    """
    if not is_sm121():
        return
    
    major, minor = get_compute_capability()
    
    print(f"\nWARNING: WARNING: {feature_name} on sm_{major}{minor} (GB10)")
    print(f"    This feature is experimental and may have limited support.")
    print(f"    Please report any issues encountered.\n")


if __name__ == "__main__":
    # Test the detection
    if torch.cuda.is_available():
        major, minor = get_compute_capability()
        print(f"Detected GPU: sm_{major}{minor}")
        print(f"Is sm_121: {is_sm121()}")
        
        if is_sm121():
            print("\nTesting skip message:")
            skip_if_sm121_triton_issue("test_script.py")
    else:
        print("No CUDA device available")

