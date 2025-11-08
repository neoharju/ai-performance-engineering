"""CUDA extension loader for ch12 kernels."""

from pathlib import Path
import torch
import sys

from common.python.nvtx_stub import ensure_nvtx_stub

# Import build utilities to prevent hangs from stale locks
try:
    from common.python.build_utils import ensure_clean_build_directory
except ImportError:
    # Fallback if build_utils not available
    def ensure_clean_build_directory(build_dir, max_lock_age_seconds=300):
        pass

NVTX_CFLAG = "-DENABLE_NVTX_PROFILING"
_NVTX_STUB_LIB = ensure_nvtx_stub()
NVTX_LDFLAGS = [f"-L{_NVTX_STUB_LIB.parent}", "-lnvToolsExt"]

_EXTENSIONS = {}


def _get_extension_dir():
    """Get the directory containing CUDA extension files."""
    return Path(__file__).parent


def load_kernel_fusion_extension():
    """Load the kernel fusion CUDA extension."""
    if "kernel_fusion" not in _EXTENSIONS:
        try:
            from torch.utils.cpp_extension import load
            
            extension_dir = _get_extension_dir()
            cuda_source = extension_dir / "kernel_fusion_kernels.cu"
            
            common_headers = Path(__file__).parent.parent.parent / "common" / "headers"
            build_dir = extension_dir / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            # Clean stale locks before building to prevent hangs
            ensure_clean_build_directory(build_dir)
            cuda_flags = ["-lineinfo", f"-I{common_headers}", NVTX_CFLAG]
            _EXTENSIONS["kernel_fusion"] = load(
                name="kernel_fusion_kernels",
                sources=[str(cuda_source)],
                extra_cuda_cflags=cuda_flags,
                extra_ldflags=list(NVTX_LDFLAGS),
                verbose=False,
                build_directory=str(build_dir),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load kernel_fusion CUDA extension: {e}"
            ) from e
    
    return _EXTENSIONS["kernel_fusion"]


def load_graph_bandwidth_extension():
    """Load the graph bandwidth CUDA extension."""
    if "graph_bandwidth" not in _EXTENSIONS:
        try:
            from torch.utils.cpp_extension import load
            
            extension_dir = _get_extension_dir()
            cuda_source = extension_dir / "graph_bandwidth_kernels.cu"
            
            common_headers = Path(__file__).parent.parent.parent / "common" / "headers"
            build_dir = extension_dir / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            # Clean stale locks before building to prevent hangs
            ensure_clean_build_directory(build_dir)
            cuda_flags = ["-lineinfo", f"-I{common_headers}", NVTX_CFLAG]
            _EXTENSIONS["graph_bandwidth"] = load(
                name="graph_bandwidth_kernels",
                sources=[str(cuda_source)],
                extra_cuda_cflags=cuda_flags,
                extra_ldflags=list(NVTX_LDFLAGS),
                verbose=False,
                build_directory=str(build_dir),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load graph_bandwidth CUDA extension: {e}"
            ) from e
    
    return _EXTENSIONS["graph_bandwidth"]


def load_work_queue_extension():
    """Load the work queue CUDA extension."""
    if "work_queue" not in _EXTENSIONS:
        from torch.utils.cpp_extension import load
        import shutil
        
        extension_dir = _get_extension_dir()
        cuda_source = extension_dir / "work_queue_kernels.cu"
        
        # Use absolute path for common_headers to avoid build directory issues
        common_headers = Path(__file__).parent.parent.parent / "common" / "headers"
        common_headers = common_headers.resolve()  # Convert to absolute path
        build_dir = extension_dir / "build"
        build_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean stale locks before building to prevent hangs
        ensure_clean_build_directory(build_dir)
        
        # Clean up any existing build artifacts that might be corrupted
        so_file = build_dir / "work_queue_kernels.so"
        o_file = build_dir / "work_queue_kernels.cuda.o"
        dep_file = build_dir / "work_queue_kernels.cuda.o.d"
        ninja_file = build_dir / "build.ninja"
        ninja_deps = build_dir / "build.ninja.d"
        
        # Clean up all build artifacts before first attempt
        for f in [so_file, o_file, dep_file, ninja_file, ninja_deps]:
            if f.exists():
                f.unlink(missing_ok=True)
        
        # Try loading with retry on undefined symbol errors
        max_retries = 3
        last_error = None
        for attempt in range(max_retries):
            try:
                cuda_flags = ["-lineinfo", f"-I{common_headers}", NVTX_CFLAG]
                _EXTENSIONS["work_queue"] = load(
                    name="work_queue_kernels",
                    sources=[str(cuda_source)],
                    extra_cuda_cflags=cuda_flags,
                    extra_ldflags=list(NVTX_LDFLAGS),
                    verbose=False,  # Set to True for debugging
                    build_directory=str(build_dir),
                )
                # Verify the extension actually works by checking if it has the expected methods
                if not hasattr(_EXTENSIONS["work_queue"], "static_work_distribution"):
                    raise RuntimeError("Extension loaded but missing expected methods")
                break  # Success, exit retry loop
            except Exception as e:
                last_error = e
                error_str = str(e)
                # Check if this is an undefined symbol error
                is_undefined_symbol = (
                    "undefined symbol" in error_str.lower() or 
                    "_ZN2at10TensorBase" in error_str or
                    "_ZNK2at10TensorBase" in error_str or
                    "cannot open shared object file" in error_str.lower()
                )
                
                if is_undefined_symbol and attempt < max_retries - 1:
                    # Clean up ALL build artifacts and force complete rebuild
                    for f in [so_file, o_file, dep_file, ninja_file, ninja_deps]:
                        if f.exists():
                            f.unlink(missing_ok=True)
                    # Also clean up any lock files
                    ensure_clean_build_directory(build_dir)
                    continue  # Retry
                elif attempt < max_retries - 1:
                    # Other error, but still retry once more
                    for f in [so_file, o_file, dep_file]:
                        if f.exists():
                            f.unlink(missing_ok=True)
                    continue
                else:
                    # Out of retries or non-retryable error
                    raise RuntimeError(
                        f"Failed to load work_queue CUDA extension after {max_retries} attempts: {last_error}"
                    ) from last_error
        
        if "work_queue" not in _EXTENSIONS:
            raise RuntimeError(
                f"Failed to load work_queue CUDA extension after {max_retries} attempts: {last_error}"
            )
    
    return _EXTENSIONS["work_queue"]


def load_cuda_graphs_extension():
    """Load the CUDA graphs extension."""
    if "cuda_graphs" not in _EXTENSIONS:
        try:
            from torch.utils.cpp_extension import load
            
            extension_dir = _get_extension_dir()
            cuda_source = extension_dir / "cuda_graphs_kernels.cu"
            
            common_headers = Path(__file__).parent.parent.parent / "common" / "headers"
            build_dir = extension_dir / "build"
            build_dir.mkdir(parents=True, exist_ok=True)
            # Clean stale locks before building to prevent hangs
            ensure_clean_build_directory(build_dir)
            cuda_flags = ["-lineinfo", f"-I{common_headers}", NVTX_CFLAG]
            _EXTENSIONS["cuda_graphs"] = load(
                name="cuda_graphs_kernels",
                sources=[str(cuda_source)],
                extra_cuda_cflags=cuda_flags,
                extra_ldflags=list(NVTX_LDFLAGS),
                verbose=False,
                build_directory=str(build_dir),
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load cuda_graphs CUDA extension: {e}"
            ) from e
    
    return _EXTENSIONS["cuda_graphs"]
