# Makefile for Chapter 4: Multi-GPU Communication Examples
#
# Compiles NVSHMEM examples for 8x B200 GPU configuration
#
# Usage:
#   make all              # Compile all examples
#   make nvshmem          # Compile NVSHMEM examples (requires NVSHMEM)
#   make clean            # Clean build artifacts

NVCC = nvcc
CUDA_ARCH = sm_100  # Blackwell B200
CXX_FLAGS = -O3 -std=c++17
CUDA_FLAGS = -arch=$(CUDA_ARCH) $(CXX_FLAGS)

# NVSHMEM paths (adjust as needed)
NVSHMEM_HOME ?= /usr/local/nvshmem
NVSHMEM_INCLUDE = -I$(NVSHMEM_HOME)/include
NVSHMEM_LIB = -L$(NVSHMEM_HOME)/lib -lnvshmem

# Targets
NVSHMEM_TARGETS = nvshmem_8gpu_examples nvshmem_advanced_8gpu

# Check if NVSHMEM is available
HAS_NVSHMEM := $(shell test -d $(NVSHMEM_HOME) && echo yes || echo no)

.PHONY: all clean nvshmem help

all: help
	@echo ""
	@echo "Note: NVSHMEM examples require NVSHMEM 3.4+ to be installed"
	@echo "Run 'make nvshmem' if NVSHMEM is available"

help:
	@echo "Available targets:"
	@echo "  make nvshmem    - Compile NVSHMEM examples (requires NVSHMEM)"
	@echo "  make clean      - Clean build artifacts"
	@echo ""
	@echo "NVSHMEM status: $(HAS_NVSHMEM)"
ifeq ($(HAS_NVSHMEM),no)
	@echo "  → NVSHMEM not found at $(NVSHMEM_HOME)"
	@echo "  → Set NVSHMEM_HOME to correct path if installed"
endif

nvshmem: $(NVSHMEM_TARGETS)
ifeq ($(HAS_NVSHMEM),yes)
	@echo "✓ NVSHMEM examples compiled successfully"
else
	@echo "⚠ NVSHMEM not found - compiling in educational mode"
endif

# NVSHMEM examples
nvshmem_8gpu_examples: nvshmem_8gpu_examples.cu
ifeq ($(HAS_NVSHMEM),yes)
	$(NVCC) $(CUDA_FLAGS) -DUSE_NVSHMEM $(NVSHMEM_INCLUDE) $< $(NVSHMEM_LIB) -o $@
	@echo "✓ Basic examples compiled with NVSHMEM support"
	@echo "  Run: nvshmemrun -np 8 ./nvshmem_8gpu_examples"
else
	$(NVCC) $(CUDA_FLAGS) $< -o $@
	@echo "Compiled in educational mode (no NVSHMEM library)"
	@echo "This version shows API patterns but won't execute communication"
endif

# Advanced NVSHMEM patterns (production-quality)
nvshmem_advanced_8gpu: nvshmem_advanced_8gpu.cu
ifeq ($(HAS_NVSHMEM),yes)
	$(NVCC) $(CUDA_FLAGS) -DUSE_NVSHMEM $(NVSHMEM_INCLUDE) $< $(NVSHMEM_LIB) -o $@
	@echo "✓ Advanced patterns compiled with NVSHMEM support"
	@echo "  Run: nvshmemrun -np 8 ./nvshmem_advanced_8gpu"
	@echo "  Includes: Ring AllReduce, Double-Buffered, Recursive H/D"
else
	$(NVCC) $(CUDA_FLAGS) $< -o $@
	@echo "Compiled in educational mode (no NVSHMEM library)"
endif

clean:
	rm -f $(NVSHMEM_TARGETS) *.o
	@echo "Cleaned build artifacts"

# Help for first-time users
install-help:
	@echo "To install NVSHMEM:"
	@echo "  1. Download from NVIDIA: https://developer.nvidia.com/nvshmem"
	@echo "  2. Extract and install to /usr/local/nvshmem"
	@echo "  3. Set environment variable: export NVSHMEM_HOME=/usr/local/nvshmem"
	@echo "  4. Run: make nvshmem"

