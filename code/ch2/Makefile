# Chapter 2 Makefile (includes Blackwell-only NVLink C2C demo)
# Target architecture: Blackwell B200/B300 (SM100)
ARCH ?= sm_100

# CUDA configuration
CUDA_VERSION = 13.0
NVCC = nvcc
NVCC_FLAGS = -O3 -std=c++17 -arch=$(ARCH) --expt-relaxed-constexpr -DCUDA_VERSION=$(CUDA_VERSION)

TARGETS =

ifeq ($(ARCH),sm_100)
TARGETS += nvlink_c2c_p2p_blackwell gb200_coherency
endif

all: $(TARGETS)
	@echo "Building Chapter 2 targets with $(ARCH) (CUDA $(CUDA_VERSION))"
	@echo "✓ Targeting Blackwell B200/B300 (Compute Capability 10.0)"
	@echo "✓ GB200/GB300 Grace-Blackwell examples included"

nvlink_c2c_p2p_blackwell: nvlink_c2c_p2p_blackwell.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

gb200_coherency: gb200_grace_blackwell_coherency.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $<

blackwell: ARCH=sm_100
blackwell: all
	@echo "Built Chapter 2 utilities for Blackwell B200/B300"

profile-hta: all
	@echo "HTA (Holistic Tracing Analysis) profiling..."
	@for exe in $(TARGETS); do 		if [ -x "$$exe" ]; then 			echo "Profiling $$exe..."; 			nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl 			  -o hta_profile_$$exe_$(ARCH) ./$$exe 2>/dev/null || 			  echo "HTA profiling failed for $$exe"; 		fi; 	done

profile-perf: all
	@echo "Perf system-level profiling..."
	@for exe in $(TARGETS); do 		if [ -x "$$exe" ]; then 			echo "Profiling $$exe with perf..."; 			perf record -g -o perf.data_$(ARCH)_$$exe ./$$exe 2>/dev/null || 			  echo "perf failed for $$exe"; 			perf report -i perf.data_$(ARCH)_$$exe 2>/dev/null || true; 		fi; 	done

profile-all: all
	@echo "Comprehensive profiling with all tools..."
	@for exe in $(TARGETS); do 		if [ -x "$$exe" ]; then 			echo "1. Nsight Systems timeline (HTA)..."; 			nsys profile --force-overwrite=true -t cuda,nvtx,osrt,cudnn,cublas,nccl 			  -o comprehensive_hta_$$exe_$(ARCH) ./$$exe 2>/dev/null || true; 			echo "2. Perf system-level analysis..."; 			perf record -g -o perf.data_$(ARCH)_$$exe ./$$exe 2>/dev/null || true; 			perf report -i perf.data_$(ARCH)_$$exe 2>/dev/null || true; 		fi; 	done

clean:
	rm -f $(TARGETS) *.o *.so *.a
	rm -f nsys_profile_* hta_profile_* perf.data_* comprehensive_*

.PHONY: all blackwell profile-hta profile-perf profile-all clean $(TARGETS)
