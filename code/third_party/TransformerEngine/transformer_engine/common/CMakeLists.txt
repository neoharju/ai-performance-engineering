# Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# See LICENSE for license information.

cmake_minimum_required(VERSION 3.21)

# Language options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")
endif()

# Hide non-necessary symbols in shared object.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libtransformer_engine.version")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/libtransformer_engine.version")

# Transformer Engine library
project(transformer_engine LANGUAGES CUDA CXX)

# CUDA Toolkit
find_package(CUDAToolkit REQUIRED)
if (CUDAToolkit_VERSION VERSION_LESS 12.1)
  message(FATAL_ERROR "CUDA 12.1+ is required, but found CUDA ${CUDAToolkit_VERSION}")
endif()

# Process GPU architectures: force Blackwell arch-specific FP4 target only.
set(CMAKE_CUDA_ARCHITECTURES 121f CACHE STRING "CUDA arch list" FORCE)
set(NVTE_GENERIC_ARCHS "121f")
set(NVTE_SPECIFIC_ARCHS)

# cuDNN frontend API
set(CUDNN_FRONTEND_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cudnn-frontend/include")
if(NOT EXISTS "${CUDNN_FRONTEND_INCLUDE_DIR}")
    message(FATAL_ERROR
            "Could not find cuDNN frontend API at ${CUDNN_FRONTEND_INCLUDE_DIR}. "
            "Try running 'git submodule update --init --recursive' "
            "within the Transformer Engine source.")
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cudnn-frontend/cmake/cuDNN.cmake)

set(CUTLASS_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cutlass/include")
set(CUTLASS_TOOLS_INCLUDE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/cutlass/tools/util/include")

# Python
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Configure Transformer Engine library
include_directories(${PROJECT_SOURCE_DIR}/..)
set(transformer_engine_SOURCES)
set(transformer_engine_cpp_sources)
set(transformer_engine_cuda_sources)
set(transformer_engine_cuda_arch_specific_sources)

list(APPEND transformer_engine_cpp_sources
     cudnn_utils.cpp
     transformer_engine.cpp
     fused_attn/fused_attn.cpp
     gemm/config.cpp
     normalization/common.cpp
     normalization/layernorm/ln_api.cpp
     normalization/rmsnorm/rmsnorm_api.cpp
     util/cuda_driver.cpp
     util/cuda_nvml.cpp
     util/cuda_runtime.cpp
     util/multi_stream.cpp
     util/rtc.cpp
     comm_gemm_overlap/userbuffers/ipcsocket.cc
     comm_gemm_overlap/userbuffers/userbuffers-host.cpp
     comm_gemm_overlap/comm_gemm_overlap.cpp)

list(APPEND transformer_engine_cuda_sources
     common.cu
     multi_tensor/adam.cu
     multi_tensor/compute_scale.cu
     multi_tensor/l2norm.cu
     multi_tensor/scale.cu
     multi_tensor/sgd.cu
     transpose/cast_transpose.cu
     transpose/transpose.cu
     transpose/cast_transpose_fusion.cu
     transpose/transpose_fusion.cu
     transpose/multi_cast_transpose.cu
     transpose/quantize_transpose_vector_blockwise.cu
     transpose/swap_first_dims.cu
     dropout/dropout.cu
     fused_attn/flash_attn.cu
     fused_attn/context_parallel.cu
     fused_attn/kv_cache.cu
     fused_attn/fused_attn_f16_max512_seqlen.cu
     fused_attn/fused_attn_f16_arbitrary_seqlen.cu
     fused_attn/fused_attn_fp8.cu
     fused_attn/utils.cu
     gemm/cublaslt_gemm.cu
     normalization/layernorm/ln_bwd_semi_cuda_kernel.cu
     normalization/layernorm/ln_fwd_cuda_kernel.cu
     normalization/rmsnorm/rmsnorm_bwd_semi_cuda_kernel.cu
     normalization/rmsnorm/rmsnorm_fwd_cuda_kernel.cu
     permutation/permutation.cu
     util/padding.cu
     swizzle/swizzle.cu
     swizzle/swizzle_block_scaling.cu
     fused_softmax/scaled_masked_softmax.cu
     fused_softmax/scaled_upper_triang_masked_softmax.cu
     fused_softmax/scaled_aligned_causal_masked_softmax.cu
     fused_rope/fused_rope.cu
     fused_router/fused_moe_aux_loss.cu
     fused_router/fused_score_for_moe_aux_loss.cu
     fused_router/fused_topk_with_score_function.cu
     recipe/current_scaling.cu
     recipe/delayed_scaling.cu
     recipe/fp8_block_scaling.cu
     recipe/nvfp4.cu
     comm_gemm_overlap/userbuffers/userbuffers.cu)

list(APPEND transformer_engine_cuda_arch_specific_sources
     gemm/cutlass_grouped_gemm.cu
     cast/cast.cu
     activation/gelu.cu
     activation/relu.cu
     activation/swiglu.cu
     transpose/quantize_transpose_square_blockwise.cu
     transpose/quantize_transpose_vector_blockwise_fp4.cu
     hadamard_transform/hadamard_transform.cu
     hadamard_transform/hadamard_transform_cast_fusion.cu)

# Build FP4/NVFP4 sources in a dedicated object library targeting only sm_121f.
set(nvfp4_sources
    activation/gelu.cu
    activation/relu.cu
    activation/swiglu.cu
    transpose/quantize_transpose_vector_blockwise_fp4.cu
    recipe/nvfp4.cu)
add_library(transformer_engine_fp4 OBJECT ${nvfp4_sources})
set_target_properties(transformer_engine_fp4 PROPERTIES CUDA_ARCHITECTURES "121f")
target_include_directories(transformer_engine_fp4 PRIVATE
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${CUTLASS_INCLUDE_DIR}
  ${CUTLASS_TOOLS_INCLUDE_DIR}
  "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Remove FP4 sources from the main source lists to avoid sm_100 codegen
foreach(fp4src IN LISTS nvfp4_sources)
  list(REMOVE_ITEM transformer_engine_cuda_arch_specific_sources ${fp4src})
  list(REMOVE_ITEM transformer_engine_cuda_sources ${fp4src})
endforeach()

# Compiling the files with the worst compilation time first to hopefully overlap
# better with the faster-compiling cpp files
list(APPEND transformer_engine_SOURCES ${transformer_engine_cuda_arch_specific_sources}
                                       ${transformer_engine_cuda_sources}
                                       ${transformer_engine_cpp_sources})

# Set compile options for CUDA sources with generic architectures
foreach(cuda_source IN LISTS transformer_engine_cuda_sources)
  set(arch_compile_options)
  foreach(arch IN LISTS NVTE_GENERIC_ARCHS)
    list(APPEND arch_compile_options "--generate-code=arch=compute_${arch},code=sm_${arch}")
  endforeach()

  if(arch_compile_options)
    set_property(
      SOURCE ${cuda_source}
      APPEND
      PROPERTY
      COMPILE_OPTIONS ${arch_compile_options}
    )
  endif()
endforeach()

# Set compile options for CUDA sources with specific architectures
foreach(cuda_source IN LISTS transformer_engine_cuda_arch_specific_sources)
  set(arch_compile_options)
  foreach(arch IN LISTS NVTE_SPECIFIC_ARCHS)
    list(APPEND arch_compile_options "--generate-code=arch=compute_${arch},code=sm_${arch}")
  endforeach()

  if(arch_compile_options)
    set_property(
      SOURCE ${cuda_source}
      APPEND
      PROPERTY
      COMPILE_OPTIONS ${arch_compile_options}
    )
  endif()
endforeach()

if (NVTE_WITH_CUBLASMP)
list(APPEND transformer_engine_SOURCES
     comm_gemm/comm_gemm.cpp)
endif()

add_library(transformer_engine SHARED ${transformer_engine_SOURCES})
target_link_libraries(transformer_engine PRIVATE transformer_engine_fp4)
target_include_directories(transformer_engine PUBLIC
                           "${CMAKE_CURRENT_SOURCE_DIR}/include")

# CUTLASS kernels require SM90a and cause hang in debug build
set_property(
  SOURCE gemm/cutlass_grouped_gemm.cu
  APPEND
  PROPERTY
  COMPILE_OPTIONS "--generate-code=arch=compute_90a,code=sm_90a;-g0")

# Configure dependencies
target_link_libraries(transformer_engine PUBLIC
                      CUDA::cublas
                      CUDA::cudart
                      CUDNN::cudnn_all)

target_include_directories(transformer_engine PRIVATE
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_include_directories(transformer_engine SYSTEM PRIVATE
                           ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/cccl)
target_include_directories(transformer_engine PRIVATE "${CUDNN_FRONTEND_INCLUDE_DIR}")
target_include_directories(transformer_engine PRIVATE
                          ${CUTLASS_INCLUDE_DIR}
                          ${CUTLASS_TOOLS_INCLUDE_DIR})

# Compiling Userbuffers with native MPI bootstrapping requires linking against MPI
option(NVTE_UB_WITH_MPI "Bootstrap Userbuffers with MPI" OFF)
if (NVTE_UB_WITH_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(transformer_engine PUBLIC MPI::MPI_CXX)
    target_include_directories(transformer_engine PRIVATE ${MPI_CXX_INCLUDES})
    target_compile_definitions(transformer_engine PUBLIC NVTE_UB_WITH_MPI)
endif()

option(NVTE_ENABLE_NVSHMEM "Compile with NVSHMEM library" OFF)
if (NVTE_ENABLE_NVSHMEM)
    add_subdirectory(nvshmem_api)
    target_link_libraries(transformer_engine PUBLIC nvshmemapi)
    target_include_directories(transformer_engine PUBLIC ${NVSHMEMAPI_INCLUDE_DIR})
endif()

option(NVTE_WITH_CUBLASMP "Use cuBLASMp for tensor parallel GEMMs" OFF)
if (NVTE_WITH_CUBLASMP)
    target_compile_definitions(transformer_engine PRIVATE NVTE_WITH_CUBLASMP)
    target_include_directories(transformer_engine PRIVATE ${CUBLASMP_DIR}/include ${NVSHMEM_DIR}/include)
    find_library(CUBLASMP_LIB
                 NAMES cublasmp libcublasmp
                 PATHS ${CUBLASMP_DIR}
                 PATH_SUFFIXES lib
                 REQUIRED)
    find_library(NVSHMEM_HOST_LIB
                 NAMES nvshmem_host libnvshmem_host.so.3
                 PATHS ${NVSHMEM_DIR}
                 PATH_SUFFIXES lib
                 REQUIRED)
  target_link_libraries(transformer_engine PUBLIC ${CUBLASMP_LIB} ${NVSHMEM_HOST_LIB})
  message(STATUS "Using cuBLASMp at: ${CUBLASMP_DIR}")
  message(STATUS "Using nvshmem at: ${NVSHMEM_DIR}")
endif()

# Hack to enable dynamic loading in cuDNN frontend
target_compile_definitions(transformer_engine PUBLIC NV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING)

# Helper functions to make header files with C++ strings
function(make_string_header STRING STRING_NAME)
    configure_file(util/string_header.h.in
                   "string_headers/${STRING_NAME}.h"
                   @ONLY)
endfunction()
function(make_string_header_from_file file_ STRING_NAME)
    file(READ "${file_}" STRING)
    configure_file(util/string_header.h.in
                   "string_headers/${STRING_NAME}.h"
                   @ONLY)
endfunction()

# Header files with C++ strings
list(GET CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES 0 cuda_include_path)
make_string_header("${cuda_include_path}"
                   string_path_cuda_include)
make_string_header_from_file(transpose/rtc/cast_transpose_fusion.cu
                             string_code_transpose_rtc_cast_transpose_fusion_cu)
make_string_header_from_file(transpose/rtc/cast_transpose.cu
                             string_code_transpose_rtc_cast_transpose_cu)
make_string_header_from_file(transpose/rtc/transpose.cu
                             string_code_transpose_rtc_transpose_cu)
make_string_header_from_file(transpose/rtc/swap_first_dims.cu
                             string_code_transpose_rtc_swap_first_dims_cu)
make_string_header_from_file(utils.cuh
                             string_code_utils_cuh)
make_string_header_from_file(util/math.h
                             string_code_util_math_h)
target_include_directories(transformer_engine PRIVATE
                           "${CMAKE_CURRENT_BINARY_DIR}/string_headers")

# Compiler options
set(nvte_sources_with_fast_math)
list(APPEND nvte_sources_with_fast_math fused_softmax/scaled_masked_softmax.cu
                                        fused_softmax/scaled_upper_triang_masked_softmax.cu
                                        fused_softmax/scaled_aligned_causal_masked_softmax.cu
                                        multi_tensor/adam.cu
                                        multi_tensor/compute_scale.cu
                                        multi_tensor/l2norm.cu
                                        multi_tensor/scale.cu
                                        multi_tensor/sgd.cu
                                        fused_attn/flash_attn.cu
                                        fused_attn/context_parallel.cu
                                        fused_attn/kv_cache.cu)

option(NVTE_BUILD_ACTIVATION_WITH_FAST_MATH "Compile activation kernels with --use_fast_math option" OFF)
if (NVTE_BUILD_ACTIVATION_WITH_FAST_MATH)
  list(APPEND nvte_sources_with_fast_math activation/gelu.cu
                                          activation/relu.cu
                                          activation/swiglu.cu)
endif()

foreach(cuda_source IN LISTS nvte_sources_with_fast_math)
  set_property(
    SOURCE ${cuda_source}
    APPEND
    PROPERTY
    COMPILE_OPTIONS "--use_fast_math")
endforeach()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Number of parallel build jobs
if($ENV{MAX_JOBS})
  set(BUILD_JOBS_STR $ENV{MAX_JOBS})
elseif($ENV{NVTE_BUILD_MAX_JOBS})
  set(BUILD_JOBS_STR $ENV{NVTE_BUILD_MAX_JOBS})
else()
  set(BUILD_JOBS_STR "max")
endif()
message(STATUS "Parallel build jobs: ${BUILD_JOBS_STR}")

# Number of threads per parallel build job
set(BUILD_THREADS_PER_JOB $ENV{NVTE_BUILD_THREADS_PER_JOB})
if (NOT BUILD_THREADS_PER_JOB)
  set(BUILD_THREADS_PER_JOB 1)
endif()
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads ${BUILD_THREADS_PER_JOB}")
message(STATUS "Threads per parallel build job: ${BUILD_THREADS_PER_JOB}")

# Install library
install(TARGETS transformer_engine DESTINATION .)
