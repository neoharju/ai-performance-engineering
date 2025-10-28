# NVIDIA Support Ticket – Blackwell TMA Descriptor Failures

## Summary

- **Issue 1 – cuTensorMapEncodeTiled (1D)**: Returns `CUDA_ERROR_INVALID_VALUE` on Blackwell B200 when encoding a 1‑D FP32 tensor map that matches the CUDA 13.0 programming guide requirements (aligned base pointer, 16‑byte stride, 1024‑element box).
- **Issue 2 – cuTensorMapEncodeTiled (2D)**: Encoding succeeds for a 4096×4096 FP32 tensor, but the first `cp_async_bulk_tensor_2d_*` transfer triggers `cudaErrorIllegalAddress`.

These regressions prevent the CUDA 13.0 TMA demos from running and block Nsight Compute profiling pipelines.

## Environment

- GPU: 8× NVIDIA B200 (SM 10.0) – confirmed via `nvidia-smi -L`
- CUDA Toolkit / Driver: 13.0
- Nsight Systems / Nsight Compute: 2024.1
- Operating System: Ubuntu 22.04 (Lambda reference image)

## Reproduction Steps

1. Build the demos:
   ```bash
   ./scripts/build_tma_demos.sh --arch sm_100
   ```
2. Run the 1‑D prefetch sample (fails immediately):
   ```bash
   ./ch7/async_prefetch_tma
   ```
   Output:
   ```
   [TMA] cuTensorMapEncodeTiled (1D) failed: invalid argument
          dataType=7 (FLOAT32)
          rank=1
          base=0x70d2df200000 / 0x70d2df240000
          elements=65536
          stride_bytes=4
          box=1024
          elem_stride=1
          swizzle=0
          l2=2 (CU_TENSOR_MAP_L2_PROMOTION_L2_128B)
   ```
3. Run the 2‑D pipeline (descriptor encodes, kernel aborts):
   ```bash
   ./ch10/tma_2d_pipeline_blackwell
   ```
   Log:
   ```
   CUDA error (warm-up sync): an illegal memory access was encountered
   ```

Both tests use device pointers returned by `cudaMalloc`, tensor ranks/strides/boxes identical to the CUDA 13.0 guide (§10.29), and the helper prints confirm the exact parameter values passed to `cuTensorMapEncodeTiled`.

## Attachments / References

- `ch7/async_prefetch_tma.cu`
- `ch10/tma_2d_pipeline_blackwell.cu`
- Logging implemented in `cuda13_feature_examples.cuh:226-257` to dump descriptor parameters.

## Request

Please investigate the driver/runtime handling of `cuTensorMapEncodeTiled` on Blackwell B200 for the configurations above. If additional traces or a reduced repro are required, let us know.

Thanks! A fallback path is in place for now (`ENABLE_BLACKWELL_TMA` env var), but we’d like to restore the descriptor-backed pipeline once a fixed driver is available.
