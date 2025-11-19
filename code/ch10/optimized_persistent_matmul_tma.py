"""optimized_persistent_matmul_tma.py

Persistent matmul using Triton TMA multicast + DSMEM on thread-block clusters.
Assumes SM100/Blackwell-class GPU with cluster support. Falls back to standard
execution if clusters are not available by letting the kernel launch; runtime
will raise if unsupported.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise ImportError("Triton is required for this example") from exc


@triton.jit
def persistent_matmul_tma(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    dsmem_a = tl.dsmem((BLOCK_M, BLOCK_K), dtype=tl.float16)
    dsmem_b = tl.dsmem((BLOCK_K, BLOCK_N), dtype=tl.float16)

    tma_a = tl.make_tensor_descriptor(stride_am, stride_ak, BLOCK_M, BLOCK_K)
    tma_b = tl.make_tensor_descriptor(stride_bk, stride_bn, BLOCK_K, BLOCK_N)

    for k0 in range(0, K, BLOCK_K):
        a_ptr = A + offs_m[:, None] * stride_am + (k0 + tl.arange(0, BLOCK_K))[None, :] * stride_ak
        b_ptr = B + (k0 + tl.arange(0, BLOCK_K))[:, None] * stride_bk + offs_n[None, :] * stride_bn

        tl.tma_async_copy(dsmem_a, a_ptr, tma_a, multicast=True)
        tl.tma_async_copy(dsmem_b, b_ptr, tma_b, multicast=True)

        tl.cluster_barrier()

        a_tile = tl.load(dsmem_a)
        b_tile = tl.load(dsmem_b)
        acc += tl.dot(a_tile, b_tile)

        tl.cluster_barrier()

    c_ptr = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptr, acc)


def run_optimized(M=1024, N=1024, K=1024, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    persistent_matmul_tma[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=8,
        num_stages=2,
        enable_warp_specialization=True,
        cluster_dims=(2, 1, 1),
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    out = run_optimized()
    print(f"Optimized TMA/DSMEM matmul completed, output mean={out.mean().item():.3f}")
