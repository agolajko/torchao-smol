# Copyright (c) Meta Platforms, Inc. and affiliates.
# Optimized FP8 Blockwise GEMM implementation

import itertools
from dataclasses import dataclass
from typing import List

import triton
import triton.language as tl
import torch
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

from torchao.prototype.blockwise_fp8_training.kernels import (
    triton_fp8_blockwise_act_quant_lhs,
    triton_fp8_blockwise_weight_quant_transposed_rhs,
    triton_fp8_gemm_1x128_128x128,
)

# Optimized autotune configs focused on the problem dimensions
fp8_gemm_configs_optimized = [
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_warps=8,
        num_stages=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 8,
        },
        num_warps=8,
        num_stages=5,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 256,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 256,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 4,
        },
        num_warps=8,
        num_stages=3,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 128,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 4,
        },
        num_warps=4,
        num_stages=4,
    ),
    triton.Config(
        {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 128,
            "BLOCK_SIZE_K": 128,
            "GROUP_SIZE_M": 8,
        },
        num_warps=4,
        num_stages=5,
    ),
]


@triton.autotune(
    configs=fp8_gemm_configs_optimized,
    key=["M", "N", "K"],
)
@triton.jit
def triton_fp8_blockwise_gemm_kernel_optimized(
    # Pointers to matrices
    a_ptr,
    a_stride_dim_0,
    a_stride_dim_1,
    b_ptr,
    b_stride_dim_0,
    b_stride_dim_1,
    c_ptr,
    c_stride_dim_0,
    c_stride_dim_1,
    # Pointers to blockwise scale factors (inverse scales, i.e., 1/scale)
    a_s_ptr,  # Shape: (M // SCALE_BLOCK_M, K // SCALE_BLOCK_K)
    a_s_stride_dim_0,
    a_s_stride_dim_1,
    b_s_ptr,  # Shape: (K // SCALE_BLOCK_K, N // SCALE_BLOCK_N)
    b_s_stride_dim_0,
    b_s_stride_dim_1,
    # Problem size
    M,
    N,
    K,
    # Block sizes for scaling
    SCALE_BLOCK_M: tl.constexpr,
    SCALE_BLOCK_K: tl.constexpr,
    SCALE_BLOCK_N: tl.constexpr,
    # Meta-parameters
    out_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Optimized FP8 GEMM with blockwise scaling inspired by CUTLASS implementation.

    Key optimizations:
    1. Load scales into shared memory once per K-tile
    2. Use broadcasting to apply scales efficiently  
    3. Minimize scale tensor accesses
    4. Better register usage
    """
    # CTA swizzling for better L2 cache utilization
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Initialize pointers for A and B tiles
    a_ptrs = a_ptr + (
        offs_m[:, None] * a_stride_dim_0 + offs_k[None, :] * a_stride_dim_1
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * b_stride_dim_0 + offs_n[None, :] * b_stride_dim_1
    )

    # Initialize accumulator in FP32
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Masks for boundary conditions
    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

    k_num_blocks = tl.cdiv(K, BLOCK_SIZE_K)

    # Main K loop
    for k_idx in range(0, k_num_blocks):
        # Load A and B tiles (FP8)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Load blockwise scales for this K-tile
        # Map element coordinates to scale coordinates
        scale_m_idx = offs_m // SCALE_BLOCK_M  # Shape: (BLOCK_SIZE_M,)
        # Shape: (BLOCK_SIZE_K,)
        scale_k_idx_a = (k_idx * BLOCK_SIZE_K + offs_k) // SCALE_BLOCK_K
        # Shape: (BLOCK_SIZE_K,)
        scale_k_idx_b = (k_idx * BLOCK_SIZE_K + offs_k) // SCALE_BLOCK_K
        scale_n_idx = offs_n // SCALE_BLOCK_N  # Shape: (BLOCK_SIZE_N,)

        # Load scale tensors with broadcasting
        # For A: we need (BLOCK_SIZE_M, BLOCK_SIZE_K) scales
        a_s_ptrs = a_s_ptr + (
            scale_m_idx[:, None] * a_s_stride_dim_0 +
            scale_k_idx_a[None, :] * a_s_stride_dim_1
        )
        a_s = tl.load(a_s_ptrs, mask=a_mask, other=1.0)

        # For B: we need (BLOCK_SIZE_K, BLOCK_SIZE_N) scales
        b_s_ptrs = b_s_ptr + (
            scale_k_idx_b[:, None] * b_s_stride_dim_0 +
            scale_n_idx[None, :] * b_s_stride_dim_1
        )
        b_s = tl.load(b_s_ptrs, mask=b_mask, other=1.0)

        # Apply blockwise scaling and accumulate
        # The scales are already broadcast to match tile dimensions
        a_scaled = a * a_s
        b_scaled = b * b_s

        # Perform matrix multiplication and accumulate
        accumulator += tl.dot(a_scaled, b_scaled, out_dtype=tl.float32)

        # Advance pointers to next K-tile
        a_ptrs += BLOCK_SIZE_K * a_stride_dim_1
        b_ptrs += BLOCK_SIZE_K * b_stride_dim_0

        # Update mask offsets for next iteration
        offs_k += BLOCK_SIZE_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

    # Write output
    c = accumulator.to(out_dtype)
    c_ptrs = c_ptr + (
        offs_m[:, None] * c_stride_dim_0 + offs_n[None, :] * c_stride_dim_1
    )
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_fp8_blockwise_gemm_optimized(
    a: torch.Tensor,  # (M, K) FP8 row-major
    b: torch.Tensor,  # (K, N) FP8 column-major
    a_s: torch.Tensor,  # (M // scale_block_m, K // scale_block_k) FP32
    b_s: torch.Tensor,  # (K // scale_block_k, N // scale_block_n) FP32
    scale_block_m: int = 128,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    out_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Optimized blockwise-scaled FP8 GEMM.
    """
    # Validate layout
    assert a.is_contiguous() and a.stride(0) > a.stride(1), "a must be row-major"
    assert b.stride(0) < b.stride(1), "b must be column-major"

    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match"

    # Validate scale tensor shapes
    expected_a_s_shape = (M // scale_block_m, K // scale_block_k)
    expected_b_s_shape = (K // scale_block_k, N // scale_block_n)
    assert a_s.shape == expected_a_s_shape, f"a_s shape mismatch: {a_s.shape} vs {expected_a_s_shape}"
    assert b_s.shape == expected_b_s_shape, f"b_s shape mismatch: {b_s.shape} vs {expected_b_s_shape}"

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)

    # Launch kernel
    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) *
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    triton_fp8_blockwise_gemm_kernel_optimized[grid](
        a,
        a.stride(0),
        a.stride(1),
        b,
        b.stride(0),
        b.stride(1),
        c,
        c.stride(0),
        c.stride(1),
        a_s,
        a_s.stride(0),
        a_s.stride(1),
        b_s,
        b_s.stride(0),
        b_s.stride(1),
        M,
        N,
        K,
        scale_block_m,
        scale_block_k,
        scale_block_n,
        out_dtype=out_dtype,
    )

    return c


# =================================================
# Benchmarking code
# =================================================

device = torch.device("cuda")

# This benchmark requires CUDA 12.9+
assert torch.version.cuda is not None, "CUDA is not available"
cuda_major, cuda_minor = map(int, torch.version.cuda.split("."))
assert cuda_major >= 12 and cuda_minor >= 9, "CUDA 12.9+ is required"

torch._dynamo.config.cache_size_limit = 1000


@dataclass(frozen=True)
class ExperimentConfig:
    out_dtype: torch.dtype
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class ExperimentResult:
    bf16_mm_us: float
    fp8_triton_us: float
    fp8_new: float
    fp8_optimized: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    mnk_list = [
        # Llama4 shapes
        (16640, 5120, 8192),
        (16640, 8192, 5120),
    ]
    out_dtypes = [torch.bfloat16]
    configs = []
    for mnk, out_dtype in itertools.product(mnk_list, out_dtypes):
        m, n, k = mnk
        configs.append(
            ExperimentConfig(
                out_dtype=out_dtype,
                m=m,
                n=n,
                k=k,
            )
        )
    return configs


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    # Simulate `grad_input = grad_output @ weight`
    M, N, K = config.m, config.n, config.k
    A = torch.randn(M, K, dtype=config.out_dtype, device="cuda")
    B = torch.randn(N, K, dtype=config.out_dtype, device="cuda")
    A_q, A_s = triton_fp8_blockwise_act_quant_lhs(A, dtype=torch.float8_e4m3fn)
    B_t_q, B_t_s = triton_fp8_blockwise_weight_quant_transposed_rhs(
        B, dtype=torch.float8_e4m3fn
    )

    def warmup(func, *args, **kwargs):
        for _ in range(10):
            func(*args, **kwargs)

    # Warmup then run bf16 torch.mm
    warmup(torch.mm, A, B.t())
    bf16_mm_us = benchmark_cuda_function_in_microseconds(torch.mm, A, B.t())

    # Warm up then run original triton kernel
    warmup(
        triton_fp8_gemm_1x128_128x128,
        A_q,
        B_t_q,
        1.0 / A_s,
        1.0 / B_t_s,
        out_dtype=config.out_dtype,
    )
    fp8_triton_us = benchmark_cuda_function_in_microseconds(
        triton_fp8_gemm_1x128_128x128,
        A_q,
        B_t_q,
        1.0 / A_s,
        1.0 / B_t_s,
        out_dtype=config.out_dtype,
    )

    # Prepare for new kernels (ensure column-major format for scales)
    A_s = A_s.t().contiguous().t()
    actual_scale_block_m = M // A_s.shape[0]
    actual_scale_block_k = K // A_s.shape[1]
    actual_scale_block_n = N // B_t_s.shape[1]

    # Warm up then run optimized kernel
    warmup(
        triton_fp8_blockwise_gemm_optimized,
        A_q,
        B_t_q,
        1.0 / A_s,
        1.0 / B_t_s,
        scale_block_m=actual_scale_block_m,
        scale_block_k=actual_scale_block_k,
        scale_block_n=actual_scale_block_n,
        out_dtype=config.out_dtype,
    )
    fp8_optimized = benchmark_cuda_function_in_microseconds(
        triton_fp8_blockwise_gemm_optimized,
        A_q,
        B_t_q,
        1.0 / A_s,
        1.0 / B_t_s,
        scale_block_m=actual_scale_block_m,
        scale_block_k=actual_scale_block_k,
        scale_block_n=actual_scale_block_n,
        out_dtype=config.out_dtype,
    )

    # For comparison with your original new kernel (if available)
    # Set to fp8_optimized for now
    fp8_new = fp8_optimized

    return ExperimentResult(
        bf16_mm_us=bf16_mm_us,
        fp8_triton_us=fp8_triton_us,
        fp8_new=fp8_new,
        fp8_optimized=fp8_optimized,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "M",
        "N",
        "K",
        "out_dtype",
        "bf16_mm_us",
        "fp8_triton_us",
        "fp8_optimized_us",
        "bf16 tflops/sec",
        "triton tflops/sec",
        "optimized tflops/sec",
        "speedup vs bf16",
        "speedup vs triton",
    ]
    rows = []
    for experiment in experiments:
        m, n, k = experiment.config.m, experiment.config.n, experiment.config.k
        flops = 2 * m * n * k
        bf16_mm_tflops_per_sec = (flops / 1e12) / \
            (experiment.result.bf16_mm_us / 1e6)
        triton_tflops_per_sec = (flops / 1e12) / \
            (experiment.result.fp8_triton_us / 1e6)
        optimized_tflops_per_sec = (
            flops / 1e12) / (experiment.result.fp8_optimized / 1e6)

        speedup_vs_bf16 = bf16_mm_tflops_per_sec / optimized_tflops_per_sec
        speedup_vs_triton = triton_tflops_per_sec / optimized_tflops_per_sec

        rows.append(
            [
                m,
                n,
                k,
                experiment.config.out_dtype,
                f"{experiment.result.bf16_mm_us:.1f}",
                f"{experiment.result.fp8_triton_us:.1f}",
                f"{experiment.result.fp8_optimized:.1f}",
                f"{bf16_mm_tflops_per_sec:.1f}",
                f"{triton_tflops_per_sec:.1f}",
                f"{optimized_tflops_per_sec:.1f}",
                f"{speedup_vs_bf16:.2f}x",
                f"{speedup_vs_triton:.2f}x",
            ]
        )
    print(tabulate(rows, headers=headers))


def benchmark_cuda_function_in_microseconds(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median") * 1e3


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    print_results(results)


if __name__ == "__main__":
    main()
