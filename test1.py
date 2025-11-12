# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

import itertools
from dataclasses import dataclass
from typing import List

import torch
import triton
import triton.language as tl
from tabulate import tabulate
from tqdm import tqdm
from triton.testing import do_bench

from torchao.prototype.blockwise_fp8_training.kernels import (
    triton_fp8_blockwise_act_quant_lhs,
    triton_fp8_blockwise_weight_quant_transposed_rhs,
)


device = torch.device("cuda")

# This benchmark requires CUDA 12.9+
assert torch.version.cuda is not None, "CUDA is not available"
cuda_major, cuda_minor = map(int, torch.version.cuda.split("."))
assert cuda_major >= 12 and cuda_minor >= 9, "CUDA 12.9+ is required"

# Needed since changing args to function causes recompiles
torch._dynamo.config.cache_size_limit = 1000


# ============================================================================
# Helper functions
# ============================================================================

def _is_row_major(tensor: torch.Tensor) -> bool:
    return tensor.stride(0) > tensor.stride(1)


def _is_column_major(tensor: torch.Tensor) -> bool:
    return tensor.stride(0) < tensor.stride(1)


# ============================================================================
# Autotuning configs
# ============================================================================

fp8_gemm_configs_max_autotune = [
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128},
        num_stages=4,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
        num_stages=4,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64},
        num_stages=5,
        num_warps=4,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 128},
        num_stages=4,
        num_warps=8,
    ),
    triton.Config(
        {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 128},
        num_stages=3,
        num_warps=8,
    ),
]


# ============================================================================
# OLD KERNEL: triton_fp8_gemm_1x128_128x128
# ============================================================================

@triton.autotune(configs=fp8_gemm_configs_max_autotune, key=["N", "K", "BLOCK_SIZE_K"])
@triton.jit
def triton_fp8_gemm_1x128_128x128_kernel(
    a_ptr,  # (M, K)
    a_stride_dim_0,
    a_stride_dim_1,
    b_ptr,  # (K, N)
    b_stride_dim_0,
    b_stride_dim_1,
    c_ptr,
    c_stride_dim_0,
    c_stride_dim_1,
    a_s_ptr,  # (M, K // block_size) reciprocals of scales
    a_s_stride_dim_0,
    a_s_stride_dim_1,
    b_s_ptr,  # (K // block_size, N // block_size) reciprocals of scales
    b_s_stride_dim_0,
    b_s_stride_dim_1,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    out_dtype: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (
        offs_m[:, None] * a_stride_dim_0 + offs_k[None, :] * a_stride_dim_1
    )
    b_ptrs = b_ptr + (
        offs_k[:, None] * b_stride_dim_0 + offs_n[None, :] * b_stride_dim_1
    )

    k_num_blocks = tl.cdiv(K, BLOCK_SIZE_K)

    # Scale base pointers start at row offsets for A, and column offsets for B.
    a_s_base_ptr = a_s_ptr + offs_m * a_s_stride_dim_0
    b_s_base_ptr = b_s_ptr + (offs_n // BLOCK_SIZE_K) * b_s_stride_dim_1
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    for k in range(0, k_num_blocks):
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Reciprocal scales to scale back to dynamic range of output dtype
        a_s = tl.load(a_s_base_ptr + k * a_s_stride_dim_1)
        b_s = tl.load(b_s_base_ptr + k * b_s_stride_dim_0)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s

        a_ptrs += BLOCK_SIZE_K * a_stride_dim_1
        b_ptrs += BLOCK_SIZE_K * b_stride_dim_0

    c = accumulator.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * \
        c_stride_dim_0 + offs_n[None, :] * c_stride_dim_1
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_fp8_gemm_1x128_128x128(
    a: torch.Tensor,  # (M, K)
    b: torch.Tensor,  # (K, N)
    a_s: torch.Tensor,  # (M, K // block_size)
    b_s: torch.Tensor,  # (K // block_size, N // block_size)
    block_size: int = 128,
    out_dtype: torch.dtype = torch.float32,
):
    # 'a' must be in row-major layout, 'b' must be in column-major layout
    assert _is_row_major(a), "a must be row-major"
    assert _is_column_major(b), "b must be column-major"

    # a_scales must be col-major, b_scales must be row-major
    assert _is_column_major(a_s), "a_s must be column-major"
    assert _is_column_major(b_s), "b_s must be column-major"

    M = a.size(0)
    K = a.size(1)
    N = b.size(1)
    c = a.new_empty(M, N, dtype=out_dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    triton_fp8_gemm_1x128_128x128_kernel[grid](
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
        out_dtype=out_dtype,
        BLOCK_SIZE_K=block_size,
    )
    return c


# ============================================================================
# NEW KERNEL: triton_fp8_blockwise_gemm with swizzling
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256,
                      'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128,
                      'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128,
                      'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256,
                      'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128,
                      'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K', 'SCALE_BLOCK_M', 'SCALE_BLOCK_K', 'SCALE_BLOCK_N'],
)
@triton.jit
def triton_fp8_blockwise_gemm_kernel(
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
    # Pointers to blockwise scale factors
    a_s_ptr,  # Shape: (M // SCALE_BLOCK_M, K // SCALE_BLOCK_K)
    a_s_stride_dim_0,
    a_s_stride_dim_1,
    b_s_ptr,  # Shape: (K // SCALE_BLOCK_K, N // SCALE_BLOCK_N)
    b_s_stride_dim_0,
    b_s_stride_dim_1,
    # Problem size
    M,
    N: tl.constexpr,
    K: tl.constexpr,
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
    FP8 GEMM with blockwise scaling and CTA swizzling.
    """
    # Program ID with swizzling for better L2 cache utilization
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

    # Iterate over K dimension
    for k in range(0, k_num_blocks):
        # Load A and B tiles (FP8)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Load blockwise scales
        # For A: map tile coordinates to scale coordinates
        scale_m_idx = offs_m // SCALE_BLOCK_M  # Shape: (BLOCK_SIZE_M,)
        # Shape: (BLOCK_SIZE_K,)
        scale_k_idx_a = (k * BLOCK_SIZE_K + offs_k) // SCALE_BLOCK_K

        a_s_ptrs = a_s_ptr + (
            scale_m_idx[:, None] * a_s_stride_dim_0 +
            scale_k_idx_a[None, :] * a_s_stride_dim_1
        )
        a_s = tl.load(a_s_ptrs, mask=a_mask, other=1.0)

        # For B: map tile coordinates to scale coordinates
        # Shape: (BLOCK_SIZE_K,)
        scale_k_idx_b = (k * BLOCK_SIZE_K + offs_k) // SCALE_BLOCK_K
        scale_n_idx = offs_n // SCALE_BLOCK_N  # Shape: (BLOCK_SIZE_N,)

        b_s_ptrs = b_s_ptr + (
            scale_k_idx_b[:, None] * b_s_stride_dim_0 +
            scale_n_idx[None, :] * b_s_stride_dim_1
        )
        b_s = tl.load(b_s_ptrs, mask=b_mask, other=1.0)

        # Apply blockwise scaling and accumulate
        a_scaled = a * a_s
        b_scaled = b * b_s
        accumulator += tl.dot(a_scaled, b_scaled, out_dtype=tl.float32)

        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * a_stride_dim_1
        b_ptrs += BLOCK_SIZE_K * b_stride_dim_0

    # Write output
    c = accumulator.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + (
        offs_m[:, None] * c_stride_dim_0 + offs_n[None, :] * c_stride_dim_1
    )
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def triton_fp8_blockwise_gemm(
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
    Blockwise-scaled FP8 GEMM with CTA swizzling.
    """
    # Validate layout
    assert a.is_contiguous() and a.stride(0) > a.stride(1), "a must be row-major"
    assert b.stride(0) < b.stride(1), "b must be column-major"
    assert a_s.is_contiguous() and b_s.is_contiguous()

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
            triton.cdiv(M, META['BLOCK_SIZE_M']) *
            triton.cdiv(N, META['BLOCK_SIZE_N']),
        )

    triton_fp8_blockwise_gemm_kernel[grid](
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


# ============================================================================
# Benchmarking
# ============================================================================

@dataclass(frozen=True)
class ExperimentConfig:
    out_dtype: torch.dtype
    m: int
    n: int
    k: int


@dataclass(frozen=True)
class ExperimentResult:
    bf16_mm_us: float
    fp8_triton_old_us: float
    fp8_triton_new_us: float
    fp8_scaled_mm_us: float


@dataclass(frozen=True)
class Experiment:
    config: ExperimentConfig
    result: ExperimentResult


def get_configs() -> List[ExperimentConfig]:
    mnk_list = [
        # Llama4 shapes
        (16640, 5120, 8192),
        (16640, 8192, 5120),
        # Additional shapes
        (8192, 8192, 8192),
        (4096, 4096, 4096),
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

    # Warm up then run OLD triton kernel
    warmup(
        triton_fp8_gemm_1x128_128x128,
        A_q,
        B_t_q,
        1.0 / A_s,
        1.0 / B_t_s,
        out_dtype=config.out_dtype,
    )
    fp8_triton_old_us = benchmark_cuda_function_in_microseconds(
        triton_fp8_gemm_1x128_128x128,
        A_q,
        B_t_q,
        1.0 / A_s,
        1.0 / B_t_s,
        out_dtype=config.out_dtype,
    )

    # Warm up then run NEW triton kernel
    warmup(
        triton_fp8_blockwise_gemm,
        A_q,
        B_t_q,
        1.0 / A_s,
        1.0 / B_t_s,
        out_dtype=config.out_dtype,
    )
    fp8_triton_new_us = benchmark_cuda_function_in_microseconds(
        triton_fp8_blockwise_gemm,
        A_q,
        B_t_q,
        1.0 / A_s,
        1.0 / B_t_s,
        out_dtype=config.out_dtype,
    )

    # Warm up then run torch bench
    # scaled_mm requires A_s and B_t_s be in column-major format
    A_s = A_s.t().contiguous().t()
    warmup(
        torch._scaled_mm,
        A_q,
        B_t_q,
        1.0 / A_s,
        1.0 / B_t_s,
        out_dtype=config.out_dtype,
    )
    fp8_scaled_mm_us = benchmark_cuda_function_in_microseconds(
        torch._scaled_mm,
        A_q,
        B_t_q,
        1.0 / A_s,
        1.0 / B_t_s,
        out_dtype=config.out_dtype,
    )

    return ExperimentResult(
        bf16_mm_us=bf16_mm_us,
        fp8_triton_old_us=fp8_triton_old_us,
        fp8_triton_new_us=fp8_triton_new_us,
        fp8_scaled_mm_us=fp8_scaled_mm_us,
    )


def print_results(experiments: List[Experiment]):
    headers = [
        "M",
        "N",
        "K",
        "out_dtype",
        "bf16_mm_us",
        "fp8_old_us",
        "fp8_new_us",
        "fp8_scaled_us",
        "bf16 tflops",
        "old tflops",
        "new tflops",
        "scaled tflops",
        "speedup_old",
        "speedup_new",
    ]
    rows = []
    for experiment in experiments:
        m, n, k = experiment.config.m, experiment.config.n, experiment.config.k
        flops = 2 * m * n * k
        bf16_mm_tflops = (flops / 1e12) / (experiment.result.bf16_mm_us / 1e6)
        old_tflops = (flops / 1e12) / \
            (experiment.result.fp8_triton_old_us / 1e6)
        new_tflops = (flops / 1e12) / \
            (experiment.result.fp8_triton_new_us / 1e6)
        scaled_tflops = (flops / 1e12) / \
            (experiment.result.fp8_scaled_mm_us / 1e6)

        speedup_old = experiment.result.bf16_mm_us / experiment.result.fp8_triton_old_us
        speedup_new = experiment.result.bf16_mm_us / experiment.result.fp8_triton_new_us

        rows.append(
            [
                m,
                n,
                k,
                str(experiment.config.out_dtype).split(".")[-1],
                f"{experiment.result.bf16_mm_us:.1f}",
                f"{experiment.result.fp8_triton_old_us:.1f}",
                f"{experiment.result.fp8_triton_new_us:.1f}",
                f"{experiment.result.fp8_scaled_mm_us:.1f}",
                f"{bf16_mm_tflops:.1f}",
                f"{old_tflops:.1f}",
                f"{new_tflops:.1f}",
                f"{scaled_tflops:.1f}",
                f"{speedup_old:.2f}x",
                f"{speedup_new:.2f}x",
            ]
        )
    print(tabulate(rows, headers=headers, tablefmt="grid"))


def benchmark_cuda_function_in_microseconds(f, *args, **kwargs):
    return do_bench(lambda: f(*args, **kwargs), return_mode="median") * 1e3


def main():
    torch.random.manual_seed(123)
    configs = get_configs()
    results = []
    for config in tqdm(configs):
        result = run_experiment(config)
        results.append(Experiment(config=config, result=result))

    # Use Tabulate to print results
    print_results(results)


if __name__ == "__main__":
    main()
