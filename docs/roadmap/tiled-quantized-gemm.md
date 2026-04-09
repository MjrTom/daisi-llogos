# Tiled Quantized GEMM: Closing the 42-64x Prefill Gap

## The Problem

LLogos prefill is 42-64x slower than llama.cpp on the same GPU:

| Model | LLogos CUDA | llama.cpp CUDA | Gap |
|-------|:-----------:|:--------------:|:---:|
| Qwen3.5-0.8B Q8_0 | 424 tok/s | 17,848 tok/s | 42x |
| Qwen3.5-4B Q8_0 | 143 tok/s | 6,776 tok/s | 47x |
| Qwen3-8B Q8_0 | 83 tok/s | 5,282 tok/s | 64x |

Decode (M=1) is at parity because both use dp4a kernels. The gap is entirely in batched matmul (M>1, used by prefill and training).

## Root Cause

LLogos `BatchMatMul` dequantizes the full weight tensor to a temporary FP32 buffer, then calls cuBLAS SGEMM:

```
Q8_0 weight [N×K] → GPU dequant → FP32 temp [N×K] → cuBLAS SGEMM → output [M×N]
```

This reads the weight data twice (once for dequant, once for SGEMM) and requires 4x the memory bandwidth (1 byte Q8_0 → 4 bytes FP32). llama.cpp reads the weight once, in quantized form, directly inside the matmul kernel.

## The Solution: Tiled Quantized GEMM

A CUDA kernel that computes `output[M×N] = activation[M×K] × weight^T[N×K]` where:
- Activation is pre-quantized to Q8_1 (36-byte blocks, same as M=1 path)
- Weight stays in Q8_0 format (no dequantization)
- dp4a computes 4 int8×int8 multiply-adds per instruction
- Shared memory tiles enable data reuse across output elements

### Kernel Architecture

```
Grid:  (ceil(N/TILE_N), ceil(M/TILE_M), 1)
Block: (WARP_SIZE=32, NWARPS=8, 1) = 256 threads

Each block computes a TILE_M × TILE_N output tile:

for each K_CHUNK along K dimension:
  1. Load weight tile [TILE_N × K_CHUNK blocks] into shared memory (cooperative)
  2. Load activation tile [TILE_M × K_CHUNK blocks] into shared memory (cooperative)
  3. __syncthreads()
  4. Each thread computes partial sums for its output elements:
     for each (i, j) in thread's output region:
       for each block in K_CHUNK:
         sum[i][j] += dp4a(a_qs, w_qs) * a_scale * w_scale
  5. __syncthreads()

Write TILE_M × TILE_N results to global memory
```

### Tiling Parameters (RTX 5080 / Blackwell)

| Parameter | Value | Rationale |
|-----------|------:|-----------|
| TILE_M | 64 | Rows of output per block |
| TILE_N | 32 | Columns of output per block |
| K_CHUNK | 8 blocks (256 elements) | Fits in shared memory with tiles |
| NWARPS | 8 | 256 threads per block |
| Output per thread | 8×4 = 32 elements | Each thread handles 8 M rows × 4 N cols |

### Shared Memory Layout

```
A tile: TILE_M × K_CHUNK_BLOCKS × 36 bytes = 64 × 8 × 36 = 18,432 bytes
B tile: TILE_N × K_CHUNK_BLOCKS × 36 bytes = 32 × 8 × 36 = 9,216 bytes
Total: 27,648 bytes (~27 KB) — well within 48 KB shared memory limit
```

### Thread Work Assignment

Each of the 256 threads computes a `(TILE_M/NWARPS) × (TILE_N/WARP_SIZE)` region:
- Thread at `(threadIdx.x, threadIdx.y)` computes `M_rows = 64/8 = 8` and `N_cols = 32/32 = 1`
- Actually better: each warp handles one N column, threads in warp handle M rows
- `threadIdx.y` (warp index) selects N column within tile
- `threadIdx.x` (lane) selects M row within tile

Wait — this gives 32 M rows per warp (one per lane) but we need 64. Two warps per N column.

Revised:
- 8 warps total
- 4 warps per N column group (processing 8 N columns each? No...)
- Better: each warp handles 64/8 = 8 M rows, processes ALL K for one N column
- 8 warps handle 8 N columns → TILE_N = 8? Too small.

The llama.cpp approach:
- Each thread accumulates `(TILE_M / WARP_SIZE) × (TILE_N / NWARPS)` output elements
- TILE_M=64, TILE_N=64, WARP_SIZE=32, NWARPS=8
- Per thread: 2 × 8 = 16 output elements in registers

### Key Implementation Details

1. **Q8_0 block loading**: Read 36 aligned bytes per block (scale + 32 quants). Store scale separately from quants in shared memory for efficient access patterns.

2. **Q8_1 activation pre-quantization**: Same kernel already exists for M=1 path (`quantize_f32_q8_1`). Extend to handle M rows in one launch.

3. **dp4a accumulation**: `__dp4a(int a, int b, int acc)` processes 4 int8 pairs per instruction. Each Q8_0/Q8_1 block has 32 quants = 8 int32 loads = 8 dp4a calls per block pair.

4. **Scale application**: `float result = a_scale * w_scale * (float)dp4a_sum`. Applied once per block pair, not per element.

5. **Bank conflict avoidance**: Pad shared memory arrays to avoid 32-way conflicts on row access.

6. **Double buffering**: Overlap K_CHUNK loading with computation from previous chunk.

## Implementation Plan

### Phase 1: Basic Tiled Kernel (~200 lines CUDA)
- Fixed tile sizes (TILE_M=32, TILE_N=8, K_CHUNK=8)
- No double buffering
- Verify correctness against F32 reference
- Expected: 2-4x over current dequant→SGEMM

### Phase 2: Tuned Kernel (~400 lines)
- Architecture-aware tile sizes
- Register-level output accumulation
- Bank conflict avoidance
- Expected: 5-10x over dequant→SGEMM

### Phase 3: Full Optimization (~600 lines)
- Double-buffered K_CHUNK loading
- Stream-K work distribution for better load balancing
- Multiple quant format support (Q4_0, Q4_K, Q6_K)
- Expected: 10-20x over dequant→SGEMM (approaching llama.cpp)

### Phase 4: Integration
- Add to both CudaBackend (inference prefill) and CudaTrainingBackend
- Benchmark prefill against llama.cpp
- Benchmark training speed improvement

## Files to Create/Modify

| File | Change |
|------|--------|
| `kernels/batched_mmq.cu` | **New** — tiled quantized GEMM kernel |
| `CudaTrainingBackend.cs` | Use batched_mmq in BatchMatMul for Q8_0 |
| `CudaBackend.cs` | Use batched_mmq in BatchMatMul for Q8_0 (inference prefill) |

## Success Criteria

| Metric | Current | Target | Stretch |
|--------|--------:|-------:|--------:|
| Prefill 0.8B Q8_0 | 424 tok/s | 4,000 tok/s | 12,000 tok/s |
| Prefill 8B Q8_0 | 83 tok/s | 1,000 tok/s | 4,000 tok/s |
| Training 0.8B | 2.9 seq/s | 20 seq/s | 50 seq/s |

## References

- llama.cpp `mmq.cuh`: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/mmq.cuh
- llama.cpp `vecdotq.cuh`: https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/vecdotq.cuh
- NVIDIA dp4a documentation: https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__INT.html
- CUTLASS GEMM templates: https://github.com/NVIDIA/cutlass
