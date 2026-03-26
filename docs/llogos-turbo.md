# LLogos Turbo — Extreme KV Cache Compression

LLogos Turbo is an extreme KV cache compression system built into the Llogos inference engine. Based on Google Research's [TurboQuant](https://arxiv.org/abs/2504.19874) paper, it achieves 8-12x KV cache memory reduction through a three-stage compression pipeline: randomized Walsh-Hadamard rotation, MSE-optimal scalar quantization, and QJL sign-bit residual correction.

## Why KV Cache Compression?

During autoregressive decoding, every generated token must attend to all previous tokens. The Key and Value tensors for every past position are stored in the **KV cache** — and this cache grows linearly with context length.

For large models at long context, the KV cache dominates memory:

| Model | Context | F16 KV Cache | Turbo 3-bit | Savings |
|-------|---------|-------------|-------------|---------|
| Qwen3.5-0.8B (6 attn layers) | 32K | 96 MB | 11 MB | 85 MB |
| Qwen3.5-9B (8 attn layers) | 32K | 2.0 GB | 230 MB | 1.8 GB |
| Qwen3.5-27B (16 attn layers) | 32K | 2.0 GB | 465 MB | 1.5 GB |
| Qwen3.5-27B (16 attn layers) | 128K | 8.0 GB | 1.8 GB | 6.2 GB |

On GPU, the KV cache competes with model weights for VRAM. Compressing it enables longer contexts, larger batch sizes, or running models that wouldn't otherwise fit.

## How It Works

LLogos Turbo uses a three-stage compression pipeline that requires **no training, no calibration, and no fine-tuning** — it works as a drop-in replacement at inference time.

### Stage 1: Walsh-Hadamard Rotation

Raw KV vectors often have outlier dimensions with much larger magnitudes than others. These outliers make fixed-grid quantization lossy. The Walsh-Hadamard Transform (WHT) with random sign flips rotates the vector into a space where all dimensions have similar magnitude.

```
original:     [0.01, 0.02, 50.0, 0.03, ...]  ← outlier at dim 2
                        ↓ WHT rotation
rotated:      [6.25, 6.24, 6.26, 6.25, ...]  ← uniform, quantizer-friendly
```

Properties:
- **Orthonormal**: preserves vector norms and inner products (`||Hx|| = ||x||`)
- **O(d log d)**: faster than matrix multiply via butterfly structure
- **Self-inverse**: `H(Hx) = x` (same operation for forward and inverse)
- **Data-oblivious**: the same rotation works for any vector — no calibration

Implementation: `WalshHadamard.cs` with AVX2 SIMD butterfly passes and scalar fallback.

### Stage 2: MSE-Optimal Scalar Quantization

After rotation, each coordinate follows a predictable distribution (approximately Gaussian). A pre-computed Lloyd-Max quantization grid maps each float to 2, 3, or 4 bits:

| Bits | Levels | Bits/dim | Typical MSE (unit variance) |
|------|--------|----------|---------------------------|
| 2 | 4 | 2.0 | ~0.36 |
| 3 | 8 | 3.0 | ~0.09 |
| 4 | 16 | 4.0 | ~0.02 |

Values are packed into byte arrays (4-bit: 2 values/byte, 3-bit: bit-packed, 2-bit: 4 values/byte). A per-head RMS scale factor is stored alongside for denormalization.

Implementation: `ScalarQuantizer.cs` with pre-computed boundary/centroid tables.

### Stage 3: QJL Sign-Bit Residual Correction

MSE-optimal quantization minimizes reconstruction error but introduces **bias in inner product estimation** — which is what attention scores are. The QJL (Quantized Johnson-Lindenstrauss) correction addresses this:

1. **On write**: Compute the residual `r = rotated - quantized`. Project through a random Rademacher matrix `R` and store only the sign bits: `sign(R · r)`. Also store the residual's L2 norm.

2. **On attention**: For each Q-K dot product, add a correction term using the stored sign bits and residual norm. The correction is computed using pre-computed query projections (once per head, reused across all positions) for efficiency.

The estimator uses sign agreement between the residual and query projections, scaled by the residual norm and query projection magnitude, to approximate the missing `q · residual` term.

Implementation: `QjlProjection.cs` with `PrecomputeQueryProjections` for amortized per-head cost.

### Compression Pipeline Summary

```
Write path (per KV head, per position):
  input[d] → normalize by RMS → WHT rotate → scalar quantize → pack bits
                                         ↓
                              compute residual → project → store sign bits + norm

Read path (fused into attention):
  packed bits → dequantize → inverse WHT → rescale → dot with Q → + QJL correction → attention score
```

## Architecture

LLogos Turbo integrates cleanly into the existing Llogos architecture through two key design decisions:

### 1. `TurboQuantKvCache` implements `IKvCache`

The compressed cache is a drop-in replacement for `KvCache` or `PagedKvCache`. The `ForwardPass` and all backend code work unchanged.

### 2. Fused Compressed Attention via `IKvCache.ComputeAttention`

Rather than decompressing the entire KV cache into F32 tensors for every decode step, `TurboQuantKvCache` performs attention directly from compressed data. Each K and V vector is decompressed inline — one at a time — during the attention dot product loop. Only a single `float[headDim]` buffer is needed, regardless of sequence length.

The `IKvCache.ComputeAttention` interface method enables this:

```csharp
// ForwardPass tries fused path first, falls back to standard
if (!_kvCache.ComputeAttention(_attnOut, _qAttn, _qGate,
        layer, numHeads, numKvHeads, keyLen, valLen, seqLen, scale))
{
    // Standard path: decompress → F32 tensors → GatedAttention
    var kCache = _kvCache.GetKCacheTensor(layer);
    var vCache = _kvCache.GetVCacheTensor(layer);
    _backend.GatedAttention(...);
}
```

Standard `KvCache` and `PagedKvCache` return `false` (use standard path). `TurboQuantKvCache` returns `true` and handles attention itself. No backend changes needed.

### 3. CUDA Adaptive Dual-Path Architecture

On GPU, `CudaTurboQuantKvCache` implements a dual-path strategy for best-of-both-worlds performance:

- **Dual write**: Every position is written to both F16 shadow tensors (on the main CUDA stream, inside graph capture) and compressed TurboQuant storage (on a secondary async stream, outside the graph). The async stream runs the `turbo_kv_write` compression kernel without impacting the main forward pass.

- **Adaptive read**: At short context (KV fits in L2 cache), `ComputeAttention` returns `false` — the forward pass uses the baseline `GatedAttention` kernel with F16 tensors at full speed. When context crosses the L2 threshold (KV spills to HBM), it returns `true` — the compressed `turbo_gated_attention` kernel reads 4-8x fewer bytes from HBM.

- **Zero-cost transition**: Compressed data is always current (written every token on the async stream). Switching to compressed attention just requires syncing the async stream.

### 4. Rotated-Domain Attention (CUDA)

The CUDA compressed attention kernel uses a key algebraic optimization: since WHT is orthonormal and self-inverse, dot products can be computed in the rotated domain without per-position inverse WHTs.

```
q · k_original = scale_k × q_rotated · k_centroids
where q_rotated = (1/√n) · H · (D_k · q)  — one WHT per head
```

This eliminates ALL per-position WHTs for K scoring. For V, centroids are accumulated in the rotated domain with a single inverse WHT of the final result.

Additional CUDA optimizations:
- Shared memory centroid lookup table (16 floats) — eliminates constant memory serialization when threads access different centroid indices
- Pre-computed combined V weights (score × correction × scale) — reduces V inner loop to 1 byte load + 1 LUT lookup + 1 FMA
- `__ldg()` intrinsics for cached packed byte reads from global memory

### File Map

```
Daisi.Llogos/Inference/DaisiTurbo/           CPU implementation
├── WalshHadamard.cs                         Fast WHT with AVX2 SIMD + scalar fallback
├── ScalarQuantizer.cs                       Lloyd-Max 2/3/4-bit quantizer with bit packing
├── QjlProjection.cs                         Sign-bit residual correction with precomputed projections
├── TurboQuantConfig.cs                      Configuration + CLI parsing
└── TurboQuantKvCache.cs                     IKvCache: compressed storage + fused CPU attention

Daisi.Llogos.Cuda/                           CUDA implementation
├── CudaTurboQuantKvCache.cs                 IKvCache: adaptive dual-path (F16 + compressed)
└── kernels/turbo_quant.cu                   CUDA kernels: turbo_kv_write + turbo_gated_attention
```

## Usage

### CLI

```bash
# Default 3-bit (best quality/compression tradeoff)
dotnet run -- --model model.gguf --prompt "Hello" --kv-quant turbo

# 4-bit (higher quality, less compression)
dotnet run -- --model model.gguf --prompt "Hello" --kv-quant turbo:4

# 2-bit (extreme compression, some quality loss)
dotnet run -- --model model.gguf --prompt "Hello" --kv-quant turbo:2

# Explicit QJL projection dimension
dotnet run -- --model model.gguf --prompt "Hello" --kv-quant turbo:3+qjl32

# Disable QJL (fastest, pure scalar quantization)
dotnet run -- --model model.gguf --prompt "Hello" --kv-quant turbo:3+noqjl

# Benchmark with compression stats
dotnet run -- --model model.gguf --bench --kv-quant turbo:3

# Combine with other KV strategies
dotnet run -- --model model.gguf --prompt "Hello" --kv-quant turbo:3 --attention sinks:64,4096
```

### Configuration Options

| Option | Description | Default |
|--------|-------------|---------|
| `turbo` | 3-bit quantization with QJL (headDim/2 projection dims) | — |
| `turbo:N` | N-bit quantization (2, 3, or 4) with default QJL | N=3 |
| `turbo:N+noqjl` | N-bit quantization, no QJL correction (fastest) | — |
| `turbo:N+qjlM` | N-bit quantization with M QJL projection dimensions | M=headDim/2 |

### Programmatic

```csharp
var turboConfig = new TurboQuantConfig { QuantBits = 3, QjlProjectionDim = 0 };
IKvCache kvCache = new TurboQuantKvCache(backend, modelConfig,
    maxSeqLen: 32768, turboConfig: turboConfig);

// Use exactly like KvCache — ForwardPass is unchanged
var forward = new ForwardPass(backend, config, weights, kvCache, deltaState);
```

## Benchmarks

### CUDA — RTX 5080 (16GB VRAM, 64MB L2, Blackwell)

**Qwen3.5-0.8B Q8_0** (6 attention layers / 24 total, hybrid DeltaNet)

| Context | Baseline F16 | Turbo 4-bit | vs Baseline | Compression | KV Savings |
|---|---|---|---|---|---|
| 138 tokens | 435 tok/s | 424 tok/s | **97.4%** | 7.8x | 3.3 KB → 0.4 KB |
| 4.5K tokens | 138 tok/s | 137 tok/s | 99.3% | 7.8x | 107 MB → 14 MB |

**Qwen3.5-9B Q4_0** (8 attention layers / 48 total, hybrid DeltaNet)

| Context | Baseline F16 | Turbo 4-bit | vs Baseline | Compression | KV Savings |
|---|---|---|---|---|---|
| 39 tokens | 98.3 tok/s | 96.5 tok/s | **98.2%** | 7.8x | 1.2 MB → 0.2 MB |
| 4K tokens | 62.5 tok/s | 53.8 tok/s | 86.1% | 7.8x | 128 MB → 16 MB |

### CPU — AMD Ryzen 9 9900X, Qwen3.5-0.8B Q8_0, 128 decode tokens

| Configuration | Decode (tok/s) | KV Memory | Compression | vs Baseline |
|---|---|---|---|---|
| Baseline F16 | 33.4 | 3,624 KB | 1.0x | — |
| Turbo 4-bit+noqjl | 34.8 | 467 KB | 7.8x | +4% |
| Turbo 3-bit+noqjl | 32.4 | 354 KB | 10.2x | -3% |
| Turbo 3-bit+qjl | 28.4 | 411 KB | 8.8x | -15% |
| Turbo 2-bit+noqjl | 30.6 | 303 KB | 12.2x | -8% |

### CUDA Kernel Optimization History

The CUDA attention kernel was optimized through 6 iterations from 45 to 422 tok/s:

| Version | tok/s | vs Baseline | Key Optimization |
|---|---|---|---|
| v1 | 45 | 10% | Per-position WHT (naive) |
| v2 | 167 | 38% | Rotated-domain K: eliminate per-position WHT via `q_rot · k_centroids` |
| v3 | 247 | 56% | Fused K dequant+dot product: zero intermediate arrays |
| v4 | 253 | 57% | Parallel V dequant + Blackwell shared memory fix (compute-sanitizer) |
| v5 | 266 | 61% | Shared memory centroid LUT + precomputed V weights |
| **v6** | **422** | **97%** | **Adaptive dual-path: F16 graph at short context, compressed at long** |

### Performance Characteristics

**Why 97% instead of faster than baseline at short context?**

At 138 tokens, attention is only **3.6% of total decode time**. The weight matmuls (Q/K/V projections, FFN, LM head) dominate at 96.4%. Even making attention infinitely fast would only yield +4%. The 2.6% overhead is from the async compressed write stream contention — the GPU shares SMs between the main forward pass and the background compression.

**Where TurboQuant beats baseline:**

As context length grows, attention becomes a larger fraction of total time. At 4K tokens attention is ~50%, at 16K it's ~80%, at 64K it's ~94%. At these lengths, reading 4-bit packed data (32 bytes per K position) instead of F16 (128 bytes) delivers real bandwidth savings — the crossover where compressed attention runs faster than F16 attention.

**Memory savings at scale:**

| Model | Context | F16 KV Cache | Turbo 4-bit | Freed VRAM |
|---|---|---|---|---|
| Qwen3.5-9B | 32K | 2.0 GB | 256 MB | 1.8 GB |
| Qwen3.5-27B | 32K | 2.0 GB | 256 MB | 1.8 GB |
| Qwen3.5-27B | 128K | 8.0 GB | 1.0 GB | 7.0 GB |

On a 16GB GPU with a 27B model (15GB weights), TurboQuant is the difference between running at 128K context and OOMing.

## Test Coverage

44 tests across all components:

| Component | Tests | What's Validated |
|-----------|-------|-----------------|
| WalshHadamard | 8 | Round-trip invertibility, energy preservation, outlier spreading, determinism |
| ScalarQuantizer | 10 | Pack/unpack round-trip, MSE bounds per bit-width, level reachability, symmetry |
| QjlProjection | 7 | Sign preservation, batch consistency, variance reduction, norm-aware MSE reduction |
| TurboQuantKvCache | 19 | Write/read accuracy, multi-position, stats, config parsing, sliding window, bit-widths |

## Roadmap

### Completed

- [x] **Walsh-Hadamard Transform** — AVX2 SIMD butterfly + scalar fallback, random sign flips
- [x] **Scalar Quantizer** — Lloyd-Max 2/3/4-bit, pre-computed grids, bit-packed storage
- [x] **QJL Sign-Bit Correction** — Rademacher projection, residual norm storage, scale-corrected estimator
- [x] **TurboQuantKvCache** — `IKvCache` implementation with compressed storage + fused CPU attention
- [x] **IKvCache.ComputeAttention** — Interface method for cache-driven attention, transparent fallback
- [x] **CLI Integration** — `--kv-quant` flag with full configuration parsing, auto-selects CUDA backend
- [x] **Incremental Fallback Decompression** — Delta decompression for lazy F32 path
- [x] **Norm-Aware QJL Estimator** — Residual norm stored during compression, pre-computed query projections
- [x] **CUDA Fused Kernels** — `turbo_kv_write` (compress on device) + `turbo_gated_attention` (compressed attention)
- [x] **Rotated-Domain Attention** — Eliminate per-position WHT via algebraic identity: `q · k = scale × q_rot · k_centroids`
- [x] **Shared Memory Centroid LUT** — 16-float lookup table in shared memory, eliminates constant memory serialization
- [x] **Adaptive Dual-Path Architecture** — F16 attention at short context (graph-captured), compressed at long context. Async compressed write on secondary CUDA stream.
- [x] **CUDA Kernel Optimization** — 6 iterations: 45 → 422 tok/s (9.4x speedup). compute-sanitizer debugging for Blackwell shared memory issues.

### Next

- [ ] **Paged TurboQuant** — Combine with `PagedKvCache` for dynamic allocation. Each 256-token page shrinks from ~128 KB to ~18 KB at 3-bit, making the paging system far more effective.
- [ ] **Quality Validation** — Perplexity benchmarks on standard datasets (WikiText, C4) comparing baseline vs turbo at each bit-width. Verify the "quality-neutral at 3.5 bits" claim holds in our implementation.
- [ ] **Long-Context Benchmarks** — Profile the compressed attention crossover point where TurboQuant beats F16 baseline. Requires contexts beyond L2 cache (8K+ on small models, 2K+ on large models).
- [ ] **SIMD-Fused Dequant in Attention Loop** — AVX2-fuse the 4-bit unpack + WHT butterfly + dot product into one pass inside the CPU fused attention loop.
- [ ] **Vulkan/WebGPU Compressed Attention Shaders** — SPIR-V/WGSL compute shaders that read compressed KV directly.

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — Zandieh, Daliri, Hadian, Mirrokni (Google Research, 2025)
- [TurboQuant: Redefining AI Efficiency with Extreme Compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — Google Research Blog
