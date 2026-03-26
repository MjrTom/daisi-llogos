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

### File Map

```
Daisi.Llogos/Inference/DaisiTurbo/
├── WalshHadamard.cs       Fast WHT with AVX2 SIMD + scalar fallback
├── ScalarQuantizer.cs     Lloyd-Max 2/3/4-bit quantizer with bit packing
├── QjlProjection.cs       Sign-bit residual correction with precomputed projections
├── TurboQuantConfig.cs    Configuration + CLI parsing
└── TurboQuantKvCache.cs   IKvCache implementation: compressed storage + fused attention
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

### CPU (AMD Ryzen 9 9900X), Qwen3.5-0.8B Q8_0, 128 decode tokens

| Configuration | Decode (tok/s) | KV Memory | Compression | vs Baseline |
|---|---|---|---|---|
| Baseline F16 | 33.4 | 3,624 KB | 1.0x | — |
| Turbo 4-bit+noqjl | 34.8 | 467 KB | 7.8x | +4% |
| Turbo 3-bit+noqjl | 32.4 | 354 KB | 10.2x | -3% |
| Turbo 3-bit+qjl | 28.4 | 411 KB | 8.8x | -15% |
| Turbo 2-bit+noqjl | 30.6 | 303 KB | 12.2x | -8% |

### Performance Characteristics

**On CPU, LLogos Turbo is a memory-capacity optimization, not a speed optimization.** CPU attention is compute-bound (dot products), not memory-bandwidth-bound. The per-position dequant+WHT cost adds overhead that offsets any cache-locality benefit at current sequence lengths.

**On GPU (future CUDA implementation), LLogos Turbo will be a speed optimization.** GPU attention is memory-bandwidth-bound — the bottleneck is reading KV from HBM. Reading 3-bit packed data instead of F16 directly reduces memory traffic, which is why the TurboQuant paper reports up to 8x speedup on H100.

**The primary value today is enabling longer contexts and larger models:**
- 27B model at 128K context: 8 GB F16 KV → 1.8 GB Turbo 3-bit
- On a 16GB GPU, this is the difference between fitting and OOMing

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
- [x] **TurboQuantKvCache** — `IKvCache` implementation with compressed storage
- [x] **Fused Compressed Attention** — Inline dequant during dot product, no F32 shadow buffers
- [x] **IKvCache.ComputeAttention** — Interface method for cache-driven attention, transparent fallback
- [x] **CLI Integration** — `--kv-quant` flag with full configuration parsing
- [x] **Incremental Fallback Decompression** — Delta decompression for lazy F32 path
- [x] **Norm-Aware QJL Estimator** — Residual norm stored during compression, pre-computed query projections

### Next

- [ ] **CUDA Fused Kernels** — Fused `rotate → quantize → store` on write, compressed attention on read. GPU memory bandwidth is the bottleneck where TurboQuant delivers its headline 8x speedup. This is the highest-impact item.
- [ ] **Paged TurboQuant** — Combine with `PagedKvCache` for dynamic allocation. Each 256-token page shrinks from ~128 KB to ~18 KB at 3-bit, making the paging system far more effective.
- [ ] **Quality Validation** — Perplexity benchmarks on standard datasets (WikiText, C4) comparing baseline vs turbo at each bit-width. Verify the "quality-neutral at 3.5 bits" claim holds in our implementation.
- [ ] **SIMD-Fused Dequant in Attention Loop** — AVX2-fuse the 4-bit unpack + WHT butterfly + dot product into one pass inside the fused attention loop. Would close the remaining ~10% CPU decode gap.
- [ ] **Vulkan/WebGPU Compressed Attention Shaders** — SPIR-V/WGSL compute shaders that read compressed KV directly. Smaller buffer reads = fewer GPU memory transactions.

## References

- [TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate](https://arxiv.org/abs/2504.19874) — Zandieh, Daliri, Hadian, Mirrokni (Google Research, 2025)
- [TurboQuant: Redefining AI Efficiency with Extreme Compression](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) — Google Research Blog
