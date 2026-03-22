# Known Issues

All original bugs (issues 1-5) are **fixed**. This document tracks remaining optimization barriers and known limitations.

---

## Resolved Issues

1. **K-Quant dequantization** (Q6_K scale, Q5_K nibble/bit layout, Q3_K type size) — Fixed
2. **Qwen3.5-9B conv buffer overflow** — Fixed
3. **Qwen3 HasGatedQ detection** — Fixed
4. **CausalConv1d buffer overflow** — Fixed
5. **Qwen3.5-9B Q/K head expansion** (repeat vs repeat_interleave) — Fixed
6. **Vulkan DeltaNet batching crash** — Fixed (CausalConv1d temp buffer lifetime + CopyBuffer batch recording)
7. **Vulkan SplitSwiGLU CPU fallback** — Fixed (GPU composite op, was doing GPU↔CPU round-trips per layer)
8. **Vulkan SplitUnequalQKV CPU fallback** — Fixed (GPU CopyTensorRegion with srcOffset)
9. **Vulkan RepeatTile CPU fallback** — Fixed (GPU compute shader)
10. **Vulkan CopyTensorRegion ignoring srcOffset** — Fixed (was copying from offset 0 regardless)
11. **Vulkan ArgMax CPU fallback** — Fixed (GPU composite op, was downloading 600KB logits per token)

---

## Resolved: dp4a Precision Loss

### Original Symptom
The `__dp4a` integer dot product approach (CUDA) produced garbage output. Initially believed to be precision loss from int8 activation quantization compounding across layers.

### Root Cause
**Stale Q8_1 cache.** The Q8_1 quantized activation was cached by device pointer address, but the same tensor (`_normOut`) is reused across layers — same pointer, different data. The cache served stale Q8_1 data from the previous layer.

### Fix
Generation-based cache invalidation: a counter is incremented by every operation that writes to a tensor (RmsNorm, AddRmsNorm, etc.). The cache checks both pointer AND generation. Additionally, three fused RmsNorm+Q8_1 kernels pre-compute the Q8_1 data inside the normalization pass, eliminating the separate quantization kernel entirely.

dp4a is now the default Q4_0 path on pre-Blackwell GPUs, with architecture-adaptive dispatch selecting the float path on Blackwell (SM 12.x).

---

## Performance vs llama.cpp

### Current Standing (RTX 5080, decode tok/s, 128 tokens)

**Exceeding llama.cpp on 4 of 6 CUDA models tested.**

| Model | Llogos CUDA | llama.cpp CUDA | % |
|-------|--------:|--------:|--------:|
| 0.8B Q8_0 | **436** | 399 | **109%** |
| 4B Q8_0 | **142** | 135 | **105%** |
| 8B Q8_0 | 90 | 92 | 98% |
| 8B Q4_K_M | 122 | 138 | 88% |
| 9B Q8_0 | **86** | 84 | **102%** |
| 9B Q4_0 | 100 | 123 | 81% |

### Remaining Gaps
- **CUDA 4-bit quants** (81-88%): Q4_0 float kernel is compute-bound on nibble extraction; dp4a matches float on Blackwell but doesn't exceed it. Q4_K needs dp4a implementation.
- **Vulkan** (33-58%): matmul bandwidth utilization gap, no dp4a equivalent in GLSL. Needs further shader optimization.

### Key Optimizations
See [Inference Optimization White Paper](inference-optimization.md) for full details on partial vocab logits, per-quant row tuning, fused RmsNorm+Q8_1, and other techniques.

---

## Architecture Reference: Qwen3.5-9B DeltaNet

See previous documentation for tensor mapping, layer schedule, and architecture details.
