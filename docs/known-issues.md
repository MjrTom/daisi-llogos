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
12. **SwiGLU Q8_1 scratch overflow** — Fixed (fused swiglu_q8_1 wrote M×N elements to single-token scratch for batched M>1)
13. **Training arena corruption at seqLen>208** — Fixed (arena disabled, per-tensor Pool allocation)
14. **Non-power-of-2 attention reduction** — Fixed (training attention kernels rounded threads to power-of-2)
15. **Duplicate embedding loop in ForwardBatchedPrefill** — Fixed

---

## Resolved: dp4a Precision Loss

### Original Symptom
The `__dp4a` integer dot product approach (CUDA) produced garbage output. Initially believed to be precision loss from int8 activation quantization compounding across layers.

### Root Cause
**Stale Q8_1 cache.** The Q8_1 quantized activation was cached by device pointer address, but the same tensor (`_normOut`) is reused across layers — same pointer, different data. The cache served stale Q8_1 data from the previous layer.

### Fix
Generation-based cache invalidation: a counter is incremented by every operation that writes to a tensor (RmsNorm, AddRmsNorm, etc.). The cache checks both pointer AND generation. Additionally, three fused RmsNorm+Q8_1 kernels pre-compute the Q8_1 data inside the normalization pass, eliminating the separate quantization kernel entirely.

dp4a is now the default Q4_0 path on all GPUs. Q4_K also has a cooperative dp4a kernel (v2) with 128 threads and 16 threads per super-block, used alongside a fused MatMulSwiGLU kernel for Q4_K FFN layers.

---

## Performance vs llama.cpp

### Current Standing (RTX 5080, decode tok/s, 128 tokens)

**Exceeding llama.cpp on 4 of 8 CUDA models across three architectures.**

| Model | Llogos CUDA | llama.cpp CUDA | % |
|-------|--------:|--------:|--------:|
| 0.8B Q8_0 | **441** | 399 | **110%** |
| TinyLlama 1.1B Q8_0 | **448** | 443 | **101%** |
| 4B Q8_0 | **144** | 135 | **107%** |
| 8B Q8_0 | 91 | 92 | 99% |
| DeepSeek R1 8B Q8_0 | 94 | 95 | 99% |
| 8B Q4_K_M | **127** | 138 | **92%** |
| 9B Q8_0 | **88** | 84 | **105%** |
| 9B Q4_0 | 101 | 123 | 82% |

### Remaining Gaps
- **CUDA Q4_K_M** (92%): Closed from 90% via fused SwiGLU matmul, Q6_K kernel optimization (10→3 rows), and cooperative dp4a kernels. Remaining ~8% gap is from pipeline overhead and mixed-quant layer efficiency.
- **CUDA Q4_0** (82%): dp4a kernel limited by DeltaNet overhead in Qwen3.5 hybrid models. Pure attention Q4_0 models would be closer.
- **Vulkan** (33-58%): matmul bandwidth utilization gap, no dp4a equivalent in GLSL. Needs further shader optimization.

### Resolved: Q4_0 dp4a Stale Activation Bug

The on-demand Q8_1 quantization condition was `!_q8_1FusedReady && (ptr/gen mismatch)`. When `_q8_1FusedReady` was stale from a previous RmsNorm but the current activation was different (o_proj, down_proj matmuls), the condition evaluated false and quantization was skipped — the dp4a kernel ran on stale Q8_1 data, producing garbled output. This was a latent bug masked by the Blackwell float fallback path for Q4_0. Fixed by removing the `!_q8_1FusedReady` guard: always re-quantize when the activation pointer or generation changes.

### Key Optimizations
See [Inference Optimization White Paper](inference-optimization.md) for full details on partial vocab logits, per-quant row tuning, fused RmsNorm+Q8_1, and other techniques.

---

## Architecture Reference: Qwen3.5-9B DeltaNet

See previous documentation for tensor mapping, layer schedule, and architecture details.
