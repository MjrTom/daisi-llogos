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

## Open: dp4a Precision Loss

### Symptom
The `__dp4a` integer dot product approach (CUDA) quantizes the activation to int8, which loses precision that compounds across 36 layers, producing garbage for 8B+ models.

### Status
dp4a path disabled. llama.cpp handles this with a more sophisticated Q8_1 format that preserves per-block activation sums for error compensation. A proper implementation would require matching their exact precision handling.

---

## Open: Performance Gap vs llama.cpp

### Current Standing (RTX 5080, decode tok/s, 128 tokens)

| Model | llama.cpp CUDA | Llogos CUDA | % | llama.cpp Vulkan | Llogos Vulkan | % |
|-------|--------:|--------:|--------:|--------:|--------:|--------:|
| 0.8B Q8_0 | 399 | 363 | 91% | 466 | 150 | 32% |
| 8B Q8_0 | 92 | **83** | **90%** | 96 | 54 | 56% |
| 8B Q4_K_M | 138 | 86 | 62% | 142 | 53 | 37% |
| 9B Q8_0 | 84 | **76** | **90%** | —* | 51 | — |

*llama.cpp Vulkan b8461 has a regression on DeltaNet models (~11 tok/s).

### Remaining Gaps
- **CUDA Q4_K_M** (62%): llama.cpp's Q4_K kernel is significantly more optimized (dp4a, specific memory patterns)
- **Vulkan** (32-56%): matmul bandwidth utilization gap, descriptor set overhead vs llama.cpp's buffer device addresses
- **CUDA Q8_0** (90%): target achieved — remaining gap is fundamental memory bandwidth efficiency

---

## Architecture Reference: Qwen3.5-9B DeltaNet

See previous documentation for tensor mapping, layer schedule, and architecture details.
