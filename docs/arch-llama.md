# Architecture: LLaMA (Standard Transformer)

GGUF architecture: `llama`

## Overview

The LLaMA architecture is the baseline transformer used by Meta's LLaMA 2/3 family, TinyLlama, DeepSeek R1 distillations, and many fine-tuned derivatives. It uses standard multi-head or grouped-query attention without gating, biases, or hybrid layers.

```
  Standard LLaMA Layer
  =====================

  hidden ──> RMSNorm ──> Q/K/V projections
                              |
                              v
                         RoPE(Q, K)
                              |
                              v
                         KV Cache Write
                              |
                              v
                     Causal Attention (GQA)
                              |
                              v
                      O projection + Residual
                              |
                              v
                    RMSNorm ──> FFN (SwiGLU)
                              |
                              v
                         + Residual ──> next layer
```

## Key Features

| Feature | Implementation |
|---------|---------------|
| GQA | Grouped Query Attention (LLaMA 3: 32/8, TinyLlama: 32/4) |
| RoPE | Rotary Position Embeddings |
| SwiGLU | Fused gate+up FFN with SiLU activation |
| No biases | No attention biases (unlike Qwen2) |
| No gated Q | Standard Q projection (unlike Qwen3) |

## What Worked

- **Fused SwiGLU kernel**: Single CUDA kernel for gate+up MatMul + SiLU activation for Q4_K weights. Eliminates intermediate buffer.
- **dp4a integer matmul**: INT8 dot product for Q8_0 and Q4_0 weights. Matches llama.cpp precision.
- **Aligned Q8_0 repacking**: 34-byte blocks repacked to 36-byte for 4-byte aligned quant reads. Required for dp4a correctness.
- **TinyLlama as test model**: Small enough for CI, exercises all code paths. LoRA training verified with "perfect recall".

## What Didn't Work

- **Unaligned Q8_0 for small models**: Models with HiddenDim < 2048 weren't getting Q8_0 repacking, causing NaN in the dp4a kernel. Fixed by removing the K >= 2048 threshold.

## Benchmarks (RTX 5080)

| Model | Quant | CUDA tok/s | llama.cpp | vs llama.cpp |
|-------|-------|--------:|--------:|--------:|
| TinyLlama 1.1B | Q8_0 | 448 | 443 | **101%** |
| DeepSeek R1 8B | Q8_0 | 94 | 95 | 99% |
| Llama 3.2-1B | Q8_0 | ~300 | — | — |

## Supported Models

| Model | Params | Quants | LoRA Training |
|-------|--------|--------|---------------|
| TinyLlama 1.1B | 1.1B | Q8_0, Q4_0 | Yes (perfect recall) |
| Llama 3.2-1B | 1B | Q8_0 | Untested |
| DeepSeek R1 Distill 8B | 8B | Q8_0 | Untested |
