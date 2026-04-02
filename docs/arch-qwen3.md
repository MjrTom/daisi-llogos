# Architecture: Qwen 3 (Standard Attention + Gated Q)

GGUF architecture: `qwen3`

## Overview

Qwen 3 is a standard transformer with Grouped Query Attention, gated Q projections, and thinking mode (`<think>` tags). All layers are standard attention — no DeltaNet.

```
  All 36 layers: Standard Gated Attention
  ========================================

  normOut ──> Q projection (2x output: Q_attn + Q_gate interleaved)
                  |
                  v
           DeInterleaveQ ──> Q_attn, Q_gate
                  |
                  v
           PerHeadRmsNorm(Q_attn, K)
                  |
                  v
              RoPE(Q, K)
                  |
                  v
           KV Cache Write ──> GatedAttention
                  |
                  v
           O projection ──> hidden
```

## Key Features

| Feature | Implementation |
|---------|---------------|
| Gated Q | Q output is 2x dim, de-interleaved into Q_attn and Q_gate |
| Q/K norms | Per-head RMSNorm after projection |
| GQA | 32 query heads / 8 KV heads (4:1 ratio) |
| Thinking mode | Model generates `<think>...</think>` before responding |
| RoPE theta | 1,000,000 (long context capable) |

## What Worked

- **Q8_0 block repacking**: All Q8_0 weights repacked to 36-byte aligned blocks regardless of K dimension. Critical for the 8B model where many weights have K < 2048.
- **Tiled flash attention**: Online softmax with 256-position tiles. Constant shared memory regardless of context length.
- **Per-model tool prompts**: Qwen3 expects specific preamble text (`# Tools\n\nYou may call one or more functions...`) for tool calling. Using the Qwen3 tokenizer.chat_template format improves tool-following quality.

## What Didn't Work

- **Generic tool prompt**: The default tool block format (`<tools>` without the Qwen3-specific preamble) causes quality degradation with long tool definitions. The model was trained on a specific format.
- **Partial vocab without remapper for large models**: The vocab remapper is important for models with large vocabularies (151K tokens). Without it, argmax searches the full vocab unnecessarily.

## Benchmarks (RTX 5080)

| Model | Quant | CUDA tok/s | llama.cpp | vs llama.cpp |
|-------|-------|--------:|--------:|--------:|
| Qwen3-8B | Q8_0 | 91 | 92 | 99% |
| Qwen3-8B | Q4_K_M | 127 | 138 | 92% |
| Bonsai-8B | Q1_0_g128 | 90 | — | — |

### Bonsai-8B (1-bit Qwen3-8B)

PrismML's Bonsai-8B is Qwen3-8B quantized to 1-bit (Q1_0_g128). 8B parameters in 1.1 GB.

```
  Q1_0_g128 kernel optimizations:
  ================================

  Naive:      25.7 tok/s  (per-bit branching)
  Optimized:  90.2 tok/s  (3.5x speedup)

  Key optimizations:
  - Float4 activation loads via __ldg (read-only texture cache)
  - 4-row multi-output blocks (activation reuse across rows)
  - Branchless FMA sign multiply: (2*bit - 1) * scale
  - Scale hoisted outside inner loop (1 mul per 128 elements)
  - 32 sign bits loaded as uint32, processed in float4 groups
```

## Supported Models

| Model | Params | Quants | Notes |
|-------|--------|--------|-------|
| Qwen3-8B | 8B | Q8_0, Q4_K_M | Standard Qwen3 |
| Bonsai-8B | 8B | Q1_0_g128 | 1-bit, 1.1 GB, 90 tok/s |
