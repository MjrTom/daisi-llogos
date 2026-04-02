# Architecture: Qwen 2 / 2.5 (Attention Biases)

GGUF architecture: `qwen2`

## Overview

Qwen 2 and 2.5 are standard transformers with Grouped Query Attention. The key difference from Qwen 3+ is that Q, K, and V projections have **learned bias vectors** added after the linear projection.

```
  Attention with Biases (Qwen2-specific)
  =======================================

  normOut ──> Q projection ──> + Q_bias ──> RoPE ──> ...
              K projection ──> + K_bias ──> RoPE ──> KV cache
              V projection ──> + V_bias ──────────> KV cache
```

## Key Features

| Feature | Implementation |
|---------|---------------|
| Attention biases | Q/K/V projections have per-dimension bias vectors |
| GQA | Grouped Query Attention |
| ChatML | `<\|im_start\|>role\ncontent<\|im_end\|>` format |
| No gated Q | Standard Q projection (no DeInterleaveQ) |
| No Q/K norms | No per-head RMSNorm on Q/K |

## What Worked

- **Loading biases from GGUF**: `attn_q.bias`, `attn_k.bias`, `attn_v.bias` tensors loaded via `TryLoadTensor` in both `ModelLoader` and `MmapModelLoader`.
- **ElementAdd for bias application**: Simple `ElementAdd(q, q, q_bias)` after projection, before RoPE. Applied in the common attention path so both single-token and batched prefill benefit.
- **WebGPU parity**: The WebGPU backend already supported biases — the .NET fix brought CPU/CUDA/Vulkan to parity.

## What Didn't Work

- **Without biases**: Model produced complete garbage output. The bias vectors contain critical offset information that the model depends on. Even small models (0.5B) fail completely without them.

## Benchmarks (RTX 5080)

| Model | Quant | CUDA tok/s | Notes |
|-------|-------|--------:|-------|
| Qwen2.5-0.5B | Q8_0 | ~200 | Small model, fast inference |

## Supported Models

| Model | Params | Quants | Notes |
|-------|--------|--------|-------|
| Qwen2.5-0.5B | 0.5B | Q8_0 | Requires attention biases |
