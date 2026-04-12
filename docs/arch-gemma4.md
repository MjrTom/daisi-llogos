# Architecture: Gemma 4 (Per-Layer Embeddings + Sliding-Window Attention)

GGUF architecture: `gemma4`

## Overview

Gemma 4 is Google's latest open model family. The E4B ("Efficient 4 Billion") variant uses a standard transformer with Per-Layer Embeddings (PLE), GeGLU activations, NEOX-style RoPE, and a mix of sliding-window and full-attention layers. KV cache sharing across non-KV layers significantly reduces memory usage.

```
  Layer structure (42 layers total):
  ==================================

  Layers 0..23: Own KV cache (7 full-attention + 17 sliding-window)
  Layers 24..41: Shared KV cache (reuse layers 22/23)

  Per token (before layer loop):
  ──────────────────────────────
  EmbeddingLookup(token_embd) + sqrt(hidden_dim) scale
  PLE Setup:
    per_layer_token_embd[token] * sqrt(n_embd_per_layer)
    per_layer_model_proj @ hidden * (1/sqrt(hidden_dim))
    PerHeadRmsNorm(per_layer_proj_norm)
    ElementAdd + scale(1/sqrt(2))

  Per layer:
  ──────────────────────────────
  RmsNorm → Q projection
            PerHeadRmsNorm(Q, Q_norm)
            K/V projection (layers with KV only)
            PerHeadRmsNorm(K, K_norm)
            PerHeadRmsNormUnit(V)     ← no learned weight
            RoPENeox / RoPENeoxWithFreqFactors
            KV Cache Write
  GatedAttention(Q, K_cache, V_cache)
  O projection
  Post-attn norm + residual

  RmsNorm → FFN:
    GeGLU(gate, up)   ← GeLU(tanh) instead of SiLU
    down projection
  Post-FFN norm + residual

  PLE Block:
    inp_gate @ hidden → GeluTanh → ElementMul(pleBase[layer])
    per_layer_proj @ gate → RmsNorm → residual add

  LayerOutScale (per-layer scalar)

  Final: RmsNorm → lm_head → LogitSoftcap(30.0)
```

## Key Features

| Feature | Implementation |
|---------|---------------|
| Per-Layer Embeddings (PLE) | Separate token embedding per layer (256-dim), projected and gated |
| GeGLU | GeLU(tanh approximation) replaces SiLU in the FFN gate |
| NEOX RoPE | Pairs at distance ropeDim/2 (not interleaved) |
| Proportional RoPE | Full-attention layers use frequency factor table |
| V-projection norm | PerHeadRmsNormUnit (no learned weight) |
| KV sharing | Layers 24..41 reuse KV cache from layers 22/23 |
| Sliding-window attention | 35 of 42 layers use sliding window |
| Logit softcap | Final logits clamped via 30.0 * tanh(logits / 30.0) |
| GQA | 8 query heads / 4 KV heads (2:1 ratio) |
| Head dims | 256 (sliding) / 512 (full attention) |

## What Worked

- **Q4_0x4 repacked weight format**: Interleave 4 weight rows with FP32 pre-converted scales. Eliminates per-block Half-to-float conversion in the inner tile. 1.36x faster than standard Q4_0 matmul at M=32.
- **VNNI 2x4 tile kernel**: AVX-VNNI `vpdpbusd` replaces sign+maddubs+madd chain on Alder Lake. Marginal gain in isolation but free on CPUs with VNNI support.
- **Q8_0_F32 activation format**: Store activation scale as FP32 (36 bytes/block vs 34) so the inner tile loop has zero Half-to-float operations.
- **Repacked path for all Q4_0 matmuls**: The Q4_0x4 kernel beats Q4_0 even at M=1 (10% faster), so it's enabled for FFN, attention Q/K/V/O, and decode.
- **Single-row embedding fallback**: CUDA embedding lookup for Q5_K per_layer_token_embd downloads only the ~7 KB row needed instead of the entire 1.6 GB table.

## What Didn't Work

- **4x4 register tile**: .NET JIT spills 16 YMM accumulators to the stack. 2x4 (8 accumulators) is the ceiling for RyuJIT.
- **Persistent thread pool**: Idle workers spin and burn memory bandwidth on Alder Lake E-cores. Parallel.For with 8 P-core affinity is faster.
- **4-accumulator BF16 dot**: No measurable improvement on the PLE BF16 matmuls — they're dispatch-overhead-bound, not FMA-throughput-bound.

## Benchmarks (RTX 5080 / Alder Lake)

| Model | Quant | CUDA tok/s | CPU tok/s |
|-------|-------|--------:|--------:|
| Gemma 4 E4B-it | Q4_0 | 75 | 13 |

### CPU Optimization History

Starting from the initial pure-C# implementation:

| Stage | Decode | Prefill (M=32) |
|-------|-------:|-------:|
| Baseline (scalar) | 1.2 tok/s | 1.1 tok/s |
| + SIMD int8 kernels + batched prefill | 5 tok/s | 10 tok/s |
| + 2Mx4N register-tiled gemm | 8 tok/s | 18 tok/s |
| + Q4_0x4 repacked format | 11 tok/s | 22 tok/s |
| + FP32 scales + Q8_0_F32 + repack all | 13 tok/s | 26 tok/s |

## Supported Models

| Model | Params | Quants | Notes |
|-------|--------|--------|-------|
| Gemma 4 E4B-it | 4B | Q4_0, Q8_0 | Text-only inference (vision/audio not supported) |

## New CUDA Kernels

Added for Gemma 4 support (also available to future architectures):

| Kernel | File | Purpose |
|--------|------|---------|
| `gelu_tanh` | elementwise.cu | GeLU activation (tanh approximation) |
| `geglu` | elementwise.cu | Fused GeGLU: GeLU(gate) * up |
| `logit_softcap` | elementwise.cu | Tanh-based logit clamping |
| `rope_neox` | elementwise.cu | NEOX-style RoPE (non-interleaved pairs) |
| `rope_neox_with_freq` | elementwise.cu | NEOX RoPE with frequency factor table |
| `rope_with_freq` | elementwise.cu | Standard RoPE with frequency factors |
| `per_head_rms_norm_unit` | composite_ops.cu | RmsNorm without learned weight |
