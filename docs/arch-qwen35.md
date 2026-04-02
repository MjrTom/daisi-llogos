# Architecture: Qwen 3.5 (Hybrid DeltaNet)

GGUF architecture: `qwen35`

## Overview

Qwen 3.5 is a hybrid transformer that interleaves standard gated attention layers with DeltaNet linear attention layers. This gives it O(1) memory per token for most layers while preserving full attention quality at regular intervals.

```
  Layer Schedule (24-layer 0.8B example)
  =======================================

  Layer:  0  1  2  3  4  5  6  7  8  9  10 11 ...
  Type:   D  D  D  A  D  D  D  A  D  D  D  A  ...

  D = DeltaNet (linear attention, O(1) memory)
  A = Standard Attention (full attention, O(n) memory)

  Pattern: every 4th layer is standard attention
  (FullAttentionInterval = 4)
```

## Key Features

| Feature | Implementation |
|---------|---------------|
| Gated Q | Q projection output is 2x, split into Q_attn + Q_gate via DeInterleaveQ |
| Q/K norms | Per-head RMSNorm on Q and K after projection |
| DeltaNet | Recurrent state-space model with decay-weighted delta rule |
| Causal conv1d | Applied to QKV before split in DeltaNet layers |
| GQA | Grouped Query Attention (e.g., 32 query heads, 8 KV heads) |

## DeltaNet Layer Forward

```
  Input (normOut)
       |
       v
  QKV Projection ──> CausalConv1d ──> SiLU
       |
       v
  Split Q, K, V (unequal: keyDim + keyDim + valueDim)
       |
       v
  L2NormGroups(Q, K) ──> RepeatTile (GQA expansion)
       |
       v
  Alpha/Beta Projections ──> ComputeDecayBeta
       |
       v
  DeltaNetStep (recurrent state update + query)
       |                    |
       |              State matrix [heads x headDim x headDim]
       |              (persistent across tokens, O(1) memory)
       v
  Gate Projection ──> SiLUGate
       |
       v
  Output Projection ──> hidden
```

## What Worked

- **Real DeltaNet forward for training**: Using the actual inference kernels (CausalConv1d, DeltaNetStep, etc.) produces correct hidden states. The earlier approximation (`silu(qkv) * silu(gate)`) produced loss 23 — completely wrong.
- **F32 weight dequant for training**: Caching dequantized F32 weights avoids Q8_0 dp4a alignment bugs and BatchMatMul dequant issues. No extra memory since backward already uses F32.
- **Token-by-token DeltaNet in training**: Processing DeltaNet layers sequentially (one token at a time) while batching attention+FFN layers matches inference behavior exactly.

## What Didn't Work

- **Approximate DeltaNet forward** (`silu(qkv) * silu(gate)` element-wise): Doesn't capture recurrent state dynamics. Produces completely different hidden states from the real model. Loss never converges.
- **DeltaNet backward through SSM**: Backpropagating through the recurrent state update (DeltaNetStep) requires BPTT through matrix operations with decay. Too complex for the benefit. Gradient through residual + FFN LoRA is sufficient for knowledge injection.

## Benchmarks (RTX 5080)

| Model | Quant | CUDA tok/s | llama.cpp | vs llama.cpp |
|-------|-------|--------:|--------:|--------:|
| Qwen3.5-0.8B | Q8_0 | 441 | 399 | **110%** |
| Qwen3.5-4B | Q8_0 | 144 | 135 | **107%** |
| Qwen3.5-9B | Q8_0 | 88 | 84 | **105%** |
| Qwen3.5-9B | Q4_0 | 101 | 123 | 82% |

## Supported Models

| Model | Params | Quants | LoRA Training |
|-------|--------|--------|---------------|
| Qwen3.5-0.8B | 0.8B | Q8_0, BF16 | Yes (4.3 seq/s, loss -> 0.001) |
| Qwen3.5-4B | 4B | Q8_0 | Untested |
| Qwen3.5-9B | 9B | Q8_0, Q4_0, Q4_K_M | Untested |
