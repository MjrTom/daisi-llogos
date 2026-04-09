# LoRA Training

Native LoRA (Low-Rank Adaptation) fine-tuning for GGUF models. Train small adapter weights to inject domain-specific knowledge into quantized models without modifying base weights.

## Overview

```
daisi-llogos train --model Qwen3.5-0.8B-Q8_0.gguf \
                    --data training-data.jsonl \
                    --rank 8 --targets qkvofd --backend cuda
```

LoRA decomposes weight updates into low-rank matrices: `W' = W + scaling * B @ A` where A is `[rank x in]` and B is `[out x rank]`. Only A and B are trained (typically <1% of model parameters).

## Architecture

```
                    Training Pipeline
 +--------------------------------------------------+
 |                                                    |
 |  GGUF Model          Training Data                |
 |  (quantized)         (JSONL/text)                  |
 |       |                    |                       |
 |       v                    v                       |
 |  +-----------+      +-----------+                  |
 |  | ModelWeights|     | Tokenizer |                  |
 |  +-----------+      +-----------+                  |
 |       |                    |                       |
 |       v                    v                       |
 |  +----------------------------------------+       |
 |  |         Training Forward Pass           |       |
 |  |  +----------------------------------+   |       |
 |  |  | For each token:                   |   |       |
 |  |  |   Embedding -> Layers -> Logits   |   |       |
 |  |  |         |                         |   |       |
 |  |  |   LoRA applied at:               |   |       |
 |  |  |   - Attention Q/K/V/O            |   |       |
 |  |  |   - DeltaNet QKV/Out             |   |       |
 |  |  |   - FFN Gate/Up/Down             |   |       |
 |  |  +----------------------------------+   |       |
 |  +----------------------------------------+       |
 |       |                                            |
 |       v                                            |
 |  +------------+    +----------+    +---------+     |
 |  | Cross-Entropy|-->| Backward |-->| AdamW   |     |
 |  | Loss         |   | Pass     |   | Optimizer|    |
 |  +------------+    +----------+    +---------+     |
 |                                         |          |
 |                                         v          |
 |                                    +---------+     |
 |                                    | .llra   |     |
 |                                    | adapter |     |
 |                                    +---------+     |
 +--------------------------------------------------+
```

## LoRA Targets

The `--targets` flag controls which projections get LoRA adapters:

| Flag | Target | Description |
|------|--------|-------------|
| `q` | Attention Q | Query projection (standard attention layers) |
| `k` | Attention K | Key projection (standard attention layers) |
| `v` | Attention V | Value projection (standard attention layers) |
| `o` | Attention O | Output projection (standard attention layers) |
| `f` | FFN Gate/Up/Down | Feed-forward network (all layers) |
| `d` | DeltaNet QKV/Out | DeltaNet projections (hybrid models) |

**Recommended:** `--targets qkvofd` for Qwen 3.5 (covers all layer types).

## LoRA Forward Pass

```
     Standard Weight Path              LoRA Path
     ==================              =========

     input [M x K]                   input [M x K]
         |                               |
         v                               v
    W [K x N]                       A [rank x K]
    (frozen,                        (trainable)
     quantized)                          |
         |                               v
         v                          inter [M x rank]
    base_out [M x N]                     |
         |                               v
         |                          B [N x rank]
         |                          (trainable)
         |                               |
         |                               v
         +--------> output = base_out + scaling * lora_out
```

## DeltaNet Training

Qwen 3.5 uses a hybrid architecture with both standard attention and DeltaNet layers. The training forward pass handles both:

```
   Layer Type Detection
   ====================

   For each layer (0..23):
       |
       +-- Standard Attention? ---> Batched attention forward
       |   (layers 3,7,11,15,19,23)  + LoRA on Q/K/V/O
       |                              + full backward
       |
       +-- DeltaNet? ------------> Real DeltaNet forward
           (layers 0,1,2,4,5,...)    (token-by-token recurrence)
                                     + gradient through residual
                                     + FFN LoRA trains on
                                       correct hidden states
```

### DeltaNet Forward (Training)

The training uses the **real DeltaNet forward** (not an approximation) to produce correct hidden states for FFN LoRA:

```
  normOut ──> QKV projection (F32 weights via cuBLAS)
                   |
                   v
            CausalConv1d ──> SiLU
                   |
                   v
            Split Q, K, V ──> L2Norm ──> RepeatTile
                   |
                   v
            Alpha/Beta projections
                   |
                   v
            ComputeDecayBeta ──> DeltaNetStep (state update)
                   |
                   v
            Gate + SiLUGate ──> Output projection
                   |
                   v
               hidden (correct, matches inference)
```

## GPU Training Pipeline

All training operations run on GPU via CUDA:

```
  +------------------+     +------------------+     +------------------+
  |   Forward Pass   |     |   Backward Pass  |     |    Optimizer     |
  |                  |     |                  |     |                  |
  | - Embedding      |     | - Cross-entropy  |     | - GPU AdamW      |
  | - All layers     | --> |   gradient       | --> | - Gradient clip  |
  | - LoRA forward   |     | - Layer backprop |     | - Weight update  |
  | - Cross-entropy  |     | - LoRA dA/dB     |     | - Zero gradients |
  |                  |     |                  |     |                  |
  +------------------+     +------------------+     +------------------+
        GPU                       GPU                      GPU
   (no CPU round-trip)      (no CPU round-trip)      (no CPU round-trip)
```

Key optimizations:
- **FP16 tensor core GemmEx**: Batch matmuls use `dequant_to_f16` + `cublasGemmEx` (tensor cores) with persistent FP16 weight cache
- **Batched DeltaNet projections**: 5 linear projections (QKV, Alpha, Beta, Gate, Out) batched for all T tokens; only conv1d + state update remain sequential
- **GPU AdamW**: Entire optimizer step on device with contiguous LoRA params (3 kernel launches)
- **Gradient clipping**: Computed on GPU via `GradNormSq` kernel

## Data Formats

### Plain Text
```
Raw text file. Tokenized with sliding window.
```

### JSONL
```json
{"text": "The full text to learn from."}
```

### Chat JSONL (Completion-Only)
```json
{"prompt": "What is X?", "completion": "X is the answer."}
```

With chat format, loss is computed only on completion tokens (prompt tokens are masked):

```
  Tokens:  [prompt prompt prompt | completion completion completion]
  Loss:    [masked masked masked | computed  computed  computed   ]
```

## Training Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rank` | 8 | LoRA rank (4, 8, 16, 32) |
| `--alpha` | 16 | LoRA scaling alpha (effective scale = alpha/rank) |
| `--lr` | 1e-4 | Learning rate |
| `--epochs` | 3 | Training epochs |
| `--seq-len` | 512 | Sequence length |
| `--warmup` | 50 | Warmup steps (linear ramp) |
| `--weight-decay` | 0.01 | AdamW weight decay |
| `--targets` | qkvo | Projection targets (see table above) |
| `--backend` | cpu | Compute backend (cpu, cuda) |

## Inference with LoRA

After training, merge the adapter into the model for full-speed inference:

```bash
daisi-llogos --model Qwen3.5-0.8B-Q8_0.gguf \
             --lora trained-adapter.llra \
             --prompt "What did I train you on?"
```

The merge dequantizes base weights to F32, adds `scaling * B @ A`, and runs inference with the merged weights. No adapter overhead at runtime.

## Adapter File Format (.llra)

Binary format storing LoRA A/B matrices per layer:

```
  +------------------+
  | Header           |
  | - rank           |
  | - alpha          |
  | - layer count    |
  +------------------+
  | Layer 0          |
  | - name (string)  |
  | - A [rank x in]  |
  | - B [out x rank] |
  +------------------+
  | Layer 1          |
  | ...              |
  +------------------+
```

## Performance

Benchmarks on RTX 5080 with Qwen 3.5 0.8B Q8_0:

| Metric | seq-len 128 | seq-len 256 |
|--------|------------|------------|
| Training speed | 2.5 seq/s | 0.6 seq/s |
| Trainable params | 540K (rank=8, targets=qkvo) | 540K |
| GPU memory | ~2 GB | ~2.5 GB |
| Loss convergence | 2.0 → 0.92 (80 steps, 10 epochs) | Stable (2.82 avg) |

Training is 96x faster than CPU (0.026 seq/s).

## Bug Fixes Applied

### SwiGLU Q8_1 Scratch Overflow (Critical)
The fused `swiglu_q8_1` kernel wrote Q8_1 data for ALL M×N elements during batched FFN, but `_q8_1Scratch` was sized for single-token only. This caused GPU memory corruption and garbage output from all batched prefill and training forward passes with M>1. Fixed by checking `n <= _q8_1ScratchK` before using the fused path.

### Arena Corruption at seqLen>208
The training CudaArenaAllocator caused non-deterministic forward results at seqLen>208 (loss diverged from 2.7 to 23.0 even with LR=0). Arena disabled in favor of per-tensor Pool dict allocation until root cause in tensor adjacency is resolved.

### Non-Power-of-2 Attention Reduction
`training_causal_gated_attention` and its backward kernel launched with `threads=T`, but the shared-memory parallel reduction (`blockDim.x / 2`) drops elements when blockDim.x is not a power of 2. Fixed by rounding thread count to next power of 2.

### DeltaNet Arena Registrations
Arena registered DeltaNet single-token buffers with massively oversized dimensions (`T * ssmInner * 3 = 1.5M` for a tensor needing only `qkvDim = 6K`). Fixed by using actual weight dimensions from model tensors.

### Q8_0 Alignment (K < 2048)
The dp4a matmul kernel requires 36-byte aligned Q8_0 blocks. `LoadTensor` only repacked for K >= 2048, causing NaN on the 0.8B model (K=1024). Fixed by repacking all Q8_0 2D tensors.
