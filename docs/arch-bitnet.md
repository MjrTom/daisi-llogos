# Architecture: BitNet b1.58 (Ternary Weights)

GGUF architecture: `bitnet-b1.58`

## Overview

BitNet b1.58 uses ternary weights ({-1, 0, +1}) stored in the I2_S format (2 bits per weight). This enables extreme compression and potentially fast inference through integer-only arithmetic.

```
  I2_S Weight Encoding
  =====================

  Each byte stores 4 ternary values (2 bits each):

  Byte: [7:6] [5:4] [3:2] [1:0]
         |      |      |      |
         v      v      v      v
      elem+0  elem+32 elem+64 elem+96
      *32     *32     *32     *32

  Encoding: 00 = -1, 01 = 0, 10 = +1, 11 = 0

  128-element interleaved groups
  Per-tensor float32 scale at byte offset (nelements/4)
```

## Key Features

| Feature | Implementation |
|---------|---------------|
| I2_S quantization | 2-bit packed ternary with per-tensor scale |
| Interleaved groups | 128 elements in 32-byte groups |
| Per-tensor scale | Single float32 scale for entire tensor |
| Special model loader | `BitNetModelLoader` for I2_S weight handling |

## What Worked

- **Dedicated CUDA kernel**: `dequant_matmul_i2s` with branchless ternary decoding (`(c == 2) - (c == 0)`). Avoids branching in the inner loop.
- **AVX2 CPU kernel**: `DotI2SAvx2` with SIMD ternary decode for CPU inference.

## What Didn't Work

- **Standard model loading**: BitNet requires a separate loading path (`BitNetModelLoader`) because the I2_S format has a per-tensor scale at the end of the packed data, unlike per-block scales in other formats.

## Benchmarks (RTX 5080)

| Model | Quant | CUDA tok/s | Notes |
|-------|-------|--------:|-------|
| BitNet b1.58 0.7B | I2_S | ~150 | Ternary model |

## Supported Models

| Model | Params | Quants | Notes |
|-------|--------|--------|-------|
| BitNet b1.58 large | 0.7B | I2_S (ternary) | Custom GGUF build required |
