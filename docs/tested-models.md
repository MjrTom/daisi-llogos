# Tested Models

Models verified with daisi-llogos across CPU, CUDA, and Vulkan backends.

## Fully Working

These models produce correct, coherent output on all available backends.

| Model | Architecture | Params | Quantization | Backends | Download |
|-------|-------------|--------|-------------|----------|----------|
| [Qwen3-8B](https://huggingface.co/unsloth/Qwen3-8B-GGUF) | qwen3 | 8B | Q8_0, Q4_K_M | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf) (8.7 GB), [Q4_K_M](https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf) (5.0 GB) |
| [DeepSeek R1 Distill Llama 8B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF) | llama | 8B | Q8_0 | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf) (8.5 GB) |
| [Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) | qwen35 (hybrid DeltaNet) | 0.8B | Q8_0 | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf) (812 MB) |
| [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | qwen35 (hybrid DeltaNet) | 9B | Q8_0, Q4_K_M | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q8_0.gguf) (9.8 GB), [Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf) (5.8 GB) |
| [TinyLlama 1.1B Chat v1.0](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) | llama | 1.1B | Q8_0 | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf) (1.1 GB) |
| [BitNet b1.58 (ggml-model-i2_s)](https://huggingface.co/1bitLLM/bitnet_b1_58-large) | bitnet-b1.58 | 0.7B | I2_S (ternary) | CPU, CUDA | Custom build (1.2 GB) |

## Performance (NVIDIA RTX 5080)

Measured with `--bench`, 128 decode tokens, FP16 KV cache. llama.cpp b8461 comparison included.

### Llogos vs llama.cpp — CUDA

| Model | llama.cpp CUDA | Llogos CUDA | % of llama.cpp |
|-------|--------:|--------:|--------:|
| Qwen3.5-0.8B Q8_0 | 423 | 249 | 59% |
| Qwen3-8B Q8_0 | 91 | 78 | 86% |
| Qwen3-8B Q4_K_M | 139 | 81 | 58% |
| Qwen3.5-9B Q8_0 | 83 | 73 | 88% |

### Llogos vs llama.cpp — Vulkan

| Model | llama.cpp Vulkan | Llogos Vulkan | % of llama.cpp |
|-------|--------:|--------:|--------:|
| Qwen3.5-0.8B Q8_0 | 476 | 151 | 32% |
| Qwen3-8B Q8_0 | 97 | 55 | 57% |
| Qwen3-8B Q4_K_M | 142 | 53 | 37% |
| Qwen3.5-9B Q8_0 | 89 | 51 | 57% |

### All Llogos Benchmarks

| Model | Backend | Decode (tok/s) |
|-------|---------|---------------:|
| Qwen3.5-0.8B Q8_0 | CUDA | 249 |
| Qwen3.5-0.8B Q8_0 | Vulkan | 151 |
| Qwen3-8B Q8_0 | CUDA | 78 |
| Qwen3-8B Q8_0 | Vulkan | 55 |
| Qwen3-8B Q4_K_M | CUDA | 81 |
| Qwen3-8B Q4_K_M | Vulkan | 53 |
| Qwen3.5-9B Q8_0 | CUDA | 73 |
| Qwen3.5-9B Q8_0 | Vulkan | 51 |
| Qwen3.5-0.8B Q8_0 | CPU | 22 |
| TinyLlama 1.1B Q8_0 | CPU | 13 |

### CUDA Optimizations Applied

- **PTX inline assembly** for fp16↔fp32 conversion (single `cvt` instruction vs 20-instruction software)
- **`__ldg` read-only cache hints** on all activation and weight loads
- **uint32 weight reads** for aligned Q8_0 blocks (native 4-byte coalesced loads)
- **Multi-row activation reuse** — process 4-8 output neurons per block, load activation once
- **Aligned Q8_0 repacking** — 36-byte blocks with 4-byte aligned quants for direct int loads
- **cuBLAS** SGEMV for F32 matmul (part of CUDA Toolkit, no extra dependency)
- **GPU-side argmax** — download 4 bytes instead of 600KB per token
- **Fused kernels** — RmsNormResidual, SwiGLU, AddRmsNorm (saves ~108 launches/token)
- **GPU-native DeltaNet ops** — SplitUnequalQKV, RepeatTile on device (no CPU round-trips)
- **Architecture-specific NVRTC** — compiles for detected GPU compute capability
- **PTX disk cache** — skip NVRTC JIT on repeat startup (~0.6s faster)
- **Candidate-based sampler** — O(k) sort instead of O(N log N) for 152K vocab

### Vulkan Optimizations Applied

- **uint32 buffer view** — native 4-byte coalesced reads via `uint weight_u32[]` binding
- **Aligned Q8_0 repacking** — 34-byte → 36-byte blocks for uint32-aligned quant reads (+20% for Q8_0)
- **8 rows/workgroup** for aligned Q8_0, Q4_K, Q6_K (activation reuse across rows)
- **Subgroup arithmetic reduction** — `subgroupAdd` replaces shared memory reduction tree
- **Fused composite ops** — RmsNormResidual, AddRmsNorm, SplitSwiGLU, RepeatTile, ArgMax on GPU
- **GPU-side ArgMax** — avoids 600KB logit download per token (+20% for small models)
- **Lazy transfer barriers** — CopyTensorRegion defers barrier to next compute dispatch
- **Pipeline bind caching** within batched command buffers
- **Vulkan 1.2** with int8/fp16 features, SPIR-V 1.3

## Supported Quantization Formats

### Native GPU Kernels (CUDA + Vulkan)

These formats have dedicated GPU kernels for maximum performance:

| Format | MatMul | EmbeddingLookup | Notes |
|--------|--------|-----------------|-------|
| F32 | GPU (cuBLAS/uint32) | GPU | Full precision |
| F16 | GPU (uint32 pairs) | GPU | Half precision |
| Q8_0 | GPU (aligned uint32) | GPU | 8-bit — recommended |
| Q4_K | GPU (uint32) | GPU | 4-bit K-quant |
| Q5_K | GPU | — | 5-bit K-quant |
| Q6_K | GPU | — | 6-bit K-quant |
| I2_S | GPU | — | BitNet ternary |
| TQ1_0 | GPU | — | Ternary base-3 |

### Generic Fallback (CPU dequant)

Any format with dequantization support works via automatic CPU fallback. This is functional but slower than native GPU kernels.

| Format | Dequant | Notes |
|--------|---------|-------|
| Q4_0 | Yes | Simple 4-bit |
| Q4_1 | Yes | 4-bit with min |
| Q5_0 | Yes | 5-bit |
| Q5_1 | Yes | 5-bit with min |
| Q2_K | Yes | 2-bit K-quant |
| Q3_K | Yes | 3-bit K-quant |
| BF16 | Yes | Brain float16 |

## Known Issues

See [known-issues.md](known-issues.md) for details. All original issues are **fixed**. Remaining items are performance optimization opportunities.

## Recommended Models

For the best experience:

1. **Best quality**: [Qwen3-8B Q8_0](https://huggingface.co/unsloth/Qwen3-8B-GGUF) — 8.7 GB, 78 tok/s CUDA / 55 tok/s Vulkan
2. **Best speed**: [Qwen3.5-0.8B Q8_0](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) — 812 MB, 249 tok/s CUDA / 151 tok/s Vulkan
3. **Best value**: [Qwen3-8B Q4_K_M](https://huggingface.co/unsloth/Qwen3-8B-GGUF) — 5.0 GB, 81 tok/s CUDA / 53 tok/s Vulkan
4. **DeltaNet hybrid**: [Qwen3.5-9B Q8_0](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) — 9.8 GB, 73 tok/s CUDA / 51 tok/s Vulkan
5. **Reasoning**: [DeepSeek R1 8B Q8_0](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF) — 8.5 GB, ~65 tok/s CUDA
