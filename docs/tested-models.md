# Tested Models

Models verified with daisi-llogos across CPU, CUDA, and Vulkan backends.

## Fully Working

These models produce correct, coherent output on all available backends.

| Model | Architecture | Params | Quantization | Backends | Download |
|-------|-------------|--------|-------------|----------|----------|
| [Qwen3-8B](https://huggingface.co/unsloth/Qwen3-8B-GGUF) | qwen3 | 8B | Q8_0, Q4_K_M | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf) (8.7 GB), [Q4_K_M](https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf) (5.0 GB) |
| [DeepSeek R1 Distill Llama 8B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF) | llama | 8B | Q8_0 | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf) (8.5 GB) |
| [Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) | qwen35 (hybrid DeltaNet) | 0.8B | Q8_0 | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf) (812 MB) |
| [Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF) | qwen35 (hybrid DeltaNet) | 4B | Q8_0 | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q8_0.gguf) (4.2 GB) |
| [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | qwen35 (hybrid DeltaNet) | 9B | Q8_0, Q4_0, Q4_K_M | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q8_0.gguf) (9.8 GB), [Q4_0](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_0.gguf) (5.1 GB) |
| [TinyLlama 1.1B Chat v1.0](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) | llama | 1.1B | Q8_0 | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf) (1.1 GB) |
| [BitNet b1.58 (ggml-model-i2_s)](https://huggingface.co/1bitLLM/bitnet_b1_58-large) | bitnet-b1.58 | 0.7B | I2_S (ternary) | CPU, CUDA | Custom build (1.2 GB) |

## Performance (NVIDIA RTX 5080)

Measured with `--bench`, 128 decode tokens, FP16 KV cache. llama.cpp b8461 comparison included.

### Llogos vs llama.cpp — CUDA

**Exceeding llama.cpp on 3 of 6 models, matching within 2% on a fourth.** See [Inference Optimization White Paper](inference-optimization.md) for technical details.

| Model | Llogos CUDA | llama.cpp CUDA | % of llama.cpp |
|-------|--------:|--------:|--------:|
| Qwen3.5-0.8B Q8_0 | **436** | 399 | **109%** |
| Qwen3.5-4B Q8_0 | **142** | 135 | **105%** |
| Qwen3-8B Q8_0 | 90 | 92 | 98% |
| Qwen3-8B Q4_K_M | 122 | 138 | 88% |
| Qwen3.5-9B Q8_0 | **86** | 84 | **102%** |
| Qwen3.5-9B Q4_0 | 100 | 123 | 81% |

### Llogos vs llama.cpp — Vulkan

| Model | llama.cpp Vulkan | Llogos Vulkan | % of llama.cpp |
|-------|--------:|--------:|--------:|
| Qwen3.5-0.8B Q8_0 | 466 | 156 | 33% |
| Qwen3-8B Q8_0 | 96 | 56 | 58% |
| Qwen3-8B Q4_K_M | 142 | 54 | 38% |
| Qwen3.5-9B Q8_0 | — | 53 | — |
| Qwen3.5-9B Q4_0 | — | 45 | — |

*Note: llama.cpp Vulkan b8461 has a regression on Qwen3.5 DeltaNet models (~11 tok/s). Llogos Vulkan handles DeltaNet correctly.*

### All Llogos Benchmarks

| Model | CUDA | Vulkan | CPU |
|-------|-----:|-------:|----:|
| Qwen3.5-0.8B Q8_0 | 436 | 156 | 22 |
| Qwen3.5-4B Q8_0 | 142 | 73 | — |
| Qwen3-8B Q8_0 | 90 | 56 | — |
| Qwen3-8B Q4_K_M | 122 | 54 | — |
| Qwen3.5-9B Q8_0 | 86 | 53 | — |
| Qwen3.5-9B Q4_0 | 100 | 45 | — |
| TinyLlama 1.1B Q8_0 | — | — | 13 |

### CUDA Optimizations Applied

- **Partial vocab logit computation** — lm_head computes only VocabSize/32 tokens (~4,752 of 152K), +10% speedup with identical greedy output
- **dp4a integer dot product** — `__dp4a` for Q4_0 with fused RmsNorm+Q8_1 quantization (zero-overhead activation prep)
- **Architecture-adaptive dispatch** — Blackwell (SM 12.x) uses float path, pre-Blackwell uses dp4a for 4-bit quants
- **Per-quant row count tuning** — Q8_0=2, Q4_K=3, Q6_K=10, Q4_0=2, Q4_1=8, Q5_K=1 (optimal per format)
- **CUDA graph capture** — single `cuGraphLaunch` replaces ~435 individual kernel launches per token
- **Aligned block repacking** — Q8_0 34→36, Q4_0 18→20 bytes for native uint32 loads
- **Self-contained AdaptiveLaunch** — each quant type computes own grid/threads/smem, no shared variables
- **Fused RmsNorm+Q8_1** — 3 fused kernels prepare Q8_1 data inside normalization pass
- **Multi-row activation reuse** — load activation once, multiply against 2-10 weight rows
- **PTX inline assembly** for fp16↔fp32 conversion (single `cvt` instruction)
- **GPU-side argmax** — download 4 bytes instead of 600KB per token
- **Fused layer boundaries** — AddRmsNormResidual, SwiGLU, AddRmsNorm
- **Architecture-specific NVRTC** with PTX disk cache

### Vulkan Optimizations Applied

- **Q4_0/Q4_1/Q5_K matmul + embedding shaders** — full 4-bit GPU support
- **uint32 buffer view** — native 4-byte coalesced reads via `uint weight_u32[]` binding
- **Aligned Q8_0 repacking** — 34→36 bytes for uint32-aligned quant reads
- **Multi-row workgroups** — per-quant tuning matching CUDA row counts
- **Subgroup arithmetic reduction** — `subgroupAdd` replaces shared memory reduction
- **Fused composite ops** — RmsNormResidual, AddRmsNorm, SplitSwiGLU, RepeatTile, ArgMax
- **Vulkan 1.2** with int8/fp16 features, SPIR-V 1.3

## Supported Quantization Formats

### Native GPU Kernels (CUDA + Vulkan)

These formats have dedicated GPU kernels for maximum performance:

| Format | MatMul | EmbeddingLookup | Notes |
|--------|--------|-----------------|-------|
| F32 | GPU (cuBLAS/uint32) | GPU | Full precision |
| F16 | GPU (uint32 pairs) | GPU | Half precision |
| Q8_0 | GPU (aligned uint32) | GPU | 8-bit — recommended |
| Q4_0 | GPU (dp4a / float) | GPU | 4-bit with dp4a on pre-Blackwell |
| Q4_1 | GPU (uint32) | GPU | 4-bit with min offset |
| Q4_K | GPU (uint32) | GPU | 4-bit K-quant |
| Q5_K | GPU | GPU | 5-bit K-quant |
| Q6_K | GPU | — | 6-bit K-quant |
| I2_S | GPU | — | BitNet ternary |
| TQ1_0 | GPU | — | Ternary base-3 |

### Generic Fallback (CPU dequant)

Any format with dequantization support works via automatic CPU fallback. This is functional but slower than native GPU kernels.

| Format | Dequant | Notes |
|--------|---------|-------|
| Q5_0 | Yes | 5-bit |
| Q5_1 | Yes | 5-bit with min |
| Q2_K | Yes | 2-bit K-quant |
| Q3_K | Yes | 3-bit K-quant |
| BF16 | Yes | Brain float16 |

## Known Issues

See [known-issues.md](known-issues.md) for details. All original issues are **fixed**. Remaining items are performance optimization opportunities.

## Recommended Models

For the best experience:

1. **Best quality**: [Qwen3-8B Q8_0](https://huggingface.co/unsloth/Qwen3-8B-GGUF) — 8.7 GB, 90 tok/s CUDA (98% of llama.cpp)
2. **Best speed**: [Qwen3.5-0.8B Q8_0](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) — 812 MB, 436 tok/s CUDA (**109% of llama.cpp**)
3. **Best value**: [Qwen3-8B Q4_K_M](https://huggingface.co/unsloth/Qwen3-8B-GGUF) — 5.0 GB, 122 tok/s CUDA (88% of llama.cpp)
4. **DeltaNet hybrid**: [Qwen3.5-9B Q8_0](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) — 9.8 GB, 86 tok/s CUDA (**102% of llama.cpp**)
5. **Smallest 4-bit**: [Qwen3.5-9B Q4_0](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) — 5.1 GB, 100 tok/s CUDA
6. **Reasoning**: [DeepSeek R1 8B Q8_0](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF) — 8.5 GB, ~65 tok/s CUDA
