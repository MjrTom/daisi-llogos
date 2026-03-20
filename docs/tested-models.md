# Tested Models

Models verified with daisi-llogos across CPU, CUDA, and Vulkan backends.

## Fully Working

These models produce correct, coherent output on all available backends.

| Model | Architecture | Params | Quantization | Backends | Download |
|-------|-------------|--------|-------------|----------|----------|
| [Qwen3-8B](https://huggingface.co/unsloth/Qwen3-8B-GGUF) | qwen3 | 8B | Q8_0, Q4_K_M | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf) (8.7 GB), [Q4_K_M](https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf) (5.0 GB) |
| [DeepSeek R1 Distill Llama 8B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF) | llama | 8B | Q8_0 | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf) (8.5 GB) |
| [Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) | qwen35 (hybrid DeltaNet) | 0.8B | Q8_0 | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf) (812 MB) |
| [Qwen3.5-9B](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) | qwen35 (hybrid DeltaNet) | 9B | Q8_0, Q4_K_M | CPU, CUDA | [Q8_0](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q8_0.gguf) (9.8 GB), [Q4_K_M](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf) (5.8 GB) |
| [TinyLlama 1.1B Chat v1.0](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) | llama | 1.1B | Q8_0 | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf) (1.1 GB) |
| [BitNet b1.58 (ggml-model-i2_s)](https://huggingface.co/1bitLLM/bitnet_b1_58-large) | bitnet-b1.58 | 0.7B | I2_S (ternary) | CPU, CUDA | Custom build (1.2 GB) |

## Performance (NVIDIA RTX 5080)

Measured with `--bench`, 32–64 decode tokens, FP16 KV cache.

| Model | Backend | Prefill (tok/s) | Decode (tok/s) |
|-------|---------|----------------:|---------------:|
| Qwen3.5-0.8B Q8_0 | CUDA | 115 | 218 |
| Qwen3-8B Q8_0 | CUDA | 67 | 68 |
| Qwen3-8B Q4_K_M | CUDA | 49 | 50 |
| Qwen3.5-9B Q8_0 | CUDA | 57 | 62 |
| Qwen3.5-9B Q4_K_M | CUDA | 37 | 38 |
| Qwen3.5-0.8B Q8_0 | CPU | 15 | 22 |
| TinyLlama 1.1B Q8_0 | CPU | 5 | 13 |
| DeepSeek R1 8B Q8_0 | CPU | 2 | 3 |

### CUDA Optimizations Applied

- **`__dp4a` integer dot product** for Q8_0 matmul (inspired by llama.cpp's approach)
- **Aligned Q8_0 repacking** — 36-byte blocks with 4-byte aligned quants for direct int loads
- **Q8_1 activation cache** — quantize once, reuse across Q/K/V and gate/up matmuls
- **cuBLAS** SGEMV for F32 matmul (part of CUDA Toolkit, no extra dependency)
- **GPU-side argmax** — download 4 bytes instead of 600KB per token
- **Fused kernels** — RmsNormResidual, SwiGLU, AddRmsNorm (saves ~108 launches/token)
- **GPU-native DeltaNet ops** — SplitUnequalQKV, RepeatTile on device (no CPU round-trips)
- **Architecture-specific NVRTC** — compiles for detected GPU compute capability
- **PTX disk cache** — skip NVRTC JIT on repeat startup (~0.6s faster)
- **Candidate-based sampler** — O(k) sort instead of O(N log N) for 152K vocab

## Supported Quantization Formats

### Native GPU Kernels (CUDA + Vulkan)

These formats have dedicated GPU kernels for maximum performance:

| Format | MatMul | EmbeddingLookup | Notes |
|--------|--------|-----------------|-------|
| F32 | GPU (cuBLAS) | GPU | Full precision, cuBLAS SGEMV |
| F16 | GPU | GPU | Half precision |
| Q8_0 | GPU (__dp4a) | GPU | 8-bit, `__dp4a` int dot product — recommended |
| Q4_K | GPU | GPU | 4-bit K-quant |
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

See [known-issues.md](known-issues.md) for details. All 5 original issues are **fixed**.

## Recommended Models

For the best experience:

1. **Best quality**: [Qwen3-8B Q8_0](https://huggingface.co/unsloth/Qwen3-8B-GGUF) — 8.7 GB, 68 tok/s on CUDA
2. **Best speed**: [Qwen3.5-0.8B Q8_0](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) — 812 MB, 218 tok/s on CUDA
3. **Best value**: [Qwen3-8B Q4_K_M](https://huggingface.co/unsloth/Qwen3-8B-GGUF) — 5.0 GB, 50 tok/s on CUDA (half the VRAM of Q8_0)
4. **DeltaNet hybrid**: [Qwen3.5-9B Q8_0](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) — 9.8 GB, 62 tok/s on CUDA
5. **Reasoning**: [DeepSeek R1 8B Q8_0](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF) — 8.5 GB, ~65 tok/s on CUDA
