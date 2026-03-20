# Tested Models

Models verified with daisi-llogos across CPU, CUDA, and Vulkan backends.

## Fully Working

These models produce correct, coherent output on all available backends.

| Model | Architecture | Params | Quantization | Backends | Download |
|-------|-------------|--------|-------------|----------|----------|
| [Qwen3-8B](https://huggingface.co/unsloth/Qwen3-8B-GGUF) | qwen3 | 8B | Q8_0 | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q8_0.gguf) (8.7 GB) |
| [DeepSeek R1 Distill Llama 8B](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF) | llama | 8B | Q8_0 | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q8_0.gguf) (8.5 GB) |
| [Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) | qwen35 (hybrid DeltaNet) | 0.8B | Q8_0 | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/resolve/main/Qwen3.5-0.8B-Q8_0.gguf) (812 MB) |
| [TinyLlama 1.1B Chat v1.0](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) | llama | 1.1B | Q8_0 | CPU, CUDA, Vulkan | [Q8_0](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf) (1.1 GB) |
| [BitNet b1.58 (ggml-model-i2_s)](https://huggingface.co/1bitLLM/bitnet_b1_58-large) | bitnet-b1.58 | 0.7B | I2_S (ternary) | CPU, CUDA | Custom build (1.2 GB) |

## Performance (NVIDIA RTX 5080)

Measured with `--bench`, 256 decode tokens.

| Model | Backend | Prefill (tok/s) | Decode (tok/s) |
|-------|---------|----------------:|---------------:|
| Qwen3-8B Q8_0 | CUDA | 24.6 | 17.1 |
| DeepSeek R1 8B Q8_0 | CUDA | 25.3 | 19.0 |
| Qwen3.5-0.8B Q8_0 | CUDA | 132.4 | 29.5 |
| TinyLlama 1.1B Q8_0 | CPU | 5.2 | 13.4 |
| Qwen3.5-0.8B Q8_0 | CPU | 5.5 | 8.5 |
| DeepSeek R1 8B Q8_0 | CPU | 2.2 | 3.2 |

## Supported Quantization Formats

### Native GPU Kernels (CUDA + Vulkan)

These formats have dedicated GPU kernels for maximum performance:

| Format | MatMul | EmbeddingLookup | Notes |
|--------|--------|-----------------|-------|
| F32 | GPU | GPU | Full precision |
| F16 | GPU | GPU | Half precision |
| Q8_0 | GPU | GPU | 8-bit quantized — recommended |
| Q4_K | GPU | GPU | 4-bit K-quant (correctness issue — see Known Issues) |
| Q5_K | GPU | — | 5-bit K-quant (correctness issue) |
| Q6_K | GPU | — | 6-bit K-quant (correctness issue) |
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

### K-Quant Models (Q4_K_M, Q5_K_M, Q6_K)

Models quantized with K-quant formats (Q4_K, Q5_K, Q6_K) produce incorrect output when used for full model inference, despite individual dequantization operations being verified correct against the ggml reference implementation. The error appears to accumulate across layers in deep networks (36+ layers). **Use Q8_0 quantization for reliable results.**

### Qwen3.5-9B DeltaNet Architecture

The Qwen3.5-9B model uses a DeltaNet variant with an unequal Q/K/V split (Q=2048, K=2048, V=4096) and head repeat-interleave (16→32 heads) that differs from the 0.8B model. This architecture is not yet fully implemented. The Qwen3.5-0.8B model (which uses a standard 3-way equal QKV split) works correctly.

## Recommended Models

For the best experience:

1. **Best quality**: [Qwen3-8B Q8_0](https://huggingface.co/unsloth/Qwen3-8B-GGUF) — 8.7 GB, 17-25 tok/s on CUDA
2. **Best speed**: [Qwen3.5-0.8B Q8_0](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) — 812 MB, 30-132 tok/s on CUDA
3. **Reasoning**: [DeepSeek R1 8B Q8_0](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF) — 8.5 GB, 19-25 tok/s on CUDA
