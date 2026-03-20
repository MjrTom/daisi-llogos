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
| Qwen3.5-0.8B Q8_0 | CUDA | 131 | 80 |
| Qwen3-8B Q8_0 | CUDA | 53 | 41 |
| Qwen3-8B Q4_K_M | CUDA | 23 | 22 |
| Qwen3.5-9B Q8_0 | CUDA | 28 | 30 |
| Qwen3.5-9B Q4_K_M | CUDA | — | ~20 |
| Qwen3.5-0.8B Q8_0 | CPU | 8 | 22 |
| TinyLlama 1.1B Q8_0 | CPU | 5 | 13 |
| DeepSeek R1 8B Q8_0 | CPU | 2 | 3 |

## Supported Quantization Formats

### Native GPU Kernels (CUDA + Vulkan)

These formats have dedicated GPU kernels for maximum performance:

| Format | MatMul | EmbeddingLookup | Notes |
|--------|--------|-----------------|-------|
| F32 | GPU | GPU | Full precision |
| F16 | GPU | GPU | Half precision |
| Q8_0 | GPU | GPU | 8-bit quantized — recommended |
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

See [known-issues.md](known-issues.md) for details. Issues 1–4 are fixed. The remaining open issue:

- **Qwen3.5-9B output quality**: The 9B DeltaNet model produces coherent English output but with lower quality than the 0.8B model. May require further DeltaNet numerical tuning or chat template formatting.

## Recommended Models

For the best experience:

1. **Best quality**: [Qwen3-8B Q8_0](https://huggingface.co/unsloth/Qwen3-8B-GGUF) — 8.7 GB, 41–53 tok/s on CUDA
2. **Best speed**: [Qwen3.5-0.8B Q8_0](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF) — 812 MB, 80–131 tok/s on CUDA
3. **Best value**: [Qwen3-8B Q4_K_M](https://huggingface.co/unsloth/Qwen3-8B-GGUF) — 5.0 GB, 22 tok/s on CUDA (half the VRAM of Q8_0)
4. **Reasoning**: [DeepSeek R1 8B Q8_0](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-GGUF) — 8.5 GB, ~40 tok/s on CUDA
