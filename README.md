# daisi-llama

A ground-up C# reimplementation of llama.cpp targeting .NET 10. Native performance through direct hardware access — SIMD intrinsics on CPU, raw P/Invoke to CUDA/Vulkan/Metal on GPU. No managed wrapper libraries, no ONNX, no Python.

## Platform Support

| Platform | Backend | Status |
|----------|---------|--------|
| Windows x64 | CPU (AVX2/AVX-512) | Priority |
| Windows x64 | CUDA 13 (NVIDIA) | Priority |
| Linux x64 | CPU (AVX2/AVX-512) | Planned |
| Linux x64 | Vulkan (NVIDIA/AMD/Intel) | Planned |
| macOS arm64 | Metal (Apple Silicon) | Planned |
| macOS x64 | Metal (Intel/AMD) | Planned |
| iOS arm64 | Metal (XCFramework) | Planned |

## Quick Start

```bash
# Build
dotnet build

# Run tests (requires Qwen 3.5 0.8B Q8_0 in C:\GGUFS)
dotnet test

# Generate text
dotnet run --project src/Daisi.Llama.Cli -- \
    --model C:\GGUFS\Qwen3.5-0.8B-Q8_0.gguf \
    --prompt "Hello, world"
```

### Test model

Tests validate against [Qwen 3.5 0.8B Q8_0](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF). Download the GGUF file to `C:\GGUFS\Qwen3.5-0.8B-Q8_0.gguf`. Tests that require the model skip gracefully if the file is not present.

## Current Status

**End-to-end text generation produces coherent output on CPU. CUDA kernel operations validated on GPU.** 113 passing tests across GGUF parsing, dequantization, math ops, tokenization, forward pass, sampling, generation, and CUDA kernels.

What works today:
- Parse any GGUF v2/v3 file (header, metadata, tensor info)
- Query model metadata (architecture, dimensions, vocab)
- Read tensor data on demand (lazy loading)
- Full quantization type support (41 GgmlType variants with block/type size calculation)
- `IComputeBackend` and `ITensor` compute abstraction interfaces
- CPU backend with dequantization for Q8_0, Q4_0, Q4_K (AVX2 SIMD + scalar fallback)
- Load quantized tensors from GGUF and dequantize to FP32
- CPU SIMD math operations: MatMul (FP32 + fused Q8_0), RMSNorm, softmax, SiLU, RoPE, element-wise add/mul
- BPE tokenizer from GGUF metadata (GPT-2 byte encoding + direct UTF-8 modes)
- Complete hybrid forward pass: standard gated attention + DeltaNet layers
- Model loading from GGUF into CPU backend with polymorphic layer weights
- KV cache for attention layers, recurrent state + conv1d buffers for DeltaNet layers
- Sampler with temperature, top-k, top-p, repetition penalty, greedy/random sampling
- Streaming text generation with prefill + decode loop
- CLI with full argument parsing (model path, prompt, sampling parameters)
- CUDA backend: P/Invoke to CUDA Driver API, NVRTC JIT compilation, SafeHandle resource management
- GPU kernels: FP32 matmul, fused Q8_0 dequant+matmul, RMSNorm, softmax, SiLU, RoPE, element-wise ops, embedding lookup
- GPU kernel correctness validated against CPU at model-scale dimensions (1024d hidden, 3584d FFN, 151936 vocab)

## Roadmap

```mermaid
flowchart LR
    P1["Phase 1\nDequantization"]
    P2["Phase 2\nMath Ops"]
    P3["Phase 3\nTokenizer"]
    P4["Phase 4\nForward Pass"]
    P5["Phase 5\nGeneration"]
    P6["Phase 6\nCUDA"]
    P7["Phase 7\nDeltaNet"]
    P8["Phase 8\nOptimization"]
    P9["Phase 9\nVulkan"]
    P10["Phase 10\nMetal"]

    P1 --> P2 --> P4
    P3 --> P4 --> P5
    P5 --> P6
    P5 --> P7
    P6 --> P8
    P7 --> P8
    P8 --> P9 --> P10

    style P1 fill:#2d6a4f,color:#fff
    style P2 fill:#2d6a4f,color:#fff
    style P3 fill:#2d6a4f,color:#fff
    style P4 fill:#2d6a4f,color:#fff
    style P5 fill:#2d6a4f,color:#fff
    style P6 fill:#2d6a4f,color:#fff
    style P7 fill:#e76f51,color:#fff
    style P8 fill:#e76f51,color:#fff
    style P9 fill:#e76f51,color:#fff
    style P10 fill:#e76f51,color:#fff
```

| Phase | Name | Goal | Status |
|-------|------|------|--------|
| 0 | [GGUF Parser](#current-status) | Parse GGUF files, read metadata and tensor info | Done |
| 1 | [Dequantization](docs/roadmap/phase-01-dequantization.md) | `IComputeBackend` + CPU dequantization (Q8_0, Q4_0, Q4_K) | Done |
| 2 | [Math Ops](docs/roadmap/phase-02-math-ops.md) | CPU SIMD matmul, RMSNorm, softmax, SiLU, RoPE | Done |
| 3 | [Tokenizer](docs/roadmap/phase-03-tokenizer.md) | BPE tokenizer from GGUF metadata | Done |
| 4 | [Forward Pass](docs/roadmap/phase-04-forward-pass.md) | Model loading + hybrid forward pass (attention + DeltaNet) | Done |
| 5 | [Generation](docs/roadmap/phase-05-generation.md) | Sampling, text generation loop, CLI | Done |
| 6 | [CUDA](docs/roadmap/phase-06-cuda.md) | NVIDIA GPU backend with fused kernels | Done |
| 7 | [DeltaNet](docs/roadmap/phase-07-deltanet.md) | Qwen 3.5 hybrid DeltaNet architecture | Done (folded into Phase 4) |
| 8 | [Optimization](docs/roadmap/phase-08-optimization.md) | Mmap loading, batch prefill, KV cache quantization | Not started |
| 9 | [Vulkan](docs/roadmap/phase-09-vulkan.md) | Cross-platform GPU backend (Windows/Linux) | Not started |
| 10 | [Metal](docs/roadmap/phase-10-metal.md) | Apple GPU backend (macOS/iOS) | Not started |

## Documentation

| Document | Description |
|----------|-------------|
| [Definitions](docs/definitions.md) | Glossary of all key terms |
| [Architecture](docs/architecture.md) | Solution structure, backend abstraction, data flow |
| [GGUF Format](docs/gguf-format.md) | Binary format deep dive with byte-level layouts |
| [Inference Pipeline](docs/inference-pipeline.md) | Complete walkthrough: tokenize → forward pass → sample |
| [CUDA Backend](docs/cuda-backend.md) | P/Invoke design, kernel compilation, fused operations |
| [DeltaNet](docs/deltanet.md) | Gated DeltaNet linear attention and hybrid architecture |

## Solution Structure

```
daisi-llama/
├── src/
│   ├── Daisi.Llama/            Core library (GGUF, model, inference, tokenizer)
│   ├── Daisi.Llama.Cpu/        CPU compute backend (SIMD)
│   ├── Daisi.Llama.Cuda/       NVIDIA CUDA backend
│   ├── Daisi.Llama.Vulkan/     Vulkan compute backend
│   ├── Daisi.Llama.Metal/      Apple Metal backend
│   └── Daisi.Llama.Cli/        Command-line interface
├── tests/
│   └── Daisi.Llama.Tests/      Unit and integration tests
└── docs/                        Architecture and roadmap documentation
```

## License

MIT License. Copyright 2026 DAISI.
