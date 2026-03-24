# WebGPU Compute Backend

> Browser-native GPU inference via WebGPU compute shaders.
> [Definitions](definitions.md) | [Architecture](architecture.md) | [CUDA Backend](cuda-backend.md) | [Vulkan Backend](vulkan-backend.md)

---

## Overview

The WebGPU backend (`Daisi.Llogos.WebGpu`) runs GGUF model inference entirely in the browser using WebGPU compute shaders. Written in TypeScript, it is the browser counterpart to the .NET CUDA and Vulkan backends. It loads GGUF models via HTTP, runs inference on the GPU, and integrates with the DAISI network via gRPC-web.

Published as `@daisinet/llogos-webgpu` on npm.

Key differences from the .NET backends:
- **Language**: TypeScript (not C#). Runs in any WebGPU-capable browser or Node.js via Dawn bindings.
- **Shader language**: WGSL (WebGPU Shading Language) — similar in capability to GLSL/SPIR-V but browser-native.
- **No shared memory model**: Each compute dispatch is its own pass with explicit barrier semantics.
- **Model loading**: HTTP fetch with Cache API persistence (no file system access in browser).
- **Architectures**: Supports Llama, Qwen 2/2.5, and Qwen 3.5 (DeltaNet hybrid).

---

## Architecture

```
Browser Tab / Node.js
┌──────────────────────────────────────────────┐
│  LlogosEngine                                 │
│  ┌──────────────────────────────────────────┐ │
│  │ GGUF Parser → Tensor Loader → Model      │ │
│  │ BPE Tokenizer (from GGUF metadata)       │ │
│  │ Sampler (temperature, top-k, top-p)      │ │
│  └──────────────────────────────────────────┘ │
│                      │                        │
│  ┌──────────────────────────────────────────┐ │
│  │ ComputeEngine                             │ │
│  │ ┌─────────────┐ ┌──────────────────────┐ │ │
│  │ │ BufferPool   │ │ ShaderCache          │ │ │
│  │ │ (VRAM track) │ │ (pipeline cache)     │ │ │
│  │ └─────────────┘ └──────────────────────┘ │ │
│  │ 20+ WGSL compute shaders                 │ │
│  └──────────────────────────────────────────┘ │
│                      │                        │
│              WebGPU API                       │
│              (navigator.gpu)                  │
└──────────────────────────────────────────────┘
                       │
                   GPU Hardware
            (NVIDIA / AMD / Intel / Apple)
```

## Files

| Directory | File | Purpose |
|-----------|------|---------|
| `src/` | `engine.ts` | Main `LlogosEngine` class — init GPU, load model, generate tokens |
| `src/` | `index.ts` | Package exports |
| `src/gguf/` | `gguf-parser.ts` | GGUF v2/v3 binary parser (header, metadata, tensor info) |
| `src/gguf/` | `quantization.ts` | GgmlType enum, block sizes, type sizes |
| `src/gpu/` | `compute.ts` | `ComputeEngine` — dispatches all GPU operations |
| `src/gpu/` | `device.ts` | WebGPU adapter/device initialization, capability detection |
| `src/gpu/` | `buffer-pool.ts` | GPU buffer allocation, reuse, VRAM tracking |
| `src/gpu/` | `shader-cache.ts` | Compiled pipeline cache |
| `src/model/` | `llama-model.ts` | Llama/Qwen2 forward pass (standard attention + GQA) |
| `src/model/` | `qwen35-model.ts` | Qwen 3.5 hybrid forward pass (DeltaNet + gated attention) |
| `src/model/` | `kv-cache.ts` | GPU-resident KV cache per layer |
| `src/model/` | `sampler.ts` | Temperature, top-k, top-p, repetition penalty sampling |
| `src/tokenizer/` | `bpe-tokenizer.ts` | BPE tokenizer from GGUF metadata |
| `src/tokenizer/` | `chat-template.ts` | Jinja2-style chat template interpreter |
| `src/storage/` | `download-manager.ts` | HTTP streaming download with Cache API persistence |

## Compute Shaders (WGSL)

| Shader | Operation | Workgroup |
|--------|-----------|-----------|
| `matmul.wgsl` | F32 matrix-vector multiply | 256 threads, 2D dispatch for large M |
| `matmul_q4.wgsl` | Fused Q4_0 dequant + matmul | 256 threads |
| `matmul_q8.wgsl` | Native Q8_0 matmul (unpack2x16float) | 256 threads |
| `attention.wgsl` | Multi-head attention with GQA | 64 threads per head |
| `rmsnorm.wgsl` | RMS normalization | 256-thread reduction |
| `rope.wgsl` | Rotary position embeddings | 1 thread per pair |
| `embedding.wgsl` | Token embedding lookup | 256 threads |
| `silu_mul.wgsl` | Fused SiLU gate × up (SwiGLU) | 256 threads |
| `add.wgsl` | Element-wise add (residual connections) | 256 threads |
| `add_bias.wgsl` | In-place bias addition | 256 threads |
| `copy_rmsnorm.wgsl` | Fused copy + RMSNorm | 256 threads |
| `softmax.wgsl` | Numerically stable softmax | 256 threads |
| **DeltaNet shaders** | | |
| `conv1d_silu.wgsl` | Fused causal conv1d + SiLU with persistent state | 256 threads |
| `l2_norm_groups.wgsl` | Per-group L2 normalization | 256-thread reduction per group |
| `compute_decay_beta.wgsl` | Softplus + exp decay, sigmoid beta | 64 threads |
| `deltanet_step.wgsl` | Full state update: mat-vec, rank-1 update, output, per-head RMSNorm | 128 threads per group |
| `silu_gate.wgsl` | Element-wise SiLU gating | 256 threads |
| `silu_inplace.wgsl` | In-place SiLU activation | 256 threads |

## Supported Models

| Architecture | Models | Features |
|-------------|--------|----------|
| Llama | TinyLlama 1.1B, Llama 3.2 1B | GQA, tied weights, 128K vocab |
| Qwen 2/2.5 | Qwen 2.5 0.5B | Attention biases, ChatML |
| Qwen 3.5 | Qwen 3.5 0.8B | **DeltaNet hybrid** (SSM + gated attention), 248K vocab |

### Quantization Support

| Format | GPU Native | CPU Dequant |
|--------|-----------|-------------|
| F32 | Matmul | — |
| Q8_0 | Matmul (native i8 dot) | Dequant |
| Q4_0 | Matmul (fused dequant) | Dequant |
| Q4_1 | — | Dequant |
| Q5_0 | — | Dequant |
| Q5_1 | — | Dequant |
| F16 | — | Dequant |
| Q4_K | — | Dequant (partial) |
| Q6_K | — | Dequant |

## Benchmarks

Measured via Dawn WebGPU (Node.js), NVIDIA RTX 5090 (Blackwell), 32 decode tokens.

| Model | Prefill | Decode | VRAM |
|-------|---------|--------|------|
| TinyLlama 1.1B Q8_0 | 45 tok/s | — | 1570 MB |
| Llama 3.2 1B Q8_0 | 61 tok/s | 54 tok/s | 2787 MB |
| Qwen 2.5 0.5B Q8_0 | 42 tok/s | 37 tok/s | 1592 MB |
| Qwen 3.5 0.8B Q8_0 | 17 tok/s | 17 tok/s | 1592 MB |

Qwen 3.5 DeltaNet runs entirely on GPU — zero CPU readbacks during inference.

## Chat Template Support

The engine automatically detects and applies chat templates:

1. **Llama 3** — detected via `<|start_header_id|>` token. Uses `<|begin_of_text|>` + role headers.
2. **ChatML** (Qwen 2/2.5/3.5) — detected via `<|im_start|>` token. Standard `<|im_start|>role\ncontent<|im_end|>` format.
3. **Llama 2** — detected via `[INST]` in template. Simple instruction wrapping.
4. **Jinja2** — generic parser for `tokenizer.chat_template` from GGUF metadata. Supports `{% for %}`, `{% if %}`, `{{ var }}`, filters.

## Integration with DAISI Network

The WebGPU engine integrates with the DAISI Manager as a **Browser Host**:

- Loads models from HuggingFace via HTTP, caches in browser Cache API
- Connects to the ORC (Orchestrator) directly via gRPC-web
- Processes inference requests from the network
- Streams tokens back in batches (10 per response) for efficient throughput
- Auto-reconnects with exponential backoff

See the [Browser Host Learn Page](https://daisinet.com/learn/browser-host) for user-facing documentation.

## Testing

```bash
cd src/Daisi.Llogos.WebGpu

# Run all tests (72 tests including GPU inference via Dawn)
npm test

# Run benchmarks
npx vitest run test/benchmark.test.ts

# Build the package
npm run build
```

Tests use the `webgpu` npm package (Dawn bindings) for GPU compute in Node.js, enabling automated CI testing without a browser.

## Quick Start

```typescript
import { LlogosEngine } from '@daisinet/llogos-webgpu';

const engine = new LlogosEngine();
await engine.initGpu();
await engine.loadModel('https://huggingface.co/.../model.gguf');

for await (const token of engine.generate('Hello, world')) {
  process.stdout.write(token);
}
```
