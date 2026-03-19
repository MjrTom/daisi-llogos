# Architecture

> High-level system design for daisi-llama.
> [Definitions](definitions.md) | [Roadmap](../README.md#roadmap)

---

## Solution Structure

daisi-llama is organized as a multi-project .NET 10 solution. The core library has zero dependencies on any backend — backends are separate assemblies loaded by the CLI or host application.

```mermaid
graph TD
    CLI["Daisi.Llama.Cli\n(Console app)"]
    CORE["Daisi.Llama\n(Core library)"]
    CPU["Daisi.Llama.Cpu\n(CPU/SIMD backend)"]
    CUDA["Daisi.Llama.Cuda\n(CUDA backend)"]
    VULKAN["Daisi.Llama.Vulkan\n(Vulkan backend)"]
    METAL["Daisi.Llama.Metal\n(Metal backend)"]
    TESTS["Daisi.Llama.Tests\n(xUnit v3)"]

    CLI --> CORE
    CLI --> CPU
    CLI --> CUDA
    CLI --> VULKAN
    CLI --> METAL
    CPU --> CORE
    CUDA --> CORE
    VULKAN --> CORE
    METAL --> CORE
    TESTS --> CORE
    TESTS --> CPU

    style CORE fill:#2d6a4f,color:#fff
    style CLI fill:#264653,color:#fff
    style CPU fill:#e76f51,color:#fff
    style CUDA fill:#e76f51,color:#fff
    style VULKAN fill:#e76f51,color:#fff
    style METAL fill:#e76f51,color:#fff
    style TESTS fill:#6c757d,color:#fff
```

### Project responsibilities

| Project | Role |
|---------|------|
| **Daisi.Llama** | GGUF parser, model loader, tokenizer, inference engine, sampling. Defines `IComputeBackend` and `ITensor` interfaces. Contains no hardware-specific code. |
| **Daisi.Llama.Cpu** | CPU compute backend using .NET SIMD intrinsics (`Vector256<T>`, `Vector512<T>`). Implements dequantization, matmul, RMSNorm, softmax, SiLU, RoPE. |
| **Daisi.Llama.Cuda** | NVIDIA GPU backend. Raw P/Invoke to CUDA Driver API, pre-compiled .cubin kernels, fused dequant+matmul. |
| **Daisi.Llama.Vulkan** | Cross-platform GPU backend using Vulkan compute shaders (SPIR-V). Targets Windows and Linux. |
| **Daisi.Llama.Metal** | Apple GPU backend using Metal compute shaders. Targets macOS (arm64/x64) and iOS via XCFramework. |
| **Daisi.Llama.Cli** | Command-line interface. Model loading, text generation, interactive chat. Selects backend based on available hardware. |
| **Daisi.Llama.Tests** | Unit and integration tests. Uses a real Qwen 3.5 0.8B Q8_0 model for integration validation. |

---

## Compute Backend Abstraction

The core library defines interfaces that all backends implement. The inference engine works exclusively through these interfaces — it never touches hardware-specific APIs.

```mermaid
classDiagram
    class IComputeBackend {
        <<interface>>
        +string Name
        +ITensor CreateTensor(string name, GgmlType type, ReadOnlySpan~long~ dimensions)
        +ITensor LoadTensor(string name, GgmlType type, ReadOnlySpan~long~ dimensions, ReadOnlySpan~byte~ data)
        +void MatMul(ITensor output, ITensor a, ITensor b)
        +void RmsNorm(ITensor output, ITensor input, ITensor weight, float eps)
        +void Softmax(ITensor output, ITensor input)
        +void SiLU(ITensor output, ITensor input)
        +void RoPE(ITensor q, ITensor k, int positionOffset, float ropeTheta)
        +void ElementMul(ITensor output, ITensor a, ITensor b)
        +void ElementAdd(ITensor output, ITensor a, ITensor b)
        +void Dispose()
    }

    class ITensor {
        <<interface>>
        +string Name
        +GgmlType Type
        +ReadOnlySpan~long~ Dimensions
        +long ElementCount
        +void CopyFrom(ReadOnlySpan~byte~ data)
        +void CopyTo(Span~float~ destination)
        +void Dispose()
    }

    class CpuBackend {
        +Implements IComputeBackend
        -AVX2/AVX-512 SIMD paths
    }

    class CudaBackend {
        +Implements IComputeBackend
        -P/Invoke to CUDA Driver API
        -Pre-compiled .cubin kernels
    }

    class VulkanBackend {
        +Implements IComputeBackend
        -SPIR-V compute shaders
    }

    class MetalBackend {
        +Implements IComputeBackend
        -MSL compute kernels
    }

    IComputeBackend --> ITensor : creates/operates on
    CpuBackend ..|> IComputeBackend
    CudaBackend ..|> IComputeBackend
    VulkanBackend ..|> IComputeBackend
    MetalBackend ..|> IComputeBackend
```

### Design principles

- **Backend owns tensors.** The backend allocates and manages all tensor memory. CPU tensors are managed arrays; CUDA tensors are device pointers behind `SafeHandle` wrappers. The inference engine never directly accesses tensor memory.
- **No cross-backend tensors.** A tensor created by one backend cannot be passed to another. Model loading binds to a single backend for the session.
- **Fused operations are optional.** Backends may implement fused operations (e.g., dequant+matmul) as optimizations. The inference engine calls primitive operations; the backend decides whether to fuse them internally.
- **Quantized math where possible.** Backends perform dequantization inside operations (e.g., matmul reads Q8_0 data and dequantizes on the fly) rather than dequantizing all weights upfront.

---

## Data Flow

From GGUF file on disk to generated text output:

```mermaid
flowchart LR
    subgraph Loading
        FILE["GGUF File\n(.gguf on disk)"]
        PARSER["GgufFile.Read()\n(parse header,\nmetadata, tensor info)"]
        LOADER["Model Loader\n(create tensors on backend,\nload weights)"]
    end

    subgraph Inference
        TOKENIZER["Tokenizer\n(text → token IDs)"]
        ENGINE["Inference Engine\n(forward pass through\ntransformer layers)"]
        SAMPLER["Sampler\n(logits → next token)"]
    end

    subgraph Output
        DETOK["Detokenizer\n(token IDs → text)"]
    end

    FILE --> PARSER --> LOADER
    LOADER --> ENGINE
    TOKENIZER --> ENGINE
    ENGINE --> SAMPLER
    SAMPLER -->|"loop until EOS"| ENGINE
    SAMPLER --> DETOK
```

### Step-by-step

1. **Parse** — `GgufFile.Read()` reads the binary header, all metadata KV pairs, and tensor info descriptors. Tensor data is not loaded yet.
2. **Load** — The model loader iterates tensor info, reads raw bytes from the tensor data section, and calls `backend.LoadTensor()` to allocate and populate each tensor on the target device.
3. **Tokenize** — The BPE tokenizer (vocabulary and merge rules extracted from GGUF metadata) converts input text to token IDs.
4. **Prefill** — All prompt tokens are processed in one batched forward pass, populating the KV cache.
5. **Decode loop** — Each iteration: run forward pass for the last token, sample next token from logits, append to KV cache. Repeat until EOS or max tokens.
6. **Detokenize** — Generated token IDs are converted back to text and streamed to the user.

---

## Memory Model

```mermaid
flowchart TD
    subgraph "Inference Engine (managed)"
        MODEL["Model config\n(layer count, dims, etc.)"]
        LAYERS["Layer references\n(pointers to weight tensors)"]
        KVC["KV cache handles"]
        BUF["Scratch buffer handles"]
    end

    subgraph "Backend (native or managed)"
        WEIGHTS["Weight tensors\n(quantized, read-only)"]
        CACHE["KV cache tensors\n(FP16/FP8, read-write)"]
        SCRATCH["Scratch tensors\n(FP32, reused per layer)"]
    end

    LAYERS --> WEIGHTS
    KVC --> CACHE
    BUF --> SCRATCH
```

**Ownership rules:**

| Resource | Owner | Lifetime |
|----------|-------|----------|
| Weight tensors | Backend | Model load → model dispose |
| KV cache | Backend | Model load → model dispose (grows with context) |
| Scratch buffers | Backend | Allocated at model load, reused every forward pass |
| Model config | Inference engine | Extracted from GGUF metadata at load time |
| Token buffers | Inference engine | Per-generation, managed arrays |

---

## Complete Generation Request

End-to-end sequence for a text generation call:

```mermaid
sequenceDiagram
    participant App as Application
    participant Loader as Model Loader
    participant GF as GgufFile
    participant Backend as IComputeBackend
    participant Tok as Tokenizer
    participant Eng as Inference Engine
    participant Samp as Sampler

    App->>Loader: LoadModel(path, backend)
    Loader->>GF: GgufFile.Read(stream)
    GF-->>Loader: GgufFile (header, metadata, tensor info)
    Loader->>Loader: Extract model config from metadata
    Loader->>Tok: Build tokenizer from vocab metadata

    loop For each tensor
        Loader->>GF: ReadTensorData(stream, tensorInfo)
        GF-->>Loader: Raw bytes
        Loader->>Backend: LoadTensor(name, type, dims, data)
        Backend-->>Loader: ITensor handle
    end

    Loader->>Backend: Allocate KV cache + scratch buffers
    Loader-->>App: Model ready

    App->>Tok: Encode("Hello, world")
    Tok-->>App: [token IDs]

    App->>Eng: Generate(tokenIds, params)
    Note over Eng: Prefill phase
    Eng->>Backend: MatMul, RmsNorm, RoPE, Softmax, SiLU...
    Eng->>Eng: KV cache populated

    loop Decode (token by token)
        Eng->>Backend: Forward pass (single token)
        Backend-->>Eng: Logits
        Eng->>Samp: Sample(logits, params)
        Samp-->>Eng: Next token ID
        Eng-->>App: Stream token
    end

    Eng-->>App: Generation complete
```
