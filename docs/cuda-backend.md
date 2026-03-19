# CUDA Backend

> Architecture and design for the NVIDIA CUDA compute backend.
> [Definitions](definitions.md) | [Architecture](architecture.md) | [Phase 6 Roadmap](roadmap/phase-06-cuda.md)

---

## Overview

The CUDA backend provides GPU-accelerated inference on NVIDIA GPUs using CUDA 13. Unlike most .NET CUDA libraries, daisi-llama uses **raw P/Invoke** to the CUDA Driver API — no managed wrappers, no CUDA Runtime API, no cuDNN. This gives full control over memory management, kernel loading, and stream orchestration.

Key design choices:
- **CUDA Driver API** (not Runtime API) — explicit context management, direct kernel loading
- **Pre-compiled .cubin kernels** — no JIT compilation at startup, deterministic performance
- **SafeHandle wrappers** — managed RAII for all CUDA resources (contexts, modules, streams, device memory)
- **Fused kernels** — dequantize and compute in a single kernel to minimize memory traffic

---

## P/Invoke Layer

### CUDA Driver API Bindings

```mermaid
classDiagram
    class CudaApi {
        <<static>>
        +cuInit(uint flags) CUresult
        +cuDeviceGet(out int device, int ordinal) CUresult
        +cuCtxCreate(out CUcontext ctx, uint flags, int device) CUresult
        +cuCtxDestroy(CUcontext ctx) CUresult
        +cuModuleLoadData(out CUmodule module, byte[] image) CUresult
        +cuModuleGetFunction(out CUfunction func, CUmodule module, string name) CUresult
        +cuMemAlloc(out CUdeviceptr ptr, ulong bytesize) CUresult
        +cuMemFree(CUdeviceptr ptr) CUresult
        +cuMemcpyHtoD(CUdeviceptr dst, IntPtr src, ulong byteCount) CUresult
        +cuMemcpyDtoH(IntPtr dst, CUdeviceptr src, ulong byteCount) CUresult
        +cuLaunchKernel(...) CUresult
        +cuStreamCreate(out CUstream stream, uint flags) CUresult
        +cuStreamSynchronize(CUstream stream) CUresult
    }

    class CudaContext {
        -CUcontext _handle
        +CudaContext(int deviceOrdinal)
        +MakeCurrent()
        +Dispose()
    }

    class CudaModule {
        -CUmodule _handle
        +CudaModule(byte[] cubinData)
        +GetFunction(string name) CudaFunction
        +Dispose()
    }

    class CudaDeviceMemory {
        -CUdeviceptr _handle
        -ulong _byteSize
        +CudaDeviceMemory(ulong byteSize)
        +CopyFromHost(ReadOnlySpan~byte~ data)
        +CopyToHost(Span~byte~ destination)
        +Dispose()
    }

    class CudaStream {
        -CUstream _handle
        +Launch(CudaFunction func, dim3 grid, dim3 block, void** args)
        +Synchronize()
        +Dispose()
    }

    CudaContext --> CudaModule
    CudaContext --> CudaDeviceMemory
    CudaContext --> CudaStream
    CudaModule --> CudaFunction
```

### SafeHandle Pattern

Every CUDA resource is wrapped in a `SafeHandle`-derived class that guarantees cleanup:

```csharp
// Conceptual pattern — actual implementation will follow this structure
class CudaDeviceMemoryHandle : SafeHandleZeroOrMinusOneIsInvalid
{
    protected override bool ReleaseHandle()
    {
        return CudaApi.cuMemFree(handle) == CUresult.CUDA_SUCCESS;
    }
}
```

This ensures GPU memory is freed even if exceptions occur or the GC collects the object.

---

## Memory Management

```mermaid
flowchart TD
    subgraph Host["Host Memory (CPU)"]
        GGUF["GGUF tensor data\n(quantized bytes)"]
        LOGITS_H["Logits output buffer"]
    end

    subgraph Device["Device Memory (GPU)"]
        WEIGHTS["Weight tensors\n(quantized, read-only)"]
        KV["KV cache\n(FP16, read-write)"]
        SCRATCH["Scratch buffers\n(FP32, reused)"]
        LOGITS_D["Logits buffer"]
    end

    GGUF -->|"cuMemcpyHtoD\n(one-time at model load)"| WEIGHTS
    LOGITS_D -->|"cuMemcpyDtoH\n(once per forward pass)"| LOGITS_H
```

### Transfer strategy

| Data | Direction | When | Frequency |
|------|-----------|------|-----------|
| Model weights | Host → Device | Model load | Once |
| KV cache | Device only | Inference | Never transferred |
| Scratch buffers | Device only | Inference | Never transferred |
| Input token IDs | Host → Device | Each generate call | Once per call |
| Logits | Device → Host | Each forward pass | Once per decode step |

**Key principle:** Minimize host-device transfers. Weights are uploaded once. All intermediate computation stays on device. Only the final logits vector (vocab_size floats) is copied back per step.

---

## Kernel Compilation and Loading

### Build pipeline

```mermaid
flowchart LR
    CU[".cu source files"]
    NVCC["nvcc compiler\n--cubin -arch=sm_120"]
    CUBIN[".cubin binary"]
    EMBED["Embedded resource\nin Daisi.Llama.Cuda.dll"]
    LOAD["cuModuleLoadData()\nat runtime"]
    FUNC["cuModuleGetFunction()\nper kernel name"]

    CU --> NVCC --> CUBIN --> EMBED --> LOAD --> FUNC
```

### Why pre-compiled cubin?

| Approach | Startup time | Runtime overhead | Deployment |
|----------|-------------|------------------|------------|
| **PTX (JIT)** | Slow (compile on first run) | None after compile | Single binary, any GPU arch |
| **cubin (AOT)** | Instant (no compilation) | None | Must ship per target arch |
| **Fat binary** | Instant | None | Larger file, multiple archs |

daisi-llama ships pre-compiled cubin for target architectures (sm_120 for Blackwell, sm_89 for Ada Lovelace, sm_86 for Ampere). A fat binary approach may be used to bundle multiple architectures.

### Target architectures

| sm_arch | GPU Family | Examples |
|---------|-----------|----------|
| sm_86 | Ampere | RTX 3060-3090, A100 |
| sm_89 | Ada Lovelace | RTX 4060-4090, L40 |
| sm_100 | Blackwell | RTX 5060-5090, B200 |
| sm_120 | Blackwell Ultra | B300 |

---

## Fused Dequant + MatMul Kernel

The most critical optimization: combining dequantization and matrix multiplication into a single kernel pass.

### Why fusion matters

```mermaid
flowchart TD
    subgraph Naive["Naive: Two Separate Kernels"]
        direction TB
        N_DEQ["Kernel 1: Dequantize\nRead Q8_0 → Write FP32"]
        N_GMEM1["Global Memory\n(FP32 weights, full size)"]
        N_MM["Kernel 2: MatMul\nRead FP32 weights × input"]
        N_DEQ --> N_GMEM1 --> N_MM
    end

    subgraph Fused["Fused: Single Kernel"]
        direction TB
        F_KERN["Fused Kernel:\nRead Q8_0 → Dequant in registers → MatMul accumulate"]
    end

    style Naive fill:#fee,stroke:#c00
    style Fused fill:#efe,stroke:#0a0
```

**Naive approach:** Dequantize all weights to FP32 in global memory (4× the quantized size), then read them again for matmul. Two full passes over the weight data.

**Fused approach:** Each thread block loads a tile of quantized weights, dequantizes into registers or shared memory, and immediately uses them for the matmul dot product. Weight data is read exactly once from global memory.

### Fused kernel data flow

```mermaid
flowchart TD
    subgraph ThreadBlock["Thread Block (e.g., 256 threads)"]
        LOAD_W["Load quantized weight tile\nfrom global memory to shared memory"]
        LOAD_X["Load input tile\nfrom global memory to shared memory"]
        DEQ["Dequantize weights\nin registers"]
        DOT["Dot product accumulate\n(registers)"]
        STORE["Write output tile\nto global memory"]
    end

    GMEM_W["Global Memory\n(quantized weights)"]
    GMEM_X["Global Memory\n(input activations, FP32)"]
    GMEM_O["Global Memory\n(output activations, FP32)"]

    GMEM_W --> LOAD_W --> DEQ --> DOT --> STORE --> GMEM_O
    GMEM_X --> LOAD_X --> DOT
```

### Kernel launch configuration

For a matmul of `[M × K] × [K × N] → [M × N]`:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Block size** | 256 threads | Good occupancy on most architectures |
| **Grid X** | `ceil(N / tile_N)` | One block column per output tile column |
| **Grid Y** | `ceil(M / tile_M)` | One block row per output tile row |
| **Shared memory** | `tile_K × (tile_M + tile_N) × sizeof(float)` | Tiles for both operands |
| **Tile size** | 128×128 or 64×64 | Tuned per architecture |

---

## Multi-Stream Pipeline

Multiple CUDA streams enable overlapping computation with memory transfers:

```mermaid
sequenceDiagram
    participant S1 as Stream 1 (Compute)
    participant S2 as Stream 2 (Transfer)
    participant Host

    Note over S1,Host: Layer N
    S1->>S1: Attention matmul
    S1->>S1: FFN matmul

    Note over S1,Host: Layer N+1
    S1->>S1: Attention matmul
    S2->>Host: Copy logits (if last layer)
    S1->>S1: FFN matmul

    Note over S1,Host: Synchronize
    S1->>S1: cuStreamSynchronize
```

In practice, the main benefit of multi-stream for inference is overlapping the final logits D2H transfer with the last layer's computation. The weight data is already on device, so there's no upload to overlap during inference.

---

## CudaBackend Implementation

```mermaid
classDiagram
    class CudaBackend {
        +string Name = "CUDA"
        -CudaContext _context
        -CudaModule _module
        -CudaStream _computeStream
        -CudaStream _transferStream
        +CreateTensor(...) CudaTensor
        +LoadTensor(...) CudaTensor
        +MatMul(output, a, b)
        +RmsNorm(output, input, weight, eps)
        +Softmax(output, input)
        +SiLU(output, input)
        +RoPE(q, k, posOffset, theta)
        +ElementMul(output, a, b)
        +ElementAdd(output, a, b)
        +Dispose()
    }

    class CudaTensor {
        +string Name
        +GgmlType Type
        +ReadOnlySpan~long~ Dimensions
        +long ElementCount
        -CudaDeviceMemory _memory
        +CopyFrom(ReadOnlySpan~byte~ data)
        +CopyTo(Span~float~ destination)
        +CUdeviceptr DevicePointer
        +Dispose()
    }

    CudaBackend --> CudaTensor : creates
    CudaBackend --> CudaContext
    CudaBackend --> CudaModule
    CudaBackend --> CudaStream
```

### Kernel inventory

| Kernel name | Operation | Input types | Notes |
|-------------|-----------|-------------|-------|
| `dequant_matmul_q8_0` | Fused dequant + matmul | Q8_0 × FP32 | Primary inference kernel |
| `dequant_matmul_q4_0` | Fused dequant + matmul | Q4_0 × FP32 | For 4-bit models |
| `dequant_matmul_q4_k` | Fused dequant + matmul | Q4_K × FP32 | For K-quant models |
| `rms_norm` | RMSNorm | FP32 | Block-level reduction |
| `softmax` | Softmax | FP32 | Numerically stable (max subtraction) |
| `silu` | SiLU activation | FP32 | Element-wise |
| `rope` | RoPE encoding | FP32 | Paired dimension rotation |
| `element_mul` | Element-wise multiply | FP32 | For SwiGLU gate |
| `element_add` | Element-wise add | FP32 | For residual connections |
