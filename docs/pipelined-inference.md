# Pipelined Inference: Running Models Bigger Than Your GPU

> **What this solves:** You have a 15 GB model but only 16 GB of GPU memory. Normally you'd be stuck using slow CPU inference. Pipelined inference lets you run the model on your GPU anyway — loading one layer at a time, like reading a book one page at a time instead of holding the whole book open.

For background on terms used here, see [Definitions](definitions.md).

---

## The Problem

AI language models are made up of stacked **layers** — a 27-billion parameter model like Qwen3.5-27B has 64 layers. Each layer contains **weight tensors** (large matrices of numbers the model learned during training). Loading all 64 layers into GPU memory at once requires ~15 GB.

If your GPU has enough memory, great — you load everything and run at full speed (~2.7 tokens/second on an RTX 5080). But what if the model is too big, or you want to leave room for other things in GPU memory?

```
Traditional approach:              Pipelined approach:
┌─────────────────────┐           ┌─────────────────────┐
│ GPU Memory (16 GB)  │           │ GPU Memory (16 GB)  │
│                     │           │                     │
│ ┌─────────────────┐ │           │ ┌───────┐ ┌───────┐ │
│ │ All 64 layers   │ │           │ │Layer 5│ │Layer 6│ │
│ │ (15 GB)         │ │           │ │(Slot A)│ │(Slot B)│ │
│ │                 │ │           │ │ 250 MB│ │ 250 MB│ │
│ │                 │ │           │ └───────┘ └───────┘ │
│ │                 │ │           │                     │
│ │                 │ │           │ ┌─────────────────┐ │
│ │                 │ │           │ │ KV Cache + Other │ │
│ │                 │ │           │ │ (small, shared)  │ │
│ └─────────────────┘ │           │ └─────────────────┘ │
│ ┌───────┐           │           │                     │
│ │ Other │           │           │ 14 GB FREE          │
│ └───────┘           │           │                     │
└─────────────────────┘           └─────────────────────┘
  Needs 15+ GB VRAM                 Needs ~1 GB VRAM
  2.7 tok/s                         0.4 tok/s
```

---

## How It Works

Pipelined inference processes one layer at a time:

```
For each word the model generates:

  1. EMBEDDING: Convert the input word to numbers        (permanent in GPU)
     ┌──────────┐
     │ "Hello"  │──→ [0.23, -0.47, 0.82, ...]
     └──────────┘

  2. LAYER 0: Load weights from disk → GPU, compute      (then overwrite)
     ┌──────────────┐     ┌──────────────┐
     │ Shard file   │────→│ GPU Slot A   │────→ compute ────→ hidden state
     │ layer.0      │     │ (250 MB)     │
     └──────────────┘     └──────────────┘

  3. LAYER 1: Load next weights, overwrite slot           (previous data gone)
     ┌──────────────┐     ┌──────────────┐
     │ Shard file   │────→│ GPU Slot B   │────→ compute ────→ hidden state
     │ layer.1      │     │ (250 MB)     │
     └──────────────┘     └──────────────┘

  4. ... repeat for all 64 layers ...

  5. OUTPUT HEAD: Convert final hidden state to word probabilities
     ┌──────────────┐
     │ hidden state │────→ [0.01, 0.85, 0.02, ...] ────→ "Paris"
     └──────────────┘         ↑
                          highest probability = the model's answer
```

The key insight: **only one layer's weights need to be in GPU memory at any time**. The "hidden state" that flows between layers is tiny (~20 KB) compared to the weights (~250 MB per layer).

---

## Shard Files: Splitting the Model

Before pipelined inference can work, the model file needs to be split into per-layer pieces called **shards**:

```bash
# Split a model into per-layer shard files
daisi-llogos split --model Qwen3.5-27B-Q4_0.gguf --output-dir ./shards/
```

This creates:

```
shards/
├── Qwen3.5-27B-Q4_0.gguf.header       (11 MB)   ← model metadata + structure
├── Qwen3.5-27B-Q4_0.gguf.embed        (683 MB)  ← word-to-number lookup table
├── Qwen3.5-27B-Q4_0.gguf.output       (995 MB)  ← number-to-word projection
├── Qwen3.5-27B-Q4_0.gguf.layer.0      (250 MB)  ← transformer layer 0
├── Qwen3.5-27B-Q4_0.gguf.layer.1      (250 MB)  ← transformer layer 1
├── ...
├── Qwen3.5-27B-Q4_0.gguf.layer.63     (250 MB)  ← transformer layer 63
└── Qwen3.5-27B-Q4_0.gguf.manifest.json (8 KB)   ← index of all shard files
```

Each shard file contains:
1. A **binary index** listing which weight tensors are inside and where
2. The **raw weight data** for those tensors

The `--align-gpu` flag pre-converts quantized data to GPU-ready format, saving work during inference:
```bash
daisi-llogos split --model model.gguf --output-dir ./shards/ --align-gpu
```

---

## DaisiChain: Distributed Partial Downloads

When running across multiple machines (DaisiChain pipeline parallelism), shard files enable **partial downloads**. Each machine only downloads the layers it's responsible for:

```
                    ┌─────────────────────────────────┐
                    │         Orchestrator (ORC)        │
                    │   "Split 64 layers across 4 hosts"│
                    └────┬────────┬────────┬────────┬──┘
                         │        │        │        │
                    ┌────▼──┐┌────▼──┐┌────▼──┐┌────▼──┐
                    │Host A ││Host B ││Host C ││Host D │
                    │Layers ││Layers ││Layers ││Layers │
                    │ 0-15  ││16-31  ││32-47  ││48-63  │
                    └───────┘└───────┘└───────┘└───────┘

Without shards:  Each host downloads 15 GB  = 60 GB total
With shards:     Each host downloads ~4 GB  = 16 GB total  (4x savings)
```

The hidden state (~20 KB) passes between hosts after each processes its layers. This is negligible compared to the weight data.

---

## What We Tried (And What Worked)

Building pipelined inference involved extensive experimentation. Here's what we learned:

### What Works

| Approach | Speed | Correct? | Notes |
|----------|------:|:--------:|-------|
| **Full GPU load** | 2.66 tok/s | Yes | All weights in VRAM. Fastest. Model must fit. |
| **Per-layer upload** (current) | 0.40 tok/s | Yes | Upload each layer, compute, upload next. ~1 GB VRAM. |
| **CPU layer-by-layer** | 0.24 tok/s | Yes | Pure CPU, any model size. Slowest but always works. |

### What Doesn't Work

| Approach | Speed | Problem |
|----------|------:|---------|
| **Shared GPU tensors** | 1.13 tok/s | Garbage output. Same GPU memory address reused for different layers causes stale data in CUDA's internal caches. Data is byte-identical but compute produces wrong results. |
| **Multi-context CUDA** | 1.90 tok/s | Only works for some chunk sizes. Separate CUDA contexts per layer group causes OOM or corruption depending on configuration. |
| **Single-context weight swap** | 0.17 tok/s | Slower than CPU. Re-uploading 200 MB per layer per token through PCIe is the bottleneck. |
| **Background copy thread** | 0.31 tok/s | Thread coordination overhead and EventSynchronize stalls negate the parallelism. |
| **Triple buffering** | OOM | Three weight slots exceed 16 GB VRAM on the 27B model. Would work on 24+ GB GPUs. |

### The Speed Hierarchy

```
Full CUDA load:     ████████████████████████████████████████ 2.66 tok/s
Hybrid 56 GPU+8 CPU: ████████████████████             1.34 tok/s
Per-layer pipeline:  ██████                              0.40 tok/s
CPU only:            ████                                0.24 tok/s
```

---

## Architecture: How PipelinedForwardPass Works

```
┌─────────────────────────────────────────────────────────────┐
│                   PipelinedForwardPass                        │
│                                                               │
│  ┌─────────┐    ┌─────────────────────────────────────────┐ │
│  │ Shard   │    │              GPU (CUDA)                  │ │
│  │ Files   │    │                                          │ │
│  │ (disk)  │    │  ┌──────────┐  ┌──────────┐             │ │
│  │         │    │  │ Slot A   │  │ Slot B   │  (alternate) │ │
│  │ layer.0 │───→│  │ weights  │  │ weights  │             │ │
│  │ layer.1 │    │  └──────────┘  └──────────┘             │ │
│  │ layer.2 │    │                                          │ │
│  │   ...   │    │  ┌──────────────────────────┐           │ │
│  │ layer.63│    │  │ KV Cache (all layers)    │ persistent │ │
│  │         │    │  │ ~33 MB                    │           │ │
│  │ embed   │───→│  └──────────────────────────┘           │ │
│  │ output  │───→│                                          │ │
│  └─────────┘    │  ┌──────────────────────────┐           │ │
│                  │  │ Embedding + Output Head  │ persistent │ │
│  ┌─────────┐    │  │ ~1.7 GB                  │           │ │
│  │ Header  │    │  └──────────────────────────┘           │ │
│  │ (GGUF   │    │                                          │ │
│  │ metadata│    │  ForwardPass instance                    │ │
│  │  only)  │    │  (reused for all layers)                 │ │
│  └─────────┘    └─────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

**Key design decisions:**
- **Two weight slots** alternate: while GPU computes from Slot A, Slot B can be prepared
- **Embedding and output head** are loaded once and stay in GPU memory permanently
- **KV cache** (the model's "memory" of what it's seen so far) persists across all layers
- The **ForwardPass** is a single instance with weights swapped via reflection between layers
- **Shard files are memory-mapped** — the OS manages paging data from disk on demand

---

## Code Example

```csharp
// Split a model into shards (one-time preprocessing)
GgufSplitter.Split("model.gguf", "./shards/");

// Load and run pipelined inference
using var cuda = new CudaBackend();
using var headerStream = File.OpenRead("./shards/model.gguf.header");
var gguf = GgufFile.Read(headerStream);
var config = ModelConfig.FromGguf(gguf);

using var pipeline = PipelinedForwardPass.Create(gguf, "./shards/", config, cuda);

// Generate tokens
var logits = pipeline.Forward(tokenId: 42, position: 0);
```

---

## Performance Summary

**Hardware:** NVIDIA RTX 5080 (16 GB VRAM), AMD Ryzen 9 9900X  
**Model:** Qwen3.5-27B Q4_0 (15 GB, 64 layers)

| Metric | Full GPU Load | Pipelined | CPU Only |
|--------|:------------:|:---------:|:--------:|
| **Decode speed** | 2.66 tok/s | 0.40 tok/s | 0.24 tok/s |
| **VRAM for weights** | 15 GB | ~1 GB | 0 |
| **Load time** | 11 s | 1 s | 8 s |
| **Can run 70B+?** | No | Yes | Yes |

The pipelined approach is **1.7x faster than CPU** while using only **7% of the VRAM** that full loading requires. It enables running models that physically cannot fit in GPU memory.

---

## Shard File Format

Each shard file (embed, output, layer.N) uses a simple binary format:

```
┌──────────────────────────────────────────┐
│ Magic "GSHD" (4 bytes)                   │
│ Version: 1 (4 bytes)                     │
│ Shard type: embed/output/layer (4 bytes) │
│ Layer index: -1 or 0-63 (4 bytes)        │
│ Tensor count (4 bytes)                   │
├──────────────────────────────────────────┤
│ For each tensor:                         │
│   Name length (4 bytes)                  │
│   Name (UTF-8 bytes)                     │
│   Data offset (8 bytes)                  │
│   Data size (8 bytes)                    │
├──────────────────────────────────────────┤
│ Alignment padding (to 32-byte boundary)  │
├──────────────────────────────────────────┤
│ Tensor data (contiguous raw bytes)       │
└──────────────────────────────────────────┘
```

The **manifest.json** file indexes all shards:
```json
{
  "version": 1,
  "modelFileName": "Qwen3.5-27B-Q4_0.gguf",
  "totalLayers": 64,
  "gpuAligned": false,
  "header": { "fileName": "...header", "sizeBytes": 10993568 },
  "embed":  { "fileName": "...embed",  "sizeBytes": 715161664 },
  "output": { "fileName": "...output", "sizeBytes": 1042964576 },
  "layers": [
    { "layerIndex": 0, "fileName": "...layer.0", "sizeBytes": 225517024 },
    ...
  ]
}
```

---

## Future Work

1. **Shared tensor correctness** — The 1.13 tok/s shared-tensor approach needs a fix for CUDA's activation cache to work with reused GPU memory addresses. This would nearly triple the pipelined speed.
2. **Async overlap** — True compute/upload overlap via CUDA pinned memory and separate copy streams. Theoretical max: ~1.9 tok/s.
3. **CLI integration** — Add `--pipelined` flag to the inference CLI for easy access.
4. **WebGPU layer-swap** — Browser-based layer-at-a-time inference for low-VRAM devices using the shard format.
