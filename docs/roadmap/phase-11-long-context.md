# Phase 11: Long Context Support (200K+)

> Flash attention, KV cache quantization, paged cache, RAM/SSD offloading, and streaming attention.
> [Definitions](../definitions.md) | [Inference Pipeline](../inference-pipeline.md) | [CUDA Backend](../cuda-backend.md) | [DeltaNet](../deltanet.md)

---

## Goal

Support 200K+ token context on desktop GPUs with 16GB VRAM. The Qwen 3.5 hybrid architecture (18 DeltaNet + 6 standard attention layers) gives us a natural advantage — only 6 layers need KV cache. This phase exploits that to push context length far beyond what pure-attention models can achieve in the same memory budget.

---

## Memory Analysis

### Current State (FP32 KV Cache)

| Context | KV Cache (6 layers) | Model Weights | Total VRAM | Fits 16GB? |
|---------|--------------------:|-------------:|----------:|:----------:|
| 2K | 48 MB | 850 MB | 898 MB | Yes |
| 32K | 768 MB | 850 MB | 1.6 GB | Yes |
| 200K | 4.8 GB | 850 MB | 5.65 GB | Yes |
| 1M | 24 GB | 850 MB | 24.85 GB | No |

KV cache per layer per position: `2 KV heads x 256 dim x 4 bytes x 2 (K+V)` = **4 KB**.
DeltaNet state per layer: `16 groups x 128 x 128 x 4 bytes` = **1 MB** (fixed, context-independent).

### After Optimization

| Context | FP16 KV | Q8_0 KV | Q4_0 KV | FP16 + RAM offload |
|---------|--------:|--------:|--------:|-------------------:|
| 200K | 2.4 GB | 1.28 GB | 0.68 GB | 384 MB VRAM |
| 1M | 12 GB | 6.4 GB | 3.4 GB | 384 MB VRAM |
| 10M | 120 GB | 64 GB | 34 GB | 384 MB VRAM + 12 GB RAM |

### Performance Impact (Decode tok/s)

At long context, KV cache reads during attention dominate over weight reads. RTX 5080 = 960 GB/s VRAM, ~60 GB/s system RAM.

| Context | FP32 KV | FP16 KV | Q8_0 KV | FP16 + 16K window in VRAM |
|---------|--------:|--------:|--------:|--------------------------:|
| 2K | ~42 | ~43 | ~43 | ~43 |
| 32K | ~35 | ~39 | ~41 | ~43 |
| 200K | ~25 | ~33 | ~38 | ~43 |
| 1M | — | ~15 | ~20 | ~43 |

---

## Sub-phases

### 11a: Flash Attention (Tiled/Chunked) — unblocks >12K context

**Problem:** Current `GatedAttention` kernel allocates `shared[seqLen]` for attention scores. At >12K tokens this exceeds the per-block shared memory limit (48-100KB) and crashes.

**Solution:** Tile the attention computation into chunks of T positions (e.g., T=1024). Each tile computes partial softmax, then tiles are combined using the online softmax trick (log-sum-exp correction).

```mermaid
flowchart TD
    subgraph Current["Current: Full Scores in Shared Memory"]
        Q1["q[1 x D]"]
        K1["K[S x D]"]
        SC["scores[S]\n(shared memory)"]
        SM["softmax(scores)"]
        V1["V[S x D]"]
        O1["output[D]"]
        Q1 --> SC
        K1 --> SC
        SC --> SM --> O1
        V1 --> O1
    end

    subgraph Tiled["Tiled: Online Softmax"]
        Q2["q[1 x D]"]
        T1["Tile 1: pos 0..T-1\npartial_max, partial_sum, partial_out"]
        T2["Tile 2: pos T..2T-1\npartial_max, partial_sum, partial_out"]
        TN["Tile N: pos (N-1)T..S-1\npartial_max, partial_sum, partial_out"]
        MERGE["Merge: rescale + combine\nusing log-sum-exp correction"]
        OUT["output[D]"]
        Q2 --> T1 & T2 & TN
        T1 & T2 & TN --> MERGE --> OUT
    end
```

**Algorithm per tile:**
1. Compute `scores[t] = q @ K_tile^T * scale` (only T scores in shared memory)
2. Find `tile_max = max(scores[0..T-1])`
3. Compute `tile_sum = sum(exp(scores - tile_max))`
4. Compute `tile_out = softmax(scores) @ V_tile`
5. Merge with running state: rescale previous output using `exp(prev_max - new_max)`, combine sums

**Shared memory:** O(T + blockDim) = constant regardless of context length.

**Files changed:**
- `composite_ops.cu` — new `gated_attention_tiled` kernel
- `CpuBackend.cs` — chunked attention loop
- `CudaBackend.cs` — launch tiled kernel

### 11b: FP16 KV Cache — 2x memory savings

Store K and V in FP16 instead of FP32. Dequantize to FP32 during attention score computation.

```mermaid
flowchart LR
    subgraph Write["KV Write (per token)"]
        KF["K[FP32]"] --> KH["quantize_fp16()"] --> KC["K_cache[FP16]"]
        VF["V[FP32]"] --> VH["quantize_fp16()"] --> VC["V_cache[FP16]"]
    end

    subgraph Read["Attention Read (per generated token)"]
        KC2["K_cache[FP16]"] --> KD["dequant on-the-fly\n(in attention kernel)"]
        VC2["V_cache[FP16]"] --> VD["dequant on-the-fly\n(in attention kernel)"]
    end
```

GPU has native FP16 load/convert instructions, so dequant is essentially free. Quality impact is negligible — FP16 has ~3 decimal digits of precision, and attention weights smooth out any noise.

**Files changed:**
- `KvCache.cs` — allocate as `GgmlType.F16` instead of `GgmlType.F32`
- `IComputeBackend` — `KvCacheWriteFp16()`, or modify `GatedAttention` to accept FP16 cache tensors
- `composite_ops.cu` — FP16 load in attention kernel
- `CpuBackend.cs` — FP32→FP16 quantize on write, FP16→FP32 dequant on read

### 11c: Paged KV Cache — efficient allocation

Replace the monolithic `[nKvHeads x maxSeqLen x headDim]` allocation with a page table of fixed-size blocks (e.g., 256 tokens per page).

```mermaid
flowchart TD
    subgraph Current["Current: Monolithic"]
        MC["K_cache: contiguous [maxSeqLen x dim]\n(pre-allocated for max context)"]
    end

    subgraph Paged["Paged: Block Table"]
        PT["Page Table\n[logical_page → physical_page]"]
        P0["Page 0\npos 0-255"]
        P1["Page 1\npos 256-511"]
        P2["Page 2\npos 512-767"]
        PN["Page N\npos N*256..(N+1)*256-1"]
        PT --> P0 & P1 & P2 & PN
    end
```

Benefits:
- Only allocate pages actually used (short prompts don't waste memory)
- Pages can live in different memory tiers (VRAM, RAM, SSD)
- Pages can be shared across sequences (for batched inference)
- No memory fragmentation — all pages are the same size

**Files changed:**
- New `PagedKvCache.cs` replacing `KvCache.cs`
- `IComputeBackend` — `PagedAttention()` operation with page table indirection
- `composite_ops.cu` — paged attention kernel using block table for indirect addressing

### 11d: RAM Offloading — enables 500K+ on any GPU

Tiered KV cache: keep recent pages in VRAM, older pages in pinned system RAM.

```mermaid
flowchart LR
    subgraph VRAM["VRAM (960 GB/s)"]
        HOT["Hot pages\n(last 16K tokens)"]
    end

    subgraph RAM["System RAM (60 GB/s)"]
        WARM["Warm pages\n(older tokens)"]
    end

    subgraph SSD["NVMe SSD (7 GB/s)"]
        COLD["Cold pages\n(oldest tokens)"]
    end

    HOT <-->|"cuMemcpyDtoH\ncuMemcpyHtoD"| WARM
    WARM <-->|"mmap / read"| COLD
```

During attention:
1. Compute attention over hot pages (in VRAM, full speed)
2. Stream warm pages from RAM → VRAM in chunks, compute partial attention, discard
3. Merge partial results using online softmax (same as Flash Attention tiling)

Requires Flash Attention (11a) as a prerequisite — the tiling mechanism naturally supports streaming pages from different memory tiers.

**Files changed:**
- `PagedKvCache.cs` — page eviction policy, tier management
- `CudaApi.cs` — `cuMemAllocHost` for pinned memory, async transfers
- `CudaBackend.cs` — streaming page attention with overlap compute/transfer

### 11e: Sliding Window + Attention Sinks — infinite context

For truly unbounded context, use a fixed-size rolling window:

```mermaid
flowchart LR
    subgraph Cache["Fixed KV Cache (W + S tokens)"]
        SINKS["Attention Sinks\n(first 4-128 tokens)\nAlways retained"]
        GAP["...\n(evicted)"]
        WINDOW["Sliding Window\n(last W tokens)\nRolling buffer"]
    end

    NEW["New token"] --> WINDOW
    WINDOW -->|"oldest exits"| GAP
```

Based on StreamingLLM: initial tokens ("attention sinks") receive disproportionate attention weight across all layers. Keeping them preserves generation quality. The sliding window captures recent context.

- **Memory:** Fixed at `(S + W) x per_position_cost` regardless of total context
- **Speed:** Constant tok/s regardless of context length
- **Quality:** Good for streaming/chat. Loses recall of middle-context details.
- **Configuration:** `--window-size 16384 --sink-tokens 64`

**Files changed:**
- `KvCache.cs` or `PagedKvCache.cs` — ring buffer mode with protected sink region
- `ForwardPass.cs` — position remapping for RoPE with discontinuous positions

### 11f: Token Eviction (H2O) — optional quality upgrade

Instead of evicting by recency, track cumulative attention scores per cached token. Evict the lowest-attention tokens (Heavy Hitter Oracle).

- Better quality than fixed sliding window
- Adapts to content: important context tokens survive regardless of distance
- More bookkeeping overhead per generated token

---

## Test Plan

| Test | Validates |
|------|-----------|
| `TiledAttention_MatchesFullAttention` | Tiled online softmax produces same output as full softmax |
| `Fp16KvCache_CoherentOutput` | Generation quality maintained with FP16 cache |
| `PagedKvCache_MatchesMonolithic` | Paged allocation produces identical results |
| `RamOffload_LargeContext` | 200K context generates coherent text with RAM offloading |
| `SlidingWindow_ConstantMemory` | Memory usage stays flat as context grows |
| `AttentionSinks_PreservesQuality` | Perplexity with sinks+window close to full attention |
| `LongContext_32K_Coherent` | 32K context prompt produces coherent continuation |
| `LongContext_200K_NoOOM` | 200K context runs without out-of-memory on 16GB GPU |

---

## Done Criteria

- [ ] **11a:** Flash/tiled attention — no shared memory limit on context length
- [ ] **11b:** FP16 KV cache — 2x memory reduction, <1% perplexity impact
- [ ] **11c:** Paged KV cache — dynamic allocation, no pre-allocation waste
- [ ] **11d:** RAM offloading — 500K+ context on 16GB GPU with 32GB RAM
- [ ] **11e:** Sliding window + sinks — infinite streaming with fixed memory
- [ ] 200K context generates coherent text at >25 tok/s on RTX 5080
- [ ] 1M context functional (with RAM offloading) at >8 tok/s
- [ ] Memory usage scales with actual context, not max context

---

## Dependencies

- **Phase 8** (Optimization): KV cache quantization basics, mmap loading
- **Phase 11a** (Flash Attention) is prerequisite for 11c, 11d
- **Phase 11c** (Paged Cache) is prerequisite for 11d

```mermaid
flowchart LR
    P8["Phase 8\nOptimization"]
    A["11a\nFlash Attention"]
    B["11b\nFP16 KV Cache"]
    C["11c\nPaged KV Cache"]
    D["11d\nRAM Offloading"]
    E["11e\nSliding Window"]
    F["11f\nToken Eviction"]

    P8 --> A
    P8 --> B
    A --> C --> D
    A --> E
    E --> F
    B --> C

    style A fill:#e76f51,color:#fff
    style B fill:#e76f51,color:#fff
    style C fill:#e76f51,color:#fff
    style D fill:#e76f51,color:#fff
    style E fill:#e76f51,color:#fff
    style F fill:#e76f51,color:#fff
```
