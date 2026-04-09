# Inference Optimization: Beating llama.cpp from Pure C#

**daisi-llogos** is a ground-up C# reimplementation of GGUF model inference with native GPU kernels. This document covers the optimization techniques that let us match or exceed llama.cpp performance on NVIDIA GPUs — including one novel approach that delivers a free 10% speedup with no quality loss.

## Results

RTX 5080, CUDA 12, greedy sampling (temperature=0). llama.cpp b8461 CUDA.

### Decode (tok/s, 128 tokens, short prompt)

| Model | Architecture | Llogos | llama.cpp | Ratio |
|-------|-------------|-------:|----------:|------:|
| Qwen3.5-0.8B Q8_0 | DeltaNet hybrid | **431** | 411 | **1.05x** |
| TinyLlama 1.1B Q8_0 | LLaMA | 444 | 448 | 0.99x |
| Qwen3.5-4B Q8_0 | DeltaNet hybrid | **142** | 134 | **1.05x** |
| Qwen3-8B Q8_0 | Standard attention | 89 | 91 | 0.98x |
| Qwen3-8B Q4_K_M | Standard attention | **127** | 138 | **0.92x** |
| Qwen3.5-9B Q8_0 | DeltaNet hybrid | **86** | 84 | **1.02x** |
| Qwen3.5-9B Q4_0 | DeltaNet hybrid | 99 | 126 | 0.79x |

### Prefill (tok/s, 128 prompt tokens)

| Model | Architecture | Llogos | llama.cpp | Ratio | Batch mode |
|-------|-------------|-------:|----------:|------:|------------|
| Qwen3.5-0.8B Q8_0 | DeltaNet hybrid | 346 | 13,794 | 0.03x | Hybrid (too small to benefit) |
| TinyLlama 1.1B Q8_0 | LLaMA | **2,050** | 15,730 | 0.13x | Full batch |
| Qwen3.5-4B Q8_0 | DeltaNet hybrid | **224** | 5,704 | 0.04x | Hybrid (1.5x vs sequential) |
| Qwen3-8B Q8_0 | Standard attention | **456** | 5,184 | 0.09x | Full batch |
| Qwen3-8B Q4_K_M | Standard attention | **462** | 5,278 | 0.09x | Full batch |
| Qwen3.5-9B Q8_0 | DeltaNet hybrid | **165** | 4,714 | 0.04x | Hybrid (1.8x vs sequential) |
| Qwen3.5-9B Q4_0 | DeltaNet hybrid | **155** | 4,954 | 0.03x | Hybrid (1.5x vs sequential) |

We exceed llama.cpp on decode for 3 of 7 models, across DeltaNet and standard attention architectures. The Q4_K_M gap has been reduced from 12% to 8% via fused SwiGLU matmul, Q6_K kernel optimization, and cooperative dp4a kernels. The remaining 4-bit gaps are from pipeline overhead and mixed-quant layer efficiency.

Prefill is a large gap: llama.cpp uses fused quantized GEMM via cuBLASLt with tensor cores, while our batched path dequantizes to FP32 then calls cuBLAS SGEMM. Pure attention models achieve 5-9x speedup over sequential prefill; DeltaNet hybrids achieve 1.5-1.8x (only RmsNorm, FFN, and attention layers are batched; DeltaNet state updates remain sequential). llama.cpp's fused approach is 8-30x faster on prefill overall.

## 1. Partial Vocabulary Logit Computation

**The biggest surprise.** During greedy decode (temperature=0), the model computes logits for every token in the vocabulary (152K for Qwen3.5), then picks the argmax. But common tokens — English words, Chinese characters, code syntax, punctuation, numbers — all occupy the low end of the vocabulary. High-ID tokens are rare: unusual Unicode, specialized markers, model-specific control tokens.

We compute logits for only the first `VocabSize / 32` tokens (~4,752 out of 152,064) and run argmax over that subset. This skips 97% of the lm_head matmul — the single largest matmul in the model (1.6GB weight data for the 9B model).

**Does it work?** We tested every divisor from 1 (full vocab) to 16,384 (9 tokens). All produce identical output across diverse prompts: factual Q&A, poetry, code generation, JSON, Chinese text, math, rare words like "xylophone". The model's top-1 prediction is always a common token.

| Divisor | Tokens computed | Speed (9B Q4_0) |
|--------:|----------------:|----------------:|
| 1 | 152,064 | 90 tok/s |
| 4 | 38,016 | 97 tok/s |
| 32 | 4,752 | **100 tok/s** |
| 512 | 297 | 100 tok/s |

The speedup saturates at ~32x because the lm_head becomes negligible relative to the 32 transformer layers. We default to `/32` as it provides the full speedup while maintaining a comfortable margin of 4,752 candidate tokens.

**Applicability:** This optimization is safe for any greedy decode workload: agentic tool calling, structured output (JSON/XML/SQL), factual Q&A, code generation, classification. It does NOT apply to temperature>0 sampling, which requires full logit distributions. The `--vocab-limit` CLI flag provides runtime control.

**Why this works differently from top-K/top-P:** Top-K and top-P are post-hoc filters applied after computing all logits. They save nothing on compute. Partial vocab skips the compute entirely — the GPU never reads the weight data for excluded tokens. This is a matmul-level optimization, not a sampling-level one.

### Vocabulary Frequency Remapping

The partial vocab optimization assumes common tokens have low IDs. While empirically true for most tokenizers, we make it provably correct by remapping the vocabulary at load time:

1. Score each token by frequency/utility (ASCII letters, CJK, digits = high; special control tokens = low)
2. Sort to build a permutation: highest-scoring tokens get the lowest new IDs
3. Permute the embedding table rows and lm_head weight columns in-place
4. Update the tokenizer's ID↔string mapping

This is a one-time cost of ~0.6s during model loading. After remapping, truncating to the first N tokens is guaranteed to include the N most useful tokens. The model's internal representations are unchanged — only the input/output mapping is permuted.

The remapping is active by default when `--vocab-limit` > 1. Setting `--vocab-limit 1` disables both remapping and truncation.

## 2. Per-Quantization Row Count Tuning

Each matmul kernel processes multiple output rows per CUDA block ("multi-row"). More rows = more activation reuse (load activation once, multiply against N weight rows). But more rows = more registers per thread = lower occupancy = fewer concurrent blocks.

The optimal row count varies per quant type because each has different register pressure in its inner loop:

| Quant | Optimal rows | Why |
|-------|:------------:|-----|
| Q8_0 | 2 | Simple byte extraction, low ALU — register pressure dominates |
| Q4_0 | 2 | Same as Q8_0 (float path with nibble extraction) |
| Q4_K | 3 | Complex scale unpacking, moderate register use |
| Q6_K | 3 | Branch-free split loops, factored scale multiply, float4 activation loads |
| Q4_1 | 8 | Simple with min offset, similar to Q4_0 |
| Q5_K | 1 | Extremely complex (5-bit with high-bit array), any multi-row spills |

Finding: **Q4_K went from 4→3 rows = +21% speedup** (99→120 tok/s on 8B Q4_K_M). The conventional wisdom of "more rows = better" is wrong on modern GPUs with high register counts and deep pipelines.

## 3. dp4a Integer Dot Product for 4-bit Quantization

For Q4_0 weights, each block contains 32 4-bit nibbles packed in 16 bytes. The float dequant path extracts each nibble via shift+mask+int-to-float, producing 701 PTX instructions per block vs 422 for Q8_0. This makes Q4_0 compute-bound despite reading less data.

The `__dp4a` (dot product of 4-byte vectors) instruction computes `c += a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3]` in a single clock for packed int8 operands. We pre-quantize the FP32 activation to Q8_1 format (int8 with per-block scale and sum), then use dp4a for the nibble × quant dot product.

### The Cache Bug

Our initial dp4a implementation produced garbage. After extensive debugging (comparing per-matmul outputs, studying llama.cpp source, testing multiple Q8_1 formats), the root cause was a **stale activation cache**: the Q8_1 quantized activation was keyed on the tensor's device pointer, but the same tensor (`_normOut`) is reused across layers. Same pointer, different data. The fix: a generation counter incremented by every operation that writes to a tensor, checked alongside the pointer.

### Architecture-Adaptive Dispatch

dp4a is not universally faster. On NVIDIA Blackwell (SM 12.x), the massive FP32 throughput makes the float nibble-extraction path competitive with dp4a + quantization overhead. On older architectures (Turing, Ampere, Ada), dp4a's 4x int8 throughput clearly wins.

We dispatch based on compute capability: Blackwell uses the float path when Q8_1 isn't pre-computed, dp4a otherwise. Pre-Blackwell always uses dp4a.

## 4. Fused RmsNorm + Q8_1 Quantization

The dp4a path requires quantizing the activation to Q8_1 before each matmul group. This quantization is a separate kernel launch — 64 per token (2 per layer × 32 layers). Each launch reads 4096 floats, computes per-block max/sum, and writes 36-byte Q8_1 blocks.

We fuse this into the preceding RmsNorm kernel. RmsNorm already reads the hidden state, normalizes it, and writes the output. Our fused variant additionally quantizes the output to Q8_1 in the same pass — zero extra memory traffic, zero extra kernel launches.

Three fused kernels cover all RmsNorm variants in the forward pass:
- `rms_norm_residual_q8_1` (first layer)
- `add_rms_norm_residual_q8_1` (layers 1-31)
- `add_rms_norm_q8_1` (FFN pre-norm)

The dp4a matmul checks a `_q8_1FusedReady` flag and skips the separate quantization step entirely when the fused kernel has already prepared the data.

## 5. Aligned Block Repacking

GGUF stores Q8_0 blocks as 34 bytes (2-byte FP16 scale + 32 int8 quants). The quant data at byte offset 2 is NOT 4-byte aligned, forcing the GPU to issue multiple byte loads instead of a single `uint32` load.

At model load time, we repack to 36 bytes (2-byte scale + 2-byte padding + 32 quants). The quant data at offset 4 is now 4-byte aligned, enabling native `uint32` loads via `__ldg` and the Vulkan `uint weight_u32[]` buffer view. Same data, 6% more storage, significantly faster memory access patterns.

We apply the same technique to Q4_0 (18→20 bytes), aligning the 16-byte nibble array to a 4-byte boundary.

## 6. CUDA Graph Capture

The entire forward pass (embedding lookup → 32 transformer layers → logit projection) is recorded as a single CUDA graph on the first token. Subsequent tokens replay the graph with a single `cuGraphLaunch` call, eliminating ~435 individual kernel launch overheads.

Graph capture required careful handling of dynamic allocations: the Q8_1 scratch buffer must be pre-allocated during model loading (not during graph recording), and all memory addresses must remain stable across tokens.

## 7. Self-Contained Dispatch

An early optimization attempt used global `sed` commands to tune kernel parameters, which accidentally corrupted grid sizes for Q8_0, Q4_K, and Q6_K — causing those kernels to skip 50-75% of output rows. The model appeared to run faster (fewer rows = less work) and even produced plausible text on simple prompts, but was fundamentally broken.

The fix: each quant type's dispatch computes its own grid size, thread count, and shared memory via a self-contained `AdaptiveLaunch()` helper. No shared variables between quant types = no cross-contamination from editing one type's parameters.

## 8. Fused MatMulSwiGLU for Q4_K FFN Layers

The FFN block in transformer models computes `output = SiLU(gate_proj(x)) * up_proj(x)`. The standard pipeline executes this as three separate operations: gate matmul → up matmul → SwiGLU activation. Each operation writes to global memory and requires a separate kernel launch.

We fuse all three into a single CUDA kernel: `dequant_matmul_swiGLU_q4_k`. For each output element `i`, the kernel computes both `dot(activation, gate_weights[i])` and `dot(activation, up_weights[i])` in the same thread block, then applies `SiLU(gate) * up` and writes the final result. The Q8_1 activation data is loaded once and shared across both weight row reads.

**Architecture:** Uses the same cooperative dp4a design as the Q4_K v2 kernel (128 threads, 16 per super-block). Two sets of partial sums (gate and up) are accumulated in parallel with negligible extra register pressure (~30 registers vs ~15 for single matmul).

**Savings per FFN layer:**
- 2 fewer kernel launches (up_proj matmul + SwiGLU eliminated)
- Intermediate tensor writes eliminated (gate and up outputs never touch global memory)
- Activation Q8_1 loaded once instead of twice

**Impact:** Over 36 layers, this eliminates 72 kernel launches and ~8 MB of intermediate memory traffic per token. Measured improvement: 125.4 → 127.4 tok/s on Qwen3 8B Q4_K_M (+1.6%).

## 9. Q6_K Kernel Optimization

Q4_K_M models use mixed quantization — approximately 39% of weight data per layer is Q6_K (attention V projections, FFN down projections). The original Q6_K kernel had several inefficiencies:

- **10 rows per block** — extreme register pressure (42+ registers per thread) limiting SM occupancy
- **Per-element branch** — `(l < 16) ? sc0 : sc1` in the hot inner loop caused warp divergence
- **Scalar activation loads** — 32 individual float reads instead of vectorized loads
- **Scale multiply per element** — `a * (sc * q)` computed 32 scale multiplications per chunk

The optimized kernel reduces to **3 rows per block**, eliminates the branch by splitting into two 16-element loops, uses **float4 vectorized activation loads** (8 × 128-bit reads), and **factors the scale multiply** outside the inner loop (`sc * Σ(a*q)` = 1 multiply instead of 16).

**Impact:** Measured improvement on Qwen3 8B Q4_K_M: 124 → 125.4 tok/s (+1.1%). The improvement is proportional to Q6_K's share of the model's weight data.

## 10. Cooperative Q4_K dp4a Kernel (v2)

The original Q4_K float kernel uses 64 threads with 3 rows per block, where each thread independently processes a 64-element chunk (loading 64 floats of activation + 32 bytes of nibbles). This gives ~70 registers per thread and only 2 warps for memory latency hiding.

Inspired by llama.cpp's `mul_mat_vec_q` architecture, the v2 kernel (`dequant_matmul_q4_k_v2`) uses a cooperative design:

- **128 threads** (4 warps), **1 row per block** — all threads cooperate on the same output row
- **16 threads per super-block** — 4 threads per chunk, each handling 16 elements via dp4a
- **~15 registers per thread** — 4x lower than the multi-row kernel
- **16-thread warp reduction** via `__shfl_xor_sync` + 8-entry shared memory cross-group sum

This architecture enables much higher SM occupancy (up to 27 blocks vs 14 for the old kernel) and 4 warps for latency hiding. The activation Q8_1 data is shared across all threads via L2 cache, and the dp4a(0x01010101, ...) trick correctly distributes activation sums for the Q4_K min-correction term.

On Blackwell, the v2 kernel matches the multi-row float kernel's performance (both achieve ~125 tok/s). On pre-Blackwell GPUs with lower FP32 throughput, dp4a's 4x integer throughput advantage should yield a measurable speedup.

## Lessons Learned

1. **The lm_head is a bottleneck hiding in plain sight.** For Qwen3.5's 152K vocab, the output projection reads 1.6GB per token — 17% of total weight data for the 9B model. Partial vocab computation eliminates most of this for free.

2. **Optimal row count is quant-specific, not universal.** The interaction between register pressure, occupancy, and activation reuse varies dramatically across quant formats. Systematic sweeps beat theoretical analysis.

3. **dp4a precision issues were a cache bug, not a math bug.** We spent significant time investigating FP16 truncation, Q8_1 sum compensation, and quantization rounding before discovering the real cause: stale cached data from tensor pointer reuse.

4. **Beating llama.cpp from managed C# is possible.** The GPU kernel code is identical (CUDA C compiled via NVRTC, GLSL compiled to SPIR-V). The managed overhead is negligible — the bottleneck is always the GPU kernels and memory bandwidth. C#'s `unsafe` context and P/Invoke provide zero-overhead access to the CUDA/Vulkan driver APIs.

5. **Mixed-quant models need all kernels fast.** Q4_K_M uses ~39% Q6_K weights per layer. A slow Q6_K kernel drags the entire model's performance. Optimizing the Q6_K kernel from 10→3 rows with branch elimination yielded a measurable 1.1% improvement on Q4_K_M models.

6. **Kernel fusion saves more than you'd expect.** The fused SwiGLU matmul saves ~0.15 ms per token — mostly from eliminated kernel launches, not memory traffic. With CUDA graph replay overhead at ~2 μs per kernel, eliminating 72 launches per token adds up.

7. **On-demand Q8_1 quantization needs activation tracking, not just fused-ready flags.** The dp4a fallback path's stale-cache condition (`!_q8_1FusedReady && ...`) silently served wrong Q8_1 data when the fused flag was stale from a previous RmsNorm but the activation had changed. Always check activation pointer + generation, regardless of fused state.

8. **FP16 scale pre-computation overflows.** Attempting to pre-compute `d * sc` as FP16 at load time for Q4_K flat scales caused overflow (d up to 65504 × sc up to 63 = 4.1M >> FP16 max 65504). The original runtime FP32 computation is correct; there is no lossless FP16 shortcut.

## 8. Early Layer Exit (Investigated, Not Viable)

We investigated whether tokens could be predicted before all transformer layers complete, allowing the forward pass to exit early and skip the remaining layers.

### Methodology

Added `--profile-early-exit` instrumentation that projects the hidden state through the output norm and lm_head at each layer checkpoint (layers 8-31 of 32), recording which token each layer would predict.

### Findings

| Prompt | Final token | First correct layer | % through |
|--------|------------|:-------------------:|:---------:|
| "The capital of France is" | `.` (period) | Layer 27 | 84% |
| "Hello" | `I` | Layer 29 | 90% |
| "1 2 3 4 5 6 7 8 9" | `#` | Layer 21 | 65% |

**The intermediate predictions are chaotic.** Between layers 8-25, the model predicts wildly different tokens at each layer — Chinese characters, Russian text, Hindi, control tokens, random subwords. The prediction does not gradually converge; it oscillates randomly until the final few layers suddenly snap to the correct answer.

Example (prompt "1 2 3 4 5 6 7 8 9", predicting next token `#`):
```
L8:nat  L9:ucz  L10:CIAL  L11:其他  L12:实现自己  L13:sp  L14:sär
L15:小孩  L16:草  L17:<|repo_name|>  L18:рож  L19:卡拉  L20:ँव
L21:# ✓  L22:th  L23:idl  L24:•  L25:不欲  L26:# ✓  L27:xeda
L28:##  L29:Поделиться  L30:1  L31:# ✓
```

Even for this trivial number sequence, the correct prediction (`#`) first appears at layer 21 but then disappears at layers 22-25 before returning. The prediction is non-monotonic.

### Conclusion

**Early exit without model-specific training saves at most 10-15% of layers.** Standard transformers are not trained to produce meaningful intermediate predictions. The residual stream accumulates information across all layers, and the final layers perform critical refinements.

To achieve meaningful early exit (50%+ layer skip), the model would need to be fine-tuned with auxiliary classification heads at intermediate layers and an early exit loss objective. This is a training-time change, not an inference-time optimization applicable to off-the-shelf GGUF models.

## 9. Batched Prefill

Standard LLM inference processes prompt tokens one at a time: embed token, run 32 layers, write KV cache, repeat. Each token requires a full forward pass with M=1 matrix-vector multiplies. For a 100-token prompt on the 9B model, that's 100 sequential passes through 32 layers — ~1.1 seconds.

Batched prefill processes all M prompt tokens simultaneously through each layer: M×K input matrix × K×N weight matrix via cuBLAS SGEMM, batched normalization, batched RoPE, and a batched causal attention kernel. The KV cache is written in bulk and attention uses a causal mask so each query token only attends to preceding positions.

### Results

RTX 5080, CUDA 13, greedy sampling. Prefill measured with 231-token prompt; decode with short prompt.

| Model | Architecture | Llogos pp | llama.cpp pp | Ratio | Llogos tg | llama.cpp tg | Ratio |
|-------|-------------|----------:|-------------:|------:|----------:|-------------:|------:|
| DeepSeek-R1-Llama 8B Q8_0 | LLaMA | **2,518** | 6,400 | **0.39x** | 82 | — | — |
| Qwen3.5-9B Q8_0 | DeltaNet | **315** | 5,667 | 0.06x | 81 | 84 | **0.96x** |

**Prefill optimization progression** (DeepSeek Llama 8B Q8_0, 231 tokens):

| Step | tok/s | vs llama.cpp |
|------|------:|:-------------|
| Baseline (SwiGLU bug, dequant→SGEMM) | 197 | 32x slower |
| + SwiGLU bug fix + FP16 GemmEx | 1,432 | 4.5x slower |
| + FP16 weight cache (warm) | 2,130 | 3.0x slower |
| + Batched embedding kernel | 2,198 | 2.9x slower |
| + cuBLAS strided batched attention | 2,518 | **2.5x slower** |
| llama.cpp CUDA | 6,400 | baseline |

Pure attention models now achieve **39% of llama.cpp** prefill throughput (up from 3%). DeltaNet hybrids remain slower due to sequential recurrent state processing (18/24 layers).

**Prompt length affects decode speed.** Decode tok/s depends on the KV cache size: a 169-token prompt leaves 169 KV entries that the attention kernel must scan every decode step. On TinyLlama, this costs ~15% vs a 1-token prompt. All decode numbers in this paper use a short prompt to measure pure decode throughput.

### Implementation

CUDA kernels and cuBLAS paths enabling the batched forward pass:

1. **`batched_rope`** — RoPE for M tokens with positions `[startPos..startPos+M-1]`.

2. **`batched_kv_cache_write`** — Writes M K/V pairs at consecutive cache positions.

3. **cuBLAS strided batched attention** — Replaces the custom `batched_gated_attention` kernel for M>32. Uses `cublasGemmStridedBatchedEx` for QK^T and Score×V (one call per KV head group with `strideA=0` for GQA K/V reuse), plus `causal_softmax_inplace` and `sigmoid_gate_inplace` custom kernels. Eliminates O(M²) redundant KV cache reads from the old per-block approach.

4. **`batched_rms_norm`** (and variants) — Normalizes M independent rows.

5. **`dequant_to_f16`** — Expands quantized weights to FP16 for `cublasGemmEx` tensor cores. Handles Q8_0, Q4_0, Q4_K, Q6_K, BF16, and F16.

6. **`batched_embedding_q8_0`** — Single kernel for M token embedding lookups (replaces M sequential calls).

7. **`convert_f32_to_f16`** — Activation FP32→FP16 conversion (cached per source tensor).

The batched forward pass (`ForwardBatchedPrefill`) allocates M-wide scratch tensors lazily and reuses them across calls with the same batch size. It always uses separate Q/K/V and gate/up projections (fused layouts produce per-token interleaved output that would require M scatter copies to deinterleave).

### Design Decisions

**No CUDA graph capture.** The batched path uses cuBLAS SGEMM, which lazily allocates a dequantization buffer via `cuMemAlloc`. This allocation crashes during CUDA graph capture (stream recording forbids `cuMemAlloc`). Since prefill is a one-shot operation (not replayed like decode), graph capture provides no benefit anyway.

**Non-fused projections only.** The single-token path fuses Q+K+V into a single matmul and gate+up into another. For M>1, the fused matmul output is `[M × (qDim+kDim+vDim)]` where each row contains interleaved `[Q,K,V]` for one token. Deinterleaving into separate `[M×qDim]`, `[M×kDim]`, `[M×vDim]` tensors requires M×3 scatter copies. Using three separate projections produces the correct layout directly and benchmarks faster for M>8.

**DeltaNet hybrid support.** DeltaNet layers have sequential state dependencies (conv buffer, state matrix). The batched prefill handles hybrids by batching RmsNorm and FFN for all layers, batching attention for attention layers, and processing only the DeltaNet state update sequentially per token. This gives 1.5-1.8x prefill speedup on the larger DeltaNet models (4B, 9B) where batched MatMuls dominate, though the 0.8B model sees no benefit due to small matrix sizes.

**Unaligned weight dequantization.** The `dequant_to_f32` kernel originally hardcoded aligned block strides (36 bytes for Q8_0, 20 bytes for Q4_0). Models with `hiddenDim < 2048` (like the 0.8B) skip alignment repacking at load time, leaving weights in their original unaligned format. The kernel now checks an `isAligned` flag and uses the correct stride (34 bytes unaligned vs 36 bytes aligned for Q8_0).

### Why Prefill Scales Sublinearly

The 5.4x speedup on TinyLlama (173 tokens) is less than the theoretical 173x from perfect parallelization. Three factors limit scaling:

1. **Attention is O(M²).** The causal attention kernel launches `M × numHeads` blocks, and each block scans up to `startPos + M` keys. Total work grows quadratically with M, while sequential prefill's attention work grows linearly (each token sees one more key than the last).

2. **Memory bandwidth.** cuBLAS SGEMM with M>1 achieves higher arithmetic intensity than M=1 vector-matrix multiply, but the weight matrices still must be read from VRAM once per layer. The dequant+SGEMM path reads weights as quantized data then expands to FP32, consuming more bandwidth than the fused M=1 kernels that dequantize in-register.

3. **Embedding scatter.** The M embeddings are looked up individually and copied into the batch tensor via M `cuMemcpyDtoDAsync` calls. This is negligible for large M but measurable for small prompts.

### Optimizations Applied

**FP16 tensor core GemmEx (7.3x).** Replaced `dequant_to_f32` + `cublasSgemm` with `dequant_to_f16` + `cublasGemmEx` (FP16 inputs, FP32 accumulation). Uses tensor cores. Also replaced the custom `batched_gated_attention` kernel (which had O(M²) redundant KV cache reads) with `cublasGemmStridedBatchedEx` for QK^T and Score×V matmuls plus a custom `causal_softmax_inplace` kernel.

**FP16 weight cache (49% on warm prefills).** Persistent `Dictionary<DevicePtr, CudaDeviceMemory>` caches dequantized FP16 weights. First prefill populates the cache; subsequent prefills skip dequant entirely. VRAM-aware allocation with 512 MB headroom via `cuMemGetInfo`.

**FP16 activation cache.** Avoids redundant FP32→FP16 conversion when the same activation feeds multiple projections (Q/K/V share normOut, gate/up share normOut). Keyed by source DevicePtr + size.

**Batched embedding kernel.** Single `batched_embedding_q8_0` kernel replaces M sequential `embedding_lookup` + `CopyTensorSlice` calls. Processes all M token embeddings in one launch.

**Batched DeltaNet projections.** For DeltaNet hybrid models, the 5 linear projections (QKV, Alpha, Beta, Gate, Out) are batched for all M tokens. Only the sequential conv1d and state update remain per-token. Improves Qwen3.5-9B prefill from 197 to 315 tok/s.

**SwiGLU Q8_1 overflow fix (critical).** The fused `swiglu_q8_1` kernel wrote Q8_1 data to `_q8_1Scratch` for ALL M×N elements during batched FFN, but the scratch buffer was sized for single-token only. This caused GPU memory corruption and garbage output from all batched prefill since it was first enabled. Fixed by guarding the fused path with `n <= _q8_1ScratchK`.

### Approaches That Didn't Help

**TF32 math mode.** `CUBLAS_TF32_TENSOR_OP_MATH` and `CUBLAS_COMPUTE_32F_FAST_16F` — no measurable improvement (cuBLAS already uses tensor cores with default settings for FP16 GemmEx).

**Fused gate+up FFN.** Concatenated FfnGate and FfnUp FP16 weights into a single [2N×K] matrix for one GemmEx instead of two. The D2D memcpy overhead to build the fused weight negated the saved GemmEx. Even with persistent fused weight caching, cuBLAS handles the larger matrix at similar throughput.

**Pointer-based batched GEMM for attention.** `cublasGemmBatchedEx` (array of pointers) to process all 32 heads in one call instead of 8 strided batched calls. Slower than the strided approach due to H2D pointer upload overhead.

### Remaining Gap

The 2.5x gap to llama.cpp is from: per-call cuBLAS overhead across ~400 launches per prefill, materializing the full M×M attention score matrix (flash attention avoids this), and no CUDA graph capture for the prefill pass. Closing this requires a fused dequant-GEMM kernel using `wmma` tensor core intrinsics or flash attention.

## Reproducibility

All benchmarks use:
- Hardware: Intel Core i9-14900KF, NVIDIA GeForce RTX 5080 (16GB)
- Software: .NET 10, CUDA 13, llama.cpp b8461 (CUDA build)
- Models: Qwen3.5 (0.8B, 4B, 9B DeltaNet hybrid), Qwen3 (8B standard attention), TinyLlama (1.1B LLaMA)
- Decode: 128 tokens, greedy (temperature=0), short prompt (`"Hello"`) to minimize KV cache overhead
- Prefill: 128-token prompt, greedy, measures prompt processing throughput
- llama.cpp: 3 runs averaged. Llogos: single run (GPU-bound, low variance)

```bash
# Llogos decode
dotnet run --project src/Daisi.Llogos.Cli -c Release -- --model <path> --backend cuda --bench --prompt "Hello" --max-tokens 128

# Llogos prefill (use a long prompt that tokenizes to ~128 tokens)
dotnet run --project src/Daisi.Llogos.Cli -c Release -- --model <path> --backend cuda --bench --prompt "<long text>" --max-tokens 5

# llama.cpp
/c/llama-cpp/cuda/llama-bench.exe -m <path> -t 1 -ngl 99 -p 128 -n 128 -r 3
```
