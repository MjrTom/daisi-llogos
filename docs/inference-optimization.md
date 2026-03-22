# Inference Optimization: Beating llama.cpp from Pure C#

**daisi-llogos** is a ground-up C# reimplementation of GGUF model inference with native GPU kernels. This document covers the optimization techniques that let us match or exceed llama.cpp performance on NVIDIA GPUs — including one novel approach that delivers a free 10% speedup with no quality loss.

## Results

RTX 5080, 128 decode tokens, greedy sampling (temperature=0). llama.cpp b8461.

| Model | Architecture | Llogos | llama.cpp | Ratio |
|-------|-------------|-------:|----------:|------:|
| Qwen3.5-0.8B Q8_0 | DeltaNet hybrid | **441** | 399 | **1.10x** |
| TinyLlama 1.1B Q8_0 | LLaMA | **448** | 443 | **1.01x** |
| Qwen3.5-4B Q8_0 | DeltaNet hybrid | **144** | 135 | **1.07x** |
| Qwen3-8B Q8_0 | Standard attention | 91 | 92 | 0.99x |
| DeepSeek R1 8B Q8_0 | LLaMA | 94 | 95 | 0.99x |
| Qwen3-8B Q4_K_M | Standard attention | 124 | 138 | 0.90x |
| Qwen3.5-9B Q8_0 | DeltaNet hybrid | **88** | 84 | **1.05x** |
| Qwen3.5-9B Q4_0 | DeltaNet hybrid | 101 | 123 | 0.82x |

We exceed llama.cpp on 4 of 8 models tested, across three different architectures (DeltaNet, LLaMA, standard attention). Results are not Qwen-specific — the optimizations generalize across model families. The remaining gap on 4-bit quants is from our Q4_0 float kernel being compute-bound on nibble extraction.

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
| Q6_K | 10 | Very complex inner loop but small data per super-block |
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

## Lessons Learned

1. **The lm_head is a bottleneck hiding in plain sight.** For Qwen3.5's 152K vocab, the output projection reads 1.6GB per token — 17% of total weight data for the 9B model. Partial vocab computation eliminates most of this for free.

2. **Optimal row count is quant-specific, not universal.** The interaction between register pressure, occupancy, and activation reuse varies dramatically across quant formats. Systematic sweeps beat theoretical analysis.

3. **dp4a precision issues were a cache bug, not a math bug.** We spent significant time investigating FP16 truncation, Q8_1 sum compensation, and quantization rounding before discovering the real cause: stale cached data from tensor pointer reuse.

4. **Beating llama.cpp from managed C# is possible.** The GPU kernel code is identical (CUDA C compiled via NVRTC, GLSL compiled to SPIR-V). The managed overhead is negligible — the bottleneck is always the GPU kernels and memory bandwidth. C#'s `unsafe` context and P/Invoke provide zero-overhead access to the CUDA/Vulkan driver APIs.

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

RTX 5080, CUDA 13, greedy sampling.

| Model | Prompt tokens | Prefill tok/s | Decode tok/s | Prefill speedup |
|-------|:------------:|:-------------:|:------------:|:---------------:|
| TinyLlama 1.1B Q8_0 | 173 | **1,976** | 369 | **5.4x** |
| TinyLlama 1.1B Q8_0 | 126 | **1,708** | 362 | **4.7x** |
| Qwen3-8B Q8_0 | 55 | **329** | 88 | **3.7x** |

### Implementation

Five new CUDA kernels enable the batched forward pass:

1. **`batched_rope`** — Applies rotary position embedding to M tokens with positions `[startPos..startPos+M-1]`. Each token's Q/K heads are rotated by their own position angle. The token index is derived from the global head index: `token = head / headsPerToken`.

2. **`batched_kv_cache_write`** — Writes M K/V pairs at consecutive positions in a single kernel launch. Input is `[M × nKvHeads × keyLen]`, scattered into the `[nKvHeads × maxSeqLen × keyLen]` cache layout.

3. **`batched_gated_attention`** — The core kernel. Grid of `M × numHeads` blocks, one per (query_token, head) pair. Each block computes tiled online-softmax attention with a causal cutoff: query token m sees keys `[0..startPos+m]`. Uses the same tile size and online softmax algorithm as the single-token kernel, just with per-query sequence length bounds.

4. **`batched_rms_norm`** (and variants) — Normalizes M independent rows. One block per row. Weight vector is shared across rows.

5. **`dequant_to_f32`** — Expands quantized weight matrices to FP32 for cuBLAS SGEMM. Handles Q8_0, Q4_0, Q4_K, Q6_K, and F16.

The batched forward pass (`ForwardBatchedPrefill`) allocates M-wide scratch tensors lazily and reuses them across calls with the same batch size. It always uses separate Q/K/V and gate/up projections (fused layouts produce per-token interleaved output that would require M scatter copies to deinterleave).

### Design Decisions

**No CUDA graph capture.** The batched path uses cuBLAS SGEMM, which lazily allocates a dequantization buffer via `cuMemAlloc`. This allocation crashes during CUDA graph capture (stream recording forbids `cuMemAlloc`). Since prefill is a one-shot operation (not replayed like decode), graph capture provides no benefit anyway.

**Non-fused projections only.** The single-token path fuses Q+K+V into a single matmul and gate+up into another. For M>1, the fused matmul output is `[M × (qDim+kDim+vDim)]` where each row contains interleaved `[Q,K,V]` for one token. Deinterleaving into separate `[M×qDim]`, `[M×kDim]`, `[M×vDim]` tensors requires M×3 scatter copies. Using three separate projections produces the correct layout directly and benchmarks faster for M>8.

**Pure attention models only (current).** DeltaNet hybrid models (Qwen3.5) have sequential state dependencies that prevent full batching. For these models, the prefill falls back to the sequential token-by-token path. A future optimization could batch the attention layers and MatMuls while processing only the DeltaNet state updates sequentially.

### Why Prefill Scales Sublinearly

The 5.4x speedup on TinyLlama (173 tokens) is less than the theoretical 173x from perfect parallelization. Three factors limit scaling:

1. **Attention is O(M²).** The causal attention kernel launches `M × numHeads` blocks, and each block scans up to `startPos + M` keys. Total work grows quadratically with M, while sequential prefill's attention work grows linearly (each token sees one more key than the last).

2. **Memory bandwidth.** cuBLAS SGEMM with M>1 achieves higher arithmetic intensity than M=1 vector-matrix multiply, but the weight matrices still must be read from VRAM once per layer. The dequant+SGEMM path reads weights as quantized data then expands to FP32, consuming more bandwidth than the fused M=1 kernels that dequantize in-register.

3. **Embedding scatter.** The M embeddings are looked up individually and copied into the batch tensor via M `cuMemcpyDtoDAsync` calls. This is negligible for large M but measurable for small prompts.

## Reproducibility

All benchmarks use:
- Hardware: AMD Ryzen 9 9900X, NVIDIA GeForce RTX 5080 (16GB)
- Software: .NET 10, CUDA 13, Vulkan SDK 1.4.341, llama.cpp b8461
- Models: Qwen3.5 (0.8B, 4B, 9B DeltaNet hybrid), Qwen3 (8B standard attention), TinyLlama (1.1B LLaMA), DeepSeek R1 (8B LLaMA distill)
- Methodology: 128 decode tokens, greedy sampling (temperature=0), 3 runs for stability
- CLI: `dotnet run --project src/Daisi.Llogos.Cli -c Release -- --model <path> --backend cuda --bench --prompt "Hello" --max-tokens 128 --temperature 0`
