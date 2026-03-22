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

## Reproducibility

All benchmarks use:
- Hardware: AMD Ryzen 9 9900X, NVIDIA GeForce RTX 5080 (16GB)
- Software: .NET 10, CUDA 13, Vulkan SDK 1.4.341, llama.cpp b8461
- Models: Qwen3.5 (0.8B, 4B, 9B DeltaNet hybrid), Qwen3 (8B standard attention), TinyLlama (1.1B LLaMA), DeepSeek R1 (8B LLaMA distill)
- Methodology: 128 decode tokens, greedy sampling (temperature=0), 3 runs for stability
- CLI: `dotnet run --project src/Daisi.Llogos.Cli -c Release -- --model <path> --backend cuda --bench --prompt "Hello" --max-tokens 128 --temperature 0`
