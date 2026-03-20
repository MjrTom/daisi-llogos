# Known Issues

Detailed investigation notes on models that don't produce correct output.

## 1. K-Quant Models (Q4_K_M, Q5_K_M, Q6_K) Produce Garbage

### Symptom

Models quantized with K-quant formats produce incoherent output (random tokens, mixed languages, nonsense) despite Q8_0 versions of the same model working perfectly.

Example with Qwen3-8B (no DeltaNet, pure standard attention):
- **Q8_0**: `"Paris. Which other city in the same country has..."` — correct
- **Q4_K_M**: `"鳁 Latin source ihammer轧 hit..."` — garbage

### What's Been Verified Correct

1. **Q4_K dequantization** — matches the ggml reference implementation byte-for-byte. Verified by `DequantReferenceTest` comparing our `DequantizeQ4_K` against a manual port of `dequantize_row_q4_K` from ggml-quants.c. All 1024+ elements match within float epsilon.

2. **Q6_K dequantization** — same verification against ggml reference. Uses the correct interleaved ql/qh layout: two 128-element halves, each with 4 groups of 32 using interleaved nibbles and 2-bit high values. Matches reference exactly.

3. **Q5_K dequantization** — qs[j] low nibble = element j (0..127), high nibble = element j+128 (128..255). qh bit n = 5th bit for element n. Verified correct layout.

4. **Q4_K MatMul** — `GenericDequantMatMul_MatchesF32MatMul` test: dequantize entire Q4_K weight to F32, do F32 matmul, compare to Q4_K generic dequant matmul. Results match exactly.

5. **Individual operations** — EmbeddingLookup, RmsNorm, MatMul, Softmax all produce correct results for Q4_K tensors when tested individually.

6. **Embedding→Norm→Logits pipeline** — Loading a Q4_K_M model, doing embedding lookup, RmsNorm, and output projection (without any transformer layers) produces reasonable values: embedding sum=42.27, norm sum=6833, logit sum=414,500, argmax=96187. Values are finite and non-zero.

### Root Cause Hypothesis

The error accumulates across layers. Each Q4_K matmul introduces small numerical differences compared to Q8_0 (expected from lower precision). But through 36 transformer layers with residual connections, attention, and FFN, these errors compound into completely wrong hidden states.

This suggests the dequant may be slightly wrong in a way that doesn't show up in isolated tests but snowballs across layers. Potential areas:

1. **Unpack6BitScalesMins for sub-blocks 4-7** — The high sub-blocks use a complex bit-field extraction:
   ```csharp
   scalesRaw[j] = (packed[j + 4] & 0xF) | ((packed[j - 4] >> 6) << 4);
   minsRaw[j] = (packed[j + 4] >> 4) | ((packed[j] >> 6) << 4);
   ```
   This was verified against the ggml reference for 1024+ elements covering all 8 sub-blocks. But the verification compared our dequant to our own reference implementation — if both have the same bug, the test would pass.

2. **Q4_K nibble layout** — ggml stores Q4_K nibbles in 64-element chunks: each 32-byte chunk's low nibbles map to the first 32 elements and high nibbles to the next 32. This is different from a naive "each byte's nibbles are 16 apart" layout. We verified this matches the ggml `quantize_row_q4_K_ref` encoding function.

3. **CUDA kernel divergence** — The CUDA Q4_K kernel was updated to match the CPU dequant, but floating-point reduction order differs between CPU (sequential) and GPU (warp reduction). This shouldn't cause garbage but could explain small differences.

### How to Debug Further

1. **Layer-by-layer comparison**: Run the Q8_0 and Q4_K_M models through one layer at a time, comparing hidden states after each layer. Find where the divergence becomes catastrophic.

2. **Compare against llama.cpp**: Run the same model in llama.cpp, dump intermediate activations, and compare. llama.cpp's Q4_K implementation is battle-tested.

3. **Test with a 1-layer model**: If a model existed with only 1 transformer layer, we could verify if a single Q4_K layer produces correct output.

4. **Binary search layers**: Modify the forward pass to use Q8_0 for all layers except one Q4_K layer. If output is correct, the individual Q4_K matmul is fine and the issue is accumulation. If garbage, the Q4_K matmul has a subtle bug.

### Affected Models

Any model using Q4_K, Q5_K, or Q6_K quantization:
- Qwen3-8B-Q4_K_M
- Qwen3.5-9B-Q4_K_M (also has DeltaNet issues)
- Any *_K_M, *_K_S, *_K_L GGUF files

### Workaround

Use Q8_0 quantization. All Q8_0 models work correctly.

---

## 2. Qwen3.5-9B DeltaNet Architecture

### Symptom

Qwen3.5-9B produces garbage output regardless of quantization format (Q8_0, Q4_K_M, Q4_0 all tested). The Qwen3.5-0.8B model with the same architecture family works correctly.

### Architecture Differences: 0.8B vs 9B

| Property | 0.8B (works) | 9B (broken) |
|----------|-------------|-------------|
| `SsmGroupCount` (metadata) | 16 | 16 |
| `ssm_a` tensor size | [16] | [32] |
| `ssm_alpha.weight` dims | [1024, 16] | [4096, 32] |
| Actual num_v_heads | 16 | 32 |
| Actual num_k_heads | 16 | 16 |
| `attn_qkv.weight` output dim | 6144 = 3×2048 | 8192 ≠ 3×4096 |
| QKV split | Equal: Q=K=V=2048 | Unequal: Q=2048, K=2048, V=4096 |
| Conv1d channels | 6144 (full QKV) | 8192 (full QKV) |
| Q/K repeat-interleave needed | No (16→16) | Yes (16→32) |
| `SsmStateSize` (metadata) | 128 | 128 |
| headDim | 128 | 128 |

### Research Findings (from HuggingFace Transformers)

The Qwen3.5 DeltaNet layers use `Qwen3NextGatedDeltaNet`:

1. **Projections**:
   - `in_proj_qkvz` → Q + K + V + Z (output gate), split into `attn_qkv` (Q+K+V) and `attn_gate` (Z) in GGUF
   - `in_proj_ba` → beta + alpha, stored as `ssm_alpha.weight` and `ssm_beta.weight` in GGUF

2. **Dimension mapping**:
   - `key_dim = num_k_heads × head_dim = 16 × 128 = 2048`
   - `value_dim = num_v_heads × head_dim = 32 × 128 = 4096`
   - `attn_qkv output = key_dim + key_dim + value_dim = 2048 + 2048 + 4096 = 8192`

3. **Data flow**:
   ```
   x → attn_qkv → [Q(2048), K(2048), V(4096)]
   [Q, K, V] → conv1d(8192 channels) → SiLU
   Q, K → L2 normalize per head (16 heads)
   Q, K → repeat_interleave(16 → 32 heads)
   alpha, beta → from ssm_alpha/ssm_beta projections
   decay = exp(-exp(A_log) × softplus(alpha + dt_bias))
   beta_val = sigmoid(beta)
   DeltaNet state update with 32 groups, headDim=128
   output → RMSNorm → × SiLU(Z) → ssm_out projection
   ```

4. **Metadata mismatch**: GGUF metadata `ssm.group_count = 16` but actual num_v_heads = 32. The code derives the actual group count from `ssm_alpha.Dimensions[1]`.

### What's Been Implemented

- Unequal QKV split: Q(keyDim) + K(keyDim) + V(valueDim)
- Q/K repeat-interleave from num_k_heads to num_v_heads
- Buffer sizes derived from actual weight tensor dimensions
- Correct group count from weight tensors

### What Might Still Be Wrong

1. **V handling after conv/SiLU**: In the current implementation, Q+K+V all go through conv1d and SiLU together (since conv operates on all 8192 channels). After split, V has been convolved and SiLU'd. This matches the reference (conv operates on the full concatenated Q+K+V).

2. **Split ordering**: We assume the QKV output is [Q, K, V] in that order. If it's [Q, V, K] or some other permutation, the split would be wrong.

3. **Repeat-interleave direction**: We repeat-interleave Q and K AFTER L2 normalization. The reference normalizes Q and K with num_k_heads groups, then repeat-interleaves to num_v_heads. Our implementation matches this order.

4. **Decay computation**: Current code: `decay = exp(sA[g] * softplus(alpha + dt))`. Reference: `decay = exp(-exp(A_log) * softplus(alpha + dt))`. If `sA` stores `-exp(A_log)` (pre-negated), these are equivalent. If `sA` stores `A_log` directly, we're missing the inner `exp()` and negation. The 0.8B model works with the current formula, suggesting `sA` stores the pre-computed value.

5. **SplitUnequal GPU path**: For GPU tensors, `SplitUnequal` downloads to CPU, splits, and re-uploads. This adds latency but should be functionally correct. The `RepeatInterleave` similarly downloads/re-uploads.

6. **State tensor sizing**: DeltaNetState allocates state with `config.SsmGroupCount(16) × config.SsmHeadDim(256)²` but ForwardDeltaNet uses `numVHeads(32) × headDim(128)²`. The allocated state is larger (1M vs 524K elements) so there's no overflow, but the memory layout assumptions may differ.

### Tensor Name Mapping

| HuggingFace | GGUF | Shape (9B) | Purpose |
|-------------|------|-----------|---------|
| in_proj_qkvz (Q+K+V) | blk.N.attn_qkv.weight | [4096, 8192] | Q+K+V projection |
| in_proj_qkvz (Z) | blk.N.attn_gate.weight | [4096, 4096] | Output gate |
| in_proj_ba (alpha) | blk.N.ssm_alpha.weight | [4096, 32] | Alpha projection |
| in_proj_ba (beta) | blk.N.ssm_beta.weight | [4096, 32] | Beta projection |
| conv1d | blk.N.ssm_conv1d.weight | [4, 8192] | Causal conv1d |
| A_log | blk.N.ssm_a | [32] | Decay parameter |
| dt_bias | blk.N.ssm_dt.bias | [32] | Timestep bias |
| norm | blk.N.ssm_norm.weight | [128] | Per-head RMSNorm |
| out_proj | blk.N.ssm_out.weight | [4096, 4096] | Output projection |

### Layer Schedule

FullAttentionInterval = 4. Every 4th layer (3, 7, 11, ...) is standard attention; all others are DeltaNet. Total: 24 DeltaNet + 8 standard attention = 32 layers.

### How to Debug Further

1. **Compare against llama.cpp**: llama.cpp has a working Qwen3.5 implementation (`src/models/qwen35.cpp`). Compare intermediate DeltaNet state values layer by layer.

2. **Test with Qwen3.5-0.8B**: The 0.8B model exercises the same DeltaNet code path (with equal QKV split). Since it works, the base DeltaNet logic is correct. The bug is in the unequal split / repeat-interleave path specific to the 9B model.

3. **Verify split ordering**: Dump the first few values of Q, K, V after splitting and compare to what llama.cpp produces. If the values don't match, the split is in the wrong order.

4. **Check alpha/beta ordering**: The GGUF has separate `ssm_alpha` and `ssm_beta` tensors. In the reference, `in_proj_ba` outputs [beta, alpha] (beta first). If the GGUF stores them in the same order but our code swaps them, the decay/update computation would be wrong.

### References

- [HuggingFace Qwen3Next modeling code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py)
- [llama.cpp Qwen3.5 model](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp)
- [llama.cpp DeltaNet base](https://github.com/ggml-org/llama.cpp/blob/master/src/models/delta-net-base.cpp)
- [NVLabs Gated DeltaNet](https://github.com/NVlabs/GatedDeltaNet)
- [Qwen3.5 architecture blog](https://huggingface.co/blog/mlabonne/qwen35)

---

## 3. Qwen3 HasGatedQ Detection (Fixed)

### Problem

The `HasGatedQ` property on `StandardAttentionWeights` returned true whenever Q/K norms existed, but Qwen3 models have Q/K norms WITHOUT gated Q (the Q projection output equals hiddenDim, not 2× hiddenDim).

This caused a null reference crash when trying to use `_qFull` (which is only allocated for gated Q models).

### Fix

Changed `HasGatedQ` to check if the Q weight output dimension exceeds the hidden dimension:
```csharp
public bool HasGatedQ => AttnQNorm != null && qOutDim > hiddenDim;
```

| Model | Q output | hiddenDim | HasGatedQ |
|-------|----------|-----------|-----------|
| Qwen3-8B | 4096 | 4096 | false (norms but not gated) |
| Qwen3.5-0.8B | 2048 | 1024 | true (2× hiddenDim) |
| Qwen3.5-9B | 8192 | 4096 | true (2× hiddenDim) |
| DeepSeek R1 (LLaMA) | 4096 | 4096 | false (no norms) |

Non-gated models with Q/K norms now correctly apply per-head RMSNorm without attempting the gated Q de-interleave.

---

## 4. CausalConv1d Buffer Overflow (Fixed)

### Problem

The 9B model's `ssm_alpha.weight` has dimensions [4096, 32] (outputting 32 values) but `_ssmAlpha` was allocated for `config.SsmGroupCount = 16` elements. MatMul wrote 32 floats into a 16-element buffer, corrupting adjacent managed heap memory.

This manifested as `System.AccessViolationException` at `CastHelpers.ChkCastClassSpecial` — the GC's type-checking infrastructure found corrupted object headers.

### Fix

Derive buffer sizes from actual weight tensor dimensions instead of config metadata:
```csharp
int numVHeads = deltaLayer.SsmAlpha.Dimensions[1]; // 32, not config.SsmGroupCount=16
_ssmAlpha = CreateF32("scratch_ssm_alpha", numVHeads);
```
