# Known Issues

Tracking document for bugs that have been investigated and resolved. All issues below are **fixed**.

---

## 1. K-Quant Models (Q4_K_M, Q5_K_M, Q6_K) Produce Garbage (Fixed)

### Symptom

Models quantized with K-quant formats produced incoherent output (random tokens, mixed languages, nonsense) despite Q8_0 versions of the same model working perfectly.

Example with Qwen3-8B (no DeltaNet, pure standard attention):
- **Q8_0**: `"Paris. Which other city in the same country has..."` — correct
- **Q4_K_M**: `"鳁 Latin source ihammer轧 hit..."` — garbage

### Root Cause

Multiple dequantization bugs, confirmed by comparison against ggml reference (`dequantize_row_q*_K` in ggml-quants.c):

1. **Q6_K: Missing `l/16` scale sub-indexing** — The ggml reference uses `int is = l/16` to select between two sub-group scales within each 32-element loop. For l=0..15, scales sc[0], sc[2], sc[4], sc[6] are used; for l=16..31, sc[1], sc[3], sc[5], sc[7]. Our code always used the l=0..15 scales, meaning half of all Q6_K elements were dequantized with the wrong scale. Since Q4_K_M and Q5_K_M models use Q6_K for critical weights (output projection, FFN down, attention), this corrupted the most important tensors.

2. **Q5_K: Wrong qs nibble layout** — Code assumed first 128 elements use low nibbles and next 128 use high nibbles. The ggml layout is chunked: each 32-byte qs block serves 64 elements (32 low + 32 high), identical to Q4_K. This scrambled the weight values.

3. **Q5_K: Wrong qh bit layout** — Code read qh as a flat 256-bit bitfield (`qh[n/8] >> (n%8)`). The ggml layout stores element n's high bit at `qh[n%32] >> (n/32)`, using a rotating bitmask (`u1/u2 <<= 2` per 64-element chunk). This mapped every element's 5th bit to the wrong source bit.

4. **Q3_K: Wrong type size** — Constant was 108 but the actual ggml block size is 110 (2+32+64+12). Every super-block after the first was read from the wrong byte offset, compounding by 2 bytes per block.

5. **Q3_K: Incorrect scale extraction** — The 12-byte scale packing uses a complex 6-bit scheme with high bits stored in a separate 4-byte field. The original code used a simplified 4-bit extraction that produced wrong scales for all 16 sub-blocks.

### Fix

- `Dequantize.cs:DequantizeQ6_K` — Added `int isc = l / 16` offset to scale indexing.
- `Dequantize.cs:DequantizeQ5_K` — Rewrote to use chunked 64-element loop with rotating `u1/u2` bitmasks (matching ggml `dequantize_row_q5_K`).
- `Dequantize.cs:DequantizeQ3_K` — Fixed type size to 110, rewrote scale unpacking to use full 6-bit extraction with high-bit merge, rewrote dequant loop to use rotating hmask bitmask.
- `GgmlType.cs:TypeSize(Q3_K)` — Changed from 108 to 110.

### Validated

- Qwen3-8B Q4_K_M: `"Paris, and the Eiffel Tower is in Paris..."` — coherent English
- Q4_K_M first token matches Q8_0 first token (greedy)
- Q6_K and Q5_K dequant unit tests pass against known values
- 253/253 tests pass with no regressions

---

## 2. Qwen3.5-9B DeltaNet Conv Buffer Overflow (Fixed)

### Symptom

Qwen3.5-9B produced garbage output regardless of quantization format (Q8_0, Q4_K_M, Q4_0 all tested). The Qwen3.5-0.8B model with the same architecture family worked correctly.

### Root Cause

**Conv buffer too small for unequal QKV models.** `DeltaNetState` allocated the conv1d shift buffer as `(convKernel-1) × SsmInnerSize × 3`, assuming equal Q/K/V dimensions. For the 9B model, the actual QKV output dimension is 8192 (Q:2048 + K:2048 + V:4096), but `SsmInnerSize × 3 = 2048 × 3 = 6144`. The conv buffer was 6144 channels per slot when conv1d needed 8192, causing out-of-bounds memory access that corrupted the QKV values at every DeltaNet layer.

The 0.8B model works because its QKV dimensions are equal (3 × 2048 = 6144 = SsmInnerSize × 3).

### Fix

- `DeltaNetState.cs` — Constructor now accepts optional `ModelWeights` parameter. When provided, derives the actual QKV output dimension from `AttnQkv.Dimensions[1]` instead of assuming `SsmInnerSize * 3`. Also derives `numVHeads` from `SsmAlpha.Dimensions[1]` for correct state tensor sizing.
- `Program.cs`, `DaisiLlogosModelHandle.cs` — Updated to pass `weights` to `DeltaNetState` constructor.

### Validated

- Qwen3.5-9B Q8_0 and Q4_K_M both produce coherent English output
- Conv buffer correctly sized for 8192 channels (verified by assertion)
- 253/253 tests pass with no regressions

---

## 3. Qwen3 HasGatedQ Detection (Fixed)

### Symptom

The `HasGatedQ` property on `StandardAttentionWeights` returned true whenever Q/K norms existed, but Qwen3 models have Q/K norms WITHOUT gated Q (the Q projection output equals hiddenDim, not 2× hiddenDim). This caused a null reference crash.

### Fix

Changed `HasGatedQ` to check if the Q weight output dimension exceeds the hidden dimension:
```csharp
public bool HasGatedQ => AttnQNorm != null && qOutDim > hiddenDim;
```

---

## 4. CausalConv1d Buffer Overflow (Fixed)

### Symptom

The 9B model's `ssm_alpha.weight` has dimensions [4096, 32] (outputting 32 values) but `_ssmAlpha` was allocated for `config.SsmGroupCount = 16` elements. MatMul wrote 32 floats into a 16-element buffer, corrupting adjacent managed heap memory.

### Fix

Derive buffer sizes from actual weight tensor dimensions instead of config metadata:
```csharp
int numVHeads = deltaLayer.SsmAlpha.Dimensions[1]; // 32, not config.SsmGroupCount=16
_ssmAlpha = CreateF32("scratch_ssm_alpha", numVHeads);
```

---

## 5. Qwen3.5-9B DeltaNet Output Quality

### Symptom

After fixing the conv buffer overflow (issue 2) and K-quant bugs (issue 1), the 9B model produces coherent English text but with lower quality than expected. The 0.8B model correctly answers "The capital of France is Paris" while the 9B model produces semi-coherent but wrong content like `"the name of France, which country is the one with that name was actually in 2019-2019-201"`.

### What's Been Verified Correct

1. **Decay formula** — Confirmed against llama.cpp `qwen35.cpp`: `gate = softplus(alpha + dt_bias) * ssm_a` where ssm_a stores pre-computed `-exp(A_log)`. Our formula matches.
2. **QKV split order** — Confirmed as [Q, K, V] by llama.cpp reference.
3. **Repeat-interleave** — Applied after L2 normalization, matching the reference order.
4. **DeltaNet state update math** — The delta rule (error → state update → output) matches the `fla` library formulation.
5. **Conv1d buffer management** — Stores pre-conv values, shifts correctly, conv weight ordering matches GGUF layout.
6. **SiLUGate** — `output * SiLU(gate)` matches the reference `output × SiLU(Z)`.
7. **Buffer sizes** — All scratch buffers correctly sized from actual weight tensor dimensions.

### Possible Remaining Issues

1. **Chat model without chat template** — Qwen3.5 is a chat/thinking model. Raw text completion (without the expected chat template) may produce lower quality output. The 0.8B model also degenerates quickly after saying "Paris" (going into quiz format). This may not be a code bug.

2. **Error accumulation across 24 DeltaNet layers** — The 9B has 24 DeltaNet layers vs 18 for the 0.8B. Small numerical differences in the recurrent state update could compound more aggressively with more layers.

3. **Chunked vs recurrent DeltaNet** — The reference HuggingFace implementation uses `chunk_gated_delta_rule` (processes multiple tokens at once with matrix operations). Our step-by-step recurrent implementation is mathematically equivalent but may accumulate floating-point errors differently.

### How to Debug Further

1. **Test with chat template** — Format the prompt using Qwen3.5's chat template to see if output quality improves. This would confirm whether the issue is model behavior vs code bug.

2. **Layer-by-layer comparison against llama.cpp** — Dump hidden states after each DeltaNet layer from both implementations and find where they diverge.

3. **Compare first-token logits** — If the top-5 logits after the full forward pass match llama.cpp's output, the implementation is correct and the quality issue is prompt-related.

---

## Architecture Reference: Qwen3.5-9B DeltaNet

Kept for reference since the 9B model uses an unusual unequal QKV split.

### Architecture Differences: 0.8B vs 9B

| Property | 0.8B | 9B |
|----------|------|-----|
| `SsmGroupCount` (metadata) | 16 | 16 |
| `ssm_a` tensor size | [16] | [32] |
| Actual num_v_heads | 16 | 32 |
| Actual num_k_heads | 16 | 16 |
| QKV split | Equal: Q=K=V=2048 | Unequal: Q=2048, K=2048, V=4096 |
| Conv1d channels | 6144 | 8192 |
| Q/K repeat-interleave | No (16→16) | Yes (16→32) |

### Tensor Name Mapping

| HuggingFace | GGUF | Shape (9B) | Purpose |
|-------------|------|-----------|---------|
| in_proj_qkvz (Q+K+V) | blk.N.attn_qkv.weight | [4096, 8192] | Q+K+V projection |
| in_proj_qkvz (Z) | blk.N.attn_gate.weight | [4096, 4096] | Output gate |
| in_proj_ba (alpha) | blk.N.ssm_alpha.weight | [4096, 32] | Alpha projection |
| in_proj_ba (beta) | blk.N.ssm_beta.weight | [4096, 32] | Beta projection |
| conv1d | blk.N.ssm_conv1d.weight | [4, 8192] | Causal conv1d |
| A_log | blk.N.ssm_a | [32] | Decay parameter (stores -exp(A_log)) |
| dt_bias | blk.N.ssm_dt.bias | [32] | Timestep bias |
| norm | blk.N.ssm_norm.weight | [128] | Per-head RMSNorm |
| out_proj | blk.N.ssm_out.weight | [4096, 4096] | Output projection |

### Layer Schedule

FullAttentionInterval = 4. Every 4th layer (3, 7, 11, ...) is standard attention; all others are DeltaNet. Total: 24 DeltaNet + 8 standard attention = 32 layers.

### References

- [HuggingFace Qwen3Next modeling code](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_next/modeling_qwen3_next.py)
- [llama.cpp Qwen3.5 model](https://github.com/ggml-org/llama.cpp/blob/master/src/models/qwen35.cpp)
- [llama.cpp DeltaNet base](https://github.com/ggml-org/llama.cpp/blob/master/src/models/delta-net-base.cpp)
- [NVLabs Gated DeltaNet](https://github.com/NVlabs/GatedDeltaNet)
- [Qwen3.5 architecture blog](https://huggingface.co/blog/mlabonne/qwen35)
