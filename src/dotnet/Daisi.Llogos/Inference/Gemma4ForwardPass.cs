using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Inference;

/// <summary>
/// Gemma 4 forward pass with interleaved sliding-window + full attention,
/// Per-Layer Embeddings (PLE), GeGLU FFN, and final logit softcap.
///
/// Reference: llama.cpp <c>src/models/gemma4-iswa.cpp</c> (<c>llm_build_gemma4_iswa</c>).
///
/// Implementation notes:
///  - Attention scale is hard-coded to 1.0 (matches llama.cpp's
///    <c>hparams.f_attention_scale = 1.0f</c> for gemma4).
///  - V projection is unit-RmsNorm'd at runtime (no learned weight).
///  - KV cache sharing: layers &gt;= <c>NumLayerKvFromStart</c> skip their own
///    K/V computation and read the cache written by an earlier layer (sliding
///    reuses layer <c>NumLayerKvFromStart-2</c>, full reuses
///    <c>NumLayerKvFromStart-1</c>). The "dead" wk/wv tensors for the shared
///    layers are loaded but never executed.
///  - GeGLU is a parallel-style fused matmul + activation, computed as
///    <c>down(GeluTanh(gate(x)) * up(x))</c>.
/// </summary>
public sealed class Gemma4ForwardPass : IDisposable
{
    private readonly IComputeBackend _backend;
    private readonly ModelConfig _config;
    private readonly ModelWeights _weights;
    private readonly Gemma4KvCache _kvCache;

    // ── Scratch buffers (sized for the FULL-attention head dim — sliding layers
    //    use a prefix of these buffers since their dim is smaller). ────────────
    private readonly ITensor _hidden;
    private readonly ITensor _residual;
    private readonly ITensor _normOut;
    private readonly ITensor _logits;
    private readonly float[] _logitsBuffer;

    // Attention scratch (sized for full-attn head dim)
    private readonly ITensor _qProj;    // [numHeads * keyLengthFull]
    private readonly ITensor _kProj;    // [numKvHeads * keyLengthFull]
    private readonly ITensor _vProj;    // [numKvHeads * valueLengthFull]
    private readonly ITensor _qGate;    // dummy gate (filled with 88.0 → sigmoid≈1.0)
    private readonly ITensor _attnOut;  // [numHeads * valueLengthFull]

    // Smaller-typed views — created lazily per layer with the right element count
    // by reusing the underlying tensor's memory. We can't easily do that without
    // rewriting CpuTensor; instead, we allocate small layer-specific scratch
    // buffers when the head dim is smaller. This wastes a tiny bit of memory
    // but keeps the code simple.
    private readonly ITensor _qProjSwa;
    private readonly ITensor _kProjSwa;
    private readonly ITensor _vProjSwa;
    private readonly ITensor _qGateSwa;
    private readonly ITensor _attnOutSwa;

    // FFN scratch
    private readonly ITensor _ffnGate;
    private readonly ITensor _ffnUp;

    // PLE scratch (allocated only if PLE is active)
    private readonly bool _hasPle;
    private readonly ITensor? _pleBase;       // [n_embd_per_layer * n_layer] — token embed lookup
    private readonly ITensor? _pleProjOut;    // [n_embd_per_layer * n_layer] — projection from hidden
    private readonly ITensor? _pleSlice;      // [n_embd_per_layer] — current layer slice
    private readonly ITensor? _pleGate;       // [n_embd_per_layer] — gated PLE
    private readonly ITensor? _pleScratch;    // [n_embd_per_layer] — temporary for norms

    public Gemma4KvCache KvCache => _kvCache;

    /// <summary>Debug: disable Per-Layer Embedding block (use only for debugging).</summary>
    public bool DebugDisablePle { get; set; }

    /// <summary>Debug: disable per-layer output scale (use only for debugging).</summary>
    public bool DebugDisableLayerOutScale { get; set; }

    /// <summary>Debug: disable embedding sqrt(hidden_dim) scale (use only for debugging).</summary>
    public bool DebugDisableEmbeddingScale { get; set; }

    /// <summary>Debug: disable final logit softcap (use only for debugging).</summary>
    public bool DebugDisableLogitSoftcap { get; set; }

    /// <summary>Debug: disable per-head Q/K RmsNorm.</summary>
    public bool DebugDisableQkNorm { get; set; }

    /// <summary>Debug: disable V unit RmsNorm.</summary>
    public bool DebugDisableVNorm { get; set; }

    /// <summary>Debug: disable RoPE entirely (positions ignored).</summary>
    public bool DebugDisableRope { get; set; }

    public Gemma4ForwardPass(IComputeBackend backend, ModelConfig config, ModelWeights weights, Gemma4KvCache kvCache)
    {
        _backend = backend;
        _config = config;
        _weights = weights;
        _kvCache = kvCache;

        _hidden   = CreateF32("g4_hidden", config.HiddenDim);
        _residual = CreateF32("g4_residual", config.HiddenDim);
        _normOut  = CreateF32("g4_normOut", config.HiddenDim);
        _logits   = CreateF32("g4_logits", config.VocabSize);
        _logitsBuffer = new float[config.VocabSize];

        int numH = config.NumHeads;
        int numKvH = config.NumKvHeads;
        int kFull = config.KeyLength;       // full-attn head dim (512 for E4B-it)
        int vFull = config.ValueLength;
        int kSwa  = config.KeyLengthSwa;    // sliding head dim (256 for E4B-it)
        int vSwa  = config.ValueLengthSwa;

        // Full-attention scratch
        _qProj   = CreateF32("g4_q_full", numH   * kFull);
        _kProj   = CreateF32("g4_k_full", numKvH * kFull);
        _vProj   = CreateF32("g4_v_full", numKvH * vFull);
        _qGate   = CreateF32("g4_qg_full", numH * kFull);
        _attnOut = CreateF32("g4_ao_full", numH * vFull);
        backend.FillTensor(_qGate, 88.0f); // sigmoid(88) ≈ 1.0 → ungated path

        // Sliding-attention scratch (smaller, used by SWA layers).
        // If the model has no sliding layers, kSwa == 0 — just point to the full ones.
        if (kSwa > 0)
        {
            _qProjSwa   = CreateF32("g4_q_swa", numH   * kSwa);
            _kProjSwa   = CreateF32("g4_k_swa", numKvH * kSwa);
            _vProjSwa   = CreateF32("g4_v_swa", numKvH * vSwa);
            _qGateSwa   = CreateF32("g4_qg_swa", numH * kSwa);
            _attnOutSwa = CreateF32("g4_ao_swa", numH * vSwa);
            backend.FillTensor(_qGateSwa, 88.0f);
        }
        else
        {
            _qProjSwa = _qProj;
            _kProjSwa = _kProj;
            _vProjSwa = _vProj;
            _qGateSwa = _qGate;
            _attnOutSwa = _attnOut;
        }

        _ffnGate = CreateF32("g4_ffn_gate", config.IntermediateDim);
        _ffnUp   = CreateF32("g4_ffn_up",   config.IntermediateDim);

        // PLE scratch — only if the model has per-layer embeddings.
        _hasPle = config.PerLayerInputDim > 0
            && weights.PerLayerTokenEmbd != null
            && weights.PerLayerModelProj != null
            && weights.PerLayerProjNorm != null;
        if (_hasPle)
        {
            int pleTotal = config.PerLayerInputDim * config.NumLayers; // 256 × 42 = 10752
            _pleBase    = CreateF32("g4_ple_base",    pleTotal);
            _pleProjOut = CreateF32("g4_ple_proj",    pleTotal);
            _pleSlice   = CreateF32("g4_ple_slice",   config.PerLayerInputDim);
            _pleGate    = CreateF32("g4_ple_gate",    config.PerLayerInputDim);
            _pleScratch = CreateF32("g4_ple_scratch", config.HiddenDim);
        }
    }

    public void ResetState() => _kvCache.Reset();

    /// <summary>Run the full forward pass for one token and return the (capped) logits.</summary>
    public ReadOnlySpan<float> Forward(int tokenId, int position)
    {
        ForwardTransformer(tokenId, position);

        // Final RmsNorm + lm_head
        _backend.RmsNorm(_normOut, _hidden, _weights.OutputNorm, _config.NormEps);
        ProjectLinear(_logits, _normOut, _weights.OutputWeight);

        // Final logit softcap (Gemma 2/3/4 trait)
        if (_config.FinalLogitSoftcap > 0 && !DebugDisableLogitSoftcap)
            _backend.LogitSoftcap(_logits, _config.FinalLogitSoftcap);

        _logits.DequantizeTo(_logitsBuffer);
        return _logitsBuffer;
    }

    public void ForwardHidden(int tokenId, int position) => ForwardTransformer(tokenId, position);

    private void ForwardTransformer(int tokenId, int position)
    {
        // 1. Embedding lookup + sqrt(hidden_dim) scale
        _backend.EmbeddingLookup(_hidden, _weights.TokenEmbedding, tokenId);
        if (!DebugDisableEmbeddingScale)
            _backend.ScaleInPlace(_hidden, MathF.Sqrt(_config.HiddenDim));

        // 2. Per-Layer-Embedding setup (once per token, before the layer loop)
        if (_hasPle && !DebugDisablePle)
            SetupPerLayerEmbeddings(tokenId);

        // 3. Transformer layers
        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            var lw = (GemmaAttentionWeights)_weights.Layers[layer];

            // ── Pre-attention norm ──────────────────────────────────────────
            _backend.RmsNorm(_normOut, _hidden, lw.AttnNorm, _config.NormEps);

            // Save residual (the value of inpL going into this layer's attention)
            _backend.CopyTensor(_residual, _hidden);

            // ── Attention ───────────────────────────────────────────────────
            ForwardAttention(lw, layer, position);

            // ── Post-attention norm + residual add ──────────────────────────
            // _hidden currently holds Wo·attn_output. Apply post-attn norm in place,
            // then add the saved residual (= inpL).
            _backend.RmsNorm(_normOut, _hidden, lw.PostAttnNorm, _config.NormEps);
            _backend.ElementAdd(_hidden, _normOut, _residual);

            // From here _hidden = attn_out (= post-norm(attn) + inpL)

            // ── Pre-FFN norm + GeGLU FFN ────────────────────────────────────
            _backend.CopyTensor(_residual, _hidden); // save attn_out for residual
            _backend.RmsNorm(_normOut, _hidden, lw.FfnNorm, _config.NormEps);

            ProjectLinear(_ffnGate, _normOut, lw.FfnGate);
            ProjectLinear(_ffnUp,   _normOut, lw.FfnUp);
            _backend.GeGLU(_ffnGate, _ffnGate, _ffnUp);
            ProjectLinear(_hidden, _ffnGate, lw.FfnDown);

            // ── Post-FFN norm + residual add ────────────────────────────────
            _backend.RmsNorm(_normOut, _hidden, lw.PostFfnNorm, _config.NormEps);
            _backend.ElementAdd(_hidden, _normOut, _residual);

            // ── Per-Layer Embedding block (Gemma 3n / Gemma 4 PLE) ──────────
            if (_hasPle && !DebugDisablePle && lw.PerLayerInpGate != null && lw.PerLayerProj != null && lw.PerLayerPostNorm != null)
            {
                ApplyPerLayerEmbedding(lw, layer);
            }

            // ── Per-layer scalar output multiplier ──────────────────────────
            if (lw.LayerOutScale != null && !DebugDisableLayerOutScale)
            {
                // LayerOutScale is shape [1] — broadcast multiply
                var scaleSpan = lw.LayerOutScale.AsFloatSpan();
                if (scaleSpan.Length > 0)
                    _backend.ScaleInPlace(_hidden, scaleSpan[0]);
            }
        }
    }

    /// <summary>
    /// Compute attention for one layer. Handles both sliding-window and full-attention
    /// layer types — they differ in head dim, RoPE theta, and rope_freqs application.
    ///
    /// KV cache sharing: layers with index >= <see cref="ModelConfig.NumLayerKvFromStart"/>
    /// do NOT compute K/V from their own (dead) wk/wv weights. They reuse the cache
    /// written by an earlier layer — sliding layers reuse (NumLayerKvFromStart-2) and
    /// full layers reuse (NumLayerKvFromStart-1), mirroring llama.cpp's gemma4 reuse cb.
    /// </summary>
    private void ForwardAttention(GemmaAttentionWeights w, int layer, int position)
    {
        bool isSwa = _config.IsSlidingLayer(layer);
        bool hasKv = _config.HasKv(layer);
        int numH = _config.NumHeads;
        int numKvH = _config.NumKvHeads;
        int headDim = _config.LayerKeyLength(layer);     // 256 sliding, 512 full
        int valDim  = _config.LayerValueLength(layer);   // same as headDim for gemma4
        int ropeDim = _config.LayerRopeDim(layer);
        float ropeTheta = _config.LayerRopeTheta(layer);
        const float attentionScale = 1.0f; // gemma4 hard-codes scale = 1.0

        // For shared-KV layers, pick the source layer whose cache we'll read.
        // Matches llama.cpp's reuse lambda in llama-model.cpp:
        //   sliding → (n_layer_kv_from_start - 2), full → (n_layer_kv_from_start - 1).
        int sourceLayer = hasKv ? layer
            : (_config.NumLayerKvFromStart - (isSwa ? 2 : 1));

        // Pick the right scratch buffers for this layer's head dim
        var qProj   = isSwa ? _qProjSwa   : _qProj;
        var kProj   = isSwa ? _kProjSwa   : _kProj;
        var vProj   = isSwa ? _vProjSwa   : _vProj;
        var qGate   = isSwa ? _qGateSwa   : _qGate;
        var attnOut = isSwa ? _attnOutSwa : _attnOut;

        // ── Q projection (always) ───────────────────────────────────────────
        ProjectLinear(qProj, _normOut, w.AttnQ);

        if (!DebugDisableQkNorm)
            _backend.PerHeadRmsNorm(qProj, w.AttnQNorm, numH, headDim, _config.NormEps);

        if (hasKv)
        {
            // ── K/V projections ─────────────────────────────────────────────
            ProjectLinear(kProj, _normOut, w.AttnK);
            ProjectLinear(vProj, _normOut, w.AttnV);

            if (!DebugDisableQkNorm)
                _backend.PerHeadRmsNorm(kProj, w.AttnKNorm, numKvH, headDim, _config.NormEps);
            if (!DebugDisableVNorm)
                _backend.PerHeadRmsNormUnit(vProj, numKvH, valDim, _config.NormEps);

            // ── RoPE (Q and K together) ─────────────────────────────────────
            // Gemma 4 uses NEOX-style RoPE. Full-attention layers also apply
            // proportional RoPE via the precomputed rope_freqs table.
            if (!DebugDisableRope)
            {
                if (isSwa || w.RopeFreqs == null)
                    _backend.RoPENeox(qProj, kProj, headDim, ropeDim, position, ropeTheta);
                else
                    _backend.RoPENeoxWithFreqFactors(qProj, kProj, headDim, ropeDim, position, ropeTheta, w.RopeFreqs);
            }

            // Write K/V to THIS layer's cache
            _kvCache.Write(_backend, layer, position, kProj, vProj);
        }
        else
        {
            // Shared-KV layer: no new K/V; rotate Q only. kProj is used as an
            // ignored scratch buffer (RoPE rotates it in place but we never read
            // the result — attention reads the source layer's cache instead).
            if (!DebugDisableRope)
            {
                if (isSwa || w.RopeFreqs == null)
                    _backend.RoPENeox(qProj, kProj, headDim, ropeDim, position, ropeTheta);
                else
                    _backend.RoPENeoxWithFreqFactors(qProj, kProj, headDim, ropeDim, position, ropeTheta, w.RopeFreqs);
            }
        }

        // ── Attention (reads sourceLayer's cache, which == layer when hasKv) ─
        int seqLen = _kvCache.LayerSeqLen(sourceLayer);
        var kCacheT = _kvCache.GetKCacheTensor(sourceLayer);
        var vCacheT = _kvCache.GetVCacheTensor(sourceLayer);
        int cap = _kvCache.LayerCapacity(sourceLayer);

        _backend.GatedAttention(attnOut, qProj, qGate, kCacheT, vCacheT,
            numH, numKvH, headDim, valDim, cap, seqLen, attentionScale);

        // ── Output projection ───────────────────────────────────────────────
        // attn_output weight shape: [num_heads * head_dim, hidden_dim]
        ProjectLinear(_hidden, attnOut, w.AttnO);
    }

    /// <summary>
    /// Build the global per-layer-embedding tensor for one token. This runs once
    /// per token (before the layer loop) and produces a [n_embd_per_layer × n_layer]
    /// tensor stored in <see cref="_pleBase"/>, ready to be sliced per layer.
    ///
    /// Steps (matches llama.cpp's <c>build_inp_per_layer</c> + <c>project_per_layer_inputs</c>):
    ///  1. ple_base = per_layer_token_embd[token] × sqrt(n_embd_per_layer)
    ///  2. ple_proj = (per_layer_model_proj · hidden) × (1/sqrt(hidden_dim))
    ///  3. ple_proj = RmsNorm(ple_proj, per_layer_proj_norm)   (per [256] slice)
    ///  4. ple_base = (ple_proj + ple_base) × (1/sqrt(2))
    /// </summary>
    private void SetupPerLayerEmbeddings(int tokenId)
    {
        int nEmbdPerLayer = _config.PerLayerInputDim;
        int nLayer = _config.NumLayers;
        int total = nEmbdPerLayer * nLayer;

        // Step 1: lookup the per-layer token embedding row
        // per_layer_token_embd shape: [nEmbdPerLayer*nLayer × vocab]
        _backend.EmbeddingLookup(_pleBase!, _weights.PerLayerTokenEmbd!, tokenId);
        _backend.ScaleInPlace(_pleBase!, MathF.Sqrt(nEmbdPerLayer));

        // Step 2: project hidden through per_layer_model_proj
        // per_layer_model_proj shape: [hidden_dim × (nEmbdPerLayer*nLayer)] (BF16)
        // Output: [nEmbdPerLayer*nLayer]
        ProjectLinear(_pleProjOut!, _hidden, _weights.PerLayerModelProj!);
        _backend.ScaleInPlace(_pleProjOut!, 1.0f / MathF.Sqrt(_config.HiddenDim));

        // Step 3: RmsNorm each [nEmbdPerLayer] slice with per_layer_proj_norm.
        // The norm weight has shape [nEmbdPerLayer]. Apply per "row" (each row = one layer's slice).
        // Existing PerHeadRmsNorm fits the shape: numHeads=nLayer, headDim=nEmbdPerLayer.
        _backend.PerHeadRmsNorm(_pleProjOut!, _weights.PerLayerProjNorm!, nLayer, nEmbdPerLayer, _config.NormEps);

        // Step 4: ple_base = (ple_proj + ple_base) × (1/sqrt(2))
        _backend.ElementAdd(_pleBase!, _pleProjOut!, _pleBase!);
        _backend.ScaleInPlace(_pleBase!, 1.0f / MathF.Sqrt(2.0f));
    }

    /// <summary>
    /// Apply the per-layer-embedding block at the end of one transformer layer.
    /// On entry, _hidden holds the post-FFN+residual value. On exit, _hidden has
    /// the PLE contribution added.
    /// </summary>
    private void ApplyPerLayerEmbedding(GemmaAttentionWeights w, int layer)
    {
        int nEmbdPerLayer = _config.PerLayerInputDim;

        // Save current _hidden (=pe_in, the residual to add at the end)
        _backend.CopyTensor(_residual, _hidden);

        // 1. cur (256-vec) = inp_gate · _hidden ([2560 → 256])
        ProjectLinear(_pleGate!, _hidden, w.PerLayerInpGate!);

        // 2. cur = GeluTanh(cur)
        _backend.GeluTanh(_pleGate!, _pleGate!);

        // 3. Slice the per-layer 256-vec out of _pleBase: pleBase[layer*256 : (layer+1)*256]
        var pleBaseSpan = _pleBase!.AsFloatSpan();
        var sliceSpan = _pleSlice!.AsFloatSpan();
        pleBaseSpan.Slice(layer * nEmbdPerLayer, nEmbdPerLayer).CopyTo(sliceSpan);

        // 4. cur = cur * pleSlice (element-wise)
        _backend.ElementMul(_pleGate!, _pleGate!, _pleSlice!);

        // 5. cur (2560-vec) = per_layer_proj · cur ([256 → 2560])
        ProjectLinear(_pleScratch!, _pleGate!, w.PerLayerProj!);

        // 6. cur = RmsNorm(cur, per_layer_post_norm)
        _backend.RmsNorm(_pleScratch!, _pleScratch!, w.PerLayerPostNorm!, _config.NormEps);

        // 7. _hidden = pe_in + cur
        _backend.ElementAdd(_hidden, _residual, _pleScratch!);
    }

    private void ProjectLinear(ITensor output, ITensor input, ITensor weight)
    {
        int K = (int)weight.Dimensions[0];
        int N = (int)weight.Dimensions[1];
        _backend.MatMul(output, input, weight, 1, K, N);
    }

    private ITensor CreateF32(string name, int size) =>
        _backend.CreateTensor(name, GgmlType.F32, [(long)size]);

    public void Dispose()
    {
        _hidden.Dispose();
        _residual.Dispose();
        _normOut.Dispose();
        _logits.Dispose();
        _qProj.Dispose();
        _kProj.Dispose();
        _vProj.Dispose();
        _qGate.Dispose();
        _attnOut.Dispose();
        if (_qProjSwa != _qProj)
        {
            _qProjSwa.Dispose();
            _kProjSwa.Dispose();
            _vProjSwa.Dispose();
            _qGateSwa.Dispose();
            _attnOutSwa.Dispose();
        }
        _ffnGate.Dispose();
        _ffnUp.Dispose();
        _pleBase?.Dispose();
        _pleProjOut?.Dispose();
        _pleSlice?.Dispose();
        _pleGate?.Dispose();
        _pleScratch?.Dispose();
    }
}
