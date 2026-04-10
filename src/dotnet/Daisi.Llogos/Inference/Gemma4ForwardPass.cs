using System.Diagnostics;
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
    private readonly int _maxBatch;

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

    // Single-row scratch tensors used by the per-token attention loop when running
    // in batched mode. Row m of the batched Q/K/V/attnOut is copied in/out of these
    // between batched projections and attention. These share memory with the
    // single-token decode scratch (same size) — they're only used inside
    // ForwardBatch and ForwardBatch never runs concurrently with Forward.
    private readonly ITensor _qSingle;
    private readonly ITensor _kSingle;
    private readonly ITensor _vSingle;
    private readonly ITensor _qGateSingle;
    private readonly ITensor _attnOutSingle;
    private readonly ITensor _qSingleSwa;
    private readonly ITensor _kSingleSwa;
    private readonly ITensor _vSingleSwa;
    private readonly ITensor _qGateSingleSwa;
    private readonly ITensor _attnOutSingleSwa;

    // Batched scratch buffers — sized [maxBatch × dim] row-major. Only used by
    // ForwardBatch. Keeping them separate from the decode scratch means the M=1
    // path is bit-identical to pre-batch (every op still runs against a
    // single-row tensor).
    private readonly ITensor _hiddenB;
    private readonly ITensor _residualB;
    private readonly ITensor _normOutB;
    private readonly ITensor _qProjB;
    private readonly ITensor _kProjB;
    private readonly ITensor _vProjB;
    private readonly ITensor _attnOutB;
    private readonly ITensor _qProjSwaB;
    private readonly ITensor _kProjSwaB;
    private readonly ITensor _vProjSwaB;
    private readonly ITensor _attnOutSwaB;
    private readonly ITensor _ffnGateB;
    private readonly ITensor _ffnUpB;
    private readonly ITensor? _pleGateB;
    private readonly ITensor? _pleScratchB;

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

    /// <summary>When true, Forward() accumulates per-section timings into the Profile* properties.</summary>
    public bool EnableProfiling { get; set; }

    public long ProfileEmbTicks;
    public long ProfilePleSetupTicks;
    public long ProfileAttnMatmulTicks;
    public long ProfileAttnOtherTicks;  // RoPE, kvwrite, softmax, qk-norm, v-norm
    public long ProfileFfnMatmulTicks;
    public long ProfileFfnOtherTicks;   // GeGLU activation
    public long ProfileNormTicks;
    public long ProfilePleBlockTicks;
    public long ProfileLmHeadTicks;

    public void ResetProfile()
    {
        ProfileEmbTicks = 0;
        ProfilePleSetupTicks = 0;
        ProfileAttnMatmulTicks = 0;
        ProfileAttnOtherTicks = 0;
        ProfileFfnMatmulTicks = 0;
        ProfileFfnOtherTicks = 0;
        ProfileNormTicks = 0;
        ProfilePleBlockTicks = 0;
        ProfileLmHeadTicks = 0;
    }

    public Gemma4ForwardPass(IComputeBackend backend, ModelConfig config, ModelWeights weights, Gemma4KvCache kvCache, int maxBatchSize = 64)
    {
        _backend = backend;
        _config = config;
        _weights = weights;
        _kvCache = kvCache;
        _maxBatch = Math.Max(1, maxBatchSize);

        // Single-token scratch (decode path) — unchanged.
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

        // Full-attention scratch (single-token, for decode path)
        _qProj   = CreateF32("g4_q_full", numH   * kFull);
        _kProj   = CreateF32("g4_k_full", numKvH * kFull);
        _vProj   = CreateF32("g4_v_full", numKvH * vFull);
        _qGate   = CreateF32("g4_qg_full", numH * kFull);
        _attnOut = CreateF32("g4_ao_full", numH * vFull);
        backend.FillTensor(_qGate, 88.0f);

        // Single-row scratch tensors — re-used for the per-token attention loop
        // inside ForwardBatch (they share the same size as the decode scratch).
        _qSingle = _qProj;
        _kSingle = _kProj;
        _vSingle = _vProj;
        _qGateSingle = _qGate;
        _attnOutSingle = _attnOut;

        if (kSwa > 0)
        {
            _qProjSwa   = CreateF32("g4_q_swa", numH   * kSwa);
            _kProjSwa   = CreateF32("g4_k_swa", numKvH * kSwa);
            _vProjSwa   = CreateF32("g4_v_swa", numKvH * vSwa);
            _qGateSwa   = CreateF32("g4_qg_swa", numH * kSwa);
            _attnOutSwa = CreateF32("g4_ao_swa", numH * vSwa);
            backend.FillTensor(_qGateSwa, 88.0f);

            _qSingleSwa = _qProjSwa;
            _kSingleSwa = _kProjSwa;
            _vSingleSwa = _vProjSwa;
            _qGateSingleSwa = _qGateSwa;
            _attnOutSingleSwa = _attnOutSwa;
        }
        else
        {
            _qProjSwa = _qProj;
            _kProjSwa = _kProj;
            _vProjSwa = _vProj;
            _qGateSwa = _qGate;
            _attnOutSwa = _attnOut;

            _qSingleSwa = _qSingle;
            _kSingleSwa = _kSingle;
            _vSingleSwa = _vSingle;
            _qGateSingleSwa = _qGateSingle;
            _attnOutSingleSwa = _attnOutSingle;
        }

        _ffnGate = CreateF32("g4_ffn_gate", config.IntermediateDim);
        _ffnUp   = CreateF32("g4_ffn_up",   config.IntermediateDim);

        _hasPle = config.PerLayerInputDim > 0
            && weights.PerLayerTokenEmbd != null
            && weights.PerLayerModelProj != null
            && weights.PerLayerProjNorm != null;
        if (_hasPle)
        {
            int pleTotal = config.PerLayerInputDim * config.NumLayers;
            _pleBase    = CreateF32("g4_ple_base",    pleTotal);
            _pleProjOut = CreateF32("g4_ple_proj",    pleTotal);
            _pleSlice   = CreateF32("g4_ple_slice",   config.PerLayerInputDim);
            _pleGate    = CreateF32("g4_ple_gate",    config.PerLayerInputDim);
            _pleScratch = CreateF32("g4_ple_scratch", config.HiddenDim);
        }

        // Batched scratch buffers — allocated separately and only used by ForwardBatch.
        // Keeping them separate means the decode (M=1) path is bit-identical to pre-batch.
        int mb = _maxBatch;
        _hiddenB   = CreateF32("g4b_hidden",   mb * config.HiddenDim);
        _residualB = CreateF32("g4b_residual", mb * config.HiddenDim);
        _normOutB  = CreateF32("g4b_normOut",  mb * config.HiddenDim);
        _qProjB    = CreateF32("g4b_q_full",   mb * numH   * kFull);
        _kProjB    = CreateF32("g4b_k_full",   mb * numKvH * kFull);
        _vProjB    = CreateF32("g4b_v_full",   mb * numKvH * vFull);
        _attnOutB  = CreateF32("g4b_ao_full",  mb * numH * vFull);
        _ffnGateB  = CreateF32("g4b_ffn_gate", mb * config.IntermediateDim);
        _ffnUpB    = CreateF32("g4b_ffn_up",   mb * config.IntermediateDim);
        if (_hasPle)
        {
            _pleGateB    = CreateF32("g4b_ple_gate",    mb * config.PerLayerInputDim);
            _pleScratchB = CreateF32("g4b_ple_scratch", mb * config.HiddenDim);
        }
        if (kSwa > 0)
        {
            _qProjSwaB   = CreateF32("g4b_q_swa",  mb * numH * kSwa);
            _kProjSwaB   = CreateF32("g4b_k_swa",  mb * numKvH * kSwa);
            _vProjSwaB   = CreateF32("g4b_v_swa",  mb * numKvH * vSwa);
            _attnOutSwaB = CreateF32("g4b_ao_swa", mb * numH * vSwa);
        }
        else
        {
            _qProjSwaB = _qProjB;
            _kProjSwaB = _kProjB;
            _vProjSwaB = _vProjB;
            _attnOutSwaB = _attnOutB;
        }
    }

    public void ResetState() => _kvCache.Reset();

    /// <summary>Run the full forward pass for one token and return the (capped) logits.</summary>
    public ReadOnlySpan<float> Forward(int tokenId, int position)
    {
        ForwardTransformer(tokenId, position);

        long t0 = EnableProfiling ? Stopwatch.GetTimestamp() : 0;

        // Final RmsNorm + lm_head
        _backend.RmsNorm(_normOut, _hidden, _weights.OutputNorm, _config.NormEps);
        ProjectLinear(_logits, _normOut, _weights.OutputWeight);

        // Final logit softcap (Gemma 2/3/4 trait)
        if (_config.FinalLogitSoftcap > 0 && !DebugDisableLogitSoftcap)
            _backend.LogitSoftcap(_logits, _config.FinalLogitSoftcap);

        _logits.DequantizeTo(_logitsBuffer);

        if (EnableProfiling)
            ProfileLmHeadTicks += Stopwatch.GetTimestamp() - t0;

        return _logitsBuffer;
    }

    public void ForwardHidden(int tokenId, int position) => ForwardTransformer(tokenId, position);

    /// <summary>
    /// Batched forward pass for prefill. Processes up to <see cref="_maxBatch"/>
    /// tokens in one shot, reusing weight loads across all M rows in the dense
    /// matmul ops (Q/K/V/O projections, FFN gate/up/down, PLE projections).
    /// Per-token attention is still done in a loop inside the layer — the win
    /// comes from amortising weight memory reads.
    ///
    /// Returns the logits for the LAST token in the batch only (prefill just
    /// needs the final position's distribution to sample the first decode token).
    /// </summary>
    public ReadOnlySpan<float> ForwardBatch(ReadOnlySpan<int> tokenIds, int startPosition)
    {
        int M = tokenIds.Length;
        if (M <= 0) return ReadOnlySpan<float>.Empty;
        if (M == 1) return Forward(tokenIds[0], startPosition);
        if (M > _maxBatch)
            throw new ArgumentException($"batch size {M} > maxBatch {_maxBatch}");

        int D = _config.HiddenDim;
        int IFF = _config.IntermediateDim;

        // Per-section profiling for batched path (cheap: 8 Stopwatch.GetTimestamp calls per layer)
        long tPleSetup = 0, tRmsNorms = 0, tAttnMm = 0, tAttnLoop = 0, tFfnMm = 0, tPleBlock = 0, tLmHead = 0;
        bool prof = EnableProfiling;
        long s;

        // 1. Embed all M tokens into _hiddenB (row-by-row; lookup is memcpy-fast)
        var hiddenBSpan = _hiddenB.AsFloatSpan();
        for (int i = 0; i < M; i++)
        {
            _backend.EmbeddingLookup(_hidden, _weights.TokenEmbedding, tokenIds[i]);
            if (!DebugDisableEmbeddingScale)
                _backend.ScaleInPlace(_hidden, MathF.Sqrt(D));
            _hidden.AsFloatSpan().CopyTo(hiddenBSpan.Slice(i * D, D));
        }

        // 2. PLE setup for all M tokens (per-token, runs M times)
        // TODO(perf): batch the PerLayerModelProj matmul across all M tokens.
        var pleBasePerToken = _hasPle ? new float[M * _config.PerLayerInputDim * _config.NumLayers] : null;
        if (_hasPle && !DebugDisablePle)
        {
            s = prof ? Stopwatch.GetTimestamp() : 0;
            int pleTotal = _config.PerLayerInputDim * _config.NumLayers;
            for (int i = 0; i < M; i++)
            {
                hiddenBSpan.Slice(i * D, D).CopyTo(_hidden.AsFloatSpan());
                SetupPerLayerEmbeddings(tokenIds[i]);
                _pleBase!.AsFloatSpan().CopyTo(pleBasePerToken.AsSpan().Slice(i * pleTotal, pleTotal));
            }
            if (prof) tPleSetup = Stopwatch.GetTimestamp() - s;
        }

        // 3. Transformer layers — all matmuls batched, attention is per-token inside.
        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            var lw = (GemmaAttentionWeights)_weights.Layers[layer];

            s = prof ? Stopwatch.GetTimestamp() : 0;
            for (int i = 0; i < M; i++)
                RmsNormRow(_normOutB, _hiddenB, lw.AttnNorm, i, D);
            _hiddenB.AsFloatSpan().Slice(0, M * D).CopyTo(_residualB.AsFloatSpan());
            if (prof) tRmsNorms += Stopwatch.GetTimestamp() - s;

            s = prof ? Stopwatch.GetTimestamp() : 0;
            ForwardAttentionBatched(lw, layer, startPosition, M);
            if (prof) tAttnLoop += Stopwatch.GetTimestamp() - s;

            s = prof ? Stopwatch.GetTimestamp() : 0;
            for (int i = 0; i < M; i++)
            {
                RmsNormRow(_normOutB, _hiddenB, lw.PostAttnNorm, i, D);
                ElementAddRow(_hiddenB, _normOutB, _residualB, i, D);
            }
            _hiddenB.AsFloatSpan().Slice(0, M * D).CopyTo(_residualB.AsFloatSpan());
            for (int i = 0; i < M; i++)
                RmsNormRow(_normOutB, _hiddenB, lw.FfnNorm, i, D);
            if (prof) tRmsNorms += Stopwatch.GetTimestamp() - s;

            s = prof ? Stopwatch.GetTimestamp() : 0;
            ProjectLinear(_ffnGateB, _normOutB, lw.FfnGate, M);
            ProjectLinear(_ffnUpB,   _normOutB, lw.FfnUp,   M);
            GeGLUBatch(_ffnGateB, _ffnUpB, M, IFF);
            ProjectLinear(_hiddenB, _ffnGateB, lw.FfnDown, M);
            if (prof) tFfnMm += Stopwatch.GetTimestamp() - s;

            s = prof ? Stopwatch.GetTimestamp() : 0;
            for (int i = 0; i < M; i++)
            {
                RmsNormRow(_normOutB, _hiddenB, lw.PostFfnNorm, i, D);
                ElementAddRow(_hiddenB, _normOutB, _residualB, i, D);
            }
            if (prof) tRmsNorms += Stopwatch.GetTimestamp() - s;

            // PLE block — batched for the 2 matmuls, per-row for norm/mul/add.
            if (_hasPle && !DebugDisablePle && lw.PerLayerInpGate != null && lw.PerLayerProj != null && lw.PerLayerPostNorm != null)
            {
                s = prof ? Stopwatch.GetTimestamp() : 0;
                int nEmbdPerLayer = _config.PerLayerInputDim;
                int pleTotal = nEmbdPerLayer * _config.NumLayers;

                _hiddenB.AsFloatSpan().Slice(0, M * D).CopyTo(_residualB.AsFloatSpan());
                ProjectLinear(_pleGateB!, _hiddenB, lw.PerLayerInpGate!, M);
                _backend.GeluTanh(_pleGateB!, _pleGateB!);

                var pleGateBSpan = _pleGateB!.AsFloatSpan();
                for (int i = 0; i < M; i++)
                {
                    var slice = pleBasePerToken.AsSpan().Slice(
                        i * pleTotal + layer * nEmbdPerLayer, nEmbdPerLayer);
                    var gateRow = pleGateBSpan.Slice(i * nEmbdPerLayer, nEmbdPerLayer);
                    for (int k = 0; k < nEmbdPerLayer; k++) gateRow[k] *= slice[k];
                }

                ProjectLinear(_pleScratchB!, _pleGateB!, lw.PerLayerProj!, M);
                for (int i = 0; i < M; i++)
                {
                    RmsNormRow(_pleScratchB!, _pleScratchB!, lw.PerLayerPostNorm!, i, D);
                    ElementAddRow(_hiddenB, _residualB, _pleScratchB!, i, D);
                }
                if (prof) tPleBlock += Stopwatch.GetTimestamp() - s;
            }

            // Per-layer scalar output multiplier
            if (lw.LayerOutScale != null && !DebugDisableLayerOutScale)
            {
                var scaleSpan = lw.LayerOutScale.AsFloatSpan();
                if (scaleSpan.Length > 0)
                {
                    float scale = scaleSpan[0];
                    var h = _hiddenB.AsFloatSpan().Slice(0, M * D);
                    for (int k = 0; k < h.Length; k++) h[k] *= scale;
                }
            }
        }

        // 4. Final norm + lm_head — only for the LAST token (row M-1).
        s = prof ? Stopwatch.GetTimestamp() : 0;
        hiddenBSpan.Slice((M - 1) * D, D).CopyTo(_hidden.AsFloatSpan());

        _backend.RmsNorm(_normOut, _hidden, _weights.OutputNorm, _config.NormEps);
        ProjectLinear(_logits, _normOut, _weights.OutputWeight);

        if (_config.FinalLogitSoftcap > 0 && !DebugDisableLogitSoftcap)
            _backend.LogitSoftcap(_logits, _config.FinalLogitSoftcap);

        _logits.DequantizeTo(_logitsBuffer);
        if (prof) tLmHead = Stopwatch.GetTimestamp() - s;

        if (prof)
        {
            ProfilePleSetupTicks += tPleSetup;
            ProfileNormTicks += tRmsNorms;
            ProfileAttnMatmulTicks += tAttnLoop;  // attention loop includes batched Q/K/V/O matmuls
            ProfileFfnMatmulTicks += tFfnMm;
            ProfilePleBlockTicks += tPleBlock;
            ProfileLmHeadTicks += tLmHead;
        }

        return _logitsBuffer;
    }

    private unsafe void GeGLUBatch(ITensor gate, ITensor up, int batchM, int dim)
    {
        // Run GeGLU over each of the M rows — the op is elementwise so it's safe
        // to call once on the whole M*dim span since both inputs are contiguous.
        _backend.GeGLU(gate, gate, up);  // op processes the whole tensor
    }

    private void ForwardTransformer(int tokenId, int position)
    {
        bool prof = EnableProfiling;
        long t;

        // 1. Embedding lookup + sqrt(hidden_dim) scale
        long tEmb = prof ? Stopwatch.GetTimestamp() : 0;
        _backend.EmbeddingLookup(_hidden, _weights.TokenEmbedding, tokenId);
        if (!DebugDisableEmbeddingScale)
            _backend.ScaleInPlace(_hidden, MathF.Sqrt(_config.HiddenDim));
        if (prof) ProfileEmbTicks += Stopwatch.GetTimestamp() - tEmb;

        // 2. Per-Layer-Embedding setup (once per token, before the layer loop)
        if (_hasPle && !DebugDisablePle)
        {
            t = prof ? Stopwatch.GetTimestamp() : 0;
            SetupPerLayerEmbeddings(tokenId);
            if (prof) ProfilePleSetupTicks += Stopwatch.GetTimestamp() - t;
        }

        // 3. Transformer layers
        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            var lw = (GemmaAttentionWeights)_weights.Layers[layer];

            // ── Pre-attention norm ──────────────────────────────────────────
            t = prof ? Stopwatch.GetTimestamp() : 0;
            _backend.RmsNorm(_normOut, _hidden, lw.AttnNorm, _config.NormEps);
            if (prof) ProfileNormTicks += Stopwatch.GetTimestamp() - t;

            // Save residual (the value of inpL going into this layer's attention)
            _backend.CopyTensor(_residual, _hidden);

            // ── Attention ───────────────────────────────────────────────────
            ForwardAttention(lw, layer, position);

            // ── Post-attention norm + residual add ──────────────────────────
            t = prof ? Stopwatch.GetTimestamp() : 0;
            _backend.RmsNorm(_normOut, _hidden, lw.PostAttnNorm, _config.NormEps);
            _backend.ElementAdd(_hidden, _normOut, _residual);
            if (prof) ProfileNormTicks += Stopwatch.GetTimestamp() - t;

            // ── Pre-FFN norm + GeGLU FFN ────────────────────────────────────
            _backend.CopyTensor(_residual, _hidden);
            t = prof ? Stopwatch.GetTimestamp() : 0;
            _backend.RmsNorm(_normOut, _hidden, lw.FfnNorm, _config.NormEps);
            if (prof) ProfileNormTicks += Stopwatch.GetTimestamp() - t;

            t = prof ? Stopwatch.GetTimestamp() : 0;
            ProjectLinear(_ffnGate, _normOut, lw.FfnGate);
            ProjectLinear(_ffnUp,   _normOut, lw.FfnUp);
            if (prof) ProfileFfnMatmulTicks += Stopwatch.GetTimestamp() - t;

            t = prof ? Stopwatch.GetTimestamp() : 0;
            _backend.GeGLU(_ffnGate, _ffnGate, _ffnUp);
            if (prof) ProfileFfnOtherTicks += Stopwatch.GetTimestamp() - t;

            t = prof ? Stopwatch.GetTimestamp() : 0;
            ProjectLinear(_hidden, _ffnGate, lw.FfnDown);
            if (prof) ProfileFfnMatmulTicks += Stopwatch.GetTimestamp() - t;

            // ── Post-FFN norm + residual add ────────────────────────────────
            t = prof ? Stopwatch.GetTimestamp() : 0;
            _backend.RmsNorm(_normOut, _hidden, lw.PostFfnNorm, _config.NormEps);
            _backend.ElementAdd(_hidden, _normOut, _residual);
            if (prof) ProfileNormTicks += Stopwatch.GetTimestamp() - t;

            // ── Per-Layer Embedding block ───────────────────────────────────
            if (_hasPle && !DebugDisablePle && lw.PerLayerInpGate != null && lw.PerLayerProj != null && lw.PerLayerPostNorm != null)
            {
                t = prof ? Stopwatch.GetTimestamp() : 0;
                ApplyPerLayerEmbedding(lw, layer);
                if (prof) ProfilePleBlockTicks += Stopwatch.GetTimestamp() - t;
            }

            // ── Per-layer scalar output multiplier ──────────────────────────
            if (lw.LayerOutScale != null && !DebugDisableLayerOutScale)
            {
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

        bool prof = EnableProfiling;
        long t = prof ? Stopwatch.GetTimestamp() : 0;

        // ── Q projection (always) ───────────────────────────────────────────
        ProjectLinear(qProj, _normOut, w.AttnQ);
        if (prof) { ProfileAttnMatmulTicks += Stopwatch.GetTimestamp() - t; t = Stopwatch.GetTimestamp(); }

        if (!DebugDisableQkNorm)
            _backend.PerHeadRmsNorm(qProj, w.AttnQNorm, numH, headDim, _config.NormEps);
        if (prof) { ProfileAttnOtherTicks += Stopwatch.GetTimestamp() - t; }

        if (hasKv)
        {
            // ── K/V projections ─────────────────────────────────────────────
            t = prof ? Stopwatch.GetTimestamp() : 0;
            ProjectLinear(kProj, _normOut, w.AttnK);
            ProjectLinear(vProj, _normOut, w.AttnV);
            if (prof) { ProfileAttnMatmulTicks += Stopwatch.GetTimestamp() - t; t = Stopwatch.GetTimestamp(); }

            if (!DebugDisableQkNorm)
                _backend.PerHeadRmsNorm(kProj, w.AttnKNorm, numKvH, headDim, _config.NormEps);
            if (!DebugDisableVNorm)
                _backend.PerHeadRmsNormUnit(vProj, numKvH, valDim, _config.NormEps);

            // ── RoPE (Q and K together) ─────────────────────────────────────
            if (!DebugDisableRope)
            {
                if (isSwa || w.RopeFreqs == null)
                    _backend.RoPENeox(qProj, kProj, headDim, ropeDim, position, ropeTheta);
                else
                    _backend.RoPENeoxWithFreqFactors(qProj, kProj, headDim, ropeDim, position, ropeTheta, w.RopeFreqs);
            }

            // Write K/V to THIS layer's cache
            _kvCache.Write(_backend, layer, position, kProj, vProj);
            if (prof) { ProfileAttnOtherTicks += Stopwatch.GetTimestamp() - t; }
        }
        else
        {
            // Shared-KV layer: no new K/V; rotate Q only. kProj is used as an
            // ignored scratch buffer (RoPE rotates it in place but we never read
            // the result — attention reads the source layer's cache instead).
            t = prof ? Stopwatch.GetTimestamp() : 0;
            if (!DebugDisableRope)
            {
                if (isSwa || w.RopeFreqs == null)
                    _backend.RoPENeox(qProj, kProj, headDim, ropeDim, position, ropeTheta);
                else
                    _backend.RoPENeoxWithFreqFactors(qProj, kProj, headDim, ropeDim, position, ropeTheta, w.RopeFreqs);
            }
            if (prof) { ProfileAttnOtherTicks += Stopwatch.GetTimestamp() - t; }
        }

        // ── Attention (reads sourceLayer's cache, which == layer when hasKv) ─
        t = prof ? Stopwatch.GetTimestamp() : 0;
        int seqLen = _kvCache.LayerSeqLen(sourceLayer);
        var kCacheT = _kvCache.GetKCacheTensor(sourceLayer);
        var vCacheT = _kvCache.GetVCacheTensor(sourceLayer);
        int cap = _kvCache.LayerCapacity(sourceLayer);

        _backend.GatedAttention(attnOut, qProj, qGate, kCacheT, vCacheT,
            numH, numKvH, headDim, valDim, cap, seqLen, attentionScale);
        if (prof) { ProfileAttnOtherTicks += Stopwatch.GetTimestamp() - t; t = Stopwatch.GetTimestamp(); }

        // ── Output projection ───────────────────────────────────────────────
        // attn_output weight shape: [num_heads * head_dim, hidden_dim]
        ProjectLinear(_hidden, attnOut, w.AttnO);
        if (prof) { ProfileAttnMatmulTicks += Stopwatch.GetTimestamp() - t; }
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

    private void ProjectLinear(ITensor output, ITensor input, ITensor weight, int batchM)
    {
        int K = (int)weight.Dimensions[0];
        int N = (int)weight.Dimensions[1];
        _backend.MatMul(output, input, weight, batchM, K, N);
    }

    /// <summary>
    /// RmsNorm applied to row <paramref name="row"/> of a batched activation
    /// buffer. Vectorized with <see cref="System.Numerics.Vector{T}"/> for the
    /// sum-of-squares reduction and the final scale-and-multiply pass.
    /// </summary>
    private unsafe void RmsNormRow(ITensor dest, ITensor src, ITensor weight, int row, int dim)
    {
        var srcSpan = src.AsFloatSpan().Slice(row * dim, dim);
        var dstSpan = dest.AsFloatSpan().Slice(row * dim, dim);
        var wSpan = weight.AsFloatSpan();

        // Sum of squares, vectorized
        int vlen = System.Numerics.Vector<float>.Count;
        var accV = System.Numerics.Vector<float>.Zero;
        int i = 0;
        for (; i + vlen <= dim; i += vlen)
        {
            var sv = new System.Numerics.Vector<float>(srcSpan.Slice(i, vlen));
            accV += sv * sv;
        }
        float sumSq = System.Numerics.Vector.Sum(accV);
        for (; i < dim; i++) sumSq += srcSpan[i] * srcSpan[i];

        float invRms = 1.0f / MathF.Sqrt(sumSq / dim + _config.NormEps);
        var invRmsV = new System.Numerics.Vector<float>(invRms);

        // dst = src * invRms * weight
        i = 0;
        for (; i + vlen <= dim; i += vlen)
        {
            var sv = new System.Numerics.Vector<float>(srcSpan.Slice(i, vlen));
            var wv = new System.Numerics.Vector<float>(wSpan.Slice(i, vlen));
            (sv * invRmsV * wv).CopyTo(dstSpan.Slice(i, vlen));
        }
        for (; i < dim; i++) dstSpan[i] = srcSpan[i] * invRms * wSpan[i];
    }

    /// <summary>
    /// Vectorized dest[row] = a[row] + b[row].
    /// </summary>
    private void ElementAddRow(ITensor dest, ITensor a, ITensor b, int row, int dim)
    {
        var aSpan = a.AsFloatSpan().Slice(row * dim, dim);
        var bSpan = b.AsFloatSpan().Slice(row * dim, dim);
        var dSpan = dest.AsFloatSpan().Slice(row * dim, dim);

        int vlen = System.Numerics.Vector<float>.Count;
        int i = 0;
        for (; i + vlen <= dim; i += vlen)
        {
            var av = new System.Numerics.Vector<float>(aSpan.Slice(i, vlen));
            var bv = new System.Numerics.Vector<float>(bSpan.Slice(i, vlen));
            (av + bv).CopyTo(dSpan.Slice(i, vlen));
        }
        for (; i < dim; i++) dSpan[i] = aSpan[i] + bSpan[i];
    }

    /// <summary>
    /// Batched attention for one layer. Does the Q/K/V/O projections as batched
    /// matmuls over M rows, loops per-token for the norms/RoPE/cache writes (which
    /// need per-token positions), then runs a single BATCHED causal attention
    /// call that processes all M queries against the shared K/V cache. Parallel
    /// work is distributed across (M × numH) independent (query, head) pairs, so
    /// 22 threads get ~12 tasks each at M=32.
    /// </summary>
    private void ForwardAttentionBatched(GemmaAttentionWeights w, int layer, int startPosition, int batchM)
    {
        bool isSwa = _config.IsSlidingLayer(layer);
        bool hasKv = _config.HasKv(layer);
        int numH = _config.NumHeads;
        int numKvH = _config.NumKvHeads;
        int headDim = _config.LayerKeyLength(layer);
        int valDim  = _config.LayerValueLength(layer);
        int ropeDim = _config.LayerRopeDim(layer);
        float ropeTheta = _config.LayerRopeTheta(layer);
        const float attentionScale = 1.0f;

        int sourceLayer = hasKv ? layer
            : (_config.NumLayerKvFromStart - (isSwa ? 2 : 1));

        // Batched tensors for this layer's head dim
        var qProjB = isSwa ? _qProjSwaB : _qProjB;
        var kProjB = isSwa ? _kProjSwaB : _kProjB;
        var vProjB = isSwa ? _vProjSwaB : _vProjB;
        var attnOutB = isSwa ? _attnOutSwaB : _attnOutB;

        // Single-row scratch (reuses the decode buffers)
        var qSingle = isSwa ? _qSingleSwa : _qSingle;
        var kSingle = isSwa ? _kSingleSwa : _kSingle;
        var vSingle = isSwa ? _vSingleSwa : _vSingle;
        var qGateSingle = isSwa ? _qGateSingleSwa : _qGateSingle;
        var attnOutSingle = isSwa ? _attnOutSingleSwa : _attnOutSingle;

        int qRowSize = numH * headDim;
        int kRowSize = numKvH * headDim;
        int vRowSize = numKvH * valDim;
        int oRowSize = numH * valDim;

        // ── Batched Q projection (reads batchM rows from _normOutB) ─────────
        ProjectLinear(qProjB, _normOutB, w.AttnQ, batchM);

        // ── Batched K/V projections (has_kv layers only) ────────────────────
        if (hasKv)
        {
            ProjectLinear(kProjB, _normOutB, w.AttnK, batchM);
            ProjectLinear(vProjB, _normOutB, w.AttnV, batchM);
        }

        // ── Per-token attention loop ────────────────────────────────────────
        var qProjBSpan = qProjB.AsFloatSpan();
        var kProjBSpan = kProjB.AsFloatSpan();
        var vProjBSpan = vProjB.AsFloatSpan();
        var attnOutBSpan = attnOutB.AsFloatSpan();
        var qSingleSpan = qSingle.AsFloatSpan();
        var kSingleSpan = kSingle.AsFloatSpan();
        var vSingleSpan = vSingle.AsFloatSpan();
        var attnOutSingleSpan = attnOutSingle.AsFloatSpan();

        // First pass: per-token norms, RoPE, and KV cache writes. We need per-token
        // positions here, so we still loop. RoPE'd Q is copied back into qProjB so
        // the batched attention step can read all M queries at once.
        for (int i = 0; i < batchM; i++)
        {
            int pos = startPosition + i;

            qProjBSpan.Slice(i * qRowSize, qRowSize).CopyTo(qSingleSpan);

            if (!DebugDisableQkNorm)
                _backend.PerHeadRmsNorm(qSingle, w.AttnQNorm, numH, headDim, _config.NormEps);

            if (hasKv)
            {
                kProjBSpan.Slice(i * kRowSize, kRowSize).CopyTo(kSingleSpan);
                vProjBSpan.Slice(i * vRowSize, vRowSize).CopyTo(vSingleSpan);

                if (!DebugDisableQkNorm)
                    _backend.PerHeadRmsNorm(kSingle, w.AttnKNorm, numKvH, headDim, _config.NormEps);
                if (!DebugDisableVNorm)
                    _backend.PerHeadRmsNormUnit(vSingle, numKvH, valDim, _config.NormEps);

                if (!DebugDisableRope)
                {
                    if (isSwa || w.RopeFreqs == null)
                        _backend.RoPENeox(qSingle, kSingle, headDim, ropeDim, pos, ropeTheta);
                    else
                        _backend.RoPENeoxWithFreqFactors(qSingle, kSingle, headDim, ropeDim, pos, ropeTheta, w.RopeFreqs);
                }

                _kvCache.Write(_backend, layer, pos, kSingle, vSingle);
            }
            else
            {
                if (!DebugDisableRope)
                {
                    if (isSwa || w.RopeFreqs == null)
                        _backend.RoPENeox(qSingle, kSingle, headDim, ropeDim, pos, ropeTheta);
                    else
                        _backend.RoPENeoxWithFreqFactors(qSingle, kSingle, headDim, ropeDim, pos, ropeTheta, w.RopeFreqs);
                }
            }

            // Write RoPE'd Q back into qProjB[i, :] for the batched attention step.
            qSingleSpan.Slice(0, qRowSize).CopyTo(qProjBSpan.Slice(i * qRowSize, qRowSize));
        }

        // Second pass: single batched causal attention over all M queries against
        // the shared source-layer cache. 256 independent (query, head) tasks at M=32.
        var kCacheT2 = _kvCache.GetKCacheTensor(sourceLayer);
        var vCacheT2 = _kvCache.GetVCacheTensor(sourceLayer);
        int cap2 = _kvCache.LayerCapacity(sourceLayer);

        BatchedCausalAttention(
            attnOutB, qProjB, kCacheT2, vCacheT2,
            batchM, numH, numKvH, headDim, valDim,
            cap2, startPosition, attentionScale);

        // ── Batched output projection ───────────────────────────────────────
        ProjectLinear(_hiddenB, attnOutB, w.AttnO, batchM);
    }

    /// <summary>
    /// Batched causal attention: for each (query m, head h) pair run a tiled
    /// online softmax against the K/V cache, restricted to positions
    /// &lt;= <paramref name="startPosition"/> + m.
    ///
    /// Cache layout (matching <see cref="Daisi.Llogos.Cpu.CpuBackend.KvCacheWrite"/>):
    /// K is <c>[nKvHeads * cap * keyLength]</c> head-major, V same with valueLength.
    /// GQA: query head h reads from kv head h / (numH / numKvH).
    /// </summary>
    private unsafe void BatchedCausalAttention(
        ITensor attnOut, ITensor qAttn, ITensor kCache, ITensor vCache,
        int M, int numH, int numKvH, int keyLength, int valueLength,
        int cap, int startPosition, float scale)
    {
        int headsPerGroup = numH / numKvH;
        int kHeadStride = cap * keyLength;
        int vHeadStride = cap * valueLength;
        int qRowSize = numH * keyLength;
        int oRowSize = numH * valueLength;

        // Pin the tensor spans so the parallel worker lambda can use nint-captured
        // pointers (Span<float> can't cross the lambda boundary).
        var qSpanTop = qAttn.AsFloatSpan();
        var oSpanTop = attnOut.AsFloatSpan();
        var kSpanTop = kCache.AsFloatSpan();
        var vSpanTop = vCache.AsFloatSpan();

        fixed (float* qFixed = qSpanTop)
        fixed (float* oFixed = oSpanTop)
        fixed (float* kFixed = kSpanTop)
        fixed (float* vFixed = vSpanTop)
        {
            nint qBase = (nint)qFixed;
            nint oBase = (nint)oFixed;
            nint kBasePtr = (nint)kFixed;
            nint vBasePtr = (nint)vFixed;
            int totalTasks = M * numH;
            int keyLen = keyLength;
            int valLen = valueLength;
            int startPos = startPosition;
            float scaleCap = scale;
            int hpg = headsPerGroup;
            int khs = kHeadStride;
            int vhs = vHeadStride;
            int qrs = qRowSize;
            int ors = oRowSize;

            Parallel.For(0, totalTasks, idx =>
            {
                int m = idx / numH;
                int h = idx % numH;
                int kvHead = h / hpg;
                int seqLen = startPos + m + 1;

                float* q = (float*)qBase + m * qrs + h * keyLen;
                float* o = (float*)oBase + m * ors + h * valLen;
                float* kHead = (float*)kBasePtr + kvHead * khs;
                float* vHead = (float*)vBasePtr + kvHead * vhs;

                const int tileSize = 256;
                Span<float> tileScores = stackalloc float[tileSize];

                for (int d = 0; d < valLen; d++) o[d] = 0;

                float runningMax = float.NegativeInfinity;
                float runningSum = 0;

                for (int tileStart = 0; tileStart < seqLen; tileStart += tileSize)
                {
                    int tileEnd = Math.Min(tileStart + tileSize, seqLen);
                    int tileLen = tileEnd - tileStart;

                    // Scores: q · k[p] * scale  for p in tile
                    for (int t = 0; t < tileLen; t++)
                    {
                        int p = tileStart + t;
                        float* kp = kHead + p * keyLen;
                        float dot = 0;
                        for (int d = 0; d < keyLen; d++)
                            dot += q[d] * kp[d];
                        tileScores[t] = dot * scaleCap;
                    }

                    float tileMax = float.NegativeInfinity;
                    for (int t = 0; t < tileLen; t++)
                        if (tileScores[t] > tileMax) tileMax = tileScores[t];

                    float tileSum = 0;
                    for (int t = 0; t < tileLen; t++)
                    {
                        tileScores[t] = MathF.Exp(tileScores[t] - tileMax);
                        tileSum += tileScores[t];
                    }

                    float newMax = MathF.Max(runningMax, tileMax);
                    float correctionOld = MathF.Exp(runningMax - newMax);
                    float correctionNew = MathF.Exp(tileMax - newMax);

                    for (int d = 0; d < valLen; d++)
                    {
                        float tileVal = 0;
                        for (int t = 0; t < tileLen; t++)
                            tileVal += tileScores[t] * vHead[(tileStart + t) * valLen + d];
                        o[d] = o[d] * correctionOld + tileVal * correctionNew;
                    }

                    runningSum = runningSum * correctionOld + tileSum * correctionNew;
                    runningMax = newMax;
                }

                float invSum = 1.0f / runningSum;
                for (int d = 0; d < valLen; d++)
                    o[d] *= invSum;
            });
        }
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

        _hiddenB.Dispose();
        _residualB.Dispose();
        _normOutB.Dispose();
        _qProjB.Dispose();
        _kProjB.Dispose();
        _vProjB.Dispose();
        _attnOutB.Dispose();
        _ffnGateB.Dispose();
        _ffnUpB.Dispose();
        _pleGateB?.Dispose();
        _pleScratchB?.Dispose();
        if (_qProjSwaB != _qProjB)
        {
            _qProjSwaB.Dispose();
            _kProjSwaB.Dispose();
            _vProjSwaB.Dispose();
            _attnOutSwaB.Dispose();
        }
    }
}
