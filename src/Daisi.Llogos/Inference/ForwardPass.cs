using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Inference;

/// <summary>
/// Hybrid transformer forward pass supporting both standard attention and DeltaNet layers.
/// Uses only IComputeBackend operations — no AsFloatSpan() — so it works on both CPU and GPU.
/// </summary>
public sealed class ForwardPass : IForwardPass
{
    private readonly IComputeBackend _backend;
    private readonly ModelConfig _config;
    private readonly ModelWeights _weights;
    private readonly IKvCache _kvCache;
    private readonly DeltaNetState _deltaState;

    // Scratch buffers
    private readonly ITensor _hidden;
    private readonly ITensor _residual;
    private readonly ITensor _normOut;
    private readonly ITensor _logits;
    private readonly float[] _logitsBuffer;

    // Standard attention scratch
    private readonly ITensor? _qFull;   // [numHeads × keyLength × 2] (attn + gate interleaved, gated Q only)
    private readonly ITensor _qAttn;    // [numHeads × keyLength]
    private readonly ITensor _qGate;    // [numHeads × keyLength] (dummy gate for non-gated models)
    private readonly ITensor _kProj;    // [numKvHeads × keyLength]
    private readonly ITensor _vProj;    // [numKvHeads × valueLength]
    private readonly ITensor _attnOut;  // [numHeads × valueLength]

    // DeltaNet scratch
    private readonly ITensor _qkvBuf;   // [ssmInnerSize × 3]
    private readonly ITensor _ssmQ;     // [ssmInnerSize] — view into qkvBuf not possible, so separate
    private readonly ITensor _ssmK;     // [ssmInnerSize]
    private readonly ITensor _ssmV;     // [ssmInnerSize]
    private readonly ITensor _ssmAlpha;  // [ssmGroupCount]
    private readonly ITensor _ssmBeta;   // [ssmGroupCount]
    private readonly ITensor _ssmDecay;  // [ssmGroupCount]
    private readonly ITensor _ssmBetaVal; // [ssmGroupCount]
    private readonly ITensor _ssmGate;   // [ssmInnerSize]
    private readonly ITensor _ssmOutput; // [ssmInnerSize]

    // FFN scratch
    private readonly ITensor _gate;
    private readonly ITensor _up;

    public ForwardPass(IComputeBackend backend, ModelConfig config, ModelWeights weights,
        IKvCache kvCache, DeltaNetState deltaState)
    {
        _backend = backend;
        _config = config;
        _weights = weights;
        _kvCache = kvCache;
        _deltaState = deltaState;

        _hidden = CreateF32("scratch_hidden", config.HiddenDim);
        _residual = CreateF32("scratch_residual", config.HiddenDim);
        _normOut = CreateF32("scratch_norm", config.HiddenDim);
        _logits = CreateF32("scratch_logits", config.VocabSize);
        _logitsBuffer = new float[config.VocabSize];

        // Standard attention scratch
        // Gated Q (Qwen) needs a 2× buffer for interleaved Q+gate; standard models project directly to _qAttn
        // Detect from actual weights: if any standard attention layer has Q/K norms, it's gated
        bool hasGatedQ = false;
        for (int i = 0; i < config.NumLayers; i++)
            if (weights.Layers[i] is StandardAttentionWeights saw && saw.HasGatedQ)
                { hasGatedQ = true; break; }
        if (hasGatedQ)
            _qFull = CreateF32("scratch_q_full", config.NumHeads * config.KeyLength * 2);
        _qAttn = CreateF32("scratch_q_attn", config.NumHeads * config.KeyLength);
        _qGate = CreateF32("scratch_q_gate", config.NumHeads * config.KeyLength);
        if (!hasGatedQ)
            backend.FillTensor(_qGate, 88.0f); // sigmoid(88)≈1.0 → ungated attention
        _kProj = CreateF32("scratch_k", config.NumKvHeads * config.KeyLength);
        _vProj = CreateF32("scratch_v", config.NumKvHeads * config.ValueLength);
        _attnOut = CreateF32("scratch_attn_out", config.NumHeads * config.ValueLength);

        // DeltaNet scratch — derive sizes from actual weight tensors
        if (config.SsmInnerSize > 0)
        {
            // Find first DeltaNet layer to get actual tensor dimensions
            DeltaNetWeights? deltaLayer = null;
            for (int i = 0; i < config.NumLayers; i++)
                if (!config.IsStandardAttention(i) && weights.Layers[i] is DeltaNetWeights dw)
                    { deltaLayer = dw; break; }

            int qkvOutDim = deltaLayer != null ? (int)deltaLayer.AttnQkv.Dimensions[1] : config.SsmInnerSize * 3;
            int numVHeads = deltaLayer != null ? (int)deltaLayer.SsmAlpha.Dimensions[1] : config.SsmGroupCount;
            int valueDim = numVHeads * (config.SsmStateSize > 0 ? config.SsmStateSize : config.SsmHeadDim);
            // Key dim: whatever remains after subtracting valueDim
            int keyDim = (qkvOutDim - valueDim) / 2;
            int numKHeads = keyDim > 0 ? keyDim / (valueDim / numVHeads) : numVHeads;

            _qkvBuf = CreateF32("scratch_qkv", qkvOutDim);
            // Q and K may be smaller than V if num_k_heads < num_v_heads.
            // After repeat-interleave, Q and K become valueDim-sized.
            _ssmQ = CreateF32("scratch_ssm_q", valueDim);
            _ssmK = CreateF32("scratch_ssm_k", valueDim);
            _ssmV = CreateF32("scratch_ssm_v", valueDim);
            _ssmAlpha = CreateF32("scratch_ssm_alpha", numVHeads);
            _ssmBeta = CreateF32("scratch_ssm_beta", numVHeads);
            _ssmDecay = CreateF32("scratch_ssm_decay", numVHeads);
            _ssmBetaVal = CreateF32("scratch_ssm_betaval", numVHeads);
            _ssmGate = CreateF32("scratch_ssm_gate", valueDim);
            _ssmOutput = CreateF32("scratch_ssm_out", valueDim);
        }
        else
        {
            _qkvBuf = _ssmQ = _ssmK = _ssmV = _ssmAlpha = _ssmBeta =
                _ssmDecay = _ssmBetaVal = _ssmGate = _ssmOutput = _hidden; // unused
        }

        _gate = CreateF32("scratch_ffn_gate", config.IntermediateDim);
        _up = CreateF32("scratch_ffn_up", config.IntermediateDim);
    }

    public IKvCache KvCache => _kvCache;
    public DeltaNetState DeltaState => _deltaState;

    /// <inheritdoc />
    public void ResetState()
    {
        _kvCache.Reset();
        _deltaState.Reset();
    }

    /// <summary>
    /// Run a forward pass for a single token at the given position.
    /// </summary>
    public ReadOnlySpan<float> Forward(int tokenId, int position)
    {
        // 1. Embedding lookup
        _backend.EmbeddingLookup(_hidden, _weights.TokenEmbedding, tokenId);

        // 2. Transformer layers
        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            var lw = _weights.Layers[layer];

            // Fused: save residual + pre-attention RMSNorm
            _backend.RmsNormResidual(_normOut, _residual, _hidden, lw.AttnNorm, _config.NormEps);

            // Layer-type-specific attention
            if (lw is StandardAttentionWeights saw)
                ForwardStandardAttention(saw, position, layer);
            else if (lw is DeltaNetWeights dnw)
                ForwardDeltaNet(dnw, layer);

            // Fused: residual add + save new residual + pre-FFN RMSNorm
            _backend.AddRmsNorm(_normOut, _hidden, _hidden, _residual, lw.PostAttnNorm, _config.NormEps);
            _backend.CopyTensor(_residual, _hidden);

            // SwiGLU FFN (fused SiLU + ElementMul)
            ProjectLinear(_gate, _normOut, lw.FfnGate);
            ProjectLinear(_up, _normOut, lw.FfnUp);
            _backend.SwiGLU(_gate, _gate, _up);
            ProjectLinear(_hidden, _gate, lw.FfnDown);

            // Residual add
            _backend.ElementAdd(_hidden, _hidden, _residual);
        }

        // 3. Final RMSNorm
        _backend.RmsNorm(_normOut, _hidden, _weights.OutputNorm, _config.NormEps);

        // 4. LM head
        ProjectLinear(_logits, _normOut, _weights.OutputWeight);

        _logits.DequantizeTo(_logitsBuffer);
        return _logitsBuffer;
    }

    // ── Standard Attention (Gated) ───────────────────────────────────────────

    private void ForwardStandardAttention(StandardAttentionWeights w, int position, int layer)
    {
        int numHeads = _config.NumHeads;
        int numKvHeads = _config.NumKvHeads;
        int keyLen = _config.KeyLength;
        int valLen = _config.ValueLength;
        int ropeDim = _config.RopeDimCount;
        float scale = 1.0f / MathF.Sqrt(keyLen);

        // K/V projections (same for gated and non-gated)
        ProjectLinear(_kProj, _normOut, w.AttnK);
        ProjectLinear(_vProj, _normOut, w.AttnV);

        if (w.HasGatedQ)
        {
            // Qwen3.5-style: Q projects to 2× dim (interleaved Q_attn + Q_gate)
            ProjectLinear(_qFull!, _normOut, w.AttnQ);
            _backend.DeInterleaveQ(_qAttn, _qGate, _qFull!, numHeads, keyLen);

            // Per-head Q/K norms
            _backend.PerHeadRmsNorm(_qAttn, w.AttnQNorm!, numHeads, keyLen, _config.NormEps);
            _backend.PerHeadRmsNorm(_kProj, w.AttnKNorm!, numKvHeads, keyLen, _config.NormEps);
        }
        else
        {
            // Standard Q projection (no gating)
            ProjectLinear(_qAttn, _normOut, w.AttnQ);
            // _qGate already filled with 88.0 (sigmoid=1.0) in constructor

            // Apply per-head Q/K norms if present (Qwen3-style: norms without gating)
            if (w.AttnQNorm != null)
            {
                _backend.PerHeadRmsNorm(_qAttn, w.AttnQNorm, numHeads, keyLen, _config.NormEps);
                _backend.PerHeadRmsNorm(_kProj, w.AttnKNorm!, numKvHeads, keyLen, _config.NormEps);
            }
        }

        // RoPE (partial — only first ropeDim dims)
        _backend.RoPE(_qAttn, _kProj, keyLen, ropeDim, position, _config.RopeTheta);

        // Write K/V to cache (maps position through strategy — ring buffer for sliding window)
        _kvCache.Write(_backend, layer, position, _kProj, _vProj);

        // seqLen is read AFTER write so it includes the current token
        int seqLen = _kvCache.Length;

        // Compute attention (gating is a no-op when gate=88.0)
        var kCacheTensor = _kvCache.GetKCacheTensor(layer);
        var vCacheTensor = _kvCache.GetVCacheTensor(layer);
        _backend.GatedAttention(_attnOut, _qAttn, _qGate, kCacheTensor, vCacheTensor,
            numHeads, numKvHeads, keyLen, valLen, _kvCache.MaxSeqLen, seqLen, scale);

        // Output projection
        ProjectLinear(_hidden, _attnOut, w.AttnO);
    }

    // ── DeltaNet (Gated Linear Attention) ────────────────────────────────────

    private void ForwardDeltaNet(DeltaNetWeights w, int layer)
    {
        int convKernel = _config.SsmConvKernel;
        int headDim = _config.SsmStateSize > 0 ? _config.SsmStateSize : _config.SsmHeadDim;

        // Derive dimensions from weight tensors
        int numVHeads = (int)w.SsmAlpha.Dimensions[1]; // num_v_heads (32 for 9B)
        int valueDim = numVHeads * headDim;             // 32 × 128 = 4096
        int qkvOutDim = (int)w.AttnQkv.Dimensions[1];  // 8192 for 9B
        int keyDim = (qkvOutDim - valueDim) / 2;        // (8192-4096)/2 = 2048
        int numKHeads = keyDim / headDim;                // 2048/128 = 16
        int repeatFactor = numVHeads / numKHeads;        // 32/16 = 2

        // 1. QKV projection
        ProjectLinear(_qkvBuf, _normOut, w.AttnQkv);

        // 2. Causal conv1d on full Q+K+V output
        int convChannels = (int)(w.SsmConv1d.ElementCount / convKernel);
        var convBuf = _deltaState.GetConvBufferTensor(layer);
        _backend.CausalConv1d(_qkvBuf, convBuf, w.SsmConv1d, convChannels, convKernel);

        // 3. SiLU activation
        _backend.SiLUInPlace(_qkvBuf);

        // 4. Split Q(keyDim) + K(keyDim) + V(valueDim) — possibly unequal
        if (keyDim == valueDim)
        {
            // Equal split (0.8B model: keyDim = valueDim = innerSize)
            _backend.SplitQKV(_ssmQ, _ssmK, _ssmV, _qkvBuf, keyDim);
        }
        else
        {
            // Unequal split (9B model: keyDim=2048, valueDim=4096)
            // Copy Q, K from qkvBuf, then V
            _backend.CopyTensorBytes(_ssmQ, _qkvBuf, keyDim * sizeof(float));

            // K starts at offset keyDim — need a shifted copy
            // Use the generic SplitQKV but with keyDim as the split size,
            // then copy V separately from offset 2*keyDim
            SplitUnequal(_qkvBuf, _ssmQ, _ssmK, _ssmV, keyDim, valueDim);
        }

        // 5. L2-normalize Q and K (num_k_heads groups)
        _backend.L2NormGroups(_ssmQ, numKHeads, headDim);
        _backend.L2NormGroups(_ssmK, numKHeads, headDim);

        // 6. Repeat-interleave Q and K from num_k_heads → num_v_heads
        if (repeatFactor > 1)
        {
            RepeatInterleave(_ssmQ, numKHeads, headDim, repeatFactor);
            RepeatInterleave(_ssmK, numKHeads, headDim, repeatFactor);
        }

        // 7. Compute alpha and beta projections
        ProjectLinear(_ssmAlpha, _normOut, w.SsmAlpha);
        ProjectLinear(_ssmBeta, _normOut, w.SsmBeta);

        // 8. Compute decay and beta values
        _backend.ComputeDecayBeta(_ssmDecay, _ssmBetaVal, _ssmAlpha, _ssmBeta,
            w.SsmA, w.SsmDtBias, numVHeads);

        // 9. DeltaNet state update + output + per-head norm
        var stateTensor = _deltaState.GetStateTensor(layer);
        float scale = 1.0f / MathF.Sqrt(headDim);
        _backend.DeltaNetStep(_ssmOutput, _ssmQ, _ssmK, _ssmV,
            stateTensor, _ssmDecay, _ssmBetaVal,
            w.SsmNorm, numVHeads, headDim, scale, _config.NormEps);

        // 10. Gate: output = RMSNorm(output) * SiLU(Z)
        ProjectLinear(_ssmGate, _normOut, w.AttnGate);
        _backend.SiLUGate(_ssmOutput, _ssmOutput, _ssmGate);

        // 11. Output projection
        ProjectLinear(_hidden, _ssmOutput, w.SsmOut);
    }

    /// <summary>
    /// Split QKV buffer with unequal Q/K and V sizes using backend CopyTensorBytes.
    /// Layout: [Q: keyDim] [K: keyDim] [V: valueDim]
    /// </summary>
    private void SplitUnequal(ITensor qkv, ITensor q, ITensor k, ITensor v,
        int keyDim, int valueDim)
    {
        // Download QKV, split on CPU, re-upload
        int totalElems = keyDim * 2 + valueDim;
        var buf = new float[totalElems];
        qkv.DequantizeTo(buf.AsSpan(0, totalElems));

        // Q: first keyDim elements → q tensor (which is valueDim-sized, pad with zeros)
        var qBuf = new byte[q.ByteSize];
        Buffer.BlockCopy(buf, 0, qBuf, 0, keyDim * sizeof(float));
        q.CopyFrom(qBuf);

        // K: next keyDim elements → k tensor (valueDim-sized, pad with zeros)
        var kBuf = new byte[k.ByteSize];
        Buffer.BlockCopy(buf, keyDim * sizeof(float), kBuf, 0, keyDim * sizeof(float));
        k.CopyFrom(kBuf);

        // V: last valueDim elements → v tensor
        var vBuf = new byte[v.ByteSize];
        Buffer.BlockCopy(buf, keyDim * 2 * sizeof(float), vBuf, 0, valueDim * sizeof(float));
        v.CopyFrom(vBuf);
    }

    /// <summary>
    /// Repeat-interleave tensor data from numHeads groups to numHeads*factor groups.
    /// Each head of headDim elements is repeated 'factor' times.
    /// Done in-place (tensor must be large enough for the expanded result).
    /// </summary>
    private void RepeatInterleave(ITensor tensor, int numHeads, int headDim, int factor)
    {
        // Download, repeat, re-upload
        int srcSize = numHeads * headDim;
        int dstSize = numHeads * factor * headDim;
        var src = new float[dstSize]; // full tensor size
        tensor.DequantizeTo(src);

        // Work backwards to avoid overwriting
        for (int h = numHeads - 1; h >= 0; h--)
        {
            int srcOff = h * headDim;
            for (int r = factor - 1; r >= 0; r--)
            {
                int dstOff = (h * factor + r) * headDim;
                Array.Copy(src, srcOff, src, dstOff, headDim);
            }
        }

        var bytes = new byte[dstSize * sizeof(float)];
        Buffer.BlockCopy(src, 0, bytes, 0, bytes.Length);
        tensor.CopyFrom(bytes);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

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
        _qFull?.Dispose();
        _qAttn.Dispose();
        _qGate.Dispose();
        _kProj.Dispose();
        _vProj.Dispose();
        _attnOut.Dispose();
        if (_config.SsmInnerSize > 0)
        {
            _qkvBuf.Dispose();
            _ssmQ.Dispose();
            _ssmK.Dispose();
            _ssmV.Dispose();
            _ssmAlpha.Dispose();
            _ssmBeta.Dispose();
            _ssmDecay.Dispose();
            _ssmBetaVal.Dispose();
            _ssmGate.Dispose();
            _ssmOutput.Dispose();
        }
        _gate.Dispose();
        _up.Dispose();
    }
}
