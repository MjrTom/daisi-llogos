using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Inference;

/// <summary>
/// Hybrid transformer forward pass supporting both standard attention and DeltaNet layers.
/// Uses only IComputeBackend operations — no AsFloatSpan() — so it works on both CPU and GPU.
/// </summary>
public sealed class ForwardPass : IDisposable
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
        bool hasGatedQ = config.HasDeltaNet; // Qwen hybrid uses gated Q in its attention layers
        if (hasGatedQ)
            _qFull = CreateF32("scratch_q_full", config.NumHeads * config.KeyLength * 2);
        _qAttn = CreateF32("scratch_q_attn", config.NumHeads * config.KeyLength);
        _qGate = CreateF32("scratch_q_gate", config.NumHeads * config.KeyLength);
        if (!hasGatedQ)
            backend.FillTensor(_qGate, 88.0f); // sigmoid(88)≈1.0 → ungated attention
        _kProj = CreateF32("scratch_k", config.NumKvHeads * config.KeyLength);
        _vProj = CreateF32("scratch_v", config.NumKvHeads * config.ValueLength);
        _attnOut = CreateF32("scratch_attn_out", config.NumHeads * config.ValueLength);

        // DeltaNet scratch
        if (config.SsmInnerSize > 0)
        {
            _qkvBuf = CreateF32("scratch_qkv", config.SsmInnerSize * 3);
            _ssmQ = CreateF32("scratch_ssm_q", config.SsmInnerSize);
            _ssmK = CreateF32("scratch_ssm_k", config.SsmInnerSize);
            _ssmV = CreateF32("scratch_ssm_v", config.SsmInnerSize);
            _ssmAlpha = CreateF32("scratch_ssm_alpha", config.SsmGroupCount);
            _ssmBeta = CreateF32("scratch_ssm_beta", config.SsmGroupCount);
            _ssmDecay = CreateF32("scratch_ssm_decay", config.SsmGroupCount);
            _ssmBetaVal = CreateF32("scratch_ssm_betaval", config.SsmGroupCount);
            _ssmGate = CreateF32("scratch_ssm_gate", config.SsmInnerSize);
            _ssmOutput = CreateF32("scratch_ssm_out", config.SsmInnerSize);
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

            // Save residual
            _backend.CopyTensor(_residual, _hidden);

            // Pre-attention RMSNorm
            _backend.RmsNorm(_normOut, _hidden, lw.AttnNorm, _config.NormEps);

            // Layer-type-specific attention
            if (lw is StandardAttentionWeights saw)
                ForwardStandardAttention(saw, position, layer);
            else if (lw is DeltaNetWeights dnw)
                ForwardDeltaNet(dnw, layer);

            // Residual add
            _backend.ElementAdd(_hidden, _hidden, _residual);

            // Save residual
            _backend.CopyTensor(_residual, _hidden);

            // Post-attention RMSNorm (= pre-FFN norm)
            _backend.RmsNorm(_normOut, _hidden, lw.PostAttnNorm, _config.NormEps);

            // SwiGLU FFN
            ProjectLinear(_gate, _normOut, lw.FfnGate);
            ProjectLinear(_up, _normOut, lw.FfnUp);
            _backend.SiLU(_gate, _gate);
            _backend.ElementMul(_gate, _gate, _up);
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
            // Qwen-style: Q projects to 2× dim (interleaved Q_attn + Q_gate)
            ProjectLinear(_qFull!, _normOut, w.AttnQ);
            _backend.DeInterleaveQ(_qAttn, _qGate, _qFull!, numHeads, keyLen);

            // Per-head Q/K norms
            _backend.PerHeadRmsNorm(_qAttn, w.AttnQNorm!, numHeads, keyLen, _config.NormEps);
            _backend.PerHeadRmsNorm(_kProj, w.AttnKNorm!, numKvHeads, keyLen, _config.NormEps);
        }
        else
        {
            // Standard LLaMA-style: Q projects directly to numHeads × keyLength
            ProjectLinear(_qAttn, _normOut, w.AttnQ);
            // _qGate already filled with 88.0 (sigmoid=1.0) in constructor
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
        int innerSize = _config.SsmInnerSize;
        int groupCount = _config.SsmGroupCount;
        int headDim = _config.SsmHeadDim;
        int qkvDim = innerSize * 3;
        int convKernel = _config.SsmConvKernel;

        // QKV projection
        ProjectLinear(_qkvBuf, _normOut, w.AttnQkv);

        // Causal conv1d using shift buffer
        var convBuf = _deltaState.GetConvBufferTensor(layer);
        _backend.CausalConv1d(_qkvBuf, convBuf, w.SsmConv1d, qkvDim, convKernel);

        // SiLU activation on entire conv output (Q+K+V)
        _backend.SiLUInPlace(_qkvBuf);

        // Split QKV into separate tensors
        _backend.SplitQKV(_ssmQ, _ssmK, _ssmV, _qkvBuf, innerSize);

        // L2-normalize Q and K per head
        _backend.L2NormGroups(_ssmQ, groupCount, headDim);
        _backend.L2NormGroups(_ssmK, groupCount, headDim);

        // Compute alpha and beta projections
        ProjectLinear(_ssmAlpha, _normOut, w.SsmAlpha);
        ProjectLinear(_ssmBeta, _normOut, w.SsmBeta);

        // Compute decay and beta values
        _backend.ComputeDecayBeta(_ssmDecay, _ssmBetaVal, _ssmAlpha, _ssmBeta,
            w.SsmA, w.SsmDtBias, groupCount);

        // DeltaNet state update + output + per-head norm
        var stateTensor = _deltaState.GetStateTensor(layer);
        float scale = 1.0f / MathF.Sqrt(headDim);
        _backend.DeltaNetStep(_ssmOutput, _ssmQ, _ssmK, _ssmV,
            stateTensor, _ssmDecay, _ssmBetaVal,
            w.SsmNorm, groupCount, headDim, scale, _config.NormEps);

        // Gate: output = output * silu(gate)
        ProjectLinear(_ssmGate, _normOut, w.AttnGate);
        _backend.SiLUGate(_ssmOutput, _ssmOutput, _ssmGate);

        // Output projection
        ProjectLinear(_hidden, _ssmOutput, w.SsmOut);
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
