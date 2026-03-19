using Daisi.Llama.Gguf;
using Daisi.Llama.Model;

namespace Daisi.Llama.Inference;

/// <summary>
/// Hybrid transformer forward pass supporting both standard attention and DeltaNet layers.
/// Given a token ID and position, produces logits over the vocabulary.
/// </summary>
public sealed class ForwardPass : IDisposable
{
    private readonly IComputeBackend _backend;
    private readonly ModelConfig _config;
    private readonly ModelWeights _weights;
    private readonly KvCache _kvCache;
    private readonly DeltaNetState _deltaState;

    // Scratch buffers
    private readonly ITensor _hidden;
    private readonly ITensor _residual;
    private readonly ITensor _normOut;
    private readonly ITensor _logits;

    // Standard attention scratch
    private readonly ITensor _qFull;    // [numHeads × keyLength × 2] (attn + gate)
    private readonly ITensor _kProj;    // [numKvHeads × keyLength]
    private readonly ITensor _vProj;    // [numKvHeads × valueLength]
    private readonly ITensor _attnOut;  // [numHeads × valueLength]

    // DeltaNet scratch
    private readonly ITensor _qkvBuf;   // [ssmInnerSize × 3]
    private readonly ITensor _ssmAlpha;  // [ssmGroupCount]
    private readonly ITensor _ssmBeta;   // [ssmGroupCount]
    private readonly ITensor _ssmGate;   // [ssmInnerSize]
    private readonly ITensor _ssmOutput; // [ssmInnerSize]

    // FFN scratch
    private readonly ITensor _gate;
    private readonly ITensor _up;

    public ForwardPass(IComputeBackend backend, ModelConfig config, ModelWeights weights,
        KvCache kvCache, DeltaNetState deltaState)
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

        // Standard attention scratch
        int qFullDim = config.NumHeads * config.KeyLength * 2; // attn Q + gate Q
        _qFull = CreateF32("scratch_q_full", qFullDim);
        _kProj = CreateF32("scratch_k", config.NumKvHeads * config.KeyLength);
        _vProj = CreateF32("scratch_v", config.NumKvHeads * config.ValueLength);
        _attnOut = CreateF32("scratch_attn_out", config.NumHeads * config.ValueLength);

        // DeltaNet scratch
        if (config.SsmInnerSize > 0)
        {
            _qkvBuf = CreateF32("scratch_qkv", config.SsmInnerSize * 3);
            _ssmAlpha = CreateF32("scratch_ssm_alpha", config.SsmGroupCount);
            _ssmBeta = CreateF32("scratch_ssm_beta", config.SsmGroupCount);
            _ssmGate = CreateF32("scratch_ssm_gate", config.SsmInnerSize);
            _ssmOutput = CreateF32("scratch_ssm_out", config.SsmInnerSize);
        }
        else
        {
            _qkvBuf = _ssmAlpha = _ssmBeta = _ssmGate = _ssmOutput = _hidden; // unused
        }

        _gate = CreateF32("scratch_ffn_gate", config.IntermediateDim);
        _up = CreateF32("scratch_ffn_up", config.IntermediateDim);
    }

    public KvCache KvCache => _kvCache;
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
            _hidden.AsFloatSpan().CopyTo(_residual.AsFloatSpan());

            // Pre-attention RMSNorm
            _backend.RmsNorm(_normOut, _hidden, lw.AttnNorm, _config.NormEps);

            // Layer-type-specific attention
            if (lw is StandardAttentionWeights saw)
                ForwardStandardAttention(saw, position);
            else if (lw is DeltaNetWeights dnw)
                ForwardDeltaNet(dnw, layer);

            // Residual add
            _backend.ElementAdd(_hidden, _hidden, _residual);

            // Save residual
            _hidden.AsFloatSpan().CopyTo(_residual.AsFloatSpan());

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

        return _logits.AsFloatSpan();
    }

    // ── Standard Attention (Gated) ───────────────────────────────────────────

    private void ForwardStandardAttention(StandardAttentionWeights w, int position)
    {
        int numHeads = _config.NumHeads;
        int numKvHeads = _config.NumKvHeads;
        int keyLen = _config.KeyLength;
        int valLen = _config.ValueLength;
        int headsPerGroup = _config.HeadsPerKvGroup;
        int ropeDim = _config.RopeDimCount;
        int seqLen = position + 1;
        float scale = 1.0f / MathF.Sqrt(keyLen);

        // Q/K/V projections
        ProjectLinear(_qFull, _normOut, w.AttnQ);
        ProjectLinear(_kProj, _normOut, w.AttnK);
        ProjectLinear(_vProj, _normOut, w.AttnV);

        var qFull = _qFull.AsFloatSpan();
        var kSpan = _kProj.AsFloatSpan();
        var vSpan = _vProj.AsFloatSpan();

        // Split Q into Q_attn and Q_gate (first half = attn, second half = gate)
        int qAttnSize = numHeads * keyLen;
        var qAttn = qFull.Slice(0, qAttnSize);
        var qGate = qFull.Slice(qAttnSize, qAttnSize);

        // Per-head Q norm and K norm
        var qNormW = w.AttnQNorm.AsFloatSpan();
        var kNormW = w.AttnKNorm.AsFloatSpan();
        for (int h = 0; h < numHeads; h++)
            RmsNormInPlace(qAttn.Slice(h * keyLen, keyLen), qNormW, _config.NormEps);
        for (int h = 0; h < numKvHeads; h++)
            RmsNormInPlace(kSpan.Slice(h * keyLen, keyLen), kNormW, _config.NormEps);

        // RoPE (partial — only first ropeDim dims)
        // Create temporary tensors for RoPE from the spans
        var qAttnTensor = _backend.CreateTensor("rope_q_tmp", GgmlType.F32, [(long)qAttnSize]);
        var kTensor = _backend.CreateTensor("rope_k_tmp", GgmlType.F32, [(long)(numKvHeads * keyLen)]);
        try
        {
            qAttn.CopyTo(qAttnTensor.AsFloatSpan());
            kSpan.CopyTo(kTensor.AsFloatSpan());
            _backend.RoPE(qAttnTensor, kTensor, keyLen, ropeDim, position, _config.RopeTheta);
            qAttnTensor.AsFloatSpan().CopyTo(qAttn);
            kTensor.AsFloatSpan().CopyTo(kSpan);
        }
        finally
        {
            qAttnTensor.Dispose();
            kTensor.Dispose();
        }

        // Write K/V to cache
        int layer = Array.IndexOf(_weights.Layers, w);
        _kvCache.Write(layer, position, kSpan, vSpan);

        // Compute attention
        var kCache = _kvCache.GetKCache(layer);
        var vCache = _kvCache.GetVCache(layer);
        var attnOutSpan = _attnOut.AsFloatSpan();
        int kHeadStride = _kvCache.KHeadStride;
        int vHeadStride = _kvCache.VHeadStride;

        Span<float> scores = seqLen <= 1024
            ? stackalloc float[seqLen]
            : new float[seqLen];

        for (int h = 0; h < numHeads; h++)
        {
            int kvHead = h / headsPerGroup;
            int qOff = h * keyLen;
            int kvKBase = kvHead * kHeadStride;
            int kvVBase = kvHead * vHeadStride;

            // Attention scores
            for (int p = 0; p < seqLen; p++)
            {
                float dot = 0;
                int kOff = kvKBase + p * keyLen;
                for (int d = 0; d < keyLen; d++)
                    dot += qAttn[qOff + d] * kCache[kOff + d];
                scores[p] = dot * scale;
            }

            SoftmaxInPlace(scores.Slice(0, seqLen));

            // Weighted V sum
            int outOff = h * valLen;
            for (int d = 0; d < valLen; d++)
            {
                float val = 0;
                for (int p = 0; p < seqLen; p++)
                    val += scores[p] * vCache[kvVBase + p * valLen + d];
                attnOutSpan[outOff + d] = val;
            }
        }

        // Gate attention output with sigmoid(Q_gate)
        for (int i = 0; i < numHeads * valLen; i++)
        {
            // Map attn_out index to corresponding gate index
            int h = i / valLen;
            int d = i % valLen;
            // Q_gate has keyLen dims per head, attn_out has valLen dims per head
            // Use element-wise gating where possible
            float gateVal = d < keyLen ? Sigmoid(qGate[h * keyLen + d]) : 1.0f;
            attnOutSpan[i] *= gateVal;
        }

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
        var qkv = _qkvBuf.AsFloatSpan();

        // Causal conv1d using shift buffer
        var convBuf = _deltaState.GetConvBuffer(layer);
        var convW = w.SsmConv1d.AsFloatSpan(); // [convKernel × qkvDim] but stored as [qkvDim, convKernel] in GGUF
        ApplyCausalConv1d(qkv, convBuf, convW, qkvDim, convKernel);

        // Split QKV
        var qSpan = qkv.Slice(0, innerSize);
        var kSpan = qkv.Slice(innerSize, innerSize);
        var vSpan = qkv.Slice(innerSize * 2, innerSize);

        // SiLU activation on Q
        for (int i = 0; i < innerSize; i++)
            qSpan[i] = qSpan[i] * Sigmoid(qSpan[i]);

        // Compute alpha (time step / decay) and beta (learning rate)
        ProjectLinear(_ssmAlpha, _normOut, w.SsmAlpha);
        ProjectLinear(_ssmBeta, _normOut, w.SsmBeta);
        var alphaSpan = _ssmAlpha.AsFloatSpan();
        var betaSpan = _ssmBeta.AsFloatSpan();
        var ssmA = w.SsmA.AsFloatSpan();
        var dtBias = w.SsmDtBias.AsFloatSpan();

        // Compute decay factors
        Span<float> decay = stackalloc float[groupCount];
        Span<float> beta = stackalloc float[groupCount];
        for (int g = 0; g < groupCount; g++)
        {
            float dt = SoftPlus(alphaSpan[g] + dtBias[g]);
            float a = -MathF.Exp(ssmA[g]);
            decay[g] = MathF.Exp(a * dt);
            beta[g] = Sigmoid(betaSpan[g]);
        }

        // DeltaNet state update and output computation
        var state = _deltaState.GetState(layer);
        var outSpan = _ssmOutput.AsFloatSpan();
        var normW = w.SsmNorm.AsFloatSpan();

        for (int g = 0; g < groupCount; g++)
        {
            int baseOff = g * headDim;
            int stateOff = g * headDim * headDim;

            // State update: S = decay * S + beta * outer(k, v)
            // Using simplified DeltaNet (without delta rule correction for Phase 4)
            float d = decay[g];
            float b = beta[g];

            for (int i = 0; i < headDim; i++)
            {
                float ki = kSpan[baseOff + i];
                int rowOff = stateOff + i * headDim;
                for (int j = 0; j < headDim; j++)
                {
                    state[rowOff + j] = d * state[rowOff + j] + b * ki * vSpan[baseOff + j];
                }
            }

            // Output: o = S^T × q
            for (int j = 0; j < headDim; j++)
            {
                float sum = 0;
                for (int i = 0; i < headDim; i++)
                    sum += state[stateOff + i * headDim + j] * qSpan[baseOff + i];
                outSpan[baseOff + j] = sum;
            }

            // Per-head RMSNorm
            RmsNormInPlace(outSpan.Slice(baseOff, headDim), normW, _config.NormEps);
        }

        // Gate: output = output * silu(gate)
        ProjectLinear(_ssmGate, _normOut, w.AttnGate);
        var gateSpan = _ssmGate.AsFloatSpan();
        for (int i = 0; i < innerSize; i++)
            outSpan[i] *= gateSpan[i] * Sigmoid(gateSpan[i]); // SiLU gate

        // Output projection
        ProjectLinear(_hidden, _ssmOutput, w.SsmOut);
    }

    // ── Causal Conv1D ────────────────────────────────────────────────────────

    /// <summary>
    /// Apply depthwise causal conv1d. Updates the shift buffer and modifies qkv in place.
    /// Conv weight in GGUF is [kernelSize, channels] = dim[0]=kernelSize, dim[1]=channels.
    /// </summary>
    private static void ApplyCausalConv1d(Span<float> qkv, Span<float> convBuf, ReadOnlySpan<float> convW,
        int channels, int kernelSize)
    {
        int bufSlots = kernelSize - 1;
        Span<float> result = stackalloc float[channels > 8192 ? 0 : channels];
        if (result.Length == 0) result = new float[channels];

        // Compute conv: sum over kernel positions
        // convBuf holds [bufSlots × channels], oldest first
        // convW layout: [kernelSize × channels] row-major → convW[k * channels + c]
        // But GGUF dim[0]=kernelSize means innermost is kernelSize, so convW[c * kernelSize + k]
        for (int c = 0; c < channels; c++)
        {
            float sum = 0;
            // Past values from buffer (oldest to newest)
            for (int k = 0; k < bufSlots; k++)
                sum += convBuf[k * channels + c] * convW[c * kernelSize + k];
            // Current value
            sum += qkv[c] * convW[c * kernelSize + bufSlots];
            result[c] = sum;
        }

        // Shift buffer: discard oldest, add current
        for (int k = 0; k < bufSlots - 1; k++)
            convBuf.Slice((k + 1) * channels, channels).CopyTo(convBuf.Slice(k * channels, channels));
        if (bufSlots > 0)
            qkv.Slice(0, channels).CopyTo(convBuf.Slice((bufSlots - 1) * channels, channels));

        // Write result back
        result.Slice(0, channels).CopyTo(qkv);
    }

    // ── Linear projection ────────────────────────────────────────────────────

    private void ProjectLinear(ITensor output, ITensor input, ITensor weight)
    {
        int K = (int)weight.Dimensions[0];
        int N = (int)weight.Dimensions[1];
        _backend.MatMul(output, input, weight, 1, K, N);
    }

    // ── Utility functions ────────────────────────────────────────────────────

    private static void RmsNormInPlace(Span<float> data, ReadOnlySpan<float> weight, float eps)
    {
        float sumSq = 0;
        for (int i = 0; i < data.Length; i++)
            sumSq += data[i] * data[i];
        float rms = MathF.Sqrt(sumSq / data.Length + eps);
        float invRms = 1.0f / rms;
        for (int i = 0; i < data.Length; i++)
            data[i] = data[i] * invRms * weight[i];
    }

    private static void SoftmaxInPlace(Span<float> values)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < values.Length; i++)
            if (values[i] > max) max = values[i];

        float sum = 0;
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = MathF.Exp(values[i] - max);
            sum += values[i];
        }

        float invSum = 1.0f / sum;
        for (int i = 0; i < values.Length; i++)
            values[i] *= invSum;
    }

    private static float Sigmoid(float x) => 1.0f / (1.0f + MathF.Exp(-x));
    private static float SoftPlus(float x) => MathF.Log(1.0f + MathF.Exp(x));

    private ITensor CreateF32(string name, int size) =>
        _backend.CreateTensor(name, GgmlType.F32, [(long)size]);

    public void Dispose()
    {
        _hidden.Dispose();
        _residual.Dispose();
        _normOut.Dispose();
        _logits.Dispose();
        _qFull.Dispose();
        _kProj.Dispose();
        _vProj.Dispose();
        _attnOut.Dispose();
        if (_config.SsmInnerSize > 0)
        {
            _qkvBuf.Dispose();
            _ssmAlpha.Dispose();
            _ssmBeta.Dispose();
            _ssmGate.Dispose();
            _ssmOutput.Dispose();
        }
        _gate.Dispose();
        _up.Dispose();
    }
}
