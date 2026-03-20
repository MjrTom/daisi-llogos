using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Inference;

/// <summary>
/// BitNet b1.58 forward pass. Separate from <see cref="ForwardPass"/> to avoid
/// any performance impact on the standard Qwen path.
///
/// Key differences from standard transformer:
/// - SubLN: extra RMSNorm after Q*K*V projections and in FFN
/// - Squared ReLU activation (not SiLU/SwiGLU)
/// - Standard MHA (no gated attention) — uses dummy gate (sigmoid(88)=1.0)
/// - No per-head Q/K norms
/// - Tied embeddings (output weight = token embedding)
/// </summary>
public sealed class BitNetForwardPass : IDisposable
{
    private readonly IComputeBackend _backend;
    private readonly ModelConfig _config;
    private readonly ModelWeights _weights;
    private readonly IKvCache _kvCache;

    // Scratch buffers
    private readonly ITensor _hidden;
    private readonly ITensor _residual;
    private readonly ITensor _normOut;
    private readonly ITensor _logits;
    private readonly float[] _logitsBuffer;

    // Attention scratch
    private readonly ITensor _qProj;    // [numHeads * keyLength]
    private readonly ITensor _kProj;    // [numKvHeads * keyLength]
    private readonly ITensor _vProj;    // [numKvHeads * valueLength]
    private readonly ITensor _qGate;    // dummy gate filled with 88.0 so sigmoid=1.0
    private readonly ITensor _attnOut;  // [numHeads * valueLength]

    // FFN scratch
    private readonly ITensor _gate;
    private readonly ITensor _up;
    private readonly ITensor _ffnNormOut; // SubLN output in FFN

    public BitNetForwardPass(IComputeBackend backend, ModelConfig config, ModelWeights weights, IKvCache kvCache)
    {
        _backend = backend;
        _config = config;
        _weights = weights;
        _kvCache = kvCache;

        _hidden = CreateF32("scratch_hidden", config.HiddenDim);
        _residual = CreateF32("scratch_residual", config.HiddenDim);
        _normOut = CreateF32("scratch_norm", config.HiddenDim);
        _logits = CreateF32("scratch_logits", config.VocabSize);
        _logitsBuffer = new float[config.VocabSize];

        int qDim = config.NumHeads * config.KeyLength;
        _qProj = CreateF32("scratch_q", qDim);
        _kProj = CreateF32("scratch_k", config.NumKvHeads * config.KeyLength);
        _vProj = CreateF32("scratch_v", config.NumKvHeads * config.ValueLength);
        _attnOut = CreateF32("scratch_attn_out", config.NumHeads * config.ValueLength);

        // Dummy gate: fill with 88.0 so sigmoid(88)≈1.0 exactly
        // This makes GatedAttention behave as standard (ungated) attention
        _qGate = CreateF32("scratch_q_gate", qDim);
        backend.FillTensor(_qGate, 88.0f);

        _gate = CreateF32("scratch_ffn_gate", config.IntermediateDim);
        _up = CreateF32("scratch_ffn_up", config.IntermediateDim);
        _ffnNormOut = CreateF32("scratch_ffn_norm", config.IntermediateDim);
    }

    public IKvCache KvCache => _kvCache;

    public ReadOnlySpan<float> Forward(int tokenId, int position)
    {
        // 1. Embedding lookup
        _backend.EmbeddingLookup(_hidden, _weights.TokenEmbedding, tokenId);

        // 2. Transformer layers
        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            var lw = (BitNetLayerWeights)_weights.Layers[layer];

            // Save residual
            _backend.CopyTensor(_residual, _hidden);

            // Pre-attention RMSNorm
            _backend.RmsNorm(_normOut, _hidden, lw.AttnNorm, _config.NormEps);

            // Attention
            ForwardAttention(lw, position, layer);

            // Residual add
            _backend.ElementAdd(_hidden, _hidden, _residual);

            // Save residual
            _backend.CopyTensor(_residual, _hidden);

            // Post-attention RMSNorm (ffn_norm in the GGUF)
            _backend.RmsNorm(_normOut, _hidden, lw.PostAttnNorm, _config.NormEps);

            // BitNet FFN: Squared ReLU instead of SwiGLU
            ForwardFFN(lw);

            // Residual add
            _backend.ElementAdd(_hidden, _hidden, _residual);
        }

        // 3. Final RMSNorm
        _backend.RmsNorm(_normOut, _hidden, _weights.OutputNorm, _config.NormEps);

        // 4. LM head (tied embeddings)
        ProjectLinear(_logits, _normOut, _weights.OutputWeight);

        _logits.DequantizeTo(_logitsBuffer);
        return _logitsBuffer;
    }

    private void ForwardAttention(BitNetLayerWeights w, int position, int layer)
    {
        int numHeads = _config.NumHeads;
        int numKvHeads = _config.NumKvHeads;
        int keyLen = _config.KeyLength;
        int valLen = _config.ValueLength;
        int ropeDim = _config.RopeDimCount;
        float scale = 1.0f / MathF.Sqrt(keyLen);

        // Q/K/V projections
        ProjectLinear(_qProj, _normOut, w.AttnQ);
        ProjectLinear(_kProj, _normOut, w.AttnK);
        ProjectLinear(_vProj, _normOut, w.AttnV);

        // RoPE
        _backend.RoPE(_qProj, _kProj, keyLen, ropeDim, position, _config.RopeTheta);

        // Write K/V to cache
        _kvCache.Write(_backend, layer, position, _kProj, _vProj);
        int seqLen = _kvCache.Length;

        // Standard attention via GatedAttention with dummy gate (sigmoid(88)=1.0)
        var kCacheTensor = _kvCache.GetKCacheTensor(layer);
        var vCacheTensor = _kvCache.GetVCacheTensor(layer);
        _backend.GatedAttention(_attnOut, _qProj, _qGate, kCacheTensor, vCacheTensor,
            numHeads, numKvHeads, keyLen, valLen, _kvCache.MaxSeqLen, seqLen, scale);

        // SubLN: RMSNorm on attention output BEFORE output projection
        // attn_sub_norm has dim=hidden_size, applied to full concatenated multi-head output
        _backend.RmsNorm(_attnOut, _attnOut, w.AttnSubNorm, _config.NormEps);

        // Output projection
        ProjectLinear(_hidden, _attnOut, w.AttnO);
    }

    private void ForwardFFN(BitNetLayerWeights w)
    {
        // gate = SquaredReLU(gate_proj(x))
        ProjectLinear(_gate, _normOut, w.FfnGate);
        SquaredReLU(_gate);

        // up = up_proj(x)
        ProjectLinear(_up, _normOut, w.FfnUp);

        // intermediate = relu2(gate) * up
        _backend.ElementMul(_gate, _gate, _up);

        // SubLN: RMSNorm on the product BEFORE down projection
        // ffn_sub_norm has dim=intermediate_size
        _backend.RmsNorm(_ffnNormOut, _gate, w.FfnSubNorm, _config.NormEps);

        // down projection
        ProjectLinear(_hidden, _ffnNormOut, w.FfnDown);
    }

    private void SquaredReLU(ITensor data) => _backend.SquaredReLU(data);

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
        _gate.Dispose();
        _up.Dispose();
        _ffnNormOut.Dispose();
    }
}
