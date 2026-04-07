namespace Daisi.Llogos.Model;

/// <summary>
/// Named references to all loaded weight tensors for a hybrid transformer model.
/// </summary>
public sealed class ModelWeights : IDisposable
{
    public required ITensor TokenEmbedding { get; set; }
    public required ITensor OutputNorm { get; set; }
    public required ITensor? Output { get; set; }
    public required LayerWeights[] Layers { get; init; }

    /// <summary>
    /// Output weight — falls back to token embedding if tied.
    /// </summary>
    public ITensor OutputWeight => Output ?? TokenEmbedding;

    public void Dispose()
    {
        TokenEmbedding.Dispose();
        OutputNorm.Dispose();
        Output?.Dispose();
        foreach (var layer in Layers)
            layer.Dispose();
    }
}

/// <summary>
/// Base class for per-layer weights. Shared: norms and FFN weights.
/// </summary>
public abstract class LayerWeights : IDisposable
{
    public required ITensor AttnNorm { get; set; }
    public required ITensor PostAttnNorm { get; set; }
    public required ITensor FfnGate { get; set; }
    public required ITensor FfnUp { get; set; }
    public required ITensor FfnDown { get; set; }

    public virtual void Dispose()
    {
        AttnNorm.Dispose();
        PostAttnNorm.Dispose();
        FfnGate.Dispose();
        FfnUp.Dispose();
        FfnDown.Dispose();
    }
}

/// <summary>
/// Weights for a standard (gated) multi-head attention layer.
/// Q is split into Q_attn + Q_gate. Q and K have per-head norms.
/// </summary>
public sealed class StandardAttentionWeights : LayerWeights
{
    public required ITensor AttnQ { get; set; }
    public required ITensor AttnK { get; set; }
    public required ITensor AttnV { get; set; }
    public required ITensor AttnO { get; set; }
    public ITensor? AttnQNorm { get; set; }
    public ITensor? AttnKNorm { get; set; }
    public ITensor? AttnQBias { get; set; }
    public ITensor? AttnKBias { get; set; }
    public ITensor? AttnVBias { get; set; }

    /// <summary>Fused Q+K+V weight tensor (concatenated rows). Null if types differ.</summary>
    public ITensor? FusedQKV { get; set; }

    /// <summary>Fused FFN gate+up weight tensor. Null if types differ.</summary>
    public ITensor? FusedGateUp { get; set; }

    /// <summary>
    /// True when the Q projection is gated (Qwen3.5-style: Q output is 2× expected dim,
    /// interleaved Q_attn + Q_gate). Detected by checking if Q output exceeds the
    /// hidden dim (gated Q outputs 2× NumHeads × HeadDim which exceeds hiddenDim).
    /// </summary>
    public bool HasGatedQ
    {
        get
        {
            if (AttnQNorm == null) return false;
            long qOutDim = AttnQ.Dimensions.Length > 1 ? AttnQ.Dimensions[1] : AttnQ.ElementCount;
            long hiddenDim = AttnQ.Dimensions[0]; // input dim = hidden dim
            // Gated Q: qOutDim = 2 × numHeads × headDim > hiddenDim
            // Non-gated Q: qOutDim = numHeads × headDim = hiddenDim (for non-GQA) or > hiddenDim (for GQA with large heads)
            // The safest check: gated Q always doubles the output, so qOutDim > hiddenDim
            return qOutDim > hiddenDim;
        }
    }

    public override void Dispose()
    {
        base.Dispose();
        AttnQ.Dispose();
        AttnK.Dispose();
        AttnV.Dispose();
        AttnO.Dispose();
        AttnQNorm?.Dispose();
        AttnKNorm?.Dispose();
        FusedQKV?.Dispose();
        FusedGateUp?.Dispose();
    }
}

/// <summary>
/// Weights for a DeltaNet (gated linear attention with delta rule) layer.
/// </summary>
public sealed class DeltaNetWeights : LayerWeights
{
    public required ITensor AttnQkv { get; set; }
    public required ITensor AttnGate { get; set; }
    public required ITensor SsmA { get; set; }
    public required ITensor SsmAlpha { get; set; }
    public required ITensor SsmBeta { get; set; }
    public required ITensor SsmConv1d { get; set; }
    public required ITensor SsmDtBias { get; set; }
    public required ITensor SsmNorm { get; set; }
    public required ITensor SsmOut { get; set; }

    public override void Dispose()
    {
        base.Dispose();
        AttnQkv.Dispose();
        AttnGate.Dispose();
        SsmA.Dispose();
        SsmAlpha.Dispose();
        SsmBeta.Dispose();
        SsmConv1d.Dispose();
        SsmDtBias.Dispose();
        SsmNorm.Dispose();
        SsmOut.Dispose();
    }
}
