namespace Daisi.Llogos.Model;

/// <summary>
/// Named references to all loaded weight tensors for a hybrid transformer model.
/// </summary>
public sealed class ModelWeights : IDisposable
{
    public required ITensor TokenEmbedding { get; init; }
    public required ITensor OutputNorm { get; init; }
    public required ITensor? Output { get; init; }
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
    public required ITensor AttnNorm { get; init; }
    public required ITensor PostAttnNorm { get; init; }
    public required ITensor FfnGate { get; init; }
    public required ITensor FfnUp { get; init; }
    public required ITensor FfnDown { get; init; }

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
    public required ITensor AttnQ { get; init; }
    public required ITensor AttnK { get; init; }
    public required ITensor AttnV { get; init; }
    public required ITensor AttnO { get; init; }
    public ITensor? AttnQNorm { get; init; }
    public ITensor? AttnKNorm { get; init; }

    /// <summary>
    /// True when the Q projection is gated (Qwen-style: Q output is 2× head dim,
    /// interleaved Q_attn + Q_gate). False for standard LLaMA-style attention.
    /// </summary>
    public bool HasGatedQ => AttnQNorm != null;

    public override void Dispose()
    {
        base.Dispose();
        AttnQ.Dispose();
        AttnK.Dispose();
        AttnV.Dispose();
        AttnO.Dispose();
        AttnQNorm?.Dispose();
        AttnKNorm?.Dispose();
    }
}

/// <summary>
/// Weights for a DeltaNet (gated linear attention with delta rule) layer.
/// </summary>
public sealed class DeltaNetWeights : LayerWeights
{
    public required ITensor AttnQkv { get; init; }
    public required ITensor AttnGate { get; init; }
    public required ITensor SsmA { get; init; }
    public required ITensor SsmAlpha { get; init; }
    public required ITensor SsmBeta { get; init; }
    public required ITensor SsmConv1d { get; init; }
    public required ITensor SsmDtBias { get; init; }
    public required ITensor SsmNorm { get; init; }
    public required ITensor SsmOut { get; init; }

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
