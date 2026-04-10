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

    // ── Gemma 4 Per-Layer Embedding (PLE) globals ───────────────────────────
    /// <summary>Gemma 4: concatenated per-layer token embedding table [n_embd_per_layer × n_layer × vocab].</summary>
    public ITensor? PerLayerTokenEmbd { get; set; }

    /// <summary>Gemma 4: projects hidden dim → 42-layer × 256 PLE space [hidden × (n_embd_per_layer × n_layer)].</summary>
    public ITensor? PerLayerModelProj { get; set; }

    /// <summary>Gemma 4: norm applied to per-layer projection [n_embd_per_layer].</summary>
    public ITensor? PerLayerProjNorm { get; set; }

    /// <summary>Gemma 4: precomputed RoPE freq factors for full-attention layers [head_dim/2].</summary>
    public ITensor? RopeFreqs { get; set; }

    /// <summary>
    /// Output weight — falls back to token embedding if tied.
    /// </summary>
    public ITensor OutputWeight => Output ?? TokenEmbedding;

    public void Dispose()
    {
        TokenEmbedding.Dispose();
        OutputNorm.Dispose();
        Output?.Dispose();
        PerLayerTokenEmbd?.Dispose();
        PerLayerModelProj?.Dispose();
        PerLayerProjNorm?.Dispose();
        RopeFreqs?.Dispose();
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
/// Weights for a Gemma 4 transformer layer.
///
/// Differences from <see cref="StandardAttentionWeights"/>:
///  - Four norms per layer: pre-attn (AttnNorm), post-attn (PostAttnNorm),
///    pre-FFN (FfnNorm), post-FFN (PostFfnNorm).
///  - Q/K per-head RmsNorm uses single-scalar weights broadcast across head_dim.
///  - No V weight is normalized at load time — V is unit-normalized at runtime.
///  - Optional Per-Layer Embedding (PLE) block: InpGate, Proj, PerLayerPostNorm.
///  - Optional per-layer scalar output multiplier (LayerOutScale, shape [1]).
///  - Optional per-layer rope_freqs (only on full-attention layers, all tied to one global tensor).
///  - K and V weights are loaded for every layer, but for KV-shared layers
///    (layer >= NumLayerKvFromStart) they are not used by the forward pass.
/// </summary>
public sealed class GemmaAttentionWeights : LayerWeights
{
    public required ITensor AttnQ { get; set; }
    public required ITensor AttnK { get; set; }
    public required ITensor AttnV { get; set; }
    public required ITensor AttnO { get; set; }
    public required ITensor AttnQNorm { get; set; }
    public required ITensor AttnKNorm { get; set; }

    /// <summary>Pre-FFN RmsNorm. Distinct from PostAttnNorm in Gemma 4 (4 norms per layer total).</summary>
    public required ITensor FfnNorm { get; set; }

    /// <summary>Post-FFN RmsNorm — applied to FFN output before residual add.</summary>
    public required ITensor PostFfnNorm { get; set; }

    /// <summary>Per-Layer Embedding input gate projection [hidden_dim × per_layer_input_dim]. Optional.</summary>
    public ITensor? PerLayerInpGate { get; set; }

    /// <summary>Per-Layer Embedding output projection [per_layer_input_dim × hidden_dim]. Optional.</summary>
    public ITensor? PerLayerProj { get; set; }

    /// <summary>Per-Layer Embedding post-norm [hidden_dim]. Optional.</summary>
    public ITensor? PerLayerPostNorm { get; set; }

    /// <summary>Optional per-layer scalar output multiplier, shape [1].</summary>
    public ITensor? LayerOutScale { get; set; }

    /// <summary>
    /// Optional precomputed RoPE freq factors for full-attention layers, shape [head_dim/2].
    /// All gemma4 full-attention layers share the same global tensor.
    /// </summary>
    public ITensor? RopeFreqs { get; set; }

    public override void Dispose()
    {
        base.Dispose();
        AttnQ.Dispose();
        AttnK.Dispose();
        AttnV.Dispose();
        AttnO.Dispose();
        AttnQNorm.Dispose();
        AttnKNorm.Dispose();
        FfnNorm.Dispose();
        PostFfnNorm.Dispose();
        PerLayerInpGate?.Dispose();
        PerLayerProj?.Dispose();
        PerLayerPostNorm?.Dispose();
        LayerOutScale?.Dispose();
        // Note: RopeFreqs is shared across layers — disposed by ModelWeights.RopeFreqs, not here.
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
