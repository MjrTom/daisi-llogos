namespace Daisi.Llama.Model;

/// <summary>
/// Per-layer weights for BitNet b1.58 architecture.
/// Differences from standard Qwen: SubLN norms (attn_sub_norm, ffn_sub_norm),
/// ffn_norm is the post-attention norm (not post_attention_norm),
/// no Q/K per-head norms, no gated attention (standard MHA).
/// </summary>
public sealed class BitNetLayerWeights : LayerWeights
{
    public required ITensor AttnQ { get; init; }
    public required ITensor AttnK { get; init; }
    public required ITensor AttnV { get; init; }
    public required ITensor AttnO { get; init; }
    public required ITensor AttnSubNorm { get; init; } // SubLN after attention projection
    public required ITensor FfnSubNorm { get; init; }  // SubLN in FFN

    public override void Dispose()
    {
        base.Dispose();
        AttnQ.Dispose();
        AttnK.Dispose();
        AttnV.Dispose();
        AttnO.Dispose();
        AttnSubNorm.Dispose();
        FfnSubNorm.Dispose();
    }
}
