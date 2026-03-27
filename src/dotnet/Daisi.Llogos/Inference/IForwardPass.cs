namespace Daisi.Llogos.Inference;

/// <summary>
/// Common interface for forward pass implementations (standard and BitNet).
/// </summary>
public interface IForwardPass : IDisposable
{
    ReadOnlySpan<float> Forward(int tokenId, int position);

    /// <summary>
    /// Run only the transformer layers without logit projection.
    /// Used for prefill tokens where logits aren't needed — skips the
    /// expensive RMSNorm + LM head + logit download.
    /// </summary>
    void ForwardHidden(int tokenId, int position);

    IKvCache KvCache { get; }

    /// <summary>
    /// Forward pass returning only the argmax token ID.
    /// Default: full Forward + CPU argmax. ForwardPass overrides with GPU-side partial vocab.
    /// </summary>
    int ForwardArgMax(int tokenId, int position)
    {
        var logits = Forward(tokenId, position);
        int best = 0;
        float bestVal = logits[0];
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        return best;
    }

    /// <summary>Whether this model supports batched prefill.</summary>
    bool SupportsBatchedPrefill => false;

    /// <summary>
    /// Process M tokens through all transformer layers in parallel.
    /// Only for models that support batched prefill (pure standard attention, CUDA backend).
    /// </summary>
    void ForwardBatchedPrefill(int[] tokenIds, int startPosition)
    {
        // Default: sequential fallback
        for (int i = 0; i < tokenIds.Length; i++)
            ForwardHidden(tokenIds[i], startPosition + i);
    }

    /// <summary>
    /// Reset all inference state (KV cache, SSM state, etc.) for a new sequence.
    /// </summary>
    void ResetState();
}
