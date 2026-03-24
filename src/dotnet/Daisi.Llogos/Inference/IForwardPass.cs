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
