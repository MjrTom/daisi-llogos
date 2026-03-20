namespace Daisi.Llogos.Inference;

/// <summary>
/// Common interface for forward pass implementations (standard and BitNet).
/// </summary>
public interface IForwardPass : IDisposable
{
    ReadOnlySpan<float> Forward(int tokenId, int position);
    IKvCache KvCache { get; }

    /// <summary>
    /// Reset all inference state (KV cache, SSM state, etc.) for a new sequence.
    /// </summary>
    void ResetState();
}
