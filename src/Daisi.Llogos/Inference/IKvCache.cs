using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Inference;

/// <summary>
/// Abstraction for KV cache implementations (monolithic, paged, offloaded).
/// </summary>
public interface IKvCache : IDisposable
{
    /// <summary>Number of positions visible to attention.</summary>
    int Length { get; }

    /// <summary>Maximum sequence length this cache can hold.</summary>
    int MaxSeqLen { get; }

    int NumKvHeads { get; }
    int KeyLength { get; }
    int ValueLength { get; }
    GgmlType CacheType { get; }
    AttentionStrategy Strategy { get; }

    /// <summary>Write K/V for a position, mapped through the attention strategy.</summary>
    void Write(IComputeBackend backend, int layer, int position, ITensor k, ITensor v);

    /// <summary>
    /// Write M K/V pairs at consecutive positions [startPos..startPos+M-1].
    /// k: [M × nKvHeads × keyLength], v: [M × nKvHeads × valueLength].
    /// Default: sequential fallback using single-token Write.
    /// </summary>
    void BatchedWrite(IComputeBackend backend, int layer, int startPosition, int M, ITensor k, ITensor v)
    {
        throw new NotSupportedException("BatchedWrite requires a KV cache implementation that supports it (PagedKvCache).");
    }

    /// <summary>Get the K cache tensor for an attention layer (contiguous, suitable for GatedAttention).</summary>
    ITensor GetKCacheTensor(int layer);

    /// <summary>Get the V cache tensor for an attention layer (contiguous, suitable for GatedAttention).</summary>
    ITensor GetVCacheTensor(int layer);

    void Reset();
}
