using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Inference;

/// <summary>
/// Key-value cache for standard attention layers only.
/// Layout per layer: [nKvHeads × maxSeqLen × keyLength] for K, [nKvHeads × maxSeqLen × valueLength] for V.
/// Supports FP16 storage for 2x memory savings (set cacheType to GgmlType.F16).
/// Supports sliding window + attention sinks for fixed-memory streaming (set strategy).
/// </summary>
public sealed class KvCache : IKvCache
{
    private readonly ITensor[] _kCaches;
    private readonly ITensor[] _vCaches;
    private readonly int[] _layerIndices;
    private readonly int _maxSeqLen;
    private readonly int _nKvHeads;
    private readonly int _keyLength;
    private readonly int _valueLength;
    private readonly GgmlType _cacheType;
    private readonly AttentionStrategy _strategy;

    /// <summary>Number of positions visible to attention (capped by strategy capacity).</summary>
    public int Length { get; private set; }
    public int MaxSeqLen => _maxSeqLen;
    public int NumKvHeads => _nKvHeads;
    public int KeyLength => _keyLength;
    public int ValueLength => _valueLength;
    public GgmlType CacheType => _cacheType;
    public AttentionStrategy Strategy => _strategy;

    public KvCache(IComputeBackend backend, ModelConfig config, int maxSeqLen,
        GgmlType cacheType = GgmlType.F16, AttentionStrategy? strategy = null)
    {
        _strategy = strategy ?? AttentionStrategy.Full;

        // For window/sinks strategies, clamp maxSeqLen to the cache capacity
        if (_strategy.Mode != AttentionMode.Full && _strategy.CacheCapacity > 0)
            maxSeqLen = Math.Min(maxSeqLen, _strategy.CacheCapacity);

        _maxSeqLen = maxSeqLen;
        _nKvHeads = config.NumKvHeads;
        _keyLength = config.KeyLength;
        _valueLength = config.ValueLength;
        _cacheType = cacheType;

        var attnLayers = new List<int>();
        for (int i = 0; i < config.NumLayers; i++)
        {
            if (config.IsStandardAttention(i))
                attnLayers.Add(i);
        }

        _layerIndices = attnLayers.ToArray();
        _kCaches = new ITensor[_layerIndices.Length];
        _vCaches = new ITensor[_layerIndices.Length];

        for (int i = 0; i < _layerIndices.Length; i++)
        {
            int layer = _layerIndices[i];
            long kSize = _nKvHeads * _maxSeqLen * _keyLength;
            long vSize = _nKvHeads * _maxSeqLen * _valueLength;
            _kCaches[i] = backend.CreateTensor($"kv_k_{layer}", cacheType, [kSize]);
            _vCaches[i] = backend.CreateTensor($"kv_v_{layer}", cacheType, [vSize]);
        }
    }

    private int GetCacheIndex(int layer)
    {
        for (int i = 0; i < _layerIndices.Length; i++)
            if (_layerIndices[i] == layer) return i;
        throw new ArgumentException($"Layer {layer} is not a standard attention layer.");
    }

    /// <summary>Get K cache tensor for a standard attention layer.</summary>
    public ITensor GetKCacheTensor(int layer) => _kCaches[GetCacheIndex(layer)];

    /// <summary>Get V cache tensor for a standard attention layer.</summary>
    public ITensor GetVCacheTensor(int layer) => _vCaches[GetCacheIndex(layer)];

    /// <summary>
    /// Write K and V for a single position using backend operations.
    /// The position is mapped through the attention strategy (ring buffer for sliding window).
    /// </summary>
    public void Write(IComputeBackend backend, int layer, int position, ITensor k, ITensor v)
    {
        int slot = _strategy.MapPosition(position);
        int idx = GetCacheIndex(layer);
        backend.KvCacheWrite(_kCaches[idx], _vCaches[idx], k, v,
            _nKvHeads, _keyLength, _valueLength, _maxSeqLen, slot);

        Length = _strategy.EffectiveSeqLen(position);
    }

    public void BatchedWrite(IComputeBackend backend, int layer, int startPosition, int M, ITensor k, ITensor v)
    {
        int startSlot = _strategy.MapPosition(startPosition);
        int idx = GetCacheIndex(layer);
        backend.BatchedKvCacheWrite(_kCaches[idx], _vCaches[idx], k, v,
            _nKvHeads, _keyLength, _valueLength, _maxSeqLen, startSlot, M);

        Length = _strategy.EffectiveSeqLen(startPosition + M - 1);
    }

    // Legacy span-based methods for backward compatibility
    public void Write(int layer, int position, ReadOnlySpan<float> k, ReadOnlySpan<float> v)
    {
        int slot = _strategy.MapPosition(position);
        int idx = GetCacheIndex(layer);
        var kSpan = _kCaches[idx].AsFloatSpan();
        var vSpan = _vCaches[idx].AsFloatSpan();

        for (int h = 0; h < _nKvHeads; h++)
        {
            int kCacheOff = h * _maxSeqLen * _keyLength + slot * _keyLength;
            k.Slice(h * _keyLength, _keyLength).CopyTo(kSpan.Slice(kCacheOff, _keyLength));

            int vCacheOff = h * _maxSeqLen * _valueLength + slot * _valueLength;
            v.Slice(h * _valueLength, _valueLength).CopyTo(vSpan.Slice(vCacheOff, _valueLength));
        }

        Length = _strategy.EffectiveSeqLen(position);
    }

    public Span<float> GetKCache(int layer) => _kCaches[GetCacheIndex(layer)].AsFloatSpan();
    public Span<float> GetVCache(int layer) => _vCaches[GetCacheIndex(layer)].AsFloatSpan();
    public int KHeadStride => _maxSeqLen * _keyLength;
    public int VHeadStride => _maxSeqLen * _valueLength;

    public void Reset() => Length = 0;

    public void Dispose()
    {
        foreach (var t in _kCaches) t.Dispose();
        foreach (var t in _vCaches) t.Dispose();
    }
}
