using Daisi.Llama.Gguf;
using Daisi.Llama.Model;

namespace Daisi.Llama.Inference;

/// <summary>
/// Key-value cache for standard attention layers only.
/// Layout per layer: [nKvHeads × maxSeqLen × keyLength] for K, [nKvHeads × maxSeqLen × valueLength] for V.
/// </summary>
public sealed class KvCache : IDisposable
{
    private readonly ITensor[] _kCaches;
    private readonly ITensor[] _vCaches;
    private readonly int[] _layerIndices; // maps sequential index → actual layer number
    private readonly int _maxSeqLen;
    private readonly int _nKvHeads;
    private readonly int _keyLength;
    private readonly int _valueLength;

    /// <summary>Number of positions currently filled.</summary>
    public int Length { get; private set; }

    public KvCache(IComputeBackend backend, ModelConfig config, int maxSeqLen)
    {
        _maxSeqLen = maxSeqLen;
        _nKvHeads = config.NumKvHeads;
        _keyLength = config.KeyLength;
        _valueLength = config.ValueLength;

        // Only allocate for standard attention layers
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
            _kCaches[i] = backend.CreateTensor($"kv_k_{layer}", GgmlType.F32, [(long)kSize]);
            _vCaches[i] = backend.CreateTensor($"kv_v_{layer}", GgmlType.F32, [(long)vSize]);
        }
    }

    private int GetCacheIndex(int layer)
    {
        for (int i = 0; i < _layerIndices.Length; i++)
            if (_layerIndices[i] == layer) return i;
        throw new ArgumentException($"Layer {layer} is not a standard attention layer.");
    }

    /// <summary>
    /// Write K and V for a single position.
    /// k: [nKvHeads × keyLength], v: [nKvHeads × valueLength].
    /// </summary>
    public void Write(int layer, int position, ReadOnlySpan<float> k, ReadOnlySpan<float> v)
    {
        int idx = GetCacheIndex(layer);
        var kSpan = _kCaches[idx].AsFloatSpan();
        var vSpan = _vCaches[idx].AsFloatSpan();

        for (int h = 0; h < _nKvHeads; h++)
        {
            int kCacheOff = h * _maxSeqLen * _keyLength + position * _keyLength;
            int kSrcOff = h * _keyLength;
            k.Slice(kSrcOff, _keyLength).CopyTo(kSpan.Slice(kCacheOff, _keyLength));

            int vCacheOff = h * _maxSeqLen * _valueLength + position * _valueLength;
            int vSrcOff = h * _valueLength;
            v.Slice(vSrcOff, _valueLength).CopyTo(vSpan.Slice(vCacheOff, _valueLength));
        }

        if (position >= Length)
            Length = position + 1;
    }

    /// <summary>Get K cache span for a standard attention layer.</summary>
    public Span<float> GetKCache(int layer) => _kCaches[GetCacheIndex(layer)].AsFloatSpan();

    /// <summary>Get V cache span for a standard attention layer.</summary>
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
