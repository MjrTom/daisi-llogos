using Daisi.Llama.Gguf;
using Daisi.Llama.Model;

namespace Daisi.Llama.Inference;

/// <summary>
/// Simple KV cache for BitNet models where ALL layers are standard attention.
/// Separate from <see cref="KvCache"/> to avoid polluting the hybrid attention logic.
/// </summary>
public sealed class BitNetKvCache : IKvCache
{
    private readonly ITensor[] _kCaches;
    private readonly ITensor[] _vCaches;
    private readonly int _maxSeqLen;
    private readonly int _nKvHeads;
    private readonly int _keyLength;
    private readonly int _valueLength;
    private readonly GgmlType _cacheType;

    public int Length { get; private set; }
    public int MaxSeqLen => _maxSeqLen;
    public int NumKvHeads => _nKvHeads;
    public int KeyLength => _keyLength;
    public int ValueLength => _valueLength;
    public GgmlType CacheType => _cacheType;
    public AttentionStrategy Strategy => AttentionStrategy.Full;

    public BitNetKvCache(IComputeBackend backend, ModelConfig config, int maxSeqLen,
        GgmlType cacheType = GgmlType.F16)
    {
        _maxSeqLen = maxSeqLen;
        _nKvHeads = config.NumKvHeads;
        _keyLength = config.KeyLength;
        _valueLength = config.ValueLength;
        _cacheType = cacheType;

        _kCaches = new ITensor[config.NumLayers];
        _vCaches = new ITensor[config.NumLayers];

        for (int i = 0; i < config.NumLayers; i++)
        {
            long kSize = _nKvHeads * _maxSeqLen * _keyLength;
            long vSize = _nKvHeads * _maxSeqLen * _valueLength;
            _kCaches[i] = backend.CreateTensor($"kv_k_{i}", cacheType, [kSize]);
            _vCaches[i] = backend.CreateTensor($"kv_v_{i}", cacheType, [vSize]);
        }
    }

    public ITensor GetKCacheTensor(int layer) => _kCaches[layer];
    public ITensor GetVCacheTensor(int layer) => _vCaches[layer];

    public void Write(IComputeBackend backend, int layer, int position, ITensor k, ITensor v)
    {
        backend.KvCacheWrite(_kCaches[layer], _vCaches[layer], k, v,
            _nKvHeads, _keyLength, _valueLength, _maxSeqLen, position);
        Length = position + 1;
    }

    public void Reset() => Length = 0;

    public void Dispose()
    {
        foreach (var t in _kCaches) t.Dispose();
        foreach (var t in _vCaches) t.Dispose();
    }
}
