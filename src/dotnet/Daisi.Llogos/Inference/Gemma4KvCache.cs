using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Inference;

/// <summary>
/// KV cache for Gemma 4 (interleaved sliding-window + full attention).
///
/// Per-layer geometry differs by attention type:
///  - Sliding-window layers: head_dim = <see cref="ModelConfig.KeyLengthSwa"/> (256),
///    capacity = <see cref="ModelConfig.SlidingWindow"/> (512), ring-buffer addressing.
///  - Full-attention layers: head_dim = <see cref="ModelConfig.KeyLength"/> (512),
///    capacity = maxSeqLen.
///
/// KV-cache sharing (<c>shared_kv_layers</c>) is NOT yet implemented; every layer
/// computes and stores its own K/V. The 18 layers that should share consume their
/// own cache slots (~10 MB extra vs the optimized version).
/// </summary>
public sealed class Gemma4KvCache : IDisposable
{
    private readonly ITensor[] _kCaches;
    private readonly ITensor[] _vCaches;
    private readonly int[] _layerKeyLength;
    private readonly int[] _layerValueLength;
    private readonly int[] _layerCapacity; // per-layer max sequence length
    private readonly int _maxSeqLen;
    private readonly int _slidingWindow;
    private readonly int _nKvHeads;
    private readonly GgmlType _cacheType;
    private readonly ModelConfig _config;

    /// <summary>Position of the most recently written token (advances monotonically).</summary>
    public int Position { get; private set; } = -1;

    public int MaxSeqLen => _maxSeqLen;
    public int NumKvHeads => _nKvHeads;
    public GgmlType CacheType => _cacheType;
    public ModelConfig Config => _config;

    public Gemma4KvCache(IComputeBackend backend, ModelConfig config, int maxSeqLen,
        GgmlType cacheType = GgmlType.F32)
    {
        _config = config;
        _maxSeqLen = maxSeqLen;
        _slidingWindow = config.SlidingWindow > 0 ? config.SlidingWindow : maxSeqLen;
        _nKvHeads = config.NumKvHeads;
        _cacheType = cacheType;

        _kCaches = new ITensor[config.NumLayers];
        _vCaches = new ITensor[config.NumLayers];
        _layerKeyLength = new int[config.NumLayers];
        _layerValueLength = new int[config.NumLayers];
        _layerCapacity = new int[config.NumLayers];

        for (int i = 0; i < config.NumLayers; i++)
        {
            int kLen = config.LayerKeyLength(i);
            int vLen = config.LayerValueLength(i);
            int cap = config.IsSlidingLayer(i) ? Math.Min(_slidingWindow, maxSeqLen) : maxSeqLen;

            _layerKeyLength[i] = kLen;
            _layerValueLength[i] = vLen;
            _layerCapacity[i] = cap;

            long kSize = (long)_nKvHeads * cap * kLen;
            long vSize = (long)_nKvHeads * cap * vLen;
            _kCaches[i] = backend.CreateTensor($"gemma4_k_{i}", cacheType, [kSize]);
            _vCaches[i] = backend.CreateTensor($"gemma4_v_{i}", cacheType, [vSize]);
        }
    }

    /// <summary>Per-layer key length (head_dim differs sliding vs full).</summary>
    public int LayerKeyLength(int layer) => _layerKeyLength[layer];
    /// <summary>Per-layer value length.</summary>
    public int LayerValueLength(int layer) => _layerValueLength[layer];
    /// <summary>Per-layer cache capacity (sliding window or full context).</summary>
    public int LayerCapacity(int layer) => _layerCapacity[layer];

    public ITensor GetKCacheTensor(int layer) => _kCaches[layer];
    public ITensor GetVCacheTensor(int layer) => _vCaches[layer];

    /// <summary>
    /// Number of valid positions visible to attention at the given layer
    /// for the current cursor position. For sliding layers this caps at
    /// the window size; for full layers it's just position+1.
    /// </summary>
    public int LayerSeqLen(int layer)
    {
        int curPos = Position;
        if (curPos < 0) return 0;
        int cap = _layerCapacity[layer];
        return Math.Min(curPos + 1, cap);
    }

    /// <summary>
    /// Write K/V for the given layer at the current position. The position is
    /// mapped per-layer through the layer's ring-buffer (sliding) or linear (full)
    /// addressing.
    /// </summary>
    public void Write(IComputeBackend backend, int layer, int position, ITensor k, ITensor v)
    {
        int slot = MapSlot(layer, position);
        backend.KvCacheWrite(_kCaches[layer], _vCaches[layer], k, v,
            _nKvHeads, _layerKeyLength[layer], _layerValueLength[layer],
            _layerCapacity[layer], slot);
        if (position > Position) Position = position;
    }

    /// <summary>
    /// Map a logical position to a physical cache slot for the given layer.
    /// Sliding layers use simple ring-buffer addressing (no attention sinks for now).
    /// </summary>
    private int MapSlot(int layer, int position)
    {
        int cap = _layerCapacity[layer];
        if (position < cap) return position;
        return position % cap;
    }

    public void Reset() => Position = -1;

    public void Dispose()
    {
        foreach (var t in _kCaches) t.Dispose();
        foreach (var t in _vCaches) t.Dispose();
    }
}
