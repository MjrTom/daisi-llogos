using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Inference;

/// <summary>
/// Paged KV cache that allocates memory in page-sized increments.
/// Memory grows with actual context usage instead of pre-allocating for max context.
/// Supports RAM offloading: when vramPageBudget is set, pages beyond the budget
/// are allocated in pinned host memory (accessible by GPU at reduced bandwidth).
/// </summary>
public sealed class PagedKvCache : IKvCache
{
    /// <summary>Number of token positions per page.</summary>
    public const int PageSize = 256;

    private readonly IComputeBackend _backend;
    private readonly int[] _layerIndices;
    private readonly int _nKvHeads;
    private readonly int _keyLength;
    private readonly int _valueLength;
    private readonly GgmlType _cacheType;
    private readonly AttentionStrategy _strategy;
    private readonly int _maxSeqLen;
    private readonly int _vramPageBudget; // 0 = unlimited VRAM

    // Per attention layer: list of page tensors for K and V
    private readonly List<ITensor>[] _kPages;
    private readonly List<ITensor>[] _vPages;

    // Per attention layer: contiguous scratch tensors for attention (rebuilt on grow)
    private ITensor[] _kScratch;
    private ITensor[] _vScratch;
    private int _scratchCapacity; // in positions (= allocatedPages * PageSize)

    private int _allocatedPages;

    public int Length { get; private set; }

    /// <summary>
    /// Returns the current scratch stride (grows with pages).
    /// This is the stride used in the contiguous scratch tensors passed to GatedAttention.
    /// </summary>
    public int MaxSeqLen => _scratchCapacity > 0 ? _scratchCapacity : _maxSeqLen;

    public int NumKvHeads => _nKvHeads;
    public int KeyLength => _keyLength;
    public int ValueLength => _valueLength;
    public GgmlType CacheType => _cacheType;
    public AttentionStrategy Strategy => _strategy;
    public int AllocatedPages => _allocatedPages;

    /// <summary>Total bytes allocated across all pages and scratch tensors.</summary>
    public long AllocatedBytes
    {
        get
        {
            long total = 0;
            for (int i = 0; i < _layerIndices.Length; i++)
            {
                foreach (var p in _kPages[i]) total += p.ByteSize;
                foreach (var p in _vPages[i]) total += p.ByteSize;
                if (_kScratch[i] != null) total += _kScratch[i].ByteSize;
                if (_vScratch[i] != null) total += _vScratch[i].ByteSize;
            }
            return total;
        }
    }

    public PagedKvCache(IComputeBackend backend, ModelConfig config, int maxSeqLen,
        GgmlType cacheType = GgmlType.F16, AttentionStrategy? strategy = null,
        int vramPageBudget = 0)
    {
        _backend = backend;
        _strategy = strategy ?? AttentionStrategy.Full;
        _cacheType = cacheType;
        _vramPageBudget = vramPageBudget;

        if (_strategy.Mode != AttentionMode.Full && _strategy.CacheCapacity > 0)
            maxSeqLen = Math.Min(maxSeqLen, _strategy.CacheCapacity);

        _maxSeqLen = maxSeqLen;
        _nKvHeads = config.NumKvHeads;
        _keyLength = config.KeyLength;
        _valueLength = config.ValueLength;

        var attnLayers = new List<int>();
        for (int i = 0; i < config.NumLayers; i++)
            if (config.IsStandardAttention(i))
                attnLayers.Add(i);

        _layerIndices = attnLayers.ToArray();
        int numLayers = _layerIndices.Length;
        _kPages = new List<ITensor>[numLayers];
        _vPages = new List<ITensor>[numLayers];
        _kScratch = new ITensor[numLayers];
        _vScratch = new ITensor[numLayers];

        for (int i = 0; i < numLayers; i++)
        {
            _kPages[i] = new List<ITensor>();
            _vPages[i] = new List<ITensor>();
        }
    }

    private int GetCacheIndex(int layer)
    {
        for (int i = 0; i < _layerIndices.Length; i++)
            if (_layerIndices[i] == layer) return i;
        throw new ArgumentException($"Layer {layer} is not a standard attention layer.");
    }

    public void Write(IComputeBackend backend, int layer, int position, ITensor k, ITensor v)
    {
        int slot = _strategy.MapPosition(position);
        int pageIdx = slot / PageSize;
        int pageOff = slot % PageSize;

        // Ensure pages exist
        EnsurePages(pageIdx + 1);

        int idx = GetCacheIndex(layer);

        // Ensure scratch exists and is big enough (so we can write to it in sync)
        EnsureScratch(idx);

        // Write to the page (canonical storage)
        backend.KvCacheWrite(_kPages[idx][pageIdx], _vPages[idx][pageIdx], k, v,
            _nKvHeads, _keyLength, _valueLength, PageSize, pageOff);

        // Write to the scratch (contiguous mirror, used by GatedAttention)
        backend.KvCacheWrite(_kScratch[idx], _vScratch[idx], k, v,
            _nKvHeads, _keyLength, _valueLength, _scratchCapacity, slot);

        Length = _strategy.EffectiveSeqLen(position);
    }

    public void BatchedWrite(IComputeBackend backend, int layer, int startPosition, int M, ITensor k, ITensor v)
    {
        // Ensure pages and scratch exist for the full range
        int lastSlot = _strategy.MapPosition(startPosition + M - 1);
        int neededPages = lastSlot / PageSize + 1;
        EnsurePages(neededPages);

        int idx = GetCacheIndex(layer);
        EnsureScratch(idx);

        int startSlot = _strategy.MapPosition(startPosition);

        // Write all M positions to scratch in one batched kernel call.
        // Scratch is the canonical store — GrowScratch copies old scratch to new,
        // so page-level writes are not needed here.
        backend.BatchedKvCacheWrite(_kScratch[idx], _vScratch[idx], k, v,
            _nKvHeads, _keyLength, _valueLength, _scratchCapacity, startSlot, M);

        Length = _strategy.EffectiveSeqLen(startPosition + M - 1);
    }

    public ITensor GetKCacheTensor(int layer)
    {
        int idx = GetCacheIndex(layer);
        EnsureScratch(idx);
        return _kScratch[idx];
    }

    public ITensor GetVCacheTensor(int layer)
    {
        int idx = GetCacheIndex(layer);
        EnsureScratch(idx);
        return _vScratch[idx];
    }

    public void Reset()
    {
        Length = 0;
    }

    private void EnsurePages(int neededPages)
    {
        while (_allocatedPages < neededPages)
        {
            int maxPages = (_maxSeqLen + PageSize - 1) / PageSize;
            if (_allocatedPages >= maxPages) return;

            bool useHost = _vramPageBudget > 0 && _allocatedPages >= _vramPageBudget;

            for (int i = 0; i < _layerIndices.Length; i++)
            {
                int layer = _layerIndices[i];
                long kSize = _nKvHeads * PageSize * _keyLength;
                long vSize = _nKvHeads * PageSize * _valueLength;

                ITensor kPage, vPage;
                if (useHost)
                {
                    kPage = _backend.CreateHostTensor($"paged_k_{layer}_p{_allocatedPages}", _cacheType, [kSize]);
                    vPage = _backend.CreateHostTensor($"paged_v_{layer}_p{_allocatedPages}", _cacheType, [vSize]);
                }
                else
                {
                    kPage = _backend.CreateTensor($"paged_k_{layer}_p{_allocatedPages}", _cacheType, [kSize]);
                    vPage = _backend.CreateTensor($"paged_v_{layer}_p{_allocatedPages}", _cacheType, [vSize]);
                }

                _kPages[i].Add(kPage);
                _vPages[i].Add(vPage);
            }

            _allocatedPages++;

            // Grow scratch to match new page count (all layers)
            GrowAllScratches();
        }
    }

    private void GrowAllScratches()
    {
        int neededCapacity = _allocatedPages * PageSize;
        if (neededCapacity <= _scratchCapacity) return;

        for (int i = 0; i < _layerIndices.Length; i++)
            GrowScratch(i, neededCapacity);

        _scratchCapacity = neededCapacity;
    }

    private void EnsureScratch(int idx)
    {
        int neededCapacity = _allocatedPages * PageSize;
        if (neededCapacity <= _scratchCapacity && _kScratch[idx] != null)
            return;

        // This can happen if pages were just allocated — grow all scratches
        GrowAllScratches();
    }

    private void GrowScratch(int idx, int neededCapacity)
    {
        if (_kScratch[idx] != null && neededCapacity <= _scratchCapacity)
            return;

        int layer = _layerIndices[idx];
        long kSize = _nKvHeads * (long)neededCapacity * _keyLength;
        long vSize = _nKvHeads * (long)neededCapacity * _valueLength;

        var oldK = _kScratch[idx];
        var oldV = _vScratch[idx];

        _kScratch[idx] = _backend.CreateTensor($"scratch_k_{layer}", _cacheType, [kSize]);
        _vScratch[idx] = _backend.CreateTensor($"scratch_v_{layer}", _cacheType, [vSize]);

        if (oldK != null)
        {
            _backend.CopyTensorBytes(_kScratch[idx], oldK, oldK.ByteSize);
            _backend.CopyTensorBytes(_vScratch[idx], oldV, oldV.ByteSize);
            oldK.Dispose();
            oldV.Dispose();
        }
    }

    public void Dispose()
    {
        for (int i = 0; i < _layerIndices.Length; i++)
        {
            foreach (var p in _kPages[i]) p.Dispose();
            foreach (var p in _vPages[i]) p.Dispose();
            _kPages[i].Clear();
            _vPages[i].Clear();
            _kScratch[i]?.Dispose();
            _vScratch[i]?.Dispose();
        }
    }
}
