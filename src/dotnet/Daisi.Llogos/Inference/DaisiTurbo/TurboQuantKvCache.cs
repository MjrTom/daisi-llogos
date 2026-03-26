using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Inference.DaisiTurbo;

/// <summary>
/// KV cache with TurboQuant compression. Stores K and V in compressed form using:
///   1. Walsh-Hadamard rotation (spreads outliers uniformly)
///   2. MSE-optimal scalar quantization (2-4 bits per dimension)
///   3. QJL sign-bit residual correction (unbiased inner product estimation)
///
/// Implements IKvCache so it can be used as a drop-in replacement for KvCache.
/// During Write: rotate → quantize → store compressed + sign bits.
/// During GatedAttention (via GetKCacheTensor): provides decompressed F32 tensors
/// on demand (cached per-layer for the current attention step).
/// </summary>
public sealed class TurboQuantKvCache : IKvCache
{
    private readonly int[] _layerIndices;
    private readonly int _maxSeqLen;
    private readonly int _nKvHeads;
    private readonly int _keyLength;
    private readonly int _valueLength;
    private readonly AttentionStrategy _strategy;
    private readonly TurboQuantConfig _config;

    // Compression primitives (shared across all layers)
    private readonly ScalarQuantizer _quantizer;
    private readonly QjlProjection? _kQjl;
    private readonly QjlProjection? _vQjl;
    private readonly float[] _kSigns;   // WHT sign flips for keys
    private readonly float[] _vSigns;   // WHT sign flips for values

    // Per-layer compressed storage
    private readonly CompressedLayer[] _layers;

    // Per-head scale factors (for denormalization after WHT)
    // We store one scale per head per position: scale = ||original|| before rotation
    private readonly float[][] _kScales; // [layerIdx][nKvHeads × maxSeqLen]
    private readonly float[][] _vScales;

    // Per-head residual norms for QJL correction (stored during compression)
    private readonly float[][]? _kResidualNorms; // [layerIdx][nKvHeads × maxSeqLen]
    private readonly float[][]? _vResidualNorms;

    // Decompression scratch buffers (reused across layers)
    private readonly float[] _rotatedBuf;
    private readonly float[] _reconstructedBuf;
    private readonly float[] _residualBuf;

    // Fused attention scratch: per-head dequantized K and V vectors (one position at a time)
    private readonly float[] _fusedKBuf;  // [keyLength] — one head's K for one position
    private readonly float[] _fusedVBuf;  // [valueLength] — one head's V for one position

    // Lazy fallback F32 cache tensors (only allocated if GetK/VCacheTensor is called)
    private ITensor[]? _kDecompressed;
    private ITensor[]? _vDecompressed;
    private int _lastDecompressedLength = -1;
    private IComputeBackend? _backend;

    public int Length { get; private set; }
    public int MaxSeqLen => _maxSeqLen;
    public int NumKvHeads => _nKvHeads;
    public int KeyLength => _keyLength;
    public int ValueLength => _valueLength;
    public GgmlType CacheType => GgmlType.F32; // Attention sees F32 after decompression
    public AttentionStrategy Strategy => _strategy;
    public TurboQuantConfig TurboConfig => _config;

    public TurboQuantKvCache(IComputeBackend backend, ModelConfig config, int maxSeqLen,
        TurboQuantConfig turboConfig, AttentionStrategy? strategy = null)
    {
        _strategy = strategy ?? AttentionStrategy.Full;
        _config = turboConfig;

        if (_strategy.Mode != AttentionMode.Full && _strategy.CacheCapacity > 0)
            maxSeqLen = Math.Min(maxSeqLen, _strategy.CacheCapacity);

        _maxSeqLen = maxSeqLen;
        _nKvHeads = config.NumKvHeads;
        _keyLength = config.KeyLength;
        _valueLength = config.ValueLength;

        // Build layer index map (only standard attention layers)
        var attnLayers = new List<int>();
        for (int i = 0; i < config.NumLayers; i++)
            if (config.IsStandardAttention(i))
                attnLayers.Add(i);
        _layerIndices = attnLayers.ToArray();

        // Initialize compression primitives
        _quantizer = ScalarQuantizer.Create(turboConfig.QuantBits);

        int kProjDim = turboConfig.QjlProjectionDim ?? (_keyLength / 2);
        int vProjDim = turboConfig.QjlProjectionDim ?? (_valueLength / 2);

        if (kProjDim > 0 && turboConfig.Target != TurboQuantTarget.Values)
            _kQjl = new QjlProjection(_keyLength, kProjDim, turboConfig.Seed + 1000);
        if (vProjDim > 0 && turboConfig.Target != TurboQuantTarget.Keys)
            _vQjl = new QjlProjection(_valueLength, vProjDim, turboConfig.Seed + 2000);

        _kSigns = WalshHadamard.GenerateSigns(_keyLength, turboConfig.Seed + 100);
        _vSigns = WalshHadamard.GenerateSigns(_valueLength, turboConfig.Seed + 200);

        // Allocate compressed storage per layer
        int kPackedPerHead = _quantizer.PackedBytes(_keyLength);
        int vPackedPerHead = _quantizer.PackedBytes(_valueLength);
        int kSignPerHead = _kQjl?.SignBitBytes ?? 0;
        int vSignPerHead = _vQjl?.SignBitBytes ?? 0;

        _layers = new CompressedLayer[_layerIndices.Length];
        _kScales = new float[_layerIndices.Length][];
        _vScales = new float[_layerIndices.Length][];

        if (_kQjl != null)
            _kResidualNorms = new float[_layerIndices.Length][];
        if (_vQjl != null)
            _vResidualNorms = new float[_layerIndices.Length][];

        for (int i = 0; i < _layerIndices.Length; i++)
        {
            _layers[i] = new CompressedLayer(
                _nKvHeads, maxSeqLen, kPackedPerHead, vPackedPerHead, kSignPerHead, vSignPerHead);
            _kScales[i] = new float[_nKvHeads * maxSeqLen];
            _vScales[i] = new float[_nKvHeads * maxSeqLen];
            if (_kResidualNorms != null)
                _kResidualNorms[i] = new float[_nKvHeads * maxSeqLen];
            if (_vResidualNorms != null)
                _vResidualNorms[i] = new float[_nKvHeads * maxSeqLen];
        }

        // Scratch buffers
        int maxDim = Math.Max(_keyLength, _valueLength);
        _rotatedBuf = new float[maxDim];
        _reconstructedBuf = new float[maxDim];
        _residualBuf = new float[maxDim];

        // Fused attention scratch (one K/V vector at a time)
        _fusedKBuf = new float[_keyLength];
        _fusedVBuf = new float[_valueLength];

        // Keep backend ref for lazy fallback tensor allocation
        _backend = backend;
    }

    private int GetCacheIndex(int layer)
    {
        for (int i = 0; i < _layerIndices.Length; i++)
            if (_layerIndices[i] == layer) return i;
        throw new ArgumentException($"Layer {layer} is not a standard attention layer.");
    }

    /// <summary>
    /// Write K/V for a single position. Compresses on the fly:
    /// rotate → quantize → store packed + scale + sign bits.
    /// </summary>
    public void Write(IComputeBackend backend, int layer, int position, ITensor k, ITensor v)
    {
        int slot = _strategy.MapPosition(position);
        int idx = GetCacheIndex(layer);
        var layerStore = _layers[idx];

        var kSpan = k.AsFloatSpan();
        var vSpan = v.AsFloatSpan();

        bool compressKeys = _config.Target != TurboQuantTarget.Values;
        bool compressValues = _config.Target != TurboQuantTarget.Keys;

        for (int h = 0; h < _nKvHeads; h++)
        {
            // ── Compress Key ────────────────────────────────────────
            if (compressKeys)
            {
                CompressHead(kSpan.Slice(h * _keyLength, _keyLength), _keyLength,
                    _kSigns, _kQjl, layerStore.GetKPacked(h, slot), layerStore.GetKSignBits(h, slot),
                    out float kScale, out float kResNorm);
                _kScales[idx][h * _maxSeqLen + slot] = kScale;
                if (_kResidualNorms != null)
                    _kResidualNorms[idx][h * _maxSeqLen + slot] = kResNorm;
            }

            // ── Compress Value ──────────────────────────────────────
            if (compressValues)
            {
                CompressHead(vSpan.Slice(h * _valueLength, _valueLength), _valueLength,
                    _vSigns, _vQjl, layerStore.GetVPacked(h, slot), layerStore.GetVSignBits(h, slot),
                    out float vScale, out float vResNorm);
                _vScales[idx][h * _maxSeqLen + slot] = vScale;
                if (_vResidualNorms != null)
                    _vResidualNorms[idx][h * _maxSeqLen + slot] = vResNorm;
            }
        }

        // Invalidate fallback decompression from this slot onward
        // For appending (slot == Length), the delta will be caught by DecompressIfNeeded
        // For overwrites (sliding window), full invalidation is needed
        if (_kDecompressed != null && slot != Length)
            _lastDecompressedLength = -1;

        Length = _strategy.EffectiveSeqLen(position);
    }

    /// <summary>
    /// Get decompressed K cache tensor for attention (lazy fallback path).
    /// Only used if ComputeAttention returns false. Allocates F32 tensors on first call.
    /// </summary>
    public ITensor GetKCacheTensor(int layer)
    {
        int idx = GetCacheIndex(layer);
        EnsureFallbackTensors();
        DecompressIfNeeded(idx);
        return _kDecompressed![idx];
    }

    /// <summary>
    /// Get decompressed V cache tensor for attention (lazy fallback path).
    /// </summary>
    public ITensor GetVCacheTensor(int layer)
    {
        int idx = GetCacheIndex(layer);
        EnsureFallbackTensors();
        DecompressIfNeeded(idx);
        return _vDecompressed![idx];
    }

    private void EnsureFallbackTensors()
    {
        if (_kDecompressed != null) return;

        _kDecompressed = new ITensor[_layerIndices.Length];
        _vDecompressed = new ITensor[_layerIndices.Length];
        for (int i = 0; i < _layerIndices.Length; i++)
        {
            int layer = _layerIndices[i];
            long kSize = _nKvHeads * _maxSeqLen * _keyLength;
            long vSize = _nKvHeads * _maxSeqLen * _valueLength;
            _kDecompressed[i] = _backend!.CreateTensor($"tq_k_{layer}", GgmlType.F32, [kSize]);
            _vDecompressed[i] = _backend!.CreateTensor($"tq_v_{layer}", GgmlType.F32, [vSize]);
        }
    }

    public void SetLength(int length)
    {
        Length = length;
        _lastDecompressedLength = -1;
    }

    public void Reset()
    {
        Length = 0;
        _lastDecompressedLength = -1;
    }

    public void Dispose()
    {
        if (_kDecompressed != null)
            foreach (var t in _kDecompressed) t.Dispose();
        if (_vDecompressed != null)
            foreach (var t in _vDecompressed) t.Dispose();
    }

    // ── Fused Compressed Attention ─────────────────────────────────────────

    /// <summary>
    /// Compute attention directly from compressed KV data.
    /// Dequantizes each K/V vector inline during the dot product loop —
    /// only one K and one V vector live in memory at a time.
    /// Uses tiled online softmax identical to CpuBackend.GatedAttention.
    /// </summary>
    public bool ComputeAttention(ITensor output, ITensor qAttn, ITensor qGate,
        int layer, int numHeads, int numKvHeads, int keyLength, int valueLength,
        int seqLen, float scale)
    {
        int idx = GetCacheIndex(layer);
        var layerStore = _layers[idx];

        var outSpan = output.AsFloatSpan();
        var qAttnSpan = qAttn.AsFloatSpan();
        var qGateSpan = qGate.AsFloatSpan();

        int headsPerGroup = numHeads / numKvHeads;

        const int tileSize = 256;
        Span<float> tileScores = stackalloc float[tileSize];

        // Per-head scratch for decompressed K and V (reused across positions)
        Span<float> kBuf = _fusedKBuf;
        Span<float> vBuf = _fusedVBuf;

        // Pre-compute QJL query projections once (reused across all positions for a head)
        float[]? qjlProjQuery = null;
        float[]? qjlAbsProjQuery = null;
        float qjlInvScale = 0;
        if (_kQjl != null && _kResidualNorms != null)
        {
            int projDim = _kQjl.ProjectionDim;
            qjlProjQuery = new float[projDim];
            qjlAbsProjQuery = new float[projDim];
        }

        for (int h = 0; h < numHeads; h++)
        {
            int kvHead = h / headsPerGroup;
            int qOff = h * keyLength;
            int outOff = h * valueLength;

            // Pre-compute query projections for QJL (once per head, not per position)
            if (_kQjl != null && qjlProjQuery != null && qjlAbsProjQuery != null)
            {
                _kQjl.PrecomputeQueryProjections(
                    qAttnSpan.Slice(qOff, keyLength),
                    qjlProjQuery, qjlAbsProjQuery, out qjlInvScale);
            }

            // Initialize output to zero
            for (int d = 0; d < valueLength; d++)
                outSpan[outOff + d] = 0;

            float runningMax = float.NegativeInfinity;
            float runningSum = 0;

            // Process tiles
            for (int tileStart = 0; tileStart < seqLen; tileStart += tileSize)
            {
                int tileEnd = Math.Min(tileStart + tileSize, seqLen);
                int tileLen = tileEnd - tileStart;

                // Compute attention scores: dequantize K inline, dot with Q, + QJL correction
                for (int t = 0; t < tileLen; t++)
                {
                    int p = tileStart + t;

                    // Dequantize K for this position (into kBuf)
                    DecompressHead(layerStore.GetKPacked(kvHead, p), keyLength, _kSigns,
                        _kScales[idx][kvHead * _maxSeqLen + p], kBuf);

                    // Dot product: Q · K_reconstructed
                    float dot = 0;
                    for (int d = 0; d < keyLength; d++)
                        dot += qAttnSpan[qOff + d] * kBuf[d];

                    // QJL correction: estimate Q · (K_original - K_reconstructed)
                    // Uses pre-computed query projections for speed
                    if (_kQjl != null && _kResidualNorms != null && qjlAbsProjQuery != null)
                    {
                        float resNorm = _kResidualNorms[idx][kvHead * _maxSeqLen + p];
                        if (resNorm > 1e-10f)
                        {
                            var kSignBits = layerStore.GetKSignBits(kvHead, p);
                            dot += _kQjl.ComputeCorrectionPrecomputed(
                                kSignBits, resNorm, qjlProjQuery!, qjlAbsProjQuery, qjlInvScale);
                        }
                    }

                    tileScores[t] = dot * scale;
                }

                // Tile max
                float tileMax = float.NegativeInfinity;
                for (int t = 0; t < tileLen; t++)
                    if (tileScores[t] > tileMax) tileMax = tileScores[t];

                // Exp and tile sum
                float tileSum = 0;
                for (int t = 0; t < tileLen; t++)
                {
                    tileScores[t] = MathF.Exp(tileScores[t] - tileMax);
                    tileSum += tileScores[t];
                }

                // Online softmax merge
                float newMax = MathF.Max(runningMax, tileMax);
                float correctionOld = MathF.Exp(runningMax - newMax);
                float correctionNew = MathF.Exp(tileMax - newMax);

                // Update output: rescale old + weighted V sum for tile
                for (int d = 0; d < valueLength; d++)
                    outSpan[outOff + d] *= correctionOld;

                for (int t = 0; t < tileLen; t++)
                {
                    int p = tileStart + t;
                    float w = tileScores[t] * correctionNew;

                    // Dequantize V for this position (into vBuf)
                    DecompressHead(layerStore.GetVPacked(kvHead, p), valueLength, _vSigns,
                        _vScales[idx][kvHead * _maxSeqLen + p], vBuf);

                    for (int d = 0; d < valueLength; d++)
                        outSpan[outOff + d] += w * vBuf[d];
                }

                runningSum = runningSum * correctionOld + tileSum * correctionNew;
                runningMax = newMax;
            }

            // Normalize and apply sigmoid gating
            float invSum = runningSum > 0 ? 1.0f / runningSum : 0;
            for (int d = 0; d < valueLength; d++)
            {
                float gateVal = d < keyLength
                    ? 1.0f / (1.0f + MathF.Exp(-qGateSpan[h * keyLength + d]))
                    : 1.0f;
                outSpan[outOff + d] = outSpan[outOff + d] * invSum * gateVal;
            }
        }

        return true;
    }

    // ── Compression ─────────────────────────────────────────────────────────

    private void CompressHead(ReadOnlySpan<float> input, int dim,
        float[] signs, QjlProjection? qjl, Span<byte> packedOut, Span<byte> signBitsOut,
        out float scale, out float residualNorm)
    {
        residualNorm = 0;

        // Compute scale (L2 norm) for denormalization
        float norm = 0;
        for (int d = 0; d < dim; d++)
            norm += input[d] * input[d];
        scale = MathF.Sqrt(norm / dim); // RMS rather than L2 for better numerical behavior

        // Normalize and rotate
        var rotated = _rotatedBuf.AsSpan(0, dim);
        float invScale = scale > 1e-10f ? 1.0f / scale : 0;
        for (int d = 0; d < dim; d++)
            rotated[d] = input[d] * invScale;

        WalshHadamard.ForwardInPlace(rotated, signs.AsSpan(0, dim));

        // Quantize
        var reconstructed = _reconstructedBuf.AsSpan(0, dim);
        _quantizer.QuantizeVector(rotated, packedOut, reconstructed);

        // QJL sign bits on residual (with residual norm for unbiased estimator)
        if (qjl != null && signBitsOut.Length > 0)
        {
            var residual = _residualBuf.AsSpan(0, dim);
            for (int d = 0; d < dim; d++)
                residual[d] = rotated[d] - reconstructed[d];
            qjl.ProjectAndSign(residual, signBitsOut, out residualNorm);
            // Scale residualNorm back to original space
            residualNorm *= scale;
        }
    }

    // ── Decompression (incremental fallback) ────────────────────────────────

    private void DecompressIfNeeded(int idx)
    {
        if (_lastDecompressedLength == Length)
            return;

        // Only decompress positions [_lastDecompressedLength .. Length-1]
        // If _lastDecompressedLength is -1 (reset/invalidated), decompress all
        int startPos = _lastDecompressedLength >= 0 ? _lastDecompressedLength : 0;

        for (int li = 0; li < _layerIndices.Length; li++)
            DecompressLayerRange(li, startPos, Length);

        _lastDecompressedLength = Length;
    }

    private void DecompressLayerRange(int idx, int fromPos, int toPos)
    {
        var layerStore = _layers[idx];
        var kOut = _kDecompressed![idx].AsFloatSpan();
        var vOut = _vDecompressed![idx].AsFloatSpan();

        bool compressKeys = _config.Target != TurboQuantTarget.Values;
        bool compressValues = _config.Target != TurboQuantTarget.Keys;

        for (int h = 0; h < _nKvHeads; h++)
        {
            for (int pos = fromPos; pos < toPos; pos++)
            {
                int kCacheOff = h * _maxSeqLen * _keyLength + pos * _keyLength;
                int vCacheOff = h * _maxSeqLen * _valueLength + pos * _valueLength;

                if (compressKeys)
                {
                    DecompressHead(layerStore.GetKPacked(h, pos), _keyLength, _kSigns,
                        _kScales[idx][h * _maxSeqLen + pos],
                        kOut.Slice(kCacheOff, _keyLength));
                }

                if (compressValues)
                {
                    DecompressHead(layerStore.GetVPacked(h, pos), _valueLength, _vSigns,
                        _vScales[idx][h * _maxSeqLen + pos],
                        vOut.Slice(vCacheOff, _valueLength));
                }
            }
        }
    }

    private void DecompressHead(ReadOnlySpan<byte> packed, int dim, float[] signs,
        float scale, Span<float> output)
    {
        // Dequantize to get reconstructed rotated vector
        _quantizer.DequantizeVector(packed, output);

        // Inverse WHT rotation
        WalshHadamard.InverseInPlace(output, signs.AsSpan(0, dim));

        // Rescale
        for (int d = 0; d < dim; d++)
            output[d] *= scale;
    }

    /// <summary>
    /// Reports memory usage statistics for diagnostics.
    /// </summary>
    public TurboQuantStats GetStats()
    {
        int kPackedPerHead = _quantizer.PackedBytes(_keyLength);
        int vPackedPerHead = _quantizer.PackedBytes(_valueLength);
        int kSignPerHead = _kQjl?.SignBitBytes ?? 0;
        int vSignPerHead = _vQjl?.SignBitBytes ?? 0;

        long compressedPerPos = (long)_nKvHeads * (kPackedPerHead + vPackedPerHead + kSignPerHead + vSignPerHead + 8); // +8 for scales
        long compressedTotal = compressedPerPos * Length * _layerIndices.Length;

        long uncompressedPerPos = (long)_nKvHeads * (_keyLength + _valueLength) * sizeof(float);
        long uncompressedTotal = uncompressedPerPos * Length * _layerIndices.Length;

        return new TurboQuantStats
        {
            CompressedBytes = compressedTotal,
            UncompressedBytes = uncompressedTotal,
            CompressionRatio = uncompressedTotal > 0 ? (float)uncompressedTotal / compressedTotal : 0,
            EffectiveBitsPerDim = _config.EffectiveBitsPerDim(_keyLength),
            QuantBits = _config.QuantBits,
            QjlProjectionDim = _kQjl?.ProjectionDim ?? 0,
            NumLayers = _layerIndices.Length,
            SeqLength = Length,
        };
    }
}

/// <summary>Compressed storage for one attention layer.</summary>
internal sealed class CompressedLayer
{
    private readonly byte[] _kPacked;       // [nKvHeads × maxSeqLen × kPackedPerHead]
    private readonly byte[] _vPacked;       // [nKvHeads × maxSeqLen × vPackedPerHead]
    private readonly byte[] _kSignBits;     // [nKvHeads × maxSeqLen �� kSignPerHead]
    private readonly byte[] _vSignBits;     // [nKvHeads × maxSeqLen × vSignPerHead]

    private readonly int _nKvHeads;
    private readonly int _maxSeqLen;
    private readonly int _kPackedPerHead;
    private readonly int _vPackedPerHead;
    private readonly int _kSignPerHead;
    private readonly int _vSignPerHead;

    public CompressedLayer(int nKvHeads, int maxSeqLen,
        int kPackedPerHead, int vPackedPerHead, int kSignPerHead, int vSignPerHead)
    {
        _nKvHeads = nKvHeads;
        _maxSeqLen = maxSeqLen;
        _kPackedPerHead = kPackedPerHead;
        _vPackedPerHead = vPackedPerHead;
        _kSignPerHead = kSignPerHead;
        _vSignPerHead = vSignPerHead;

        _kPacked = new byte[nKvHeads * maxSeqLen * kPackedPerHead];
        _vPacked = new byte[nKvHeads * maxSeqLen * vPackedPerHead];
        _kSignBits = kSignPerHead > 0 ? new byte[nKvHeads * maxSeqLen * kSignPerHead] : [];
        _vSignBits = vSignPerHead > 0 ? new byte[nKvHeads * maxSeqLen * vSignPerHead] : [];
    }

    public Span<byte> GetKPacked(int head, int pos) =>
        _kPacked.AsSpan((head * _maxSeqLen + pos) * _kPackedPerHead, _kPackedPerHead);

    public Span<byte> GetVPacked(int head, int pos) =>
        _vPacked.AsSpan((head * _maxSeqLen + pos) * _vPackedPerHead, _vPackedPerHead);

    public Span<byte> GetKSignBits(int head, int pos) =>
        _kSignPerHead > 0 ? _kSignBits.AsSpan((head * _maxSeqLen + pos) * _kSignPerHead, _kSignPerHead) : default;

    public Span<byte> GetVSignBits(int head, int pos) =>
        _vSignPerHead > 0 ? _vSignBits.AsSpan((head * _maxSeqLen + pos) * _vSignPerHead, _vSignPerHead) : default;
}

/// <summary>Diagnostics for TurboQuant memory usage.</summary>
public readonly struct TurboQuantStats
{
    public long CompressedBytes { get; init; }
    public long UncompressedBytes { get; init; }
    public float CompressionRatio { get; init; }
    public float EffectiveBitsPerDim { get; init; }
    public int QuantBits { get; init; }
    public int QjlProjectionDim { get; init; }
    public int NumLayers { get; init; }
    public int SeqLength { get; init; }
}
