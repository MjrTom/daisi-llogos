using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Inference.DaisiTurbo;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Cuda;

/// <summary>
/// GPU-side TurboQuant KV cache. Stores compressed K/V in device memory and
/// performs fused compressed attention entirely on GPU — the packed data never
/// leaves HBM except as compressed bytes, delivering bandwidth savings.
/// </summary>
public sealed class CudaTurboQuantKvCache : IKvCache
{
    private readonly int[] _layerIndices;
    private readonly int _maxSeqLen;
    private readonly int _nKvHeads;
    private readonly int _keyLength;
    private readonly int _valueLength;
    private readonly AttentionStrategy _strategy;
    private readonly TurboQuantConfig _config;

    // Per-layer GPU storage
    private readonly CudaDeviceMemory[] _kPacked;   // compressed K bytes
    private readonly CudaDeviceMemory[] _vPacked;   // compressed V bytes
    private readonly CudaDeviceMemory[] _kScales;   // F32 per-head per-position scales
    private readonly CudaDeviceMemory[] _vScales;

    // WHT sign flips on device
    private readonly CudaDeviceMemory _kSignsDev;
    private readonly CudaDeviceMemory _vSignsDev;

    // Packed sizes per head
    private readonly int _kPackedPerHead;
    private readonly int _vPackedPerHead;

    // CUDA module and backend ref for kernel launches
    private readonly CudaBackend _cudaBackend;
    private readonly CudaModule _turboModule;

    public int Length { get; private set; }
    public int MaxSeqLen => _maxSeqLen;
    public int NumKvHeads => _nKvHeads;
    public int KeyLength => _keyLength;
    public int ValueLength => _valueLength;
    public GgmlType CacheType => GgmlType.F32;
    public AttentionStrategy Strategy => _strategy;

    public CudaTurboQuantKvCache(CudaBackend backend, ModelConfig config, int maxSeqLen,
        TurboQuantConfig turboConfig, AttentionStrategy? strategy = null)
    {
        _cudaBackend = backend;
        _strategy = strategy ?? AttentionStrategy.Full;
        _config = turboConfig;

        if (_strategy.Mode != AttentionMode.Full && _strategy.CacheCapacity > 0)
            maxSeqLen = Math.Min(maxSeqLen, _strategy.CacheCapacity);

        _maxSeqLen = maxSeqLen;
        _nKvHeads = config.NumKvHeads;
        _keyLength = config.KeyLength;
        _valueLength = config.ValueLength;

        // Compute packed sizes
        var quantizer = ScalarQuantizer.Create(turboConfig.QuantBits);
        _kPackedPerHead = quantizer.PackedBytes(_keyLength);
        _vPackedPerHead = quantizer.PackedBytes(_valueLength);

        // Build layer index map
        var attnLayers = new List<int>();
        for (int i = 0; i < config.NumLayers; i++)
            if (config.IsStandardAttention(i))
                attnLayers.Add(i);
        _layerIndices = attnLayers.ToArray();

        // Allocate device memory
        _kPacked = new CudaDeviceMemory[_layerIndices.Length];
        _vPacked = new CudaDeviceMemory[_layerIndices.Length];
        _kScales = new CudaDeviceMemory[_layerIndices.Length];
        _vScales = new CudaDeviceMemory[_layerIndices.Length];

        for (int i = 0; i < _layerIndices.Length; i++)
        {
            _kPacked[i] = new CudaDeviceMemory((ulong)(_nKvHeads * maxSeqLen * _kPackedPerHead));
            _vPacked[i] = new CudaDeviceMemory((ulong)(_nKvHeads * maxSeqLen * _vPackedPerHead));
            _kScales[i] = new CudaDeviceMemory((ulong)(_nKvHeads * maxSeqLen * sizeof(float)));
            _vScales[i] = new CudaDeviceMemory((ulong)(_nKvHeads * maxSeqLen * sizeof(float)));
        }

        // Upload WHT signs to device
        var kSigns = WalshHadamard.GenerateSigns(_keyLength, turboConfig.Seed + 100);
        var vSigns = WalshHadamard.GenerateSigns(_valueLength, turboConfig.Seed + 200);

        _kSignsDev = new CudaDeviceMemory((ulong)(_keyLength * sizeof(float)));
        _vSignsDev = new CudaDeviceMemory((ulong)(_valueLength * sizeof(float)));
        _kSignsDev.CopyFromHost(System.Runtime.InteropServices.MemoryMarshal.AsBytes(kSigns.AsSpan()));
        _vSignsDev.CopyFromHost(System.Runtime.InteropServices.MemoryMarshal.AsBytes(vSigns.AsSpan()));

        // Load CUDA module
        var archOpts = new[] { $"--gpu-architecture=compute_{backend.ComputeCapabilityMajor}{backend.ComputeCapabilityMinor}" };
        _turboModule = CudaModule.FromEmbeddedResource("turbo_quant.cu", archOpts);
    }

    private int GetCacheIndex(int layer)
    {
        for (int i = 0; i < _layerIndices.Length; i++)
            if (_layerIndices[i] == layer) return i;
        throw new ArgumentException($"Layer {layer} is not a standard attention layer.");
    }

    /// <summary>
    /// Write K/V: launch GPU kernel to compress on device.
    /// K and V are F32 tensors already on GPU from the forward pass projections.
    /// </summary>
    public unsafe void Write(IComputeBackend backend, int layer, int position, ITensor k, ITensor v)
    {
        int slot = _strategy.MapPosition(position);
        int idx = GetCacheIndex(layer);

        var kT = (CudaTensor)k;
        var vT = (CudaTensor)v;

        ulong kPtr = kT.DevicePtr;
        ulong vPtr = vT.DevicePtr;
        ulong kPackedPtr = _kPacked[idx].DevicePtr;
        ulong vPackedPtr = _vPacked[idx].DevicePtr;
        ulong kScalesPtr = _kScales[idx].DevicePtr;
        ulong vScalesPtr = _vScales[idx].DevicePtr;
        ulong kSignsPtr = _kSignsDev.DevicePtr;
        ulong vSignsPtr = _vSignsDev.DevicePtr;
        int nKvHeads = _nKvHeads;
        int keyLength = _keyLength;
        int valueLength = _valueLength;
        int maxSeqLen = _maxSeqLen;
        int kPackedPerHead = _kPackedPerHead;
        int vPackedPerHead = _vPackedPerHead;
        int quantBits = _config.QuantBits;

        var func = _turboModule.GetFunction("turbo_kv_write");

        nint* kArgs = stackalloc nint[14];
        kArgs[0] = (nint)(&kPtr);
        kArgs[1] = (nint)(&vPtr);
        kArgs[2] = (nint)(&kPackedPtr);
        kArgs[3] = (nint)(&vPackedPtr);
        kArgs[4] = (nint)(&kScalesPtr);
        kArgs[5] = (nint)(&vScalesPtr);
        kArgs[6] = (nint)(&kSignsPtr);
        kArgs[7] = (nint)(&vSignsPtr);
        kArgs[8] = (nint)(&nKvHeads);
        kArgs[9] = (nint)(&keyLength);
        kArgs[10] = (nint)(&valueLength);
        kArgs[11] = (nint)(&maxSeqLen);
        kArgs[12] = (nint)(&slot);
        kArgs[13] = (nint)(&kPackedPerHead);
        // Need to pass vPackedPerHead and quantBits too — adjust arg count
        int vPackedPH = vPackedPerHead;
        int qBits = quantBits;

        // Re-do with all 16 args
        nint* args = stackalloc nint[16];
        args[0] = (nint)(&kPtr);
        args[1] = (nint)(&vPtr);
        args[2] = (nint)(&kPackedPtr);
        args[3] = (nint)(&vPackedPtr);
        args[4] = (nint)(&kScalesPtr);
        args[5] = (nint)(&vScalesPtr);
        args[6] = (nint)(&kSignsPtr);
        args[7] = (nint)(&vSignsPtr);
        args[8] = (nint)(&nKvHeads);
        args[9] = (nint)(&keyLength);
        args[10] = (nint)(&valueLength);
        args[11] = (nint)(&maxSeqLen);
        args[12] = (nint)(&slot);
        args[13] = (nint)(&kPackedPerHead);
        args[14] = (nint)(&vPackedPH);
        args[15] = (nint)(&qBits);

        _cudaBackend.LaunchKernel(func, (uint)nKvHeads, 1, 1, 1, 1, 1, 0, args);

        Length = _strategy.EffectiveSeqLen(position);
    }

    /// <summary>
    /// Compute attention from compressed KV directly on GPU.
    /// Launches the fused turbo_gated_attention kernel.
    /// </summary>
    public unsafe bool ComputeAttention(ITensor output, ITensor qAttn, ITensor qGate,
        int layer, int numHeads, int numKvHeads, int keyLength, int valueLength,
        int seqLen, float attnScale)
    {
        int idx = GetCacheIndex(layer);

        var outT = (CudaTensor)output;
        var qaT = (CudaTensor)qAttn;
        var qgT = (CudaTensor)qGate;

        ulong outPtr = outT.DevicePtr;
        ulong qaPtr = qaT.DevicePtr;
        ulong qgPtr = qgT.DevicePtr;
        ulong kPackedPtr = _kPacked[idx].DevicePtr;
        ulong vPackedPtr = _vPacked[idx].DevicePtr;
        ulong kScalesPtr = _kScales[idx].DevicePtr;
        ulong vScalesPtr = _vScales[idx].DevicePtr;
        ulong kSignsPtr = _kSignsDev.DevicePtr;
        ulong vSignsPtr = _vSignsDev.DevicePtr;
        int maxSeqLen = _maxSeqLen;
        int kPackedPerHead = _kPackedPerHead;
        int vPackedPerHead = _vPackedPerHead;
        int quantBits = _config.QuantBits;

        var func = _turboModule.GetFunction("turbo_gated_attention");

        const int tileSize = 256;
        const int blockSize = 256;
        // Shared: [256 scores] + [256 temp] + [128 qRot broadcast] + [128 vShared]
        uint sharedMem = (uint)((tileSize + blockSize + 128 + 128) * sizeof(float));

        nint* args = stackalloc nint[19];
        args[0] = (nint)(&outPtr);
        args[1] = (nint)(&qaPtr);
        args[2] = (nint)(&qgPtr);
        args[3] = (nint)(&kPackedPtr);
        args[4] = (nint)(&vPackedPtr);
        args[5] = (nint)(&kScalesPtr);
        args[6] = (nint)(&vScalesPtr);
        args[7] = (nint)(&kSignsPtr);
        args[8] = (nint)(&vSignsPtr);
        args[9] = (nint)(&numHeads);
        args[10] = (nint)(&numKvHeads);
        args[11] = (nint)(&keyLength);
        args[12] = (nint)(&valueLength);
        args[13] = (nint)(&maxSeqLen);
        args[14] = (nint)(&seqLen);
        args[15] = (nint)(&attnScale);
        args[16] = (nint)(&kPackedPerHead);
        args[17] = (nint)(&vPackedPerHead);
        args[18] = (nint)(&quantBits);

        _cudaBackend.LaunchKernel(func, (uint)numHeads, 1, 1,
            (uint)blockSize, 1, 1, sharedMem, args);

        return true;
    }

    // GetK/VCacheTensor not supported — always use fused attention
    public ITensor GetKCacheTensor(int layer) =>
        throw new NotSupportedException("CudaTurboQuantKvCache uses fused attention via ComputeAttention. GetKCacheTensor is not supported.");
    public ITensor GetVCacheTensor(int layer) =>
        throw new NotSupportedException("CudaTurboQuantKvCache uses fused attention via ComputeAttention. GetVCacheTensor is not supported.");

    public void SetLength(int length) => Length = length;
    public void Reset() => Length = 0;

    public void Dispose()
    {
        foreach (var m in _kPacked) m.Dispose();
        foreach (var m in _vPacked) m.Dispose();
        foreach (var m in _kScales) m.Dispose();
        foreach (var m in _vScales) m.Dispose();
        _kSignsDev.Dispose();
        _vSignsDev.Dispose();
    }

    public TurboQuantStats GetStats()
    {
        long compressedPerPos = (long)_nKvHeads * (_kPackedPerHead + _vPackedPerHead + 8);
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
            QjlProjectionDim = 0,
            NumLayers = _layerIndices.Length,
            SeqLength = Length,
        };
    }
}
