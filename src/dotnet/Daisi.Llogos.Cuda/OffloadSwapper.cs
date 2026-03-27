using Daisi.Llogos.Model;

namespace Daisi.Llogos.Cuda;

/// <summary>
/// Double-buffered DMA pipeline for layer offloading.
/// Manages two VRAM staging buffers and async H2D copies from pinned host memory.
///
/// For each offloaded layer, the swapper:
/// 1. Async-copies all tensor data from pinned RAM → VRAM staging buffer (DMA engine)
/// 2. Swaps the layer's tensor DevicePtrs to point at the staging buffer
/// 3. After GPU finishes, swaps pointers back to pinned (ready for next token)
///
/// The DMA engine runs concurrently with GPU compute — while GPU processes VRAM layers,
/// the DMA engine prefetches the next offloaded layer to staging.
/// </summary>
public sealed class OffloadSwapper : IDisposable
{
    private readonly CudaBackend _backend;
    private readonly int _gpuLayers;
    private readonly int _totalLayers;
    private readonly nint _dmaStreamHandle;

    // Per offloaded layer: list of (pinned DevicePtr, byte size) for each tensor
    private readonly LayerTensorMap[] _pinnedMaps;

    // Two VRAM staging buffers for double-buffering
    private CudaDeviceMemory? _stagingA;
    private CudaDeviceMemory? _stagingB;
    private ulong _stagingSize;
    private bool _useBufferA = true;

    public OffloadSwapper(CudaBackend backend, int gpuLayers, int totalLayers)
    {
        _backend = backend;
        _gpuLayers = gpuLayers;
        _totalLayers = totalLayers;
        _pinnedMaps = new LayerTensorMap[totalLayers];

        // Create DMA stream for async copies
        CudaApi.Check(CudaApi.StreamCreate(out _dmaStreamHandle, 0), "cuStreamCreate(dma)");
    }

    /// <summary>
    /// Register an offloaded layer's tensors. Called during model loading.
    /// Records the pinned DevicePtr and size of each tensor for later DMA copy.
    /// </summary>
    public void RegisterLayer(int layerIndex, LayerWeights layer)
    {
        if (layerIndex < _gpuLayers) return; // VRAM layer, no registration needed

        var tensors = new List<TensorRef>();
        CollectTensors(layer, tensors);

        _pinnedMaps[layerIndex] = new LayerTensorMap { Tensors = tensors.ToArray() };

        // Track max layer size for staging buffer (including 256-byte alignment padding)
        ulong layerBytes = 0;
        foreach (var t in tensors)
        {
            layerBytes += t.ByteSize;
            layerBytes = (layerBytes + 255) & ~255UL;  // Match PrefetchLayer's alignment
        }
        if (layerBytes > _stagingSize) _stagingSize = layerBytes;
    }

    /// <summary>
    /// Allocate staging buffers after all layers are registered.
    /// Must be called before first PrefetchLayer.
    /// </summary>
    public void AllocateStaging()
    {
        if (_stagingSize == 0) return;
        _stagingA = new CudaDeviceMemory(_stagingSize);
        _stagingB = new CudaDeviceMemory(_stagingSize);
    }

    /// <summary>
    /// Launch async DMA copy of an offloaded layer's weights from pinned RAM to VRAM staging.
    /// Non-blocking — returns immediately, DMA runs concurrently with GPU compute.
    /// Call SyncDma before GPU processes this layer.
    /// </summary>
    public unsafe void PrefetchLayer(int layerIndex, LayerWeights layer)
    {
        if (layerIndex < _gpuLayers) return;

        var map = _pinnedMaps[layerIndex];
        if (map.Tensors == null) return;

        var staging = _useBufferA ? _stagingA! : _stagingB!;
        ulong offset = 0;

        // Async copy each tensor from pinned → staging, and update DevicePtr
        var tensorList = new List<TensorRef>();
        CollectTensors(layer, tensorList);

        for (int i = 0; i < map.Tensors.Length && i < tensorList.Count; i++)
        {
            var src = map.Tensors[i]; // Original pinned pointer
            var dst = staging.DevicePtr + offset;
            var size = src.ByteSize;

            // Debug: validate pointers
            if (src.HostPtr == 0) throw new InvalidOperationException($"Layer {layerIndex} tensor {i}: HostPtr is null");
            if (dst == 0) throw new InvalidOperationException($"Layer {layerIndex} tensor {i}: staging dst is null");
            if (size == 0) throw new InvalidOperationException($"Layer {layerIndex} tensor {i}: ByteSize is 0");
            if (offset + size > _stagingSize) throw new InvalidOperationException($"Layer {layerIndex} tensor {i}: staging overflow ({offset + size} > {_stagingSize})");

            CudaApi.Check(CudaApi.MemcpyHtoD(dst, (void*)src.HostPtr, size), $"cuMemcpyHtoD(layer{layerIndex}.tensor{i}, host={src.HostPtr:X}, dst={dst:X}, size={size})");

            // Update the tensor's device pointer to point at staging
            tensorList[i].Tensor.SetDevicePtr(dst);

            offset += size;
            // Align to 256 bytes for GPU memory access
            offset = (offset + 255) & ~255UL;
        }

        _useBufferA = !_useBufferA;
    }

    /// <summary>
    /// Wait for all pending DMA copies to complete.
    /// Call before GPU starts processing an offloaded layer.
    /// </summary>
    public void SyncDma()
    {
        CudaApi.Check(CudaApi.StreamSynchronize(_dmaStreamHandle), "cuStreamSynchronize(dma)");
    }

    /// <summary>
    /// Restore all offloaded tensors' DevicePtrs to their pinned memory locations.
    /// Call after each token's forward pass completes.
    /// </summary>
    public void RestorePinnedPtrs(LayerWeights[] layers)
    {
        for (int i = _gpuLayers; i < _totalLayers && i < layers.Length; i++)
        {
            var map = _pinnedMaps[i];
            if (map.Tensors == null) continue;

            var tensorList = new List<TensorRef>();
            CollectTensors(layers[i], tensorList);

            for (int j = 0; j < map.Tensors.Length && j < tensorList.Count; j++)
                tensorList[j].Tensor.SetDevicePtr(map.Tensors[j].PinnedDevicePtr);
        }
    }

    public void Dispose()
    {
        _stagingA?.Dispose();
        _stagingB?.Dispose();
        if (_dmaStreamHandle != 0)
            CudaApi.StreamDestroy(_dmaStreamHandle);
    }

    // ── Internals ────────────────────────────────────────────────────────────

    private static void CollectTensors(LayerWeights layer, List<TensorRef> refs)
    {
        void Add(Daisi.Llogos.ITensor? tensor)
        {
            if (tensor is CudaTensor ct && ct.IsPinned)
                refs.Add(new TensorRef { Tensor = ct, PinnedDevicePtr = ct.DevicePtr, HostPtr = ct.PinnedHostPtr, ByteSize = (ulong)ct.ByteSize });
        }

        Add(layer.AttnNorm);
        Add(layer.PostAttnNorm);
        Add(layer.FfnGate);
        Add(layer.FfnUp);
        Add(layer.FfnDown);

        if (layer is StandardAttentionWeights saw)
        {
            Add(saw.AttnQ); Add(saw.AttnK); Add(saw.AttnV); Add(saw.AttnO);
            Add(saw.AttnQNorm); Add(saw.AttnKNorm);
            Add(saw.FusedGateUp); Add(saw.FusedQKV);
        }
        else if (layer is DeltaNetWeights dn)
        {
            Add(dn.AttnQkv); Add(dn.AttnGate);
            Add(dn.SsmA); Add(dn.SsmAlpha); Add(dn.SsmBeta);
            Add(dn.SsmConv1d); Add(dn.SsmDtBias); Add(dn.SsmNorm); Add(dn.SsmOut);
        }
    }

    private struct TensorRef
    {
        public CudaTensor Tensor;
        public ulong PinnedDevicePtr;  // Original pinned device pointer
        public nint HostPtr;           // Host pointer for DMA source
        public ulong ByteSize;
    }

    private struct LayerTensorMap
    {
        public TensorRef[]? Tensors;
    }
}
