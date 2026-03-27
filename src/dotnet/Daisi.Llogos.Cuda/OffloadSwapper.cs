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

    // Per offloaded layer: list of tensors with pinned data + VRAM mirror
    private readonly LayerTensorMap[] _pinnedMaps;

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
    /// Allocates a per-tensor VRAM mirror for each pinned tensor.
    /// </summary>
    public void RegisterLayer(int layerIndex, LayerWeights layer)
    {
        if (layerIndex < _gpuLayers) return;

        var tensors = new List<TensorRef>();
        CollectTensors(layer, tensors);

        // Allocate a VRAM mirror for each pinned tensor
        foreach (var t in tensors)
            t.VramMirror = new CudaDeviceMemory(t.ByteSize);

        _pinnedMaps[layerIndex] = new LayerTensorMap { Tensors = tensors.ToArray() };
    }

    /// <summary>No separate staging allocation needed — mirrors are per-tensor.</summary>
    public void AllocateStaging() { }

    /// <summary>
    /// Copy an offloaded layer's weights from pinned RAM to per-tensor VRAM mirrors.
    /// Uses async DMA on the copy stream for overlap with GPU compute.
    /// </summary>
    public unsafe void PrefetchLayer(int layerIndex, LayerWeights layer)
    {
        if (layerIndex < _gpuLayers) return;

        var map = _pinnedMaps[layerIndex];
        if (map.Tensors == null) return;

        for (int i = 0; i < map.Tensors.Length; i++)
        {
            var t = map.Tensors[i];
            // Async DMA: pinned host → VRAM mirror
            CudaApi.MemcpyHtoDAsync(t.VramMirror!.DevicePtr, (void*)t.HostPtr, t.ByteSize, _dmaStreamHandle);
            // Point tensor at VRAM mirror
            t.Tensor.SetDevicePtr(t.VramMirror.DevicePtr);
        }
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
        // Dispose per-tensor VRAM mirrors
        foreach (var map in _pinnedMaps)
        {
            if (map.Tensors == null) continue;
            foreach (var t in map.Tensors)
                t.VramMirror?.Dispose();
        }
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

    private class TensorRef
    {
        public CudaTensor Tensor = null!;
        public ulong PinnedDevicePtr;  // Original pinned device pointer
        public nint HostPtr;           // Host pointer for DMA source
        public ulong ByteSize;
        public CudaDeviceMemory? VramMirror;  // Per-tensor VRAM staging buffer
    }

    private struct LayerTensorMap
    {
        public TensorRef[]? Tensors;
    }
}
