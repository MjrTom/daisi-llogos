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
    /// Register an offloaded layer. Computes total layer size for contiguous allocation.
    /// </summary>
    public void RegisterLayer(int layerIndex, LayerWeights layer)
    {
        if (layerIndex < _gpuLayers) return;

        var tensors = new List<TensorRef>();
        CollectTensors(layer, tensors);
        _pinnedMaps[layerIndex] = new LayerTensorMap { Tensors = tensors.ToArray() };
    }

    /// <summary>
    /// Allocate contiguous VRAM mirrors — ONE allocation per layer for single-DMA copy.
    /// </summary>
    public void AllocateStaging()
    {
        for (int i = _gpuLayers; i < _totalLayers; i++)
        {
            var map = _pinnedMaps[i];
            if (map.Tensors == null) continue;

            // Total layer size with 256-byte alignment per tensor
            ulong totalBytes = 0;
            foreach (var t in map.Tensors)
            {
                totalBytes = (totalBytes + 255) & ~255UL;
                totalBytes += t.ByteSize;
            }

            // ONE contiguous VRAM buffer per layer
            map.VramMirror = new CudaDeviceMemory(totalBytes);
            map.TotalBytes = totalBytes;

            // Compute each tensor's offset within the contiguous buffer
            ulong offset = 0;
            foreach (var t in map.Tensors)
            {
                offset = (offset + 255) & ~255UL;
                t.MirrorOffset = offset;
                offset += t.ByteSize;
            }

            // Also allocate contiguous pinned host buffer and copy tensors into it
            map.PinnedBlob = new CudaPinnedMemory(totalBytes);
            unsafe
            {
                foreach (var t in map.Tensors)
                {
                    // Copy from individual pinned tensor → contiguous pinned blob
                    Buffer.MemoryCopy((void*)t.HostPtr, (void*)(map.PinnedBlob.HostPtr + (nint)t.MirrorOffset),
                        (long)(totalBytes - t.MirrorOffset), (long)t.ByteSize);
                }
            }
        }
    }

    /// <summary>
    /// Single DMA copy per layer: contiguous pinned blob → contiguous VRAM mirror.
    /// Then set each tensor's DevicePtr to its offset within the mirror.
    /// </summary>
    public unsafe void PrefetchLayer(int layerIndex, LayerWeights layer)
    {
        if (layerIndex < _gpuLayers) return;

        var map = _pinnedMaps[layerIndex];
        if (map.Tensors == null || map.VramMirror == null || map.PinnedBlob == null) return;

        // ONE async DMA copy for the entire layer
        CudaApi.MemcpyHtoDAsync(map.VramMirror.DevicePtr, (void*)map.PinnedBlob.HostPtr,
            map.TotalBytes, _dmaStreamHandle);

        // Set each tensor's DevicePtr to its offset within the VRAM mirror
        foreach (var t in map.Tensors)
            t.Tensor.SetDevicePtr(map.VramMirror.DevicePtr + t.MirrorOffset);
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
        foreach (var map in _pinnedMaps)
        {
            if (map == null) continue;
            map.VramMirror?.Dispose();
            map.PinnedBlob?.Dispose();
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
        public ulong MirrorOffset;     // Offset within contiguous VRAM mirror
    }

    private class LayerTensorMap
    {
        public TensorRef[]? Tensors;
        public CudaDeviceMemory? VramMirror;  // ONE contiguous VRAM buffer per layer
        public CudaPinnedMemory? PinnedBlob;  // ONE contiguous pinned buffer per layer
        public ulong TotalBytes;
    }
}
