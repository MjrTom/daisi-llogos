using Daisi.Llogos.Inference;

namespace Daisi.Llogos.Cuda;

/// <summary>
/// IForwardPass wrapper that adds double-buffered DMA prefetching for offloaded layers.
/// VRAM layers run in one graph-captured batch at full HBM bandwidth.
/// Offloaded layers run one-at-a-time with DMA prefetch overlapping GPU compute.
/// Drops into TextGenerator as a transparent replacement for ForwardPass.
/// </summary>
public sealed class OffloadForwardPass : IForwardPass
{
    private readonly ForwardPass _forward;
    private readonly OffloadSwapper _swapper;
    private readonly int _gpuLayers;
    private readonly int _totalLayers;
    private readonly float[] _logitsBuffer;

    public IKvCache KvCache => _forward.KvCache;

    public OffloadForwardPass(ForwardPass forward, OffloadSwapper swapper, int gpuLayers)
    {
        _forward = forward;
        _swapper = swapper;
        _gpuLayers = gpuLayers;
        _totalLayers = forward.NumLayers;
        _logitsBuffer = new float[forward.VocabSize];
    }

    public ReadOnlySpan<float> Forward(int tokenId, int position)
    {
        RunLayers(tokenId, position);
        _forward.ForwardOutputHead(_logitsBuffer);
        return _logitsBuffer;
    }

    public int ForwardArgMax(int tokenId, int position)
    {
        RunLayers(tokenId, position);
        // Delegate to ForwardPass.ForwardArgMax's output head (partial vocab)
        // ForwardArgMax does: RmsNorm + partial LM head + GPU argmax
        return _forward.ForwardArgMaxOutputOnly();
    }

    public void ForwardHidden(int tokenId, int position)
    {
        RunLayers(tokenId, position);
    }

    public void ResetState()
    {
        _forward.ResetState();
    }

    private void RunLayers(int tokenId, int position)
    {
        // 1. Copy ALL offloaded layers to VRAM mirrors upfront
        for (int i = _gpuLayers; i < _totalLayers; i++)
            _swapper.PrefetchLayer(i, _forward.GetLayerWeights(i));
        _swapper.SyncDma();

        // 2. Run ALL layers in one batch — VRAM layers read from device memory,
        // offloaded layers read from VRAM mirrors (SetDevicePtr redirects).
        // Single ForwardLayers call preserves correct residual connections.
        _forward.ForwardEmbedding(tokenId);
        _forward.ForwardLayers(0, _totalLayers, position);

        // 3. Restore pinned pointers for next token
        _swapper.RestorePinnedPtrs(_forward.GetAllLayerWeights());
    }

    public void Dispose()
    {
        _swapper.Dispose();
        _forward.Dispose();
    }
}
