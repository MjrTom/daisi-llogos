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
        _forward.ForwardEmbedding(tokenId);

        bool hasOffloaded = _gpuLayers < _totalLayers;

        // Kick off ALL DMA copies immediately (non-blocking, batched on DMA stream)
        if (hasOffloaded)
            for (int i = _gpuLayers; i < _totalLayers; i++)
                _swapper.PrefetchLayer(i, _forward.GetLayerWeights(i));

        // Run VRAM layers (overlaps with DMA of offloaded layers)
        if (_gpuLayers > 0)
            _forward.ForwardLayers(0, _gpuLayers, position, isFinal: !hasOffloaded);

        // Wait for ALL DMA to complete, then run all offloaded layers in one batch
        if (hasOffloaded)
        {
            _swapper.SyncDma();
            _forward.ForwardLayers(_gpuLayers, _totalLayers, position,
                continuation: true, isFinal: true);
            _swapper.RestorePinnedPtrs(_forward.GetAllLayerWeights());
        }
    }

    public void Dispose()
    {
        _swapper.Dispose();
        _forward.Dispose();
    }
}
