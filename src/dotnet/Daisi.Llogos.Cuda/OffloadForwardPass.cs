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

        // 1. VRAM layers: one fast graph-captured batch
        // isFinal=false: defer ElementAdd so offloaded layers can continue the residual chain
        bool hasOffloaded = _gpuLayers < _totalLayers;
        if (_gpuLayers > 0)
            _forward.ForwardLayers(0, _gpuLayers, position, isFinal: !hasOffloaded);

        // 2. Offloaded layers: per-layer DMA pipeline with overlap
        if (hasOffloaded)
        {
            _swapper.PrefetchLayer(_gpuLayers, _forward.GetLayerWeights(_gpuLayers));

            for (int i = _gpuLayers; i < _totalLayers; i++)
            {
                if (i + 1 < _totalLayers)
                    _swapper.PrefetchLayer(i + 1, _forward.GetLayerWeights(i + 1));

                _swapper.SyncDma();

                bool isLast = (i == _totalLayers - 1);
                _forward.ForwardLayers(i, i + 1, position,
                    continuation: true, isFinal: isLast);
            }

            _swapper.RestorePinnedPtrs(_forward.GetAllLayerWeights());
        }
    }

    public void Dispose()
    {
        _swapper.Dispose();
        _forward.Dispose();
    }
}
