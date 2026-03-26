using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Inference;

/// <summary>
/// Manages recurrent state for DeltaNet layers.
/// Each DeltaNet layer has:
///   - A state matrix per head: [groupCount × headDim × headDim]
///   - A conv1d shift buffer: [(convKernel-1) × qkvDim]
/// State is stored as ITensor so it can live on device (GPU) memory.
/// </summary>
public sealed class DeltaNetState : IDisposable
{
    private readonly IComputeBackend _backend;
    private readonly ITensor[] _states;
    private readonly ITensor[] _convBuffers;
    private readonly int[] _layerIndices;
    private readonly int _groupCount;
    private readonly int _headDim;
    private readonly int _qkvDim;
    private readonly int _convKernel;

    /// <param name="startLayer">DaisiChain: only allocate for layers >= startLayer. Default 0.</param>
    /// <param name="endLayer">DaisiChain: only allocate for layers &lt; endLayer. Default NumLayers.</param>
    public DeltaNetState(IComputeBackend backend, ModelConfig config, ModelWeights? weights = null,
        int startLayer = 0, int endLayer = -1)
    {
        _backend = backend;
        _groupCount = config.SsmGroupCount;
        _headDim = config.SsmHeadDim;
        _convKernel = config.SsmConvKernel;
        if (endLayer < 0) endLayer = config.NumLayers;

        // Derive actual QKV output dim from weight tensors if available.
        // SsmInnerSize * 3 is wrong when Q/K/V have unequal sizes (e.g. 9B: 2048+2048+4096=8192 vs 3*2048=6144).
        _qkvDim = config.SsmInnerSize * 3; // default fallback
        if (weights != null)
        {
            for (int i = startLayer; i < endLayer; i++)
            {
                if (!config.IsStandardAttention(i) && weights.Layers[i] is DeltaNetWeights dw)
                {
                    _qkvDim = (int)dw.AttnQkv.Dimensions[1]; // actual QKV output dimension
                    break;
                }
            }
        }

        var deltaLayers = new List<int>();
        for (int i = startLayer; i < endLayer; i++)
        {
            if (!config.IsStandardAttention(i))
                deltaLayers.Add(i);
        }

        _layerIndices = deltaLayers.ToArray();
        _states = new ITensor[_layerIndices.Length];
        _convBuffers = new ITensor[_layerIndices.Length];

        if (_layerIndices.Length > 0)
        {
            // Derive actual head count from weights when available (metadata SsmGroupCount may differ).
            int actualGroupCount = _groupCount;
            int actualHeadDim = _headDim;
            if (weights != null)
            {
                for (int i = startLayer; i < endLayer; i++)
                {
                    if (!config.IsStandardAttention(i) && weights.Layers[i] is DeltaNetWeights dw)
                    {
                        actualGroupCount = (int)dw.SsmAlpha.Dimensions[1]; // num_v_heads
                        actualHeadDim = config.SsmStateSize > 0 ? config.SsmStateSize : config.SsmHeadDim;
                        break;
                    }
                }
            }
            int stateSize = Math.Max(_groupCount * _headDim * _headDim,
                                     actualGroupCount * actualHeadDim * actualHeadDim);
            int convBufSize = (_convKernel - 1) * _qkvDim;

            for (int i = 0; i < _layerIndices.Length; i++)
            {
                int layer = _layerIndices[i];
                _states[i] = backend.CreateTensor($"delta_state_{layer}", GgmlType.F32, [stateSize]);
                _convBuffers[i] = backend.CreateTensor($"delta_conv_{layer}", GgmlType.F32, [convBufSize]);
            }
        }
    }

    private int GetIndex(int layer)
    {
        for (int i = 0; i < _layerIndices.Length; i++)
            if (_layerIndices[i] == layer) return i;
        throw new ArgumentException($"Layer {layer} is not a DeltaNet layer.");
    }

    /// <summary>Get the state tensor for a DeltaNet layer.</summary>
    public ITensor GetStateTensor(int layer) => _states[GetIndex(layer)];

    /// <summary>Get the conv1d shift buffer tensor for a DeltaNet layer.</summary>
    public ITensor GetConvBufferTensor(int layer) => _convBuffers[GetIndex(layer)];

    /// <summary>Get the state matrix as a span (CPU only).</summary>
    public Span<float> GetState(int layer) => _states[GetIndex(layer)].AsFloatSpan();

    /// <summary>Get the conv1d shift buffer as a span (CPU only).</summary>
    public Span<float> GetConvBuffer(int layer) => _convBuffers[GetIndex(layer)].AsFloatSpan();

    public void Reset()
    {
        foreach (var s in _states) _backend.ZeroTensor(s);
        foreach (var b in _convBuffers) _backend.ZeroTensor(b);
    }

    public void Dispose()
    {
        foreach (var s in _states) s.Dispose();
        foreach (var b in _convBuffers) b.Dispose();
    }
}
