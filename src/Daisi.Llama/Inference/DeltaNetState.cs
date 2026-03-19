using Daisi.Llama.Gguf;
using Daisi.Llama.Model;

namespace Daisi.Llama.Inference;

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

    public DeltaNetState(IComputeBackend backend, ModelConfig config)
    {
        _backend = backend;
        _groupCount = config.SsmGroupCount;
        _headDim = config.SsmHeadDim;
        _qkvDim = config.SsmInnerSize * 3;
        _convKernel = config.SsmConvKernel;

        var deltaLayers = new List<int>();
        for (int i = 0; i < config.NumLayers; i++)
        {
            if (!config.IsStandardAttention(i))
                deltaLayers.Add(i);
        }

        _layerIndices = deltaLayers.ToArray();
        _states = new ITensor[_layerIndices.Length];
        _convBuffers = new ITensor[_layerIndices.Length];

        int stateSize = _groupCount * _headDim * _headDim;
        int convBufSize = (_convKernel - 1) * _qkvDim;

        for (int i = 0; i < _layerIndices.Length; i++)
        {
            int layer = _layerIndices[i];
            _states[i] = backend.CreateTensor($"delta_state_{layer}", GgmlType.F32, [stateSize]);
            _convBuffers[i] = backend.CreateTensor($"delta_conv_{layer}", GgmlType.F32, [convBufSize]);
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
