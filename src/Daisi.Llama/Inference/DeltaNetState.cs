using Daisi.Llama.Gguf;
using Daisi.Llama.Model;

namespace Daisi.Llama.Inference;

/// <summary>
/// Manages recurrent state for DeltaNet layers.
/// Each DeltaNet layer has:
///   - A state matrix per head: [groupCount × headDim × headDim]
///   - A conv1d shift buffer: [(convKernel-1) × qkvDim]
/// </summary>
public sealed class DeltaNetState : IDisposable
{
    private readonly float[][] _states;       // per-layer state matrices
    private readonly float[][] _convBuffers;  // per-layer conv shift buffers
    private readonly int[] _layerIndices;
    private readonly int _groupCount;
    private readonly int _headDim;
    private readonly int _qkvDim;
    private readonly int _convKernel;

    public DeltaNetState(IComputeBackend backend, ModelConfig config)
    {
        _groupCount = config.SsmGroupCount;
        _headDim = config.SsmHeadDim;
        _qkvDim = config.SsmInnerSize * 3; // QKV concatenated
        _convKernel = config.SsmConvKernel;

        var deltaLayers = new List<int>();
        for (int i = 0; i < config.NumLayers; i++)
        {
            if (!config.IsStandardAttention(i))
                deltaLayers.Add(i);
        }

        _layerIndices = deltaLayers.ToArray();
        _states = new float[_layerIndices.Length][];
        _convBuffers = new float[_layerIndices.Length][];

        int stateSize = _groupCount * _headDim * _headDim;
        int convBufSize = (_convKernel - 1) * _qkvDim;

        for (int i = 0; i < _layerIndices.Length; i++)
        {
            _states[i] = new float[stateSize];
            _convBuffers[i] = new float[convBufSize];
        }
    }

    private int GetIndex(int layer)
    {
        for (int i = 0; i < _layerIndices.Length; i++)
            if (_layerIndices[i] == layer) return i;
        throw new ArgumentException($"Layer {layer} is not a DeltaNet layer.");
    }

    /// <summary>
    /// Get the state matrix for a DeltaNet layer.
    /// Layout: [groupCount × headDim × headDim].
    /// For head h: offset = h * headDim * headDim.
    /// </summary>
    public Span<float> GetState(int layer) => _states[GetIndex(layer)];

    /// <summary>
    /// Get the conv1d shift buffer for a DeltaNet layer.
    /// Layout: [(convKernel-1) × qkvDim].
    /// </summary>
    public Span<float> GetConvBuffer(int layer) => _convBuffers[GetIndex(layer)];

    /// <summary>Reset all states and conv buffers to zero.</summary>
    public void Reset()
    {
        foreach (var s in _states) Array.Clear(s);
        foreach (var b in _convBuffers) Array.Clear(b);
    }

    public void Dispose()
    {
        // Managed arrays, no explicit cleanup needed.
    }
}
