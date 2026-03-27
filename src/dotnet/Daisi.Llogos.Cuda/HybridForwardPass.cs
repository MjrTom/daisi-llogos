using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Cuda;

/// <summary>
/// Hybrid GPU+CPU forward pass: first N layers on CUDA (VRAM @ 960 GB/s),
/// remaining layers on CPU (DDR5 @ 80 GB/s). Zero PCIe weight transfers —
/// each backend reads from its native memory. Only the 20KB hidden state
/// crosses between them via GetHidden/SetHidden.
///
/// Uses DaisiChain's ForwardLayers/GetHidden/SetHidden — no modifications
/// to ForwardPass needed.
/// </summary>
public sealed class HybridForwardPass : IForwardPass
{
    private readonly ForwardPass _gpuForward;
    private readonly ForwardPass _cpuForward;
    private readonly int _gpuLayers;
    private readonly int _totalLayers;
    private readonly int _hiddenDim;
    private readonly float[] _hiddenTransfer;
    private readonly float[] _residualTransfer;

    public IKvCache KvCache => _gpuForward.KvCache;

    public HybridForwardPass(ForwardPass gpuForward, ForwardPass cpuForward,
        int gpuLayers, int totalLayers, int hiddenDim)
    {
        _gpuForward = gpuForward;
        _cpuForward = cpuForward;
        _gpuLayers = gpuLayers;
        _totalLayers = totalLayers;
        _hiddenDim = hiddenDim;
        _hiddenTransfer = new float[hiddenDim];
        _residualTransfer = new float[hiddenDim];
    }

    public ReadOnlySpan<float> Forward(int tokenId, int position)
    {
        RunLayers(tokenId, position);
        // Transfer final hidden back to GPU for output head (LM head is huge, needs GPU)
        _cpuForward.GetHidden(_hiddenTransfer);
        _gpuForward.SetHidden(_hiddenTransfer);
        var logits = new float[_gpuForward.VocabSize];
        _gpuForward.ForwardOutputHead(logits);
        return logits;
    }

    public void ForwardHidden(int tokenId, int position)
    {
        RunLayers(tokenId, position);
    }

    public void ResetState()
    {
        _gpuForward.ResetState();
        _cpuForward.ResetState();
    }

    private void RunLayers(int tokenId, int position)
    {
        // 1. GPU: embedding + first N layers (VRAM speed)
        _gpuForward.ForwardEmbedding(tokenId);
        if (_gpuLayers > 0)
            _gpuForward.ForwardLayers(0, _gpuLayers, position, isFinal: false);

        // 2. Transfer hidden + residual state: GPU → CPU (40KB total, negligible)
        _gpuForward.GetHidden(_hiddenTransfer);
        _gpuForward.GetResidual(_residualTransfer);
        _cpuForward.SetHidden(_hiddenTransfer);
        _cpuForward.SetResidual(_residualTransfer);

        // 3. CPU: remaining layers (DDR5 speed)
        _cpuForward.ForwardLayers(_gpuLayers, _totalLayers, position, continuation: true, isFinal: true);
    }

    public void Dispose()
    {
        _gpuForward.Dispose();
        _cpuForward.Dispose();
    }

    /// <summary>
    /// Create a hybrid GPU+CPU forward pass for a model.
    /// Loads first gpuLayers to CUDA, remaining to CPU.
    /// </summary>
    public static HybridForwardPass Create(
        GgufFile gguf, string modelPath, ModelConfig config,
        CudaBackend cudaBackend, int gpuLayers,
        AttentionStrategy? strategy = null)
    {
        int totalLayers = config.NumLayers;
        int cpuLayers = totalLayers - gpuLayers;

        Console.Error.WriteLine($"  Hybrid: {gpuLayers} GPU + {cpuLayers} CPU layers");

        // GPU: embedding + first gpuLayers + output head (LM head needs GPU speed)
        var gpuWeights = MmapModelLoader.LoadPartial(gguf, modelPath, cudaBackend, config,
            startLayer: 0, endLayer: gpuLayers,
            includeEmbedding: true, includeOutputHead: true);

        var gpuStrategy = strategy ?? AttentionStrategy.Full;
        int maxContext = gpuStrategy.Mode != AttentionMode.Full && gpuStrategy.CacheCapacity > 0
            ? gpuStrategy.CacheCapacity : 2048;

        var gpuKvCache = new KvCache(cudaBackend, config, maxSeqLen: maxContext, strategy: gpuStrategy);
        var gpuDeltaState = new DeltaNetState(cudaBackend, config, gpuWeights);
        var gpuForward = new ForwardPass(cudaBackend, config, gpuWeights, gpuKvCache, gpuDeltaState);

        // CPU: remaining layers + output head + embedding (needed for tied weights fallback)
        var cpuBackend = new CpuBackend();
        var cpuWeights = MmapModelLoader.LoadPartial(gguf, modelPath, cpuBackend, config,
            startLayer: gpuLayers, endLayer: totalLayers,
            includeEmbedding: true, includeOutputHead: true);

        var cpuKvCache = new KvCache(cpuBackend, config, maxSeqLen: maxContext, strategy: gpuStrategy);
        var cpuDeltaState = new DeltaNetState(cpuBackend, config, cpuWeights);
        var cpuForward = new ForwardPass(cpuBackend, config, cpuWeights, cpuKvCache, cpuDeltaState);

        return new HybridForwardPass(gpuForward, cpuForward, gpuLayers, totalLayers, config.HiddenDim);
    }
}
