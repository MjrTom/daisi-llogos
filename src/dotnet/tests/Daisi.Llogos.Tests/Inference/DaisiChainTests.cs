using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Tests.Inference;

/// <summary>
/// Tests for DaisiChain pipeline parallelism — verifies that splitting the forward pass
/// across layer ranges produces identical results to running the full forward pass.
/// </summary>
public class DaisiChainTests
{
    [Fact]
    public void SplitForward_MatchesFullForward()
    {
        if (!TestConstants.ModelExists) return;

        // Run full forward pass
        using var fullCtx = LoadModel();
        var fullLogits = fullCtx.Forward.Forward(tokenId: 42, position: 0).ToArray();

        // Run split forward pass: embedding → first half layers → second half layers → output head
        using var splitCtx = LoadModel();
        int numLayers = splitCtx.Config.NumLayers;
        int midLayer = numLayers / 2;

        splitCtx.Forward.ForwardEmbedding(tokenId: 42);
        splitCtx.Forward.ForwardLayers(0, midLayer, position: 0);
        splitCtx.Forward.ForwardLayers(midLayer, numLayers, position: 0);

        var splitLogits = new float[splitCtx.Config.VocabSize];
        splitCtx.Forward.ForwardOutputHead(splitLogits);

        // Results must match exactly (same compute backend, deterministic)
        Assert.Equal(fullLogits.Length, splitLogits.Length);
        for (int i = 0; i < fullLogits.Length; i++)
            Assert.Equal(fullLogits[i], splitLogits[i], precision: 5);
    }

    [Fact]
    public void GetSetHidden_RoundTrip()
    {
        if (!TestConstants.ModelExists) return;

        using var ctx = LoadModel();
        int hiddenDim = ctx.Config.HiddenDim;

        // Run embedding to populate hidden state
        ctx.Forward.ForwardEmbedding(tokenId: 100);

        // Extract hidden state
        var hidden = new float[hiddenDim];
        ctx.Forward.GetHidden(hidden);

        // Verify not all zeros
        bool hasNonZero = false;
        for (int i = 0; i < hidden.Length; i++)
            if (hidden[i] != 0) { hasNonZero = true; break; }
        Assert.True(hasNonZero, "Hidden state is all zeros after embedding.");

        // Inject into a fresh context and verify round-trip
        using var ctx2 = LoadModel();
        ctx2.Forward.SetHidden(hidden);

        var roundTripped = new float[hiddenDim];
        ctx2.Forward.GetHidden(roundTripped);

        for (int i = 0; i < hiddenDim; i++)
            Assert.Equal(hidden[i], roundTripped[i]);
    }

    [Fact]
    public void SimulatedTwoHostPipeline_MatchesFullForward()
    {
        if (!TestConstants.ModelExists) return;

        // Simulate two hosts: host1 does embedding + layers 0..mid, host2 does layers mid..end + output head
        using var fullCtx = LoadModel();
        var fullLogits = fullCtx.Forward.Forward(tokenId: 7, position: 0).ToArray();

        // Host 1: embedding + first half
        using var host1 = LoadModel();
        int numLayers = host1.Config.NumLayers;
        int mid = numLayers / 2;

        host1.Forward.ForwardEmbedding(tokenId: 7);
        host1.Forward.ForwardLayers(0, mid, position: 0);

        // Transfer hidden state (simulates network transfer)
        var activation = new float[host1.Config.HiddenDim];
        host1.Forward.GetHidden(activation);

        // Host 2: second half + output head
        using var host2 = LoadModel();
        host2.Forward.SetHidden(activation);
        host2.Forward.ForwardLayers(mid, numLayers, position: 0);

        var pipelineLogits = new float[host2.Config.VocabSize];
        host2.Forward.ForwardOutputHead(pipelineLogits);

        // Must match full forward pass
        Assert.Equal(fullLogits.Length, pipelineLogits.Length);
        for (int i = 0; i < fullLogits.Length; i++)
            Assert.Equal(fullLogits[i], pipelineLogits[i], precision: 5);
    }

    [Fact]
    public void ForwardLayers_UpdatesKvCache()
    {
        if (!TestConstants.ModelExists) return;

        using var ctx = LoadModel();

        Assert.Equal(0, ctx.Forward.KvCache.Length);

        ctx.Forward.ForwardEmbedding(tokenId: 0);
        ctx.Forward.ForwardLayers(0, ctx.Config.NumLayers, position: 0);

        Assert.Equal(1, ctx.Forward.KvCache.Length);
    }

    [Fact]
    public void HiddenDim_VocabSize_Exposed()
    {
        if (!TestConstants.ModelExists) return;

        using var ctx = LoadModel();

        Assert.Equal(ctx.Config.HiddenDim, ctx.Forward.HiddenDim);
        Assert.Equal(ctx.Config.VocabSize, ctx.Forward.VocabSize);
        Assert.Equal(ctx.Config.NumLayers, ctx.Forward.NumLayers);
    }

    [Fact]
    public void LoadPartial_OnlyLoadsAssignedLayers()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        using var backend = new CpuBackend();

        // Load only layers 0-12 with embedding (first stage)
        int mid = config.NumLayers / 2;
        using var weights = MmapModelLoader.LoadPartial(gguf, TestConstants.Qwen35_08B_Q8_0,
            backend, config, 0, mid, includeEmbedding: true, includeOutputHead: false);

        Assert.Equal(config.NumLayers, weights.Layers.Length);

        // First layer should have real weights (check dimension > 1)
        Assert.True(weights.Layers[0].AttnNorm.ElementCount > 1,
            "First layer should have real weights.");

        // Layer at mid should be a placeholder (1 element)
        Assert.Equal(1, weights.Layers[mid].AttnNorm.ElementCount);

        // Embedding should be real
        Assert.True(weights.TokenEmbedding.ElementCount > 1,
            "Embedding should be loaded for first stage.");
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static ModelContext LoadModel()
    {
        var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        var backend = new CpuBackend();
        var weights = ModelLoader.Load(gguf, stream, backend, config);
        var kvCache = new KvCache(backend, config, maxSeqLen: 128);
        var deltaState = new DeltaNetState(backend, config);
        var forward = new ForwardPass(backend, config, weights, kvCache, deltaState);
        return new ModelContext(stream, gguf, config, backend, weights, kvCache, deltaState, forward);
    }

    private sealed class ModelContext : IDisposable
    {
        public Stream Stream { get; }
        public GgufFile Gguf { get; }
        public ModelConfig Config { get; }
        public CpuBackend Backend { get; }
        public ModelWeights Weights { get; }
        public IKvCache KvCache { get; }
        public DeltaNetState DeltaState { get; }
        public ForwardPass Forward { get; }

        public ModelContext(Stream stream, GgufFile gguf, ModelConfig config,
            CpuBackend backend, ModelWeights weights, IKvCache kvCache,
            DeltaNetState deltaState, ForwardPass forward)
        {
            Stream = stream; Gguf = gguf; Config = config;
            Backend = backend; Weights = weights; KvCache = kvCache;
            DeltaState = deltaState; Forward = forward;
        }

        public void Dispose()
        {
            Forward.Dispose();
            DeltaState.Dispose();
            KvCache.Dispose();
            Weights.Dispose();
            Backend.Dispose();
            Stream.Dispose();
        }
    }
}
