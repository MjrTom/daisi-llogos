using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Tests.Inference;

public class ForwardPassTests
{
    [Fact]
    public void SingleToken_ProducesLogits()
    {
        if (!TestConstants.ModelExists) return;

        using var ctx = LoadModel();
        var logits = ctx.Forward.Forward(tokenId: 0, position: 0);

        Assert.Equal(ctx.Config.VocabSize, logits.Length);
    }

    [Fact]
    public void Logits_NotAllZero()
    {
        if (!TestConstants.ModelExists) return;

        using var ctx = LoadModel();
        var logits = ctx.Forward.Forward(tokenId: 0, position: 0);

        bool hasNonZero = false;
        for (int i = 0; i < logits.Length; i++)
            if (logits[i] != 0) { hasNonZero = true; break; }

        Assert.True(hasNonZero, "All logits are zero.");
    }

    [Fact]
    public void Logits_NoNaNOrInf()
    {
        if (!TestConstants.ModelExists) return;

        using var ctx = LoadModel();
        var logits = ctx.Forward.Forward(tokenId: 0, position: 0);

        for (int i = 0; i < logits.Length; i++)
        {
            Assert.False(float.IsNaN(logits[i]), $"logits[{i}] is NaN");
            Assert.False(float.IsInfinity(logits[i]), $"logits[{i}] is Infinity");
        }
    }

    [Fact]
    public void Deterministic_SameInputSameOutput()
    {
        if (!TestConstants.ModelExists) return;

        using var ctx1 = LoadModel();
        using var ctx2 = LoadModel();

        var logits1 = ctx1.Forward.Forward(tokenId: 100, position: 0).ToArray();
        var logits2 = ctx2.Forward.Forward(tokenId: 100, position: 0).ToArray();

        Assert.Equal(logits1.Length, logits2.Length);
        for (int i = 0; i < logits1.Length; i++)
            Assert.Equal(logits1[i], logits2[i]);
    }

    [Fact]
    public void MultipleTokens_SequentialPositions()
    {
        if (!TestConstants.ModelExists) return;

        using var ctx = LoadModel();

        var tokenizer = TokenizerFactory.FromGguf(ctx.Gguf);
        var ids = tokenizer.Encode("Hello");
        Assert.True(ids.Length > 0);

        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < ids.Length; i++)
            logits = ctx.Forward.Forward(ids[i], i);

        Assert.Equal(ctx.Config.VocabSize, logits.Length);

        int argmax = ArgMax(logits);
        Assert.InRange(argmax, 0, ctx.Config.VocabSize - 1);
    }

    [Fact]
    public void KvCache_LengthTracksPositions()
    {
        if (!TestConstants.ModelExists) return;

        using var ctx = LoadModel();

        Assert.Equal(0, ctx.Forward.KvCache.Length);

        ctx.Forward.Forward(tokenId: 0, position: 0);
        Assert.Equal(1, ctx.Forward.KvCache.Length);

        ctx.Forward.Forward(tokenId: 1, position: 1);
        Assert.Equal(2, ctx.Forward.KvCache.Length);
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

    private static int ArgMax(ReadOnlySpan<float> values)
    {
        int best = 0;
        float bestVal = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] > bestVal) { bestVal = values[i]; best = i; }
        return best;
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
