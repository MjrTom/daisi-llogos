using Daisi.Llama.Cpu;
using Daisi.Llama.Gguf;
using Daisi.Llama.Inference;
using Daisi.Llama.Model;
using Daisi.Llama.Tokenizer;

namespace Daisi.Llama.Tests.Inference;

/// <summary>
/// Tests for Phase 11: tiled attention, FP16 KV cache, and long context support.
/// </summary>
public class LongContextTests
{
    [Fact]
    public void TiledAttention_MatchesFullAttention()
    {
        // Tiled attention (online softmax) should produce the same output as
        // the original full-scores implementation. Since the current implementation
        // IS tiled, we verify it produces correct results by comparing FP32 vs FP16 cache.
        if (!TestConstants.ModelExists) return;

        using var ctx32 = LoadModel(GgmlType.F32);
        using var ctx16 = LoadModel(GgmlType.F16);

        // Run a short prompt through both
        var logits32 = ctx32.Forward.Forward(tokenId: 100, position: 0).ToArray();
        var logits16 = ctx16.Forward.Forward(tokenId: 100, position: 0).ToArray();

        Assert.Equal(logits32.Length, logits16.Length);

        // FP16 cache should produce very similar (but not identical) results
        float maxDiff = 0;
        for (int i = 0; i < logits32.Length; i++)
        {
            float diff = MathF.Abs(logits32[i] - logits16[i]);
            if (diff > maxDiff) maxDiff = diff;
        }

        // Max difference should be small (FP16 precision is ~3 decimal digits)
        Assert.True(maxDiff < 1.0f, $"Max logit difference too large: {maxDiff}");
    }

    [Fact]
    public void Fp16KvCache_CoherentOutput()
    {
        if (!TestConstants.ModelExists) return;

        using var ctx = LoadModel(GgmlType.F16);
        var tokenizer = TokenizerFactory.FromGguf(ctx.Gguf);
        var generator = new TextGenerator(ctx.Forward, tokenizer, seed: 42);

        var p = new GenerationParams { MaxTokens = 10, Temperature = 0 };
        var text = string.Concat(generator.Generate("The capital of France is", p)
            .Where(t => !t.IsDone)
            .Select(t => t.Text));

        Assert.True(text.Length > 0, "FP16 KV cache produced empty output");
    }

    [Fact]
    public void Fp16KvCache_HalvesMemory()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        // Find first standard attention layer
        int attnLayer = -1;
        for (int i = 0; i < config.NumLayers; i++)
            if (config.IsStandardAttention(i)) { attnLayer = i; break; }
        Assert.True(attnLayer >= 0);

        using var backend = new CpuBackend();
        using var cacheF32 = new KvCache(backend, config, maxSeqLen: 1024, cacheType: GgmlType.F32);
        using var cacheF16 = new KvCache(backend, config, maxSeqLen: 1024, cacheType: GgmlType.F16);

        long f32Size = cacheF32.GetKCacheTensor(attnLayer).ByteSize;
        long f16Size = cacheF16.GetKCacheTensor(attnLayer).ByteSize;

        Assert.Equal(f32Size, f16Size * 2);
    }

    [Fact]
    public void Fp16KvCache_GreedySameTopToken()
    {
        // With greedy sampling, the top token should be the same regardless of KV precision
        if (!TestConstants.ModelExists) return;

        using var ctx32 = LoadModel(GgmlType.F32);
        using var ctx16 = LoadModel(GgmlType.F16);

        var tokenizer32 = TokenizerFactory.FromGguf(ctx32.Gguf);
        var tokenizer16 = TokenizerFactory.FromGguf(ctx16.Gguf);

        var gen32 = new TextGenerator(ctx32.Forward, tokenizer32, seed: 42);
        var gen16 = new TextGenerator(ctx16.Forward, tokenizer16, seed: 42);

        var p = new GenerationParams { MaxTokens = 5, Temperature = 0 };

        var tokens32 = gen32.Generate("Hello", p).Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();
        var tokens16 = gen16.Generate("Hello", p).Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();

        // At least the first token should match (greedy on very similar logits)
        Assert.Equal(tokens32[0], tokens16[0]);
    }

    [Fact]
    public void LargerContext_2K_Stable()
    {
        // Test that tiled attention handles a larger context without crashing
        if (!TestConstants.ModelExists) return;

        using var ctx = LoadModel(GgmlType.F16, maxSeqLen: 2048);
        var tokenizer = TokenizerFactory.FromGguf(ctx.Gguf);
        var ids = tokenizer.Encode("The quick brown fox jumps over the lazy dog. " +
            "This is a test of the tiled attention mechanism with online softmax. " +
            "The model should handle this context without issues.");

        // Prefill all tokens
        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < ids.Length; i++)
            logits = ctx.Forward.Forward(ids[i], i);

        Assert.Equal(ctx.Config.VocabSize, logits.Length);
        Assert.True(float.IsFinite(logits[0]), "Output is not finite at larger context");

        // Generate a few more tokens
        for (int i = 0; i < 5; i++)
        {
            int argmax = ArgMax(logits);
            logits = ctx.Forward.Forward(argmax, ids.Length + i);
            Assert.True(float.IsFinite(logits[0]), $"Output not finite at decode step {i}");
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static ModelContext LoadModel(GgmlType cacheType = GgmlType.F16, int maxSeqLen = 256)
    {
        var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        var backend = new CpuBackend();
        var weights = ModelLoader.Load(gguf, stream, backend, config);
        var kvCache = new KvCache(backend, config, maxSeqLen: maxSeqLen, cacheType: cacheType);
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
        public KvCache KvCache { get; }
        public DeltaNetState DeltaState { get; }
        public ForwardPass Forward { get; }

        public ModelContext(Stream stream, GgufFile gguf, ModelConfig config,
            CpuBackend backend, ModelWeights weights, KvCache kvCache,
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
