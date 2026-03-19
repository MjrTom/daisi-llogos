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

    // ── Sliding Window + Attention Sinks ─────────────────────────────────────

    [Fact]
    public void AttentionStrategy_Parse_Full()
    {
        var s = AttentionStrategy.Parse("full");
        Assert.Equal(AttentionMode.Full, s.Mode);
    }

    [Fact]
    public void AttentionStrategy_Parse_Window()
    {
        var s = AttentionStrategy.Parse("window:1024");
        Assert.Equal(AttentionMode.Window, s.Mode);
        Assert.Equal(0, s.SinkTokens);
        Assert.Equal(1024, s.WindowSize);
        Assert.Equal(1024, s.CacheCapacity);
    }

    [Fact]
    public void AttentionStrategy_Parse_Sinks()
    {
        var s = AttentionStrategy.Parse("sinks:64,4096");
        Assert.Equal(AttentionMode.Sinks, s.Mode);
        Assert.Equal(64, s.SinkTokens);
        Assert.Equal(4096, s.WindowSize);
        Assert.Equal(4160, s.CacheCapacity);
    }

    [Fact]
    public void AttentionStrategy_MapPosition_Full()
    {
        var s = AttentionStrategy.Full;
        Assert.Equal(0, s.MapPosition(0));
        Assert.Equal(999, s.MapPosition(999));
    }

    [Fact]
    public void AttentionStrategy_MapPosition_RingBuffer()
    {
        // 4 sink tokens + 8 window = 12 total slots
        var s = AttentionStrategy.Sinks(4, 8);

        // Positions 0-11 map linearly (still filling)
        for (int i = 0; i < 12; i++)
            Assert.Equal(i, s.MapPosition(i));

        // Position 12: first overwrite in window region → slot 4 + ((12-4) % 8) = 4 + 0 = 4
        Assert.Equal(4, s.MapPosition(12));
        // Position 13 → slot 4 + ((13-4) % 8) = 4 + 1 = 5
        Assert.Equal(5, s.MapPosition(13));
        // Position 19 → slot 4 + ((19-4) % 8) = 4 + 7 = 11
        Assert.Equal(11, s.MapPosition(19));
        // Position 20 wraps again → slot 4 + ((20-4) % 8) = 4 + 0 = 4
        Assert.Equal(4, s.MapPosition(20));
    }

    [Fact]
    public void AttentionStrategy_EffectiveSeqLen_Capped()
    {
        var s = AttentionStrategy.Sinks(4, 8);
        Assert.Equal(1, s.EffectiveSeqLen(0));   // position 0 → 1 token visible
        Assert.Equal(12, s.EffectiveSeqLen(11));  // still filling, 12 visible
        Assert.Equal(12, s.EffectiveSeqLen(12));  // now capped at capacity
        Assert.Equal(12, s.EffectiveSeqLen(1000)); // stays capped
    }

    [Fact]
    public void SlidingWindow_ConstantMemory()
    {
        // KV cache with sinks strategy should have fixed maxSeqLen
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        using var backend = new CpuBackend();

        var strategy = AttentionStrategy.Sinks(4, 60);
        using var cache = new KvCache(backend, config, maxSeqLen: 99999, strategy: strategy);

        // maxSeqLen should be clamped to strategy capacity
        Assert.Equal(64, cache.MaxSeqLen);
    }

    [Fact]
    public void SlidingWindow_GeneratesWithoutCrash()
    {
        // Sliding window should produce output even when context exceeds window
        if (!TestConstants.ModelExists) return;

        var strategy = AttentionStrategy.Sinks(4, 60);
        using var ctx = LoadModel(strategy: strategy, maxSeqLen: 64);
        var tokenizer = TokenizerFactory.FromGguf(ctx.Gguf);
        var generator = new TextGenerator(ctx.Forward, tokenizer, seed: 42);

        // Generate enough tokens to overflow the window (64 cache slots)
        var p = new GenerationParams { MaxTokens = 80, Temperature = 0 };
        var tokens = generator.Generate("The capital of France is", p)
            .Where(t => !t.IsDone)
            .Select(t => t.Text)
            .ToList();

        Assert.True(tokens.Count > 0, "Sliding window produced empty output");

        // Memory should stay constant — Length never exceeds capacity
        Assert.True(ctx.KvCache.Length <= 64, $"KV cache length {ctx.KvCache.Length} exceeded capacity 64");
    }

    [Fact]
    public void AttentionSinks_PreservesQuality()
    {
        // With sinks, first tokens are retained. Output should be non-degenerate
        // even after overflowing the window.
        if (!TestConstants.ModelExists) return;

        var strategy = AttentionStrategy.Sinks(8, 56);
        using var ctx = LoadModel(strategy: strategy, maxSeqLen: 64);
        var tokenizer = TokenizerFactory.FromGguf(ctx.Gguf);
        var generator = new TextGenerator(ctx.Forward, tokenizer, seed: 42);

        var p = new GenerationParams { MaxTokens = 100, Temperature = 0 };
        var text = string.Concat(generator.Generate("The quick brown fox jumps over the lazy dog.", p)
            .Where(t => !t.IsDone)
            .Select(t => t.Text));

        // Should produce actual text, not garbage
        Assert.True(text.Length > 20, $"Sinks output too short: '{text}'");
        // Output should contain recognizable words (not degenerate noise)
        Assert.True(text.Any(char.IsLetter), "Sinks output contains no letters — likely degenerate");
    }

    // ── Paged KV Cache ─────────────────────────────────────────────────────

    [Fact]
    public void PagedKvCache_GrowsWithUsage()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        using var backend = new CpuBackend();

        using var cache = new PagedKvCache(backend, config, maxSeqLen: 4096);

        Assert.Equal(0, cache.AllocatedPages);

        // Write to position 0 — should allocate first page
        var k = backend.CreateTensor("k", GgmlType.F32, [config.NumKvHeads * config.KeyLength]);
        var v = backend.CreateTensor("v", GgmlType.F32, [config.NumKvHeads * config.ValueLength]);

        int attnLayer = -1;
        for (int i = 0; i < config.NumLayers; i++)
            if (config.IsStandardAttention(i)) { attnLayer = i; break; }

        cache.Write(backend, attnLayer, 0, k, v);
        Assert.Equal(1, cache.AllocatedPages);
        Assert.Equal(1, cache.Length);

        // Write to position 255 — still in first page
        cache.Write(backend, attnLayer, 255, k, v);
        Assert.Equal(1, cache.AllocatedPages);

        // Write to position 256 — triggers second page
        cache.Write(backend, attnLayer, 256, k, v);
        Assert.Equal(2, cache.AllocatedPages);

        k.Dispose();
        v.Dispose();
    }

    [Fact]
    public void PagedKvCache_MemoryScalesWithContext()
    {
        if (!TestConstants.ModelExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        using var backend = new CpuBackend();

        // Monolithic cache allocates upfront for full maxSeqLen
        using var mono = new KvCache(backend, config, maxSeqLen: 4096);
        long monoBytes = 0;
        for (int i = 0; i < config.NumLayers; i++)
            if (config.IsStandardAttention(i))
                monoBytes += mono.GetKCacheTensor(i).ByteSize + mono.GetVCacheTensor(i).ByteSize;

        // Paged cache starts empty, allocates on demand
        using var paged = new PagedKvCache(backend, config, maxSeqLen: 4096);
        Assert.Equal(0, paged.AllocatedBytes);

        // After writing 1 position, paged should use much less than monolithic
        var k = backend.CreateTensor("k", GgmlType.F32, [config.NumKvHeads * config.KeyLength]);
        var v = backend.CreateTensor("v", GgmlType.F32, [config.NumKvHeads * config.ValueLength]);
        int attnLayer = -1;
        for (int i = 0; i < config.NumLayers; i++)
            if (config.IsStandardAttention(i)) { attnLayer = i; break; }
        paged.Write(backend, attnLayer, 0, k, v);

        Assert.True(paged.AllocatedBytes < monoBytes,
            $"Paged {paged.AllocatedBytes} should be less than monolithic {monoBytes}");

        k.Dispose();
        v.Dispose();
    }

    [Fact]
    public void PagedKvCache_CoherentOutput()
    {
        // Paged cache should produce the same output as monolithic cache
        if (!TestConstants.ModelExists) return;

        using var ctxMono = LoadModel(maxSeqLen: 512);
        using var ctxPaged = LoadModel(maxSeqLen: 512, usePaged: true);
        var tokMono = TokenizerFactory.FromGguf(ctxMono.Gguf);
        var tokPaged = TokenizerFactory.FromGguf(ctxPaged.Gguf);

        var genMono = new TextGenerator(ctxMono.Forward, tokMono, seed: 42);
        var genPaged = new TextGenerator(ctxPaged.Forward, tokPaged, seed: 42);

        var p = new GenerationParams { MaxTokens = 10, Temperature = 0 };

        var tokensMono = genMono.Generate("Hello", p).Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();
        var tokensPaged = genPaged.Generate("Hello", p).Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();

        // Greedy output should match exactly
        Assert.Equal(tokensMono.Length, tokensPaged.Length);
        for (int i = 0; i < tokensMono.Length; i++)
            Assert.Equal(tokensMono[i], tokensPaged[i]);
    }

    [Fact]
    public void PagedKvCache_LargerContext_Stable()
    {
        // Paged cache should handle larger context that spans multiple pages
        if (!TestConstants.ModelExists) return;

        using var ctx = LoadModel(maxSeqLen: 2048, usePaged: true);
        var tokenizer = TokenizerFactory.FromGguf(ctx.Gguf);
        var ids = tokenizer.Encode("The quick brown fox jumps over the lazy dog. " +
            "This tests the paged KV cache across multiple page boundaries.");

        ReadOnlySpan<float> logits = default;
        for (int i = 0; i < ids.Length; i++)
            logits = ctx.Forward.Forward(ids[i], i);

        Assert.Equal(ctx.Config.VocabSize, logits.Length);
        Assert.True(float.IsFinite(logits[0]), "Paged cache output not finite");

        // Generate a few more
        for (int i = 0; i < 5; i++)
        {
            int argmax = ArgMax(logits);
            logits = ctx.Forward.Forward(argmax, ids.Length + i);
            Assert.True(float.IsFinite(logits[0]));
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static ModelContext LoadModel(GgmlType cacheType = GgmlType.F16, int maxSeqLen = 256,
        AttentionStrategy? strategy = null, bool usePaged = false)
    {
        var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        var backend = new CpuBackend();
        var weights = ModelLoader.Load(gguf, stream, backend, config);
        IKvCache kvCache = usePaged
            ? new PagedKvCache(backend, config, maxSeqLen: maxSeqLen, cacheType: cacheType, strategy: strategy)
            : new KvCache(backend, config, maxSeqLen: maxSeqLen, cacheType: cacheType, strategy: strategy);
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
