using Daisi.Llama.Cpu;
using Daisi.Llama.Gguf;
using Daisi.Llama.Inference;
using Daisi.Llama.Model;
using Daisi.Llama.Tokenizer;

namespace Daisi.Llama.Tests.Inference;

public class TextGeneratorTests
{
    [Fact]
    public void StopsAtMaxTokens()
    {
        if (!TestConstants.ModelExists) return;

        using var ctx = LoadModel();
        var generator = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);

        var p = new GenerationParams { MaxTokens = 5, Temperature = 0 };
        var tokens = generator.Generate("Hello", p).Where(t => !t.IsDone).ToList();

        Assert.Equal(5, tokens.Count);
    }

    [Fact]
    public void ProducesValidText()
    {
        if (!TestConstants.ModelExists) return;

        using var ctx = LoadModel();
        var generator = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);

        var p = new GenerationParams { MaxTokens = 10, Temperature = 0 };
        var text = string.Concat(generator.Generate("Hello", p)
            .Where(t => !t.IsDone)
            .Select(t => t.Text));

        Assert.True(text.Length > 0, "Generated text is empty");
    }

    [Fact]
    public void ReportsThroughput()
    {
        if (!TestConstants.ModelExists) return;

        using var ctx = LoadModel();
        var generator = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);

        var p = new GenerationParams { MaxTokens = 3, Temperature = 0 };
        var all = generator.Generate("Hello", p).ToList();

        var done = all.Last();
        Assert.True(done.IsDone);
        Assert.Equal(3, done.TotalTokens);
        Assert.True(done.TokensPerSecond > 0);
    }

    [Fact]
    public void Greedy_Deterministic()
    {
        if (!TestConstants.ModelExists) return;

        using var ctx1 = LoadModel();
        using var ctx2 = LoadModel();
        var gen1 = new TextGenerator(ctx1.Forward, ctx1.Tokenizer, seed: 42);
        var gen2 = new TextGenerator(ctx2.Forward, ctx2.Tokenizer, seed: 42);

        var p = new GenerationParams { MaxTokens = 5, Temperature = 0 };

        var tokens1 = gen1.Generate("Hello", p).Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();
        var tokens2 = gen2.Generate("Hello", p).Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();

        Assert.Equal(tokens1, tokens2);
    }

    [Fact]
    public void EndToEnd_Qwen35()
    {
        if (!TestConstants.ModelExists) return;

        using var ctx = LoadModel();
        var generator = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);

        var p = new GenerationParams { MaxTokens = 20, Temperature = 0 };
        var text = string.Concat(generator.Generate("The capital of France is", p)
            .Where(t => !t.IsDone)
            .Select(t => t.Text));

        Assert.True(text.Length > 0, "Generated text is empty");
        // With greedy sampling, we expect coherent output
        // The model should produce something related to Paris
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
        var tokenizer = TokenizerFactory.FromGguf(gguf);
        return new ModelContext(stream, gguf, config, backend, weights, kvCache, deltaState, forward, tokenizer);
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
        public BpeTokenizer Tokenizer { get; }

        public ModelContext(Stream stream, GgufFile gguf, ModelConfig config,
            CpuBackend backend, ModelWeights weights, KvCache kvCache,
            DeltaNetState deltaState, ForwardPass forward, BpeTokenizer tokenizer)
        {
            Stream = stream; Gguf = gguf; Config = config;
            Backend = backend; Weights = weights; KvCache = kvCache;
            DeltaState = deltaState; Forward = forward; Tokenizer = tokenizer;
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
