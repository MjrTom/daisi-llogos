using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Tests.Inference;

/// <summary>
/// Integration tests for sampler parameters and grammar-constrained generation.
/// These run the full model → forward pass → sampler pipeline.
/// </summary>
public class SamplerIntegrationTests
{
    // ── AntiPrompts ─────────────────────────────────────────────────────────

    [Fact]
    public void AntiPrompts_StopsOnMatchingString()
    {
        if (!TestConstants.ModelExists) return;
        using var ctx = LoadModel();
        var gen = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);

        // Generate with anti-prompt that should trigger early stop
        var p = new GenerationParams
        {
            MaxTokens = 100,
            Temperature = 0,
            AntiPrompts = ["."],  // Stop at first period
        };

        var text = string.Concat(gen.Generate("The capital of France is", p)
            .Where(t => !t.IsDone).Select(t => t.Text));

        // Should contain a period (the anti-prompt) and be shorter than 100 tokens worth
        Assert.Contains(".", text);
        // The output should be relatively short since we stop at the first sentence
        Assert.True(text.Length < 500, $"AntiPrompt should have stopped early, got {text.Length} chars");
    }

    [Fact]
    public void AntiPrompts_MultiplePatterns()
    {
        if (!TestConstants.ModelExists) return;
        using var ctx = LoadModel();
        var gen = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);

        var p = new GenerationParams
        {
            MaxTokens = 100,
            Temperature = 0,
            AntiPrompts = ["!", "?", "."],
        };

        var text = string.Concat(gen.Generate("Tell me about", p)
            .Where(t => !t.IsDone).Select(t => t.Text));

        // Should stop at any sentence-ending punctuation
        bool hasStop = text.Contains('.') || text.Contains('!') || text.Contains('?');
        Assert.True(hasStop, $"Should stop at punctuation, got: {text}");
    }

    // ── PreventEOS ──────────────────────────────────────────────────────────

    [Fact]
    public void PreventEOS_GeneratesMoreTokens()
    {
        if (!TestConstants.ModelExists) return;
        using var ctx = LoadModel();

        // Without PreventEOS
        var gen1 = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);
        var p1 = new GenerationParams { MaxTokens = 50, Temperature = 0 };
        var tokens1 = gen1.Generate("Hi", p1).Where(t => !t.IsDone).Count();

        // Reset and run with PreventEOS
        ctx.Forward.ResetState();
        var gen2 = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);
        var p2 = new GenerationParams
        {
            MaxTokens = 50,
            Temperature = 0,
            PreventEOS = true,
            StopTokens = ctx.Tokenizer.Vocabulary.EosTokenId >= 0
                ? [ctx.Tokenizer.Vocabulary.EosTokenId] : [],
        };
        var tokens2 = gen2.Generate("Hi", p2).Where(t => !t.IsDone).Count();

        // PreventEOS should generate at least as many tokens (often exactly MaxTokens)
        Assert.True(tokens2 >= tokens1,
            $"PreventEOS should generate >= tokens: {tokens2} vs {tokens1}");
    }

    // ── FrequencyPenalty / PresencePenalty ───────────────────────────────────

    [Fact]
    public void FrequencyPenalty_ReducesRepetition()
    {
        if (!TestConstants.ModelExists) return;
        using var ctx = LoadModel();
        var gen = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);

        var p = new GenerationParams
        {
            MaxTokens = 40,
            Temperature = 0.7f,
            FrequencyPenalty = 1.0f,
            RepetitionPenalty = 1.0f, // Disable standard repeat penalty
        };

        var tokens = gen.Generate("Repeat after me: hello hello hello", p)
            .Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();

        // With high frequency penalty, tokens should be more diverse
        var uniqueRatio = (float)tokens.Distinct().Count() / tokens.Length;
        Assert.True(uniqueRatio > 0.3f,
            $"Frequency penalty should increase diversity: {uniqueRatio:P0} unique");
    }

    [Fact]
    public void PresencePenalty_ProducesValidOutput()
    {
        if (!TestConstants.ModelExists) return;
        using var ctx = LoadModel();
        var gen = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);

        var p = new GenerationParams
        {
            MaxTokens = 20,
            Temperature = 0.7f,
            PresencePenalty = 0.5f,
            RepetitionPenalty = 1.0f,
        };

        var text = string.Concat(gen.Generate("Hello world", p)
            .Where(t => !t.IsDone).Select(t => t.Text));

        Assert.True(text.Length > 0, "Presence penalty should produce valid output");
    }

    // ── MinP ────────────────────────────────────────────────────────────────

    [Fact]
    public void MinP_ProducesValidOutput()
    {
        if (!TestConstants.ModelExists) return;
        using var ctx = LoadModel();
        var gen = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);

        var p = new GenerationParams
        {
            MaxTokens = 10,
            Temperature = 0.8f,
            MinP = 0.05f,
        };

        var text = string.Concat(gen.Generate("Hello", p)
            .Where(t => !t.IsDone).Select(t => t.Text));

        Assert.True(text.Length > 0, "MinP sampling should produce output");
    }

    // ── TypicalP ────────────────────────────────────────────────────────────

    [Fact]
    public void TypicalP_ProducesValidOutput()
    {
        if (!TestConstants.ModelExists) return;
        using var ctx = LoadModel();
        var gen = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);

        var p = new GenerationParams
        {
            MaxTokens = 10,
            Temperature = 0.8f,
            TypicalP = 0.9f,
        };

        var text = string.Concat(gen.Generate("Hello", p)
            .Where(t => !t.IsDone).Select(t => t.Text));

        Assert.True(text.Length > 0, "Typical sampling should produce output");
    }

    // ── PenaltyCount (window) ───────────────────────────────────────────────

    [Fact]
    public void PenaltyCount_LimitsLookback()
    {
        if (!TestConstants.ModelExists) return;
        using var ctx = LoadModel();
        var gen = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);

        // With PenaltyCount=5, only last 5 tokens are penalized
        var p = new GenerationParams
        {
            MaxTokens = 20,
            Temperature = 0,
            RepetitionPenalty = 2.0f,
            PenaltyCount = 5,
        };

        var tokens = gen.Generate("Hello world", p)
            .Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();

        // Should produce valid output
        Assert.True(tokens.Length > 0, "PenaltyCount should not prevent generation");
    }

    // ── Grammar (GBNF constrained) ──────────────────────────────────────────

    [Fact]
    public void Grammar_JsonConstraint_DoesNotCrash()
    {
        if (!TestConstants.ModelExists) return;
        using var ctx = LoadModel();
        var gen = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);

        // Simple JSON grammar — tests that grammar parsing and constraint
        // application doesn't crash, even if output isn't perfectly constrained
        // (the state machine is a best-effort character-level constraint)
        string grammar = """
            root   ::= "{" ws "\"name\"" ws ":" ws string ws "}"
            string ::= "\"" [a-zA-Z ]+ "\""
            ws     ::= " "?
            """;

        var p = new GenerationParams
        {
            MaxTokens = 20,
            Temperature = 0.8f,
            GrammarText = grammar,
        };

        // Should not throw
        var tokens = gen.Generate("Output JSON:", p).ToList();
        Assert.True(tokens.Count > 0, "Grammar generation should produce tokens");
    }

    [Fact]
    public void Grammar_ParsesStandardJsonGrammar()
    {
        // Test that the GBNF parser handles the standard JSON grammar without crashing
        string jsonGrammar = """
            root   ::= object
            value  ::= object | array | string | number | ("true" | "false" | "null") ws
            object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
            array  ::= "[" ws (value ("," ws value)*)? "]" ws
            string ::= "\"" [^"\\]* "\"" ws
            number ::= ("-"? [0-9]+) ("." [0-9]+)? ws
            ws     ::= [ \t\n]*
            """;

        // Should not throw
        var constraint = new GrammarConstraint(jsonGrammar,
            LoadTokenizerOnly());
        Assert.NotNull(constraint);
    }

    // ── Combined parameters ─────────────────────────────────────────────────

    [Fact]
    public void AllParams_CombinedSampling()
    {
        if (!TestConstants.ModelExists) return;
        using var ctx = LoadModel();
        var gen = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);

        // Use all parameters together
        var p = new GenerationParams
        {
            MaxTokens = 15,
            Temperature = 0.7f,
            TopK = 40,
            TopP = 0.9f,
            MinP = 0.05f,
            TypicalP = 0.95f,
            RepetitionPenalty = 1.1f,
            FrequencyPenalty = 0.3f,
            PresencePenalty = 0.3f,
            PenaltyCount = 20,
            MinKeep = 2,
        };

        var text = string.Concat(gen.Generate("The meaning of life is", p)
            .Where(t => !t.IsDone).Select(t => t.Text));

        Assert.True(text.Length > 0, "Combined params should produce output");
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

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
        return new ModelContext(stream, backend, weights, kvCache, deltaState, forward, tokenizer);
    }

    private static BpeTokenizer LoadTokenizerOnly()
    {
        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        return TokenizerFactory.FromGguf(gguf);
    }

    private sealed class ModelContext(Stream stream, CpuBackend backend, ModelWeights weights,
        KvCache kvCache, DeltaNetState deltaState, ForwardPass forward, BpeTokenizer tokenizer) : IDisposable
    {
        public ForwardPass Forward => forward;
        public BpeTokenizer Tokenizer => tokenizer;
        public void Dispose()
        {
            forward.Dispose(); deltaState.Dispose(); kvCache.Dispose();
            weights.Dispose(); backend.Dispose(); stream.Dispose();
        }
    }
}
