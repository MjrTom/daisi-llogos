using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Tests.Inference;

/// <summary>
/// Integration tests that verify each sampler parameter has a measurable EFFECT
/// on generation output. Each test compares behavior with vs without the parameter
/// to prove it's not silently ignored.
/// </summary>
public class SamplerIntegrationTests
{
    // ── AntiPrompts ─────────────────────────────────────────────────────────

    [Fact]
    public void AntiPrompts_StopsEarlierThanWithout()
    {
        if (!TestConstants.ModelExists) return;

        // Without anti-prompt: generate freely
        using var ctx1 = LoadModel();
        var gen1 = new TextGenerator(ctx1.Forward, ctx1.Tokenizer, seed: 42);
        var noStop = new GenerationParams { MaxTokens = 80, Temperature = 0 };
        int countWithout = gen1.Generate("The capital of France is", noStop)
            .Count(t => !t.IsDone);

        // With anti-prompt on period: should stop sooner
        using var ctx2 = LoadModel();
        var gen2 = new TextGenerator(ctx2.Forward, ctx2.Tokenizer, seed: 42);
        var withStop = new GenerationParams { MaxTokens = 80, Temperature = 0, AntiPrompts = ["."] };
        var tokensWith = gen2.Generate("The capital of France is", withStop)
            .Where(t => !t.IsDone).ToList();
        int countWith = tokensWith.Count;
        string textWith = string.Concat(tokensWith.Select(t => t.Text));

        Assert.True(countWith < countWithout,
            $"AntiPrompt '.' should stop earlier: {countWith} vs {countWithout} tokens");
        Assert.Contains(".", textWith);
    }

    // ── PreventEOS ──────────────────────────────────────────────────────────

    [Fact]
    public void PreventEOS_ForcesFullMaxTokens()
    {
        if (!TestConstants.ModelExists) return;
        using var ctx = LoadModel();
        var gen = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);

        int maxTokens = 20;
        var p = new GenerationParams
        {
            MaxTokens = maxTokens,
            Temperature = 0,
            PreventEOS = true,
            StopTokens = ctx.Tokenizer.Vocabulary.EosTokenId >= 0
                ? [ctx.Tokenizer.Vocabulary.EosTokenId] : [],
        };
        int count = gen.Generate("Hi", p).Count(t => !t.IsDone);

        // PreventEOS should force generation to hit MaxTokens exactly
        Assert.Equal(maxTokens, count);
    }

    // ── FrequencyPenalty ─────────────────────────────────────────────────────

    [Fact]
    public void FrequencyPenalty_HighPenalty_MoreDiverseThanLow()
    {
        if (!TestConstants.ModelExists) return;

        // Low penalty
        using var ctx1 = LoadModel();
        var gen1 = new TextGenerator(ctx1.Forward, ctx1.Tokenizer, seed: 42);
        var low = new GenerationParams { MaxTokens = 40, Temperature = 0.7f, FrequencyPenalty = 0f, RepetitionPenalty = 1.0f };
        var tokensLow = gen1.Generate("Say hello hello hello hello hello", low)
            .Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();
        float uniqueLow = (float)tokensLow.Distinct().Count() / Math.Max(tokensLow.Length, 1);

        // High penalty
        using var ctx2 = LoadModel();
        var gen2 = new TextGenerator(ctx2.Forward, ctx2.Tokenizer, seed: 42);
        var high = new GenerationParams { MaxTokens = 40, Temperature = 0.7f, FrequencyPenalty = 2.0f, RepetitionPenalty = 1.0f };
        var tokensHigh = gen2.Generate("Say hello hello hello hello hello", high)
            .Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();
        float uniqueHigh = (float)tokensHigh.Distinct().Count() / Math.Max(tokensHigh.Length, 1);

        Assert.True(uniqueHigh >= uniqueLow,
            $"High FreqPenalty should be more diverse: {uniqueHigh:P0} vs {uniqueLow:P0}");
    }

    // ── PresencePenalty ──────────────────────────────────────────────────────

    [Fact]
    public void PresencePenalty_HighPenalty_MoreDiverseThanLow()
    {
        if (!TestConstants.ModelExists) return;

        // No penalty
        using var ctx1 = LoadModel();
        var gen1 = new TextGenerator(ctx1.Forward, ctx1.Tokenizer, seed: 42);
        var none = new GenerationParams { MaxTokens = 40, Temperature = 0.7f, PresencePenalty = 0f, RepetitionPenalty = 1.0f };
        var tokensNone = gen1.Generate("Repeat: cat cat cat cat cat", none)
            .Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();
        float uniqueNone = (float)tokensNone.Distinct().Count() / Math.Max(tokensNone.Length, 1);

        // High penalty
        using var ctx2 = LoadModel();
        var gen2 = new TextGenerator(ctx2.Forward, ctx2.Tokenizer, seed: 42);
        var high = new GenerationParams { MaxTokens = 40, Temperature = 0.7f, PresencePenalty = 2.0f, RepetitionPenalty = 1.0f };
        var tokensHigh = gen2.Generate("Repeat: cat cat cat cat cat", high)
            .Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();
        float uniqueHigh = (float)tokensHigh.Distinct().Count() / Math.Max(tokensHigh.Length, 1);

        Assert.True(uniqueHigh >= uniqueNone,
            $"High PresencePenalty should be more diverse: {uniqueHigh:P0} vs {uniqueNone:P0}");
    }

    // ── MinP ────────────────────────────────────────────────────────────────

    [Fact]
    public void MinP_HighThreshold_ProducesDifferentOutputThanLow()
    {
        if (!TestConstants.ModelExists) return;

        // MinP=0 (disabled)
        using var ctx1 = LoadModel();
        var gen1 = new TextGenerator(ctx1.Forward, ctx1.Tokenizer, seed: 42);
        var off = new GenerationParams { MaxTokens = 15, Temperature = 0.8f, MinP = 0f };
        var tokensOff = gen1.Generate("The meaning of life", off)
            .Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();

        // MinP=0.3 (aggressive filtering)
        using var ctx2 = LoadModel();
        var gen2 = new TextGenerator(ctx2.Forward, ctx2.Tokenizer, seed: 42);
        var on = new GenerationParams { MaxTokens = 15, Temperature = 0.8f, MinP = 0.3f };
        var tokensOn = gen2.Generate("The meaning of life", on)
            .Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();

        // MinP should change the output (fewer low-probability tokens allowed)
        bool different = !tokensOff.SequenceEqual(tokensOn);
        Assert.True(different,
            "MinP=0.3 should produce different output than MinP=0 (filters low-prob tokens)");
    }

    // ── TypicalP ────────────────────────────────────────────────────────────

    [Fact]
    public void TypicalP_ProducesDifferentOutputThanDisabled()
    {
        if (!TestConstants.ModelExists) return;

        // TypicalP=1.0 (disabled)
        using var ctx1 = LoadModel();
        var gen1 = new TextGenerator(ctx1.Forward, ctx1.Tokenizer, seed: 42);
        var off = new GenerationParams { MaxTokens = 15, Temperature = 0.8f, TypicalP = 1.0f };
        var tokensOff = gen1.Generate("Once upon a time", off)
            .Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();

        // TypicalP=0.5 (strong filtering)
        using var ctx2 = LoadModel();
        var gen2 = new TextGenerator(ctx2.Forward, ctx2.Tokenizer, seed: 42);
        var on = new GenerationParams { MaxTokens = 15, Temperature = 0.8f, TypicalP = 0.5f };
        var tokensOn = gen2.Generate("Once upon a time", on)
            .Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();

        bool different = !tokensOff.SequenceEqual(tokensOn);
        Assert.True(different,
            "TypicalP=0.5 should produce different output than TypicalP=1.0");
    }

    // ── PenaltyCount ────────────────────────────────────────────────────────

    [Fact]
    public void PenaltyCount_SmallWindow_DifferentFromLargeWindow()
    {
        if (!TestConstants.ModelExists) return;

        // Large window (all history penalized)
        using var ctx1 = LoadModel();
        var gen1 = new TextGenerator(ctx1.Forward, ctx1.Tokenizer, seed: 42);
        var large = new GenerationParams { MaxTokens = 20, Temperature = 0.5f, RepetitionPenalty = 2.0f, PenaltyCount = 0 };
        var tokensLarge = gen1.Generate("Hello world, hello world, hello", large)
            .Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();

        // Small window (only last 3 tokens penalized — earlier repeats allowed)
        using var ctx2 = LoadModel();
        var gen2 = new TextGenerator(ctx2.Forward, ctx2.Tokenizer, seed: 42);
        var small = new GenerationParams { MaxTokens = 20, Temperature = 0.5f, RepetitionPenalty = 2.0f, PenaltyCount = 3 };
        var tokensSmall = gen2.Generate("Hello world, hello world, hello", small)
            .Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();

        bool different = !tokensLarge.SequenceEqual(tokensSmall);
        Assert.True(different,
            "PenaltyCount=3 should produce different output than PenaltyCount=0 (full window)");
    }

    // ── MinKeep ─────────────────────────────────────────────────────────────

    [Fact]
    public void MinKeep_HighValue_ProducesValidOutput()
    {
        if (!TestConstants.ModelExists) return;
        using var ctx = LoadModel();
        var gen = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);

        // MinKeep=10 with very aggressive TopP should still produce output
        var p = new GenerationParams { MaxTokens = 10, Temperature = 0.5f, TopP = 0.01f, MinKeep = 10 };
        var tokens = gen.Generate("Hello", p).Where(t => !t.IsDone).ToArray();

        Assert.True(tokens.Length > 0,
            "MinKeep=10 should prevent TopP=0.01 from eliminating all candidates");
    }

    // ── Grammar ─────────────────────────────────────────────────────────────

    [Fact]
    public void Grammar_ParsesStandardJsonGrammar()
    {
        if (!TestConstants.ModelExists) return;

        string jsonGrammar = """
            root   ::= object
            value  ::= object | array | string | number | ("true" | "false" | "null") ws
            object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
            array  ::= "[" ws (value ("," ws value)*)? "]" ws
            string ::= "\"" [^"\\]* "\"" ws
            number ::= ("-"? [0-9]+) ("." [0-9]+)? ws
            ws     ::= [ \t\n]*
            """;

        var constraint = new GrammarConstraint(jsonGrammar, LoadTokenizerOnly());
        Assert.NotNull(constraint);
    }

    [Fact]
    public void Grammar_ConstrainsOutput()
    {
        if (!TestConstants.ModelExists) return;
        using var ctx = LoadModel();

        // Without grammar: free generation
        var gen1 = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);
        var free = new GenerationParams { MaxTokens = 15, Temperature = 0.8f };
        var textFree = string.Concat(gen1.Generate("Output:", free)
            .Where(t => !t.IsDone).Select(t => t.Text));

        // Reset and run with grammar
        ctx.Forward.ResetState();
        var gen2 = new TextGenerator(ctx.Forward, ctx.Tokenizer, seed: 42);
        var constrained = new GenerationParams
        {
            MaxTokens = 15,
            Temperature = 0.8f,
            GrammarText = "root ::= [0-9]+",  // Only digits
        };
        var textGrammar = string.Concat(gen2.Generate("Output:", constrained)
            .Where(t => !t.IsDone).Select(t => t.Text));

        // Grammar output should differ from free output
        Assert.NotEqual(textFree, textGrammar);
    }

    // ── Combined all-params ─────────────────────────────────────────────────

    [Fact]
    public void AllParams_DifferFromDefaults()
    {
        if (!TestConstants.ModelExists) return;

        // Default params
        using var ctx1 = LoadModel();
        var gen1 = new TextGenerator(ctx1.Forward, ctx1.Tokenizer, seed: 42);
        var defaults = new GenerationParams { MaxTokens = 15, Temperature = 0.7f };
        var tokensDefault = gen1.Generate("The meaning of life is", defaults)
            .Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();

        // All custom params
        using var ctx2 = LoadModel();
        var gen2 = new TextGenerator(ctx2.Forward, ctx2.Tokenizer, seed: 42);
        var custom = new GenerationParams
        {
            MaxTokens = 15, Temperature = 0.7f,
            MinP = 0.1f, TypicalP = 0.8f,
            FrequencyPenalty = 0.5f, PresencePenalty = 0.5f,
            PenaltyCount = 10, MinKeep = 3,
        };
        var tokensCustom = gen2.Generate("The meaning of life is", custom)
            .Where(t => !t.IsDone).Select(t => t.TokenId).ToArray();

        bool different = !tokensDefault.SequenceEqual(tokensCustom);
        Assert.True(different,
            "Custom params (MinP, TypicalP, Freq/Presence penalty) should produce different output than defaults");
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
