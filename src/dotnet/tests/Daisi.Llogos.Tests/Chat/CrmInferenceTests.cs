using Daisi.Llogos.Chat;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;
using System.Text;
using System.Text.Json;

namespace Daisi.Llogos.Tests.Chat;

/// <summary>
/// Tests that replicate the CRM app's AI service request patterns against LLogos + Qwen 3.5.
/// Validates that the fixes for think-tag interference, grammar constraints, and JSON extraction work.
/// </summary>
public class CrmInferenceTests : IDisposable
{
    private readonly IComputeBackend? _backend;
    private readonly ModelConfig? _config;
    private readonly ModelWeights? _weights;
    private readonly BpeTokenizer? _tokenizer;
    private readonly ChatTemplate? _chatTemplate;

    public CrmInferenceTests()
    {
        if (!TestConstants.Model9BExists) return;

        using var stream = File.OpenRead(TestConstants.Qwen35_9B_Q8_0);
        var gguf = GgufFile.Read(stream);
        _config = ModelConfig.FromGguf(gguf);
        _backend = CreateCudaBackend();
        _weights = MmapModelLoader.Load(gguf, TestConstants.Qwen35_9B_Q8_0, _backend, _config);
        _tokenizer = TokenizerFactory.FromGguf(gguf);
        _chatTemplate = ChatTemplate.FromGguf(gguf);
    }

    private bool Ready => _backend != null;

    private static IComputeBackend CreateCudaBackend()
    {
        var type = Type.GetType("Daisi.Llogos.Cuda.CudaBackend, Daisi.Llogos.Cuda")
            ?? throw new InvalidOperationException("CUDA backend not available.");
        return (IComputeBackend)Activator.CreateInstance(type, 0)!;
    }

    private DaisiLlogosChatSession CreateSession(string? systemPrompt = null, int? seed = 42)
    {
        var kvCache = new KvCache(_backend!, _config!, maxSeqLen: 2048);
        var deltaState = new DeltaNetState(_backend!, _config!, _weights!);
        var forward = new ForwardPass(_backend!, _config!, _weights!, kvCache, deltaState);
        var renderer = new ChatTemplateRenderer(_chatTemplate!);
        var stopSequences = _chatTemplate!.GetStopSequences();
        var session = new DaisiLlogosChatSession(forward, _tokenizer!, renderer, stopSequences, seed);

        if (!string.IsNullOrEmpty(systemPrompt))
            session.AddMessage(new ChatMessage("system", systemPrompt));

        return session;
    }

    // ─── Template detection ──────────────────────────────────────────────────

    [Fact]
    public void Qwen35_ShouldUse_ChatMLTemplate()
    {
        if (!Ready) return;
        Assert.Equal(ChatTemplateFormat.ChatML, _chatTemplate!.Format);
    }

    [Fact]
    public void Qwen35_StopSequences_ShouldInclude_ImEnd()
    {
        if (!Ready) return;
        Assert.Contains("<|im_end|>", _chatTemplate!.GetStopSequences());
    }

    // ─── Think-tag-aware JSON extraction (unit tests, no model needed) ───────

    [Theory]
    [InlineData(
        "<think>Let me create a JSON with {name: test}.</think><response>{\"name\":\"Order Email\"}</response>",
        "{\"name\":\"Order Email\"}",
        "Think block contains curly braces")]
    [InlineData(
        "{\"name\":\"Order Email\"}",
        "{\"name\":\"Order Email\"}",
        "Plain JSON with no tags")]
    [InlineData(
        "<think>Reasoning here</think><response>{\"key\":\"value\"}</response>",
        "{\"key\":\"value\"}",
        "Clean think + response tags")]
    [InlineData(
        "<think>Step 1: {parse}\nStep 2: {create}</think>{\"result\":\"done\"}",
        "{\"result\":\"done\"}",
        "Think block with braces, JSON after close")]
    [InlineData(
        "<think>first</think><think>second {x}</think>{\"final\":true}",
        "{\"final\":true}",
        "Multiple think blocks, uses LastIndexOf")]
    public void ExtractJsonAfterThinkTags_ShouldWork(string input, string expectedJson, string _scenario)
    {
        var result = ExtractJsonFromResponse(input);
        Assert.Equal(expectedJson, result);
    }

    // ─── Fix validation: JSON output without think tags ──────────────────────

    /// <summary>
    /// Validates the fix: when the host's grammar override prompt tells the model
    /// NOT to use think/response tags, Qwen 3.5 outputs clean JSON.
    /// This is the path the CRM now takes with OutputFormat = Json.
    /// </summary>
    [Fact]
    public async Task JsonOutput_WithoutThinkTags_ProducesValidJson()
    {
        if (!Ready) return;

        // Grammar override prompt (prepended by host when OutputFormat = Json)
        var grammarOverride = "IMPORTANT: Do not use <think> or <response> tags. Respond with the requested format only.\n\n";

        var crmPrompt =
            "You are an automation configuration generator. " +
            "Return ONLY a valid JSON object. No explanation, no markdown.\n\n" +
            "Schema: {\"name\": \"string\", \"triggerType\": \"string\"}\n\n" +
            "Return ONLY the JSON object, nothing else";

        using var session = CreateSession(grammarOverride + crmPrompt);

        var parameters = new GenerationParams
        {
            MaxTokens = 256,
            Temperature = 0.3f,
            TopK = 40,
            TopP = 0.9f,
        };

        var tokens = new List<string>();
        await foreach (var token in session.ChatAsync(
            new ChatMessage("user", "Create an automation for new orders"),
            parameters))
        {
            tokens.Add(token);
        }

        var fullResponse = string.Join("", tokens);

        // Should NOT contain think tags
        Assert.DoesNotContain("<think>", fullResponse);

        // Should contain valid JSON
        var trimmed = fullResponse.Trim();
        var startIdx = trimmed.IndexOf('{');
        var endIdx = trimmed.LastIndexOf('}');
        Assert.True(startIdx >= 0 && endIdx > startIdx, $"Response should contain JSON braces. Got: {fullResponse}");

        var jsonStr = trimmed[startIdx..(endIdx + 1)];
        var doc = JsonDocument.Parse(jsonStr); // throws if invalid
        Assert.True(doc.RootElement.TryGetProperty("name", out _), $"JSON should have 'name' property. Got: {jsonStr}");
    }

    /// <summary>
    /// Validates that AntiPrompts now work as stop sequences in LLogos chat sessions.
    /// </summary>
    [Fact]
    public async Task AntiPrompts_ShouldStopGeneration()
    {
        if (!Ready) return;

        using var session = CreateSession("You are a helpful assistant.");

        var parameters = new GenerationParams
        {
            MaxTokens = 256,
            Temperature = 0.7f,
            AntiPrompts = ["User:", "\n\n\n"],
        };

        var tokens = new List<string>();
        await foreach (var token in session.ChatAsync(
            new ChatMessage("user", "Hello, how are you?"),
            parameters))
        {
            tokens.Add(token);
        }

        var fullResponse = string.Join("", tokens);

        // Response should not contain the anti-prompt strings (they trigger stop before being emitted)
        Assert.DoesNotContain("User:", fullResponse);
        Assert.DoesNotContain("\n\n\n", fullResponse);
        Assert.True(fullResponse.Length > 0, "Response should not be empty");
    }

    // ─── Fix validation: Grammar-constrained JSON generation ───────────────

    /// <summary>
    /// Validates that GBNF grammar constraints are applied in chat sessions with the
    /// full recursive JSON grammar (the one that previously caused a stack overflow).
    /// </summary>
    [Fact]
    public async Task GrammarConstraint_JsonGrammar_ProducesValidJson()
    {
        if (!Ready) return;

        using var session = CreateSession(
            "IMPORTANT: Do not use <think> or <response> tags. Respond with the requested format only.\n\n" +
            "You generate JSON objects. Return only valid JSON.");

        // Full recursive JSON grammar from ToolSession.jsonGrammarText
        var grammarText = "root   ::= object\r\nvalue  ::= object | array | string | number | (\"true\" | \"false\" | \"null\") ws\r\n\r\nobject ::=\r\n  \"{\" ws (\r\n            string \":\" ws value\r\n    (\",\" ws string \":\" ws value)*\r\n  )? \"}\" ws\r\n\r\narray  ::=\r\n  \"[\" ws (\r\n            value\r\n    (\",\" ws value)*\r\n  )? \"]\" ws\r\n\r\nstring ::=\r\n  \"\\\"\" (\r\n    [^\"\\\\\\x7F\\x00-\\x1F] |\r\n    \"\\\\\" ([\"\\\\bfnrt] | \"u\" [0-9a-fA-F]{4}) # escapes\r\n  )* \"\\\"\" ws\r\n\r\nnumber ::= (\"-\"? ([0-9] | [1-9] [0-9]{0,15})) (\".\" [0-9]+)? ([eE] [-+]? [0-9] [1-9]{0,15})? ws\r\n\r\n# Optional space: by convention, applied in this grammar after literal chars when allowed\r\nws ::= | \" \" | \"\\n\" [ \\t]{0,20}";

        var parameters = new GenerationParams
        {
            MaxTokens = 256,
            Temperature = 0.3f,
            GrammarText = grammarText,
        };

        var tokens = new List<string>();
        await foreach (var token in session.ChatAsync(
            new ChatMessage("user", "Generate a greeting message as JSON with a message field"),
            parameters))
        {
            tokens.Add(token);
        }

        var fullResponse = string.Join("", tokens);
        Assert.True(fullResponse.Length > 2, $"Response should not be empty. Got: '{fullResponse}'");

        // Grammar should force valid JSON
        var trimmed = fullResponse.Trim();
        var doc = JsonDocument.Parse(trimmed);
        Assert.Equal(JsonValueKind.Object, doc.RootElement.ValueKind);
    }

    // ─── Helper: Think-tag-aware JSON extraction ─────────────────────────────

    private static string? ExtractJsonFromResponse(string response)
    {
        var text = response;

        // Strip everything up to and including the LAST </think> tag
        var thinkEnd = text.LastIndexOf("</think>", StringComparison.Ordinal);
        if (thinkEnd >= 0)
            text = text[(thinkEnd + 8)..];

        // Strip response wrapper tags
        text = text.Replace("<response>", "").Replace("</response>", "").Trim();

        // Extract JSON
        var startIdx = text.IndexOf('{');
        var endIdx = text.LastIndexOf('}');
        if (startIdx < 0 || endIdx <= startIdx)
            return null;

        return text[startIdx..(endIdx + 1)];
    }

    public void Dispose()
    {
        _weights?.Dispose();
        _backend?.Dispose();
    }
}
