using Daisi.Llogos.Chat;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;
using System.Text;

namespace Daisi.Llogos.Tests.Chat;

/// <summary>
/// Find exact divergence between LLogos and llama.cpp by comparing greedy (temp=0) output.
/// </summary>
public class DivergenceTest : IDisposable
{
    private readonly IComputeBackend? _backend;
    private readonly ModelConfig? _config;
    private readonly ModelWeights? _weights;
    private readonly BpeTokenizer? _tokenizer;
    private readonly ChatTemplate? _chatTemplate;

    // Set to true to test with 0.8B (faster, helps isolate if issue is model-size dependent)
    private static readonly bool Use08B = false;

    public DivergenceTest()
    {
        var modelPath = Use08B ? TestConstants.Qwen35_08B_Q8_0 : TestConstants.Qwen35_9B_Q8_0;
        if (!File.Exists(modelPath)) return;
        using var stream = File.OpenRead(modelPath);
        var gguf = GgufFile.Read(stream);
        _config = ModelConfig.FromGguf(gguf);
        var cudaType = Type.GetType("Daisi.Llogos.Cuda.CudaBackend, Daisi.Llogos.Cuda");
        _backend = (IComputeBackend)Activator.CreateInstance(cudaType!, 0)!;
        _weights = MmapModelLoader.Load(gguf, modelPath, _backend, _config);
        _tokenizer = TokenizerFactory.FromGguf(gguf);
        _chatTemplate = ChatTemplate.FromGguf(gguf);
    }

    private bool Ready => _backend != null;

    /// <summary>
    /// Short prompt: LLogos should produce valid JSON (matches llama.cpp behavior).
    /// </summary>
    [Fact]
    public async Task Short_ProducesValidJson()
    {
        if (!Ready) return;

        var result = await GenerateGreedy(
            "Return ONLY valid JSON: {\"a\":\"string\",\"b\":\"string\",\"c\":\"string\"}",
            "Generate a test with three fields.");

        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-divergence-short.txt", result);

        var stripped = StripThink(result);
        Assert.True(stripped.Contains("{") && stripped.Contains("}"), $"No JSON. Got: {result}");
        var json = stripped[stripped.IndexOf('{')..(stripped.LastIndexOf('}') + 1)];
        var doc = System.Text.Json.JsonDocument.Parse(json);
        Assert.Equal(System.Text.Json.JsonValueKind.Object, doc.RootElement.ValueKind);
    }

    /// <summary>
    /// Medium prompt — exact match with llama.cpp baseline.
    /// llama.cpp output: {"executiveSummary": "Revenue reached $32,980 with 78 orders, driven by the top product, Virtual Desk.", "keyMetrics": [{"label": "Revenue", "value": "$32,980", "trend": "up"}, ...]}
    /// </summary>
    [Fact]
    public async Task MediumPrompt_MatchesLlamaCpp()
    {
        if (!Ready) return;

        var result = await GenerateGreedy(
            "Return ONLY valid JSON. No thinking, no explanation.\n{\"executiveSummary\":\"string\",\"keyMetrics\":[{\"label\":\"string\",\"value\":\"string\",\"trend\":\"string\"}]}",
            "Revenue: $32,980. Orders: 78. Top Product: Virtual Desk.");

        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-divergence-medium.txt", result);

        var stripped = StripThink(result);
        var startIdx = stripped.IndexOf('{');
        var endIdx = stripped.LastIndexOf('}');
        Assert.True(startIdx >= 0 && endIdx > startIdx, $"No JSON. Got: {stripped[..Math.Min(200, stripped.Length)]}");
        var json = stripped[startIdx..(endIdx + 1)];

        var doc = System.Text.Json.JsonDocument.Parse(json);
        Assert.True(doc.RootElement.TryGetProperty("executiveSummary", out _), $"Missing executiveSummary. JSON:\n{json}");
        Assert.True(doc.RootElement.TryGetProperty("keyMetrics", out var metrics), $"Missing keyMetrics. JSON:\n{json}");
        Assert.True(metrics.GetArrayLength() >= 2, $"Should have >= 2 metrics. JSON:\n{json}");
    }

    /// <summary>
    /// CRM-length prompt with greedy: find where JSON formatting breaks.
    /// llama.cpp produces perfect JSON for this. Does LLogos?
    /// </summary>
    [Fact]
    public async Task CrmLength_ProducesValidJson()
    {
        if (!Ready) return;

        var result = await GenerateGreedy(
            "You are a business analytics expert.\n\n" +
            "Return ONLY a valid JSON object with this schema:\n" +
            "{\"executiveSummary\": \"string\", \"keyMetrics\": [{\"label\": \"string\", \"value\": \"string\", \"trend\": \"up|down|neutral\"}], " +
            "\"insights\": [{\"title\": \"string\", \"description\": \"string\", \"severity\": \"success|warning|info\"}], " +
            "\"recommendations\": [{\"action\": \"string\", \"impact\": \"high|medium|low\", \"description\": \"string\"}]}\n\n" +
            "Do NOT abbreviate. All keys and values MUST be in double quotes. Return ONLY the JSON.",
            "Report: Revenue: $32,980. Orders: 78. Top Product: Virtual Desk (79 sold, $30,929). " +
            "Top Customers: Candace Hill ($2,500), Test Customer ($1,946).");

        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-divergence-crm.txt", result);

        var stripped = StripThink(result);
        Assert.True(stripped.Contains("{"), $"No JSON. Got: {stripped[..Math.Min(200, stripped.Length)]}");

        var startIdx = stripped.IndexOf('{');
        var endIdx = stripped.LastIndexOf('}');
        Assert.True(endIdx > startIdx, "No closing brace");

        var json = stripped[startIdx..(endIdx + 1)];

        // Strict JSON parse — this is what we need to match llama.cpp
        try
        {
            var doc = System.Text.Json.JsonDocument.Parse(json);
            Assert.True(doc.RootElement.TryGetProperty("executiveSummary", out _), "Missing executiveSummary");
            Assert.True(doc.RootElement.TryGetProperty("keyMetrics", out _), "Missing keyMetrics");
        }
        catch (System.Text.Json.JsonException ex)
        {
            // Find the position where JSON breaks
            Assert.Fail($"Invalid JSON at position {ex.BytePositionInLine}.\n\n" +
                $"JSON (first 500 chars):\n{json[..Math.Min(500, json.Length)]}\n\n" +
                $"Error: {ex.Message}");
        }
    }

    private async Task<string> GenerateGreedy(string system, string user)
    {
        var kvCache = new KvCache(_backend!, _config!, maxSeqLen: 4096);
        var deltaState = new DeltaNetState(_backend!, _config!, _weights!);
        var forward = new ForwardPass(_backend!, _config!, _weights!, kvCache, deltaState);
        var renderer = new ChatTemplateRenderer(_chatTemplate!);
        var stops = _chatTemplate!.GetStopSequences();
        using var session = new DaisiLlogosChatSession(forward, _tokenizer!, renderer, stops);
        session.AddMessage(new ChatMessage("system", system));

        var sb = new StringBuilder();
        await foreach (var tok in session.ChatAsync(
            new ChatMessage("user", user),
            new GenerationParams { MaxTokens = 512, Temperature = 0 }))
        {
            sb.Append(tok);
        }
        return sb.ToString();
    }

    private static string StripThink(string s)
    {
        var idx = s.LastIndexOf("</think>", StringComparison.Ordinal);
        if (idx >= 0) s = s[(idx + 8)..];
        return s.Trim();
    }

    public void Dispose()
    {
        _weights?.Dispose();
        _backend?.Dispose();
    }
}
