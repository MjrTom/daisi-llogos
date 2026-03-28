using Daisi.Llogos.Chat;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;
using System.Text;

namespace Daisi.Llogos.Tests.Chat;

/// <summary>
/// Test with REALISTIC sampling parameters (temp=0.4, repeat_penalty=1.1)
/// matching the CRM's actual settings. This should produce complete JSON
/// regardless of minor logit precision differences from llama.cpp.
/// </summary>
public class RealisticSamplingTest : IDisposable
{
    private readonly IComputeBackend? _backend;
    private readonly ModelConfig? _config;
    private readonly ModelWeights? _weights;
    private readonly BpeTokenizer? _tokenizer;
    private readonly ChatTemplate? _chatTemplate;

    public RealisticSamplingTest()
    {
        if (!TestConstants.Model9BExists) return;
        using var stream = File.OpenRead(TestConstants.Qwen35_9B_Q8_0);
        var gguf = GgufFile.Read(stream);
        _config = ModelConfig.FromGguf(gguf);
        _backend = (IComputeBackend)Activator.CreateInstance(
            Type.GetType("Daisi.Llogos.Cuda.CudaBackend, Daisi.Llogos.Cuda")!, 0)!;
        _weights = MmapModelLoader.Load(gguf, TestConstants.Qwen35_9B_Q8_0, _backend, _config);
        _tokenizer = TokenizerFactory.FromGguf(gguf);
        _chatTemplate = ChatTemplate.FromGguf(gguf);
    }

    private bool Ready => _backend != null;

    [Fact]
    public async Task CrmReport_RealisticParams_ProducesJson()
    {
        if (!Ready) return;

        var kvCache = new KvCache(_backend!, _config!, maxSeqLen: 4096);
        var deltaState = new DeltaNetState(_backend!, _config!, _weights!);
        var forward = new ForwardPass(_backend!, _config!, _weights!, kvCache, deltaState);
        var renderer = new ChatTemplateRenderer(_chatTemplate!);
        var stops = _chatTemplate!.GetStopSequences();
        using var session = new DaisiLlogosChatSession(forward, _tokenizer!, renderer, stops);

        // Full CRM system prompt (the one that was in the original BuildSystemPrompt)
        session.AddMessage(new ChatMessage("system",
            "You are a business analytics expert. Given CRM report data, produce a concise executive analysis.\n\n" +
            "Return ONLY a valid JSON object matching this exact schema. Do not include any explanation or markdown formatting.\n\n" +
            "Schema:\n" +
            "{\"executiveSummary\": \"2-3 sentence overview\", \"keyMetrics\": [{\"label\": \"string\", \"value\": \"string\", \"trend\": \"up|down|neutral\"}], " +
            "\"insights\": [{\"title\": \"string\", \"description\": \"string\", \"severity\": \"success|warning|info\"}], " +
            "\"recommendations\": [{\"action\": \"string\", \"impact\": \"high|medium|low\", \"description\": \"string\"}]}\n\n" +
            "Rules:\n- Include 3-6 key metrics, 3-5 insights, and 2-4 recommendations\n" +
            "- Be specific and reference actual numbers\n- Return ONLY the JSON object, nothing else"));

        // CRM's actual sampling parameters
        var parameters = new GenerationParams
        {
            MaxTokens = 2048,
            Temperature = 0f,
            RepetitionPenalty = 1.05f, // Lower penalty to avoid penalizing structural JSON chars
            PenaltyCount = 32, // Shorter window
        };

        var sb = new StringBuilder();
        var sw = System.Diagnostics.Stopwatch.StartNew();
        await foreach (var token in session.ChatAsync(
            new ChatMessage("user",
                "Report: Sales Report\n\nPeriod: 2026-02-26 to 2026-03-27\nRevenue: $32,980.76\n" +
                "Orders: 78\nAvg Order Value: $422.83\n\n" +
                "Top Products:\n- Virtual Desk: 79 sold, $30,929.00\n- Large Office Subscription: 4 sold, $1,896.76\n\n" +
                "Top Customers:\n- Candace Hill: 4 orders, $2,500.00\n- Test Customer: 9 orders, $1,946.76"),
            parameters))
        {
            sb.Append(token);
        }
        sw.Stop();

        var result = sb.ToString();
        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-realistic-output.txt",
            $"Time: {sw.Elapsed.TotalSeconds:F1}s\n\n{result}");

        // Strip think tags
        var stripped = result;
        var thinkEnd = stripped.LastIndexOf("</think>");
        if (thinkEnd >= 0) stripped = stripped[(thinkEnd + 8)..];
        stripped = stripped.Trim();

        Assert.True(stripped.Contains("{"), $"No JSON. Output:\n{stripped[..Math.Min(300, stripped.Length)]}");
        var json = stripped[stripped.IndexOf('{')..(stripped.LastIndexOf('}') + 1)];

        // Strict JSON parse — no repair
        var doc = System.Text.Json.JsonDocument.Parse(json);
        Assert.True(doc.RootElement.TryGetProperty("executiveSummary", out _), $"Missing executiveSummary");
        Assert.True(doc.RootElement.TryGetProperty("keyMetrics", out var m) && m.GetArrayLength() >= 2, $"Missing/empty keyMetrics");
        Assert.True(doc.RootElement.TryGetProperty("insights", out var ins) && ins.GetArrayLength() >= 1, $"Missing/empty insights");
        Assert.True(doc.RootElement.TryGetProperty("recommendations", out var rec) && rec.GetArrayLength() >= 1, $"Missing/empty recommendations");
    }

    public void Dispose()
    {
        _weights?.Dispose();
        _backend?.Dispose();
    }
}
