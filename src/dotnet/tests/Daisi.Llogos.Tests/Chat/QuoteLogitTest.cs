using Daisi.Llogos.Chat;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;
using System.Text;

namespace Daisi.Llogos.Tests.Chat;

/// <summary>
/// Dump logits at the exact positions where JSON keys should have quotes.
/// Check if the quote token is a close second or completely absent from top candidates.
/// </summary>
public class QuoteLogitTest : IDisposable
{
    private readonly IComputeBackend? _backend;
    private readonly ModelConfig? _config;
    private readonly ModelWeights? _weights;
    private readonly BpeTokenizer? _tokenizer;
    private readonly ChatTemplate? _chatTemplate;

    public QuoteLogitTest()
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
    public async Task DumpLogitsAtKeyPositions()
    {
        if (!Ready) return;

        var kvCache = new KvCache(_backend!, _config!, maxSeqLen: 4096);
        var deltaState = new DeltaNetState(_backend!, _config!, _weights!);
        var forward = new ForwardPass(_backend!, _config!, _weights!, kvCache, deltaState);
        var renderer = new ChatTemplateRenderer(_chatTemplate!);
        var stops = _chatTemplate!.GetStopSequences();
        using var session = new DaisiLlogosChatSession(forward, _tokenizer!, renderer, stops);

        session.AddMessage(new ChatMessage("system",
            "You are a business analytics expert. Given CRM report data, produce a concise executive analysis.\n\n" +
            "Return ONLY a valid JSON object matching this exact schema. Do not include any explanation or markdown formatting.\n\n" +
            "Schema:\n{\"executiveSummary\": \"2-3 sentence overview\", \"keyMetrics\": [{\"label\": \"string\", \"value\": \"string\", \"trend\": \"up|down|neutral\"}], " +
            "\"insights\": [{\"title\": \"string\", \"description\": \"string\", \"severity\": \"success|warning|info\"}], " +
            "\"recommendations\": [{\"action\": \"string\", \"impact\": \"high|medium|low\", \"description\": \"string\"}]}\n\n" +
            "Rules:\n- Include 3-6 key metrics, 3-5 insights, and 2-4 recommendations\n" +
            "- Be specific and reference actual numbers\n- Return ONLY the JSON object, nothing else"));

        var sb = new StringBuilder();
        int tokenCount = 0;
        await foreach (var token in session.ChatAsync(
            new ChatMessage("user",
                "Report: Sales Report\n\nPeriod: 2026-02-26 to 2026-03-27\nRevenue: $32,980.76\nOrders: 78\nAvg Order Value: $422.83\n\n" +
                "Top Products:\n- Virtual Desk: 79 sold, $30,929.00\n- Large Office Subscription: 4 sold, $1,896.76\n\n" +
                "Top Customers:\n- Candace Hill: 4 orders, $2,500.00\n- Test Customer: 9 orders, $1,946.76"),
            new GenerationParams { MaxTokens = 2048, Temperature = 0, RepetitionPenalty = 1.1f }))
        {
            sb.Append(token);
            tokenCount++;
        }

        var result = sb.ToString();
        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-quote-logits.txt", result);

        // Check: does it produce all keys quoted?
        var stripped = result;
        var thinkEnd = stripped.LastIndexOf("</think>");
        if (thinkEnd >= 0) stripped = stripped[(thinkEnd + 8)..];
        stripped = stripped.Trim();

        // Count quoted vs unquoted keys
        int quotedKeys = 0, unquotedKeys = 0;
        for (int i = 0; i < stripped.Length - 1; i++)
        {
            // Pattern: "key": or key: (after { or ,)
            if ((stripped[i] == '{' || stripped[i] == ',') && i + 1 < stripped.Length)
            {
                // Skip whitespace
                int j = i + 1;
                while (j < stripped.Length && char.IsWhiteSpace(stripped[j])) j++;
                if (j < stripped.Length)
                {
                    if (stripped[j] == '"') quotedKeys++;
                    else if (char.IsLetter(stripped[j])) unquotedKeys++;
                }
            }
        }

        // Try to parse the JSON strictly
        bool validJson = false;
        string? parseError = null;
        try
        {
            if (stripped.Contains("{"))
            {
                var jsonStr = stripped[stripped.IndexOf('{')..(stripped.LastIndexOf('}') + 1)];
                System.Text.Json.JsonDocument.Parse(jsonStr);
                validJson = true;
            }
        }
        catch (System.Text.Json.JsonException ex) { parseError = ex.Message; }

        Assert.True(validJson,
            $"JSON parse failed: {parseError}\nQuoted keys: {quotedKeys}, Unquoted: {unquotedKeys}\n\nOutput:\n{stripped}");
    }

    public void Dispose()
    {
        _weights?.Dispose();
        _backend?.Dispose();
    }
}
