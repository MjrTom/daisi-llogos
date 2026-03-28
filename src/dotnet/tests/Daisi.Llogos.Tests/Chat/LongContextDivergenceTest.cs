using Daisi.Llogos.Chat;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;
using System.Diagnostics;
using System.Text;

namespace Daisi.Llogos.Tests.Chat;

/// <summary>
/// Diagnose long-context divergence between LLogos and llama.cpp.
/// Compares token-by-token greedy output at increasing prompt lengths.
/// </summary>
public class LongContextDivergenceTest : IDisposable
{
    private readonly IComputeBackend? _backend;
    private readonly ModelConfig? _config;
    private readonly ModelWeights? _weights;
    private readonly BpeTokenizer? _tokenizer;
    private readonly ChatTemplate? _chatTemplate;

    public LongContextDivergenceTest()
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

    /// <summary>
    /// Generate with prompts of increasing length to find the threshold where output degrades.
    /// Uses temp=0 (greedy) for determinism.
    /// </summary>
    [Fact]
    public void FindDivergenceThreshold()
    {
        if (!Ready) return;

        var sb = new StringBuilder();

        // Increasing prompt lengths
        var prompts = new[]
        {
            ("Short ~50 tok",
             "Return ONLY valid JSON: {\"a\":\"string\",\"b\":\"string\"}",
             "Revenue: $32,980. Orders: 78."),

            ("Medium ~100 tok",
             "Return ONLY valid JSON: {\"summary\":\"string\",\"metrics\":[{\"label\":\"string\",\"value\":\"string\",\"trend\":\"string\"}]}",
             "Revenue: $32,980. Orders: 78. Top Product: Virtual Desk (79 sold, $30,929). Top Customer: Candace Hill ($2,500)."),

            ("Long ~200 tok",
             "You are a business analytics expert.\n\nReturn ONLY a valid JSON object with this schema:\n" +
             "{\"executiveSummary\": \"string\", \"keyMetrics\": [{\"label\": \"string\", \"value\": \"string\", \"trend\": \"up|down|neutral\"}], " +
             "\"insights\": [{\"title\": \"string\", \"description\": \"string\", \"severity\": \"success|warning|info\"}]}\n\n" +
             "Include 3-6 metrics, 2-4 insights. Reference actual numbers. Return ONLY JSON.",
             "Report: Sales Report\n\nPeriod: 2026-02-26 to 2026-03-27\nRevenue: $32,980.76\nOrders: 78\nAvg Order Value: $422.83\n\n" +
             "Top Products:\n- Virtual Desk: 79 sold, $30,929.00\n- Large Office Subscription: 4 sold, $1,896.76\n\n" +
             "Top Customers:\n- Candace Hill: 4 orders, $2,500.00\n- Test Customer: 9 orders, $1,946.76"),

            ("Full CRM ~350 tok",
             "You are a business analytics expert. Given CRM report data, produce a concise executive analysis.\n\n" +
             "Return ONLY a valid JSON object matching this exact schema. Do not include any explanation or markdown formatting.\n\n" +
             "Schema:\n{\"executiveSummary\": \"2-3 sentence overview\", \"keyMetrics\": [{\"label\": \"string\", \"value\": \"string\", \"trend\": \"up|down|neutral\"}], " +
             "\"insights\": [{\"title\": \"string\", \"description\": \"string\", \"severity\": \"success|warning|info\"}], " +
             "\"recommendations\": [{\"action\": \"string\", \"impact\": \"high|medium|low\", \"description\": \"string\"}]}\n\n" +
             "Rules:\n- Include 3-6 key metrics, 3-5 insights, and 2-4 recommendations\n- Be specific and reference actual numbers\n- Return ONLY the JSON object, nothing else",
             "Report: Sales Report\n\nPeriod: 2026-02-26 to 2026-03-27\nRevenue: $32,980.76\nOrders: 78\nAvg Order Value: $422.83\n\n" +
             "Top Products:\n- Virtual Desk: 79 sold, $30,929.00\n- Large Office Subscription: 4 sold, $1,896.76\n- Conference Room Hourly: 7 sold, $50.00\n\n" +
             "Top Customers:\n- Candace Hill: 4 orders, $2,500.00\n- Sanes & Larkin: 1 orders, $2,000.00\n- Test Customer: 9 orders, $1,946.76\n" +
             "- Tiffany Fairdosi: 2 orders, $1,800.00\n- Jahan Brooks: 1 orders, $1,800.00"),
        };

        foreach (var (name, system, user) in prompts)
        {
            var kvCache = new KvCache(_backend!, _config!, maxSeqLen: 4096);
            var deltaState = new DeltaNetState(_backend!, _config!, _weights!);
            var forward = new ForwardPass(_backend!, _config!, _weights!, kvCache, deltaState);
            var renderer = new ChatTemplateRenderer(_chatTemplate!);

            // Build ChatML prompt
            var prompt = $"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n";
            var tokenIds = _tokenizer!.Encode(prompt);

            // Prefill
            for (int i = 0; i < tokenIds.Length - 1; i++)
                forward.ForwardHidden(tokenIds[i], i);
            var logits = forward.Forward(tokenIds[^1], tokenIds.Length - 1);

            // Generate 64 tokens greedy
            var output = new StringBuilder();
            var sampler = new Sampler();
            int pos = tokenIds.Length;
            for (int t = 0; t < 64; t++)
            {
                int tok = sampler.Sample(logits, new GenerationParams { Temperature = 0 }, []);
                string text = _tokenizer.Decode([tok]);
                output.Append(text);
                if (tok == _tokenizer.Vocabulary.EosTokenId) break;
                logits = forward.Forward(tok, pos++);
            }

            var result = output.ToString();
            bool validJson = false;
            try
            {
                var stripped = result;
                var thinkEnd = stripped.LastIndexOf("</think>");
                if (thinkEnd >= 0) stripped = stripped[(thinkEnd + 8)..];
                stripped = stripped.Trim();
                if (stripped.Contains("{"))
                {
                    var json = stripped[stripped.IndexOf('{')..(stripped.LastIndexOf('}') + 1)];
                    System.Text.Json.JsonDocument.Parse(json);
                    validJson = true;
                }
            }
            catch { }

            sb.AppendLine($"=== {name} ({tokenIds.Length} prompt tokens) ===");
            sb.AppendLine($"Valid JSON: {validJson}");
            sb.AppendLine($"Output: {result[..Math.Min(300, result.Length)]}");
            sb.AppendLine();

            // Cleanup
            kvCache.Dispose();
            deltaState.Dispose();
            forward.Dispose();
        }

        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-divergence-threshold.txt", sb.ToString());

        // At minimum, the short and medium prompts should produce valid JSON
        Assert.Contains("Short ~50 tok", sb.ToString());
        Assert.Fail(sb.ToString()); // Dump everything
    }

    public void Dispose()
    {
        _weights?.Dispose();
        _backend?.Dispose();
    }
}
