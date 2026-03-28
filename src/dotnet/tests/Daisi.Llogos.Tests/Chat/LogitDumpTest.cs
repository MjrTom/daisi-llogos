using Daisi.Llogos.Chat;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;
using System.Text;

namespace Daisi.Llogos.Tests.Chat;

public class LogitDumpTest : IDisposable
{
    private readonly IComputeBackend? _backend;
    private readonly ModelConfig? _config;
    private readonly ModelWeights? _weights;
    private readonly BpeTokenizer? _tokenizer;
    private readonly ChatTemplate? _chatTemplate;

    public LogitDumpTest()
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
    /// Dump top-10 logits at each generated token to find where "..." gets chosen.
    /// </summary>
    [Fact]
    public void DumpTopLogitsPerToken()
    {
        if (!Ready) return;

        var kvCache = new KvCache(_backend!, _config!, maxSeqLen: 4096);
        var deltaState = new DeltaNetState(_backend!, _config!, _weights!);
        var forward = new ForwardPass(_backend!, _config!, _weights!, kvCache, deltaState);

        // Build the same ChatML prompt
        var prompt = "<|im_start|>system\nReturn ONLY valid JSON. No thinking, no explanation.\n" +
            "{\"executiveSummary\":\"string\",\"keyMetrics\":[{\"label\":\"string\",\"value\":\"string\",\"trend\":\"string\"}]}<|im_end|>\n" +
            "<|im_start|>user\nRevenue: $32,980. Orders: 78. Top Product: Virtual Desk.<|im_end|>\n" +
            "<|im_start|>assistant\n";

        var tokenIds = _tokenizer!.Encode(prompt);
        var sb = new StringBuilder();
        sb.AppendLine($"Prompt tokens: {tokenIds.Length}");

        // Prefill
        for (int i = 0; i < tokenIds.Length - 1; i++)
            forward.ForwardHidden(tokenIds[i], i);

        var logits = forward.Forward(tokenIds[^1], tokenIds.Length - 1);
        int position = tokenIds.Length;

        // Generate with greedy, logging top-10 logits
        var sampler = new Sampler(seed: null);
        for (int t = 0; t < 80; t++)
        {
            // Get top-10 tokens by logit value
            var top = new (int id, float logit, string text)[10];
            for (int i = 0; i < 10; i++) top[i] = (-1, float.NegativeInfinity, "");

            for (int i = 0; i < logits.Length; i++)
            {
                if (logits[i] > top[9].logit)
                {
                    top[9] = (i, logits[i], _tokenizer.Decode([i]));
                    Array.Sort(top, (a, b) => b.logit.CompareTo(a.logit));
                }
            }

            int chosen = top[0].id;
            string chosenText = top[0].text;

            sb.AppendLine($"[{t}] pos={position} chosen={chosen} \"{chosenText.Replace("\n","\\n")}\" logit={top[0].logit:F2}");
            for (int i = 0; i < 5; i++)
                sb.AppendLine($"    #{i+1}: {top[i].id} \"{top[i].text.Replace("\n","\\n")}\" = {top[i].logit:F2}");

            // Check for EOS
            if (chosen == _tokenizer.Vocabulary.EosTokenId) break;

            // Next token
            logits = forward.Forward(chosen, position);
            position++;
        }

        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-logit-dump.txt", sb.ToString());
        Assert.True(true); // Just dump, no assertion
    }

    public void Dispose()
    {
        _weights?.Dispose();
        _backend?.Dispose();
    }
}
