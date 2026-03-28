using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;
using System.Text;

namespace Daisi.Llogos.Tests.Chat;

public class PrecisionDiagnosticTest : IDisposable
{
    private readonly IComputeBackend? _backend;
    private readonly ModelConfig? _config;
    private readonly ModelWeights? _weights;
    private readonly BpeTokenizer? _tokenizer;

    public PrecisionDiagnosticTest()
    {
        if (!TestConstants.Model9BExists) return;
        using var stream = File.OpenRead(TestConstants.Qwen35_9B_Q8_0);
        var gguf = GgufFile.Read(stream);
        _config = ModelConfig.FromGguf(gguf);
        _backend = (IComputeBackend)Activator.CreateInstance(
            Type.GetType("Daisi.Llogos.Cuda.CudaBackend, Daisi.Llogos.Cuda")!, 0)!;
        _weights = MmapModelLoader.Load(gguf, TestConstants.Qwen35_9B_Q8_0, _backend, _config);
        _tokenizer = TokenizerFactory.FromGguf(gguf);
    }

    private bool Ready => _backend != null;

    /// <summary>
    /// Dump max logit value and logit sum at every position through prefill and first 200 gen tokens.
    /// Compare these against llama.cpp's --logits-all output to find where drift starts.
    /// </summary>
    [Fact]
    public void DumpLogitStats()
    {
        if (!Ready) return;

        var prompt = File.ReadAllText(@"C:\repos\daisinet-qwen-integration-crm\test-prompt.txt");
        var tokenIds = _tokenizer!.Encode(prompt);

        var kvCache = new KvCache(_backend!, _config!, maxSeqLen: 4096);
        var deltaState = new DeltaNetState(_backend!, _config!, _weights!);
        var forward = new ForwardPass(_backend!, _config!, _weights!, kvCache, deltaState);

        var sb = new StringBuilder();
        sb.AppendLine($"prompt_tokens={tokenIds.Length}");

        // Prefill - get logits at last prompt token
        for (int i = 0; i < tokenIds.Length - 1; i++)
            forward.ForwardHidden(tokenIds[i], i);
        var logits = forward.Forward(tokenIds[^1], tokenIds.Length - 1);

        // Log prefill result
        float maxLogit = float.NegativeInfinity;
        int maxIdx = 0;
        double logitSum = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            logitSum += logits[i];
            if (logits[i] > maxLogit) { maxLogit = logits[i]; maxIdx = i; }
        }
        sb.AppendLine($"[prefill] maxLogit={maxLogit:F4} maxId={maxIdx}(\"{_tokenizer.Decode([maxIdx])}\") logitSum={logitSum:F4}");

        // Generate 195 tokens
        var sampler = new Sampler();
        int pos = tokenIds.Length;
        for (int t = 0; t < 195; t++)
        {
            int tok = sampler.Sample(logits, new GenerationParams { Temperature = 0 }, []);
            string text = _tokenizer.Decode([tok]);

            // Compute stats for next position
            logits = forward.Forward(tok, pos++);

            maxLogit = float.NegativeInfinity;
            maxIdx = 0;
            logitSum = 0;
            for (int i = 0; i < logits.Length; i++)
            {
                logitSum += logits[i];
                if (logits[i] > maxLogit) { maxLogit = logits[i]; maxIdx = i; }
            }

            sb.AppendLine($"[{t}] tok={tok}(\"{text.Replace("\n","\\n")}\") next_max={maxLogit:F4} next_maxId={maxIdx}(\"{_tokenizer.Decode([maxIdx])}\") sum={logitSum:F2}");
        }

        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-logit-stats.txt", sb.ToString());
        Assert.True(true);
    }

    public void Dispose()
    {
        _weights?.Dispose();
        _backend?.Dispose();
    }
}
