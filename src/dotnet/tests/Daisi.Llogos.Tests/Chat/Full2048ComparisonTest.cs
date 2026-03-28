using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;
using System.Text;

namespace Daisi.Llogos.Tests.Chat;

public class Full2048ComparisonTest : IDisposable
{
    private readonly IComputeBackend? _backend;
    private readonly ModelConfig? _config;
    private readonly ModelWeights? _weights;
    private readonly BpeTokenizer? _tokenizer;

    public Full2048ComparisonTest()
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

    [Fact]
    public void Generate2048Tokens()
    {
        if (!Ready) return;

        var prompt = File.ReadAllText(@"C:\repos\daisinet-qwen-integration-crm\test-prompt.txt");
        var tokenIds = _tokenizer!.Encode(prompt);

        var kvCache = new KvCache(_backend!, _config!, maxSeqLen: 4096);
        var deltaState = new DeltaNetState(_backend!, _config!, _weights!);
        var forward = new ForwardPass(_backend!, _config!, _weights!, kvCache, deltaState);

        // Prefill
        for (int i = 0; i < tokenIds.Length - 1; i++)
            forward.ForwardHidden(tokenIds[i], i);
        var logits = forward.Forward(tokenIds[^1], tokenIds.Length - 1);

        // Generate 2048 tokens greedy
        var sampler = new Sampler();
        int pos = tokenIds.Length;
        var output = new StringBuilder();
        var tokenLog = new List<(int id, string text)>();

        for (int t = 0; t < 2048; t++)
        {
            int tok = sampler.Sample(logits, new GenerationParams { Temperature = 0 }, []);
            string text = _tokenizer.Decode([tok]);
            output.Append(text);
            tokenLog.Add((tok, text));
            if (tok == _tokenizer.Vocabulary.EosTokenId) break;
            logits = forward.Forward(tok, pos++);
        }

        // Write output
        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-llogos-2048.txt", output.ToString());

        // Write token log
        var logSb = new StringBuilder();
        for (int i = 0; i < tokenLog.Count; i++)
            logSb.AppendLine($"[{i}] id={tokenLog[i].id} \"{tokenLog[i].text.Replace("\n","\\n")}\"");
        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-llogos-2048-tokens.txt", logSb.ToString());

        Assert.True(tokenLog.Count > 100, $"Only generated {tokenLog.Count} tokens");
    }

    public void Dispose()
    {
        _weights?.Dispose();
        _backend?.Dispose();
    }
}
