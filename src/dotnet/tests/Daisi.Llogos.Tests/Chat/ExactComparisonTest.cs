using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;
using System.Text;

namespace Daisi.Llogos.Tests.Chat;

/// <summary>
/// Token-by-token comparison with llama.cpp using the exact same prompt file.
/// Both use temp=0 greedy, same model, same prompt bytes.
/// </summary>
public class ExactComparisonTest : IDisposable
{
    private readonly IComputeBackend? _backend;
    private readonly ModelConfig? _config;
    private readonly ModelWeights? _weights;
    private readonly BpeTokenizer? _tokenizer;

    public ExactComparisonTest()
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
    public void CompareWithLlamaCpp()
    {
        if (!Ready) return;

        // Read the exact same prompt file used for llama.cpp
        var prompt = File.ReadAllText(@"C:\repos\daisinet-qwen-integration-crm\test-prompt.txt");
        var tokenIds = _tokenizer!.Encode(prompt);

        var sb = new StringBuilder();
        sb.AppendLine($"Prompt: {tokenIds.Length} tokens");
        sb.AppendLine($"First 10 token IDs: [{string.Join(", ", tokenIds.Take(10))}]");
        sb.AppendLine($"Last 5 token IDs: [{string.Join(", ", tokenIds.TakeLast(5))}]");
        sb.AppendLine();

        // Prefill
        var kvCache = new KvCache(_backend!, _config!, maxSeqLen: 4096);
        var deltaState = new DeltaNetState(_backend!, _config!, _weights!);
        var forward = new ForwardPass(_backend!, _config!, _weights!, kvCache, deltaState);

        for (int i = 0; i < tokenIds.Length - 1; i++)
            forward.ForwardHidden(tokenIds[i], i);
        var logits = forward.Forward(tokenIds[^1], tokenIds.Length - 1);

        // Generate 512 tokens greedy, logging each token
        var sampler = new Sampler();
        int pos = tokenIds.Length;
        var output = new StringBuilder();
        var tokenLog = new StringBuilder();

        for (int t = 0; t < 512; t++)
        {
            int tok = sampler.Sample(logits, new GenerationParams { Temperature = 0 }, []);
            string text = _tokenizer.Decode([tok]);
            output.Append(text);
            tokenLog.AppendLine($"[{t}] id={tok} \"{text.Replace("\n","\\n").Replace("\r","\\r")}\"");

            if (tok == _tokenizer.Vocabulary.EosTokenId) break;
            logits = forward.Forward(tok, pos++);
        }

        sb.AppendLine("=== LLogos Output ===");
        sb.AppendLine(output.ToString());
        sb.AppendLine();
        sb.AppendLine("=== Token Log (first 50) ===");
        var logLines = tokenLog.ToString().Split('\n');
        for (int i = 0; i < Math.Min(50, logLines.Length); i++)
            sb.AppendLine(logLines[i]);

        // Read llama.cpp output for comparison
        var llamaOutput = File.Exists(@"C:\repos\daisinet-qwen-integration-crm\test-llamacpp-output.txt")
            ? File.ReadAllText(@"C:\repos\daisinet-qwen-integration-crm\test-llamacpp-output.txt")
            : "(not available)";

        // Extract just the generated part (after the prompt echo)
        var llamaGen = llamaOutput;
        int assistantIdx = llamaGen.LastIndexOf("assistant\n");
        if (assistantIdx >= 0) llamaGen = llamaGen[(assistantIdx + 10)..].TrimStart();
        // Strip ANSI codes
        llamaGen = System.Text.RegularExpressions.Regex.Replace(llamaGen, @"\x1B\[[0-9;]*m", "");
        llamaGen = llamaGen.Trim();

        sb.AppendLine();
        sb.AppendLine("=== llama.cpp Output (first 500 chars) ===");
        sb.AppendLine(llamaGen[..Math.Min(500, llamaGen.Length)]);

        // Find first divergence
        var llogosText = output.ToString();
        int minLen = Math.Min(llogosText.Length, llamaGen.Length);
        int divergeAt = -1;
        for (int i = 0; i < minLen; i++)
        {
            if (llogosText[i] != llamaGen[i]) { divergeAt = i; break; }
        }

        sb.AppendLine();
        if (divergeAt >= 0)
            sb.AppendLine($"DIVERGES at char {divergeAt}: LLogos='{llogosText[divergeAt]}' llama='{llamaGen[divergeAt]}'");
        else if (llogosText.Length != llamaGen.Length)
            sb.AppendLine($"Same content up to {minLen} chars, but lengths differ: LLogos={llogosText.Length} llama={llamaGen.Length}");
        else
            sb.AppendLine("IDENTICAL OUTPUT!");

        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-exact-comparison.txt", sb.ToString());
        Assert.Fail(sb.ToString());
    }

    public void Dispose()
    {
        _weights?.Dispose();
        _backend?.Dispose();
    }
}
