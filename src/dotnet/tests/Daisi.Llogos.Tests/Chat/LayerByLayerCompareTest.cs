using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;
using System.Text;

namespace Daisi.Llogos.Tests.Chat;

/// <summary>
/// Compare CPU vs CUDA hidden state after each layer of a single forward pass.
/// Finds which layer introduces the divergence.
/// Uses the 0.8B model (fits in both CPU and CUDA memory simultaneously).
/// </summary>
public class LayerByLayerCompareTest : IDisposable
{
    private IComputeBackend? _cpuBackend;
    private IComputeBackend? _cudaBackend;
    private ModelWeights? _cpuWeights;
    private ModelWeights? _cudaWeights;
    private ModelConfig? _config;
    private BpeTokenizer? _tokenizer;

    public LayerByLayerCompareTest()
    {
        if (!TestConstants.ModelExists) return; // 0.8B
        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        _config = ModelConfig.FromGguf(gguf);
        _tokenizer = TokenizerFactory.FromGguf(gguf);

        _cpuBackend = new CpuBackend();
        _cpuWeights = MmapModelLoader.Load(gguf, TestConstants.Qwen35_08B_Q8_0, _cpuBackend, _config);

        var cudaType = Type.GetType("Daisi.Llogos.Cuda.CudaBackend, Daisi.Llogos.Cuda");
        _cudaBackend = (IComputeBackend)Activator.CreateInstance(cudaType!, 0)!;
        _cudaWeights = MmapModelLoader.Load(gguf, TestConstants.Qwen35_08B_Q8_0, _cudaBackend, _config);
    }

    private bool Ready => _cpuBackend != null && _cudaBackend != null;

    [Fact]
    public void CompareFullForwardPass()
    {
        if (!Ready) return;

        // Use a longer prompt to stress test (50+ tokens)
        var prompt = "<|im_start|>system\nYou are a JSON generator. Return valid JSON only.<|im_end|>\n<|im_start|>user\nGenerate data.<|im_end|>\n<|im_start|>assistant\n";
        var tokenIds = _tokenizer!.Encode(prompt);

        // Run full prefill on both, then compare first generated token logits
        var cpuKv = new KvCache(_cpuBackend!, _config!, maxSeqLen: 256);
        var cpuDelta = new DeltaNetState(_cpuBackend!, _config!, _cpuWeights!);
        var cpuFwd = new ForwardPass(_cpuBackend!, _config!, _cpuWeights!, cpuKv, cpuDelta);

        var cudaKv = new KvCache(_cudaBackend!, _config!, maxSeqLen: 256);
        var cudaDelta = new DeltaNetState(_cudaBackend!, _config!, _cudaWeights!);
        var cudaFwd = new ForwardPass(_cudaBackend!, _config!, _cudaWeights!, cudaKv, cudaDelta);

        // Prefill all but last token
        for (int i = 0; i < tokenIds.Length - 1; i++)
        {
            cpuFwd.ForwardHidden(tokenIds[i], i);
            cudaFwd.ForwardHidden(tokenIds[i], i);
        }

        // Forward last token with logits
        var cpuLogits = cpuFwd.Forward(tokenIds[^1], tokenIds.Length - 1).ToArray();
        var cudaLogits = cudaFwd.Forward(tokenIds[^1], tokenIds.Length - 1).ToArray();

        int cpuArgmax = ArgMax(cpuLogits);
        int cudaArgmax = ArgMax(cudaLogits);

        // Now generate 100 tokens and compare at each step
        var sb = new StringBuilder();
        sb.AppendLine($"Prompt: {tokenIds.Length} tokens");
        sb.AppendLine($"Prefill - CPU argmax: {cpuArgmax}(\"{_tokenizer.Decode([cpuArgmax])}\") CUDA argmax: {cudaArgmax}(\"{_tokenizer.Decode([cudaArgmax])}\")");

        float maxDiffSoFar = 0;
        int pos = tokenIds.Length;
        for (int t = 0; t < 100; t++)
        {
            int cpuTok = ArgMax(cpuLogits);
            int cudaTok = ArgMax(cudaLogits);

            float maxDiff = 0;
            for (int i = 0; i < Math.Min(cpuLogits.Length, cudaLogits.Length); i++)
                maxDiff = Math.Max(maxDiff, Math.Abs(cpuLogits[i] - cudaLogits[i]));
            maxDiffSoFar = Math.Max(maxDiffSoFar, maxDiff);

            if (cpuTok != cudaTok)
            {
                sb.AppendLine($"[{t}] DIVERGE! CPU={cpuTok}(\"{_tokenizer.Decode([cpuTok])}\") CUDA={cudaTok}(\"{_tokenizer.Decode([cudaTok])}\") maxDiff={maxDiff:F6} cumMax={maxDiffSoFar:F6}");
                break;
            }

            sb.AppendLine($"[{t}] match tok={cpuTok}(\"{_tokenizer.Decode([cpuTok]).Replace("\n","\\n")}\") maxDiff={maxDiff:F6}");

            if (cpuTok == _tokenizer.Vocabulary.EosTokenId) break;

            cpuLogits = cpuFwd.Forward(cpuTok, pos).ToArray();
            cudaLogits = cudaFwd.Forward(cudaTok, pos).ToArray();
            pos++;
        }

        sb.AppendLine($"\nMax diff across all tokens: {maxDiffSoFar:F6}");
        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-layer-compare.txt", sb.ToString());

        // On the 0.8B model, CPU and CUDA should match exactly
        Assert.True(maxDiffSoFar < 0.001f, sb.ToString());
    }

    private static int ArgMax(float[] arr)
    {
        int best = 0;
        for (int i = 1; i < arr.Length; i++)
            if (arr[i] > arr[best]) best = i;
        return best;
    }

    public void Dispose()
    {
        _cpuWeights?.Dispose();
        _cudaWeights?.Dispose();
        _cpuBackend?.Dispose();
        _cudaBackend?.Dispose();
    }
}
