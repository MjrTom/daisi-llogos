using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;
using Daisi.Llogos.Inference;

namespace Daisi.Llogos.Tests.Chat;

/// <summary>
/// Compare individual operations between CPU and CUDA to find which one diverges.
/// </summary>
public class OperationCompareTest : IDisposable
{
    private IComputeBackend? _cpuBackend;
    private IComputeBackend? _cudaBackend;
    private ModelWeights? _cpuWeights;
    private ModelWeights? _cudaWeights;
    private ModelConfig? _config;

    public OperationCompareTest()
    {
        if (!TestConstants.Model9BExists) return;
        using var stream = File.OpenRead(TestConstants.Qwen35_9B_Q8_0);
        var gguf = GgufFile.Read(stream);
        _config = ModelConfig.FromGguf(gguf);

        _cpuBackend = new CpuBackend();
        _cpuWeights = MmapModelLoader.Load(gguf, TestConstants.Qwen35_9B_Q8_0, _cpuBackend, _config);

        var cudaType = Type.GetType("Daisi.Llogos.Cuda.CudaBackend, Daisi.Llogos.Cuda");
        _cudaBackend = (IComputeBackend)Activator.CreateInstance(cudaType!, 0)!;
        _cudaWeights = MmapModelLoader.Load(gguf, TestConstants.Qwen35_9B_Q8_0, _cudaBackend, _config);
    }

    private bool Ready => _cpuBackend != null && _cudaBackend != null;

    [Fact]
    public void CompareEmbeddingAndFirstRmsNorm()
    {
        if (!Ready) return;

        int tokenId = 248045; // <|im_start|>
        int hiddenDim = _config!.HiddenDim;

        // CPU: embedding + RmsNorm
        var cpuHidden = _cpuBackend!.CreateTensor("cpu_h", GgmlType.F32, [hiddenDim]);
        var cpuNorm = _cpuBackend.CreateTensor("cpu_n", GgmlType.F32, [hiddenDim]);
        _cpuBackend.EmbeddingLookup(cpuHidden, _cpuWeights!.TokenEmbedding, tokenId);
        _cpuBackend.RmsNorm(cpuNorm, cpuHidden, _cpuWeights.Layers[0].AttnNorm, _config.NormEps);
        var cpuH = new float[hiddenDim];
        var cpuN = new float[hiddenDim];
        cpuHidden.AsFloatSpan().CopyTo(cpuH);
        cpuNorm.AsFloatSpan().CopyTo(cpuN);

        // CUDA: embedding + RmsNorm
        var cudaHidden = _cudaBackend!.CreateTensor("cuda_h", GgmlType.F32, [hiddenDim]);
        var cudaNorm = _cudaBackend.CreateTensor("cuda_n", GgmlType.F32, [hiddenDim]);
        _cudaBackend.EmbeddingLookup(cudaHidden, _cudaWeights!.TokenEmbedding, tokenId);
        _cudaBackend.FlushCommands();
        _cudaBackend.RmsNorm(cudaNorm, cudaHidden, _cudaWeights.Layers[0].AttnNorm, _config.NormEps);
        _cudaBackend.FlushCommands();
        var cudaH = new float[hiddenDim];
        var cudaN = new float[hiddenDim];
        cudaHidden.DequantizeTo(cudaH);
        cudaNorm.DequantizeTo(cudaN);

        // Compare embeddings
        float maxEmbDiff = 0;
        for (int i = 0; i < hiddenDim; i++)
            maxEmbDiff = Math.Max(maxEmbDiff, Math.Abs(cpuH[i] - cudaH[i]));

        // Compare RmsNorm output
        float maxNormDiff = 0;
        for (int i = 0; i < hiddenDim; i++)
            maxNormDiff = Math.Max(maxNormDiff, Math.Abs(cpuN[i] - cudaN[i]));

        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"HiddenDim: {hiddenDim}");
        sb.AppendLine($"Embedding max diff: {maxEmbDiff:E6}");
        sb.AppendLine($"RmsNorm max diff: {maxNormDiff:E6}");
        sb.AppendLine($"CPU embedding[0..4]: {cpuH[0]:F6}, {cpuH[1]:F6}, {cpuH[2]:F6}, {cpuH[3]:F6}");
        sb.AppendLine($"CUDA embedding[0..4]: {cudaH[0]:F6}, {cudaH[1]:F6}, {cudaH[2]:F6}, {cudaH[3]:F6}");
        sb.AppendLine($"CPU norm[0..4]: {cpuN[0]:F6}, {cpuN[1]:F6}, {cpuN[2]:F6}, {cpuN[3]:F6}");
        sb.AppendLine($"CUDA norm[0..4]: {cudaN[0]:F6}, {cudaN[1]:F6}, {cudaN[2]:F6}, {cudaN[3]:F6}");

        File.WriteAllText(@"C:\repos\daisinet-qwen-integration-crm\test-op-compare.txt", sb.ToString());

        // Embedding should be EXACT (just a lookup)
        Assert.True(maxEmbDiff == 0, $"Embedding differs! Max diff: {maxEmbDiff}\n{sb}");
    }

    public void Dispose()
    {
        _cpuWeights?.Dispose();
        _cudaWeights?.Dispose();
        _cpuBackend?.Dispose();
        _cudaBackend?.Dispose();
    }
}
