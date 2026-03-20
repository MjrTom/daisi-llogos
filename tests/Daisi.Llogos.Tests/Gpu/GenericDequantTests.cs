using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Tests.Gpu;

/// <summary>
/// Tests that models with mixed quantization types (Q4_K_M) work end-to-end.
/// Q4_K_M models use Q4_K for most layers and Q5_K/Q6_K for important layers.
/// </summary>
public class GenericDequantTests
{
    private static readonly string Qwen9BPath = @"C:\GGUFS\Qwen3.5-9B-Q4_K_M.gguf";

    [Fact]
    public void Q4KM_Model_LoadsAndEmbeddingWorks()
    {
        if (!File.Exists(Qwen9BPath)) return;

        using var stream = File.OpenRead(Qwen9BPath);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        using var backend = new CpuBackend();
        using var weights = MmapModelLoader.Load(gguf, Qwen9BPath, backend, config);

        // Verify Q4_K embedding lookup works
        using var outT = backend.CreateTensor("out", GgmlType.F32, [config.HiddenDim]);
        backend.EmbeddingLookup(outT, weights.TokenEmbedding, 100);

        var result = new float[config.HiddenDim];
        outT.DequantizeTo(result);

        Assert.True(result.Any(v => v != 0), "Embedding should produce non-zero values");
        Assert.True(result.All(v => !float.IsNaN(v) && !float.IsInfinity(v)), "No NaN/Inf in embedding");
    }

    [Fact]
    public void Q4KM_Model_MatMulWithMixedTypes()
    {
        if (!File.Exists(Qwen9BPath)) return;

        using var stream = File.OpenRead(Qwen9BPath);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);

        using var backend = new CpuBackend();
        using var weights = MmapModelLoader.Load(gguf, Qwen9BPath, backend, config);

        // Test MatMul with a layer weight (likely Q4_K or Q5_K)
        var layer0 = weights.Layers[0];

        // Get the FFN gate weight — its type will be Q4_K or Q5_K
        var gateWeight = layer0.FfnGate;
        int K = (int)gateWeight.Dimensions[0];
        int N = (int)gateWeight.Dimensions[1];

        // Create a dummy input
        using var input = backend.CreateTensor("in", GgmlType.F32, [K]);
        using var output = backend.CreateTensor("out", GgmlType.F32, [N]);

        var inputData = new float[K];
        Array.Fill(inputData, 0.01f);
        var inputBytes = new byte[K * 4];
        Buffer.BlockCopy(inputData, 0, inputBytes, 0, inputBytes.Length);
        input.CopyFrom(inputBytes);

        // Should not throw — generic dequant fallback must handle Q4_K/Q5_K/Q6_K
        backend.MatMul(output, input, gateWeight, 1, K, N);

        var result = new float[N];
        output.DequantizeTo(result);

        Assert.True(result.Any(v => v != 0), $"MatMul with {gateWeight.Type} should produce non-zero values");
        Assert.True(result.All(v => !float.IsNaN(v) && !float.IsInfinity(v)),
            $"No NaN/Inf in MatMul result for {gateWeight.Type}");
    }
}
