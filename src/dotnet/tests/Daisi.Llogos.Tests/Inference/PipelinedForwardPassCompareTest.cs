using Daisi.Llogos.Cuda;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Tests.Inference;

/// <summary>
/// Compares CUDA PipelinedForwardPass output against standard CUDA forward pass.
/// Uses the 0.8B model (Q8_0) which fits entirely in VRAM for both paths.
/// </summary>
public class PipelinedForwardPassCompareTest : IDisposable
{
    private readonly string? _shardDir;

    public PipelinedForwardPassCompareTest()
    {
        if (!TestConstants.ModelExists) return;
        // Split the 0.8B model into shards for the pipeline path
        _shardDir = Path.Combine(Path.GetTempPath(), "llogos-pipeline-test-" + Guid.NewGuid().ToString("N")[..8]);
        GgufSplitter.Split(TestConstants.Qwen35_08B_Q8_0, _shardDir);
    }

    [Fact]
    public void Pipeline_08B_MatchesStandard_CUDA() => RunComparison(_shardDir!, false);

    [Fact]
    public void Pipeline_08B_GpuAligned_MatchesStandard_CUDA()
    {
        if (!TestConstants.ModelExists)
        {
            Console.Error.WriteLine("Skipping: 0.8B model not found.");
            return;
        }
        // Split with GPU alignment
        var alignedDir = Path.Combine(Path.GetTempPath(), "llogos-pipeline-aligned-" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            GgufSplitter.Split(TestConstants.Qwen35_08B_Q8_0, alignedDir, gpuAligned: true);
            RunComparison(alignedDir, true);
        }
        finally
        {
            try { Directory.Delete(alignedDir, true); } catch { }
        }
    }

    private void RunComparison(string shardDir, bool expectAligned)
    {
        if (!TestConstants.ModelExists)
        {
            Console.Error.WriteLine("Skipping: 0.8B model not found.");
            return;
        }
        Console.Error.WriteLine($"Testing with gpuAligned={expectAligned}");

        try { using var t = new CudaBackend(); }
        catch { Console.Error.WriteLine("Skipping: CUDA not available."); return; }

        using var stream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        int tokenId = 42;

        // Standard CUDA forward pass
        using var cuda1 = new CudaBackend();
        var weights = MmapModelLoader.Load(gguf, TestConstants.Qwen35_08B_Q8_0, cuda1, config);
        var kvCache = new KvCache(cuda1, config, maxSeqLen: 128);
        var deltaState = new DeltaNetState(cuda1, config, weights);
        using var stdForward = new ForwardPass(cuda1, config, weights, kvCache, deltaState);

        // Pipeline CUDA forward pass
        var headerPath = Path.Combine(shardDir, Path.GetFileName(TestConstants.Qwen35_08B_Q8_0) + ".header");
        using var headerStream = File.OpenRead(headerPath);
        var shardGguf = GgufFile.Read(headerStream);
        using var cuda2 = new CudaBackend();
        using var pipForward = PipelinedForwardPass.Create(shardGguf, shardDir, config, cuda2, maxContext: 128);

        // Generate 10 tokens and compare each
        float[] standardLogits = stdForward.Forward(tokenId, position: 0).ToArray();
        float[] pipelineLogits = pipForward.Forward(tokenId, position: 0).ToArray();

        for (int g = 0; g < 10; g++)
        {
            int sa = ArgMax(standardLogits);
            int pa = ArgMax(pipelineLogits);
            Console.Error.WriteLine($"Token {g} - Standard argmax: {sa} Pipeline argmax: {pa} match={sa == pa}");
            Assert.Equal(sa, pa);

            standardLogits = stdForward.Forward(sa, position: g + 1).ToArray();
            pipelineLogits = pipForward.Forward(pa, position: g + 1).ToArray();
        }

        kvCache.Dispose(); deltaState.Dispose(); weights.Dispose();
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
        if (_shardDir != null)
        {
            try { Directory.Delete(_shardDir, true); } catch { }
        }
    }
}
