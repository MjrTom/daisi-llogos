using Daisi.Llogos.Cuda;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Tests.Inference;

/// <summary>
/// Test pipeline.Forward() output for a single token on 27B.
/// Checks if the argmax is a reasonable token and if logits look sane.
/// </summary>
public class PipelineSingleTokenTest
{
    [Fact]
    public void SingleToken_27B_SaneOutput()
    {
        if (!TestConstants.Model27BShardsExist)
        {
            Console.Error.WriteLine("Skipping: 27B shards not found.");
            return;
        }
        try { using var t = new CudaBackend(); } catch { Console.Error.WriteLine("Skipping: no CUDA."); return; }

        var shardDir = TestConstants.Qwen35_27B_Shards;
        var baseName = ShardModelLoader.FindShardBaseName(shardDir);
        var headerPath = Path.Combine(shardDir, $"{baseName}.header");
        using var hs = File.OpenRead(headerPath);
        var gguf = GgufFile.Read(hs);
        var config = ModelConfig.FromGguf(gguf);

        using var cuda = new CudaBackend();
        using var pipeline = PipelinedForwardPass.Create(gguf, shardDir, config, cuda, maxContext: 128);

        // Run two tokens
        pipeline.Forward(tokenId: 42, position: 0);
        var logits = pipeline.Forward(tokenId: 100, position: 1).ToArray();

        // Check logit sanity
        float maxLogit = float.MinValue;
        float minLogit = float.MaxValue;
        int argmax = 0;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > maxLogit) { maxLogit = logits[i]; argmax = i; }
            if (logits[i] < minLogit) minLogit = logits[i];
        }

        float mean = logits.Average();
        Console.Error.WriteLine($"Token 100@pos1 forward: argmax={argmax}, max={maxLogit:F4}, min={minLogit:F4}, mean={mean:F4}, vocab={logits.Length}");
        Console.Error.WriteLine($"First 10 logits: [{string.Join(", ", logits.Take(10).Select(v => v.ToString("F4")))}]");

        // Sanity checks: logits should have reasonable range
        Assert.True(maxLogit > -100 && maxLogit < 100, $"Max logit {maxLogit} out of reasonable range");
        Assert.True(!float.IsNaN(maxLogit) && !float.IsInfinity(maxLogit), "Logits contain NaN/Inf");
    }
}
