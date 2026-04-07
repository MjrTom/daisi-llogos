using System.Diagnostics;
using Daisi.Llogos.Cuda;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Tests.Inference;

/// <summary>
/// Benchmarks the double-buffered pipelined forward pass on the 27B model.
/// Compares against full CUDA load baseline.
/// </summary>
public class PipelinedBenchmark
{
    [Fact]
    public void Pipelined_27B_Benchmark()
    {
        if (!TestConstants.Model27BShardsExist)
        {
            Console.Error.WriteLine("Skipping: 27B shard directory not found.");
            return;
        }

        try { using var t = new CudaBackend(); }
        catch { Console.Error.WriteLine("Skipping: CUDA not available."); return; }

        var shardDir = TestConstants.Qwen35_27B_Shards;
        int maxGenerate = 30;

        // Parse header
        var headerPath = Path.Combine(shardDir,
            ShardModelLoader.FindShardBaseName(shardDir) + ".header");
        using var headerStream = File.OpenRead(headerPath);
        var gguf = GgufFile.Read(headerStream);
        var config = ModelConfig.FromGguf(gguf);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        Console.Error.WriteLine($"\n=== Pipelined Forward Pass | {config.Architecture} {config.NumLayers}L ===");

        using var cuda = new CudaBackend();

        var loadSw = Stopwatch.StartNew();
        using var forward = PipelinedForwardPass.Create(gguf, shardDir, config, cuda, maxContext: 256);
        Console.Error.WriteLine($"Load: {loadSw.Elapsed.TotalSeconds:F1}s");

        // Tokenize
        var prompt = "Explain how distributed machine learning inference works across multiple nodes. The key concepts are";
        var promptTokens = tokenizer.Encode(prompt);
        Console.Error.WriteLine($"Prompt: {promptTokens.Length} tokens");

        // Warmup
        var warmSw = Stopwatch.StartNew();
        forward.Forward(promptTokens[0], position: 0);
        Console.Error.WriteLine($"Warmup: {warmSw.ElapsedMilliseconds}ms");

        // Prefill
        var prefillSw = Stopwatch.StartNew();
        ReadOnlySpan<float> logitsSpan = default;
        for (int t = 1; t < promptTokens.Length; t++)
            logitsSpan = forward.Forward(promptTokens[t], position: t);
        prefillSw.Stop();
        var logits = logitsSpan.ToArray();

        // Decode
        Console.Error.Write($"  Output: {prompt}");
        var decodeSw = Stopwatch.StartNew();
        int generated = 0;
        int eosToken = tokenizer.Vocabulary.EosTokenId;

        for (int g = 0; g < maxGenerate; g++)
        {
            int next = ArgMax(logits);
            if (next == eosToken) break;
            Console.Error.Write(tokenizer.Decode([next]));
            generated++;
            logits = forward.Forward(next, position: promptTokens.Length + g).ToArray();
        }
        decodeSw.Stop();
        Console.Error.WriteLine();

        double prefillTps = (promptTokens.Length - 1) / prefillSw.Elapsed.TotalSeconds;
        double decodeTps = generated > 0 ? generated / decodeSw.Elapsed.TotalSeconds : 0;
        double msPerToken = generated > 0 ? decodeSw.Elapsed.TotalMilliseconds / generated : 0;

        Console.Error.WriteLine($"  Prefill: {promptTokens.Length - 1} tok in {prefillSw.Elapsed.TotalSeconds:F2}s ({prefillTps:F1} tok/s)");
        Console.Error.WriteLine($"  Decode:  {generated} tok in {decodeSw.Elapsed.TotalSeconds:F2}s ({decodeTps:F2} tok/s, {msPerToken:F0}ms/tok)");

        Assert.True(generated > 0);
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        float bestVal = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] > bestVal) { bestVal = values[i]; best = i; }
        return best;
    }
}
