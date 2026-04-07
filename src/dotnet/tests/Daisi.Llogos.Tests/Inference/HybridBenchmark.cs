using System.Diagnostics;
using Daisi.Llogos.Cpu;
using Daisi.Llogos.Cuda;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Tests.Inference;

/// <summary>
/// Benchmarks hybrid GPU+CPU inference on models too large (or borderline) for VRAM.
/// Sweeps GPU layer counts to find the optimal split for maximum tok/s.
/// </summary>
public class HybridBenchmark
{
    /// <summary>
    /// Sweep GPU layer counts on the 27B model to find the speed curve.
    /// Tests: all-GPU (64), majority-GPU, half, minority-GPU, and all-CPU (0).
    /// </summary>
    [Theory]
    [InlineData(64)]  // all GPU (baseline — may OOM on long context)
    [InlineData(56)]  // 56 GPU + 8 CPU
    [InlineData(48)]  // 48 GPU + 16 CPU
    [InlineData(32)]  // half and half
    [InlineData(16)]  // 16 GPU + 48 CPU
    [InlineData(0)]   // all CPU (baseline)
    public void Hybrid_27B_Sweep(int gpuLayers)
    {
        string modelPath = TestConstants.Qwen35_27B_Q4_0;
        if (!File.Exists(modelPath))
        {
            Console.Error.WriteLine("Skipping: 27B model not found.");
            return;
        }

        bool useCuda = gpuLayers > 0;
        if (useCuda)
        {
            try { using var t = new CudaBackend(); }
            catch { Console.Error.WriteLine("Skipping: CUDA not available."); return; }
        }

        int maxGenerate = 20;
        int maxContext = 256;

        // Load model
        using var stream = File.OpenRead(modelPath);
        var gguf = GgufFile.Read(stream);
        var config = ModelConfig.FromGguf(gguf);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        Console.Error.WriteLine($"\n=== Hybrid {gpuLayers} GPU + {config.NumLayers - gpuLayers} CPU | {config.Architecture} {config.NumLayers}L ===");

        var loadSw = Stopwatch.StartNew();
        IForwardPass forward;
        IComputeBackend primaryBackend;

        if (gpuLayers > 0 && gpuLayers < config.NumLayers)
        {
            // Hybrid: GPU + CPU
            var cuda = new CudaBackend();
            primaryBackend = cuda;
            forward = HybridForwardPass.Create(gguf, modelPath, config, cuda, gpuLayers);
        }
        else if (gpuLayers >= config.NumLayers)
        {
            // All GPU
            var cuda = new CudaBackend();
            primaryBackend = cuda;
            var weights = MmapModelLoader.Load(gguf, modelPath, cuda, config);
            var kvCache = new KvCache(cuda, config, maxSeqLen: maxContext);
            var deltaState = new DeltaNetState(cuda, config, weights);
            forward = new ForwardPass(cuda, config, weights, kvCache, deltaState);
        }
        else
        {
            // All CPU
            var cpu = new CpuBackend();
            primaryBackend = cpu;
            var weights = MmapModelLoader.Load(gguf, modelPath, cpu, config);
            var kvCache = new KvCache(cpu, config, maxSeqLen: maxContext);
            var deltaState = new DeltaNetState(cpu, config, weights);
            forward = new ForwardPass(cpu, config, weights, kvCache, deltaState);
        }

        Console.Error.WriteLine($"Loaded in {loadSw.Elapsed.TotalSeconds:F1}s");

        // Tokenize
        var prompt = "Explain how distributed machine learning inference works across multiple nodes. The key concepts are";
        var promptTokens = tokenizer.Encode(prompt);

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
        Console.Error.WriteLine($"  Config:  {gpuLayers} GPU + {config.NumLayers - gpuLayers} CPU layers");

        forward.Dispose();
        primaryBackend.Dispose();
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
