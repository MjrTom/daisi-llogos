using System.Diagnostics;
using Daisi.Llogos.Cpu;
using Daisi.Llogos.Cuda;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Tests.Inference;

/// <summary>
/// Benchmark: runs the Qwen3.5-27B model layer-by-layer from shard files.
/// Each layer has its own ForwardPass + KV cache, loaded from its shard file.
/// All layers stay mmapped — the OS pages in the active layer and evicts cold ones.
/// This simulates a single machine running a model too large for VRAM by swapping
/// layer weights through memory one at a time.
/// </summary>
public class DaisiChainShardBenchmark
{
    [Fact]
    public void ShardBench_27B_FullPipeline()
    {
        if (!TestConstants.Model27BShardsExist)
        {
            Console.Error.WriteLine("Skipping: 27B shard directory not found.");
            return;
        }

        var shardDir = TestConstants.Qwen35_27B_Shards;
        var baseName = "Qwen3.5-27B-Q4_0.gguf";
        int maxGenerate = 20;

        // ── Parse header ────────────────────────────────────────────────────
        var headerPath = Path.Combine(shardDir, $"{baseName}.header");
        using var headerStream = File.OpenRead(headerPath);
        var gguf = GgufFile.Read(headerStream);
        var config = ModelConfig.FromGguf(gguf);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        Console.Error.WriteLine($"Model: {config.Architecture}, {config.NumLayers} layers, {config.HiddenDim}d, Q4_0");
        Console.Error.WriteLine($"Loading all {config.NumLayers} layer stages from shards...");

        // ── Load all pipeline stages ────────────────────────────────────────
        // Each stage: 1 layer loaded from its shard, with its own backend + KV cache.
        // All stay mmapped — the OS will page in/out as we walk through layers.
        var loadSw = Stopwatch.StartNew();

        var stages = new PipelineStage[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            var backend = new CpuBackend();
            var weights = MmapModelLoader.LoadPartialFromShards(
                gguf, shardDir, backend, config,
                startLayer: i, endLayer: i + 1,
                includeEmbedding: false, includeOutputHead: false);
            var kvCache = new KvCache(backend, config, maxSeqLen: 256,
                startLayer: i, endLayer: i + 1);
            var deltaState = new DeltaNetState(backend, config, weights,
                startLayer: i, endLayer: i + 1);
            var forward = new ForwardPass(backend, config, weights, kvCache, deltaState);
            stages[i] = new PipelineStage(backend, weights, kvCache, deltaState, forward);
        }

        // Embedding stage
        var embedBackend = new CpuBackend();
        var embedWeights = MmapModelLoader.LoadPartialFromShards(
            gguf, shardDir, embedBackend, config,
            startLayer: 0, endLayer: 0,
            includeEmbedding: true, includeOutputHead: false);
        var embedKv = new KvCache(embedBackend, config, maxSeqLen: 256, startLayer: 0, endLayer: 0);
        var embedDelta = new DeltaNetState(embedBackend, config, embedWeights, startLayer: 0, endLayer: 0);
        var embedForward = new ForwardPass(embedBackend, config, embedWeights, embedKv, embedDelta);

        // Output head stage
        var outBackend = new CpuBackend();
        var outWeights = MmapModelLoader.LoadPartialFromShards(
            gguf, shardDir, outBackend, config,
            startLayer: config.NumLayers, endLayer: config.NumLayers,
            includeEmbedding: false, includeOutputHead: true);
        var outKv = new KvCache(outBackend, config, maxSeqLen: 256,
            startLayer: config.NumLayers, endLayer: config.NumLayers);
        var outDelta = new DeltaNetState(outBackend, config, outWeights,
            startLayer: config.NumLayers, endLayer: config.NumLayers);
        var outForward = new ForwardPass(outBackend, config, outWeights, outKv, outDelta);

        Console.Error.WriteLine($"All stages loaded in {loadSw.Elapsed.TotalSeconds:F1}s");

        // ── Tokenize prompt ─────────────────────────────────────────────────
        var prompt = "Explain how distributed machine learning inference works across multiple nodes. The key concepts are";
        var promptTokens = tokenizer.Encode(prompt);
        Console.Error.WriteLine($"Prompt: \"{prompt}\"");
        Console.Error.WriteLine($"Tokens: {promptTokens.Length} [{string.Join(", ", promptTokens.Take(10))}{(promptTokens.Length > 10 ? "..." : "")}]");
        Console.Error.WriteLine();

        var hidden = new float[config.HiddenDim];
        var logits = new float[config.VocabSize];

        // ── Prefill: process all prompt tokens ──────────────────────────────
        Console.Error.WriteLine($"=== Prefill ({promptTokens.Length} tokens) ===");
        var prefillSw = Stopwatch.StartNew();

        for (int t = 0; t < promptTokens.Length; t++)
        {
            var tokenSw = Stopwatch.StartNew();
            int pos = t;

            // Embedding
            embedForward.ForwardEmbedding(promptTokens[t]);
            embedForward.GetHidden(hidden);

            // Walk through all layers
            for (int layer = 0; layer < config.NumLayers; layer++)
            {
                stages[layer].Forward.SetHidden(hidden);
                stages[layer].Forward.ForwardLayers(layer, layer + 1, position: pos);
                stages[layer].Forward.GetHidden(hidden);
            }

            // Output head (only needed for last prompt token to get first generated token)
            if (t == promptTokens.Length - 1)
            {
                outForward.SetHidden(hidden);
                outForward.ForwardOutputHead(logits);
            }

            Console.Error.WriteLine($"  Token {t + 1}/{promptTokens.Length} (id={promptTokens[t]}) in {tokenSw.Elapsed.TotalSeconds:F2}s");
        }

        prefillSw.Stop();
        double prefillTokPerSec = promptTokens.Length / prefillSw.Elapsed.TotalSeconds;
        Console.Error.WriteLine($"Prefill: {promptTokens.Length} tokens in {prefillSw.Elapsed.TotalSeconds:F1}s ({prefillTokPerSec:F2} tok/s)");

        // ── Decode: generate tokens ─────────────────────────────────────────
        Console.Error.WriteLine();
        Console.Error.WriteLine($"=== Decode (generating {maxGenerate} tokens) ===");
        Console.Write(prompt);

        var decodeSw = Stopwatch.StartNew();
        int generatedCount = 0;
        int eosToken = tokenizer.Vocabulary.EosTokenId;

        for (int g = 0; g < maxGenerate; g++)
        {
            // Sample top token from logits
            int nextToken = ArgMax(logits);

            // Check EOS
            if (nextToken == eosToken)
            {
                Console.Error.WriteLine($"\n  [EOS at token {g}]");
                break;
            }

            var text = tokenizer.Decode([nextToken]);
            Console.Write(text);
            generatedCount++;

            // Forward the generated token through the full pipeline
            var tokenSw = Stopwatch.StartNew();
            int pos = promptTokens.Length + g;

            embedForward.ForwardEmbedding(nextToken);
            embedForward.GetHidden(hidden);

            for (int layer = 0; layer < config.NumLayers; layer++)
            {
                stages[layer].Forward.SetHidden(hidden);
                stages[layer].Forward.ForwardLayers(layer, layer + 1, position: pos);
                stages[layer].Forward.GetHidden(hidden);
            }

            outForward.SetHidden(hidden);
            outForward.ForwardOutputHead(logits);

            Console.Error.WriteLine($"  gen {g + 1}: \"{text}\" (id={nextToken}) in {tokenSw.Elapsed.TotalSeconds:F2}s");
        }

        decodeSw.Stop();
        Console.WriteLine();
        Console.Error.WriteLine();

        double decodeTokPerSec = generatedCount > 0 ? generatedCount / decodeSw.Elapsed.TotalSeconds : 0;

        // ── Summary ─────────────────────────────────────────────────────────
        Console.Error.WriteLine("=== Results ===");
        Console.Error.WriteLine($"  Model:    Qwen3.5-27B Q4_0 ({config.NumLayers} layers, {config.HiddenDim}d)");
        Console.Error.WriteLine($"  Method:   Layer-by-layer shard pipeline (CPU, single machine)");
        Console.Error.WriteLine($"  Prefill:  {promptTokens.Length} tokens in {prefillSw.Elapsed.TotalSeconds:F1}s ({prefillTokPerSec:F2} tok/s)");
        Console.Error.WriteLine($"  Decode:   {generatedCount} tokens in {decodeSw.Elapsed.TotalSeconds:F1}s ({decodeTokPerSec:F2} tok/s)");
        Console.Error.WriteLine($"  Load:     {loadSw.Elapsed.TotalSeconds:F1}s (all stages from shards)");

        // Cleanup
        embedForward.Dispose(); embedKv.Dispose(); embedDelta.Dispose();
        embedWeights.Dispose(); embedBackend.Dispose();
        outForward.Dispose(); outKv.Dispose(); outDelta.Dispose();
        outWeights.Dispose(); outBackend.Dispose();
        foreach (var s in stages) s.Dispose();

        Assert.True(generatedCount > 0, "No tokens were generated.");
    }

    /// <summary>
    /// Same pipeline but loading N layers per stage instead of 1.
    /// Reduces SetHidden/GetHidden transfers from 64 to 64/N per token.
    /// </summary>
    [Theory]
    [InlineData(1)]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(16)]
    public void ShardBench_27B_ChunkedLayers(int layersPerChunk)
    {
        if (!TestConstants.Model27BShardsExist)
        {
            Console.Error.WriteLine("Skipping: 27B shard directory not found.");
            return;
        }

        var shardDir = TestConstants.Qwen35_27B_Shards;
        var baseName = "Qwen3.5-27B-Q4_0.gguf";
        int maxGenerate = 10;

        // Parse header
        var headerPath = Path.Combine(shardDir, $"{baseName}.header");
        using var headerStream = File.OpenRead(headerPath);
        var gguf = GgufFile.Read(headerStream);
        var config = ModelConfig.FromGguf(gguf);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        int numChunks = (config.NumLayers + layersPerChunk - 1) / layersPerChunk;
        Console.Error.WriteLine($"\n=== Chunk size: {layersPerChunk} layers ({numChunks} chunks for {config.NumLayers} layers) ===");

        // Load chunked stages
        var loadSw = Stopwatch.StartNew();
        var chunks = new List<ChunkedStage>();
        for (int start = 0; start < config.NumLayers; start += layersPerChunk)
        {
            int end = Math.Min(start + layersPerChunk, config.NumLayers);
            var backend = new CpuBackend();
            var weights = MmapModelLoader.LoadPartialFromShards(
                gguf, shardDir, backend, config,
                startLayer: start, endLayer: end,
                includeEmbedding: false, includeOutputHead: false);
            var kvCache = new KvCache(backend, config, maxSeqLen: 128,
                startLayer: start, endLayer: end);
            var deltaState = new DeltaNetState(backend, config, weights,
                startLayer: start, endLayer: end);
            var forward = new ForwardPass(backend, config, weights, kvCache, deltaState);
            chunks.Add(new ChunkedStage(start, end, backend, weights, kvCache, deltaState, forward));
        }

        // Embedding + output stages
        var embedBackend = new CpuBackend();
        var embedWeights = MmapModelLoader.LoadPartialFromShards(
            gguf, shardDir, embedBackend, config, 0, 0, true, false);
        var embedKv = new KvCache(embedBackend, config, maxSeqLen: 128, startLayer: 0, endLayer: 0);
        var embedDelta = new DeltaNetState(embedBackend, config, embedWeights, startLayer: 0, endLayer: 0);
        var embedFwd = new ForwardPass(embedBackend, config, embedWeights, embedKv, embedDelta);

        var outBackend = new CpuBackend();
        var outWeights = MmapModelLoader.LoadPartialFromShards(
            gguf, shardDir, outBackend, config, config.NumLayers, config.NumLayers, false, true);
        var outKv = new KvCache(outBackend, config, maxSeqLen: 128,
            startLayer: config.NumLayers, endLayer: config.NumLayers);
        var outDelta = new DeltaNetState(outBackend, config, outWeights,
            startLayer: config.NumLayers, endLayer: config.NumLayers);
        var outFwd = new ForwardPass(outBackend, config, outWeights, outKv, outDelta);

        Console.Error.WriteLine($"Loaded in {loadSw.Elapsed.TotalSeconds:F1}s");

        var hidden = new float[config.HiddenDim];
        var logits = new float[config.VocabSize];

        // Tokenize
        var prompt = "The capital of France is";
        var promptTokens = tokenizer.Encode(prompt);

        // Helper: run one token through the full pipeline
        void ForwardToken(int tokenId, int position, bool needLogits)
        {
            embedFwd.ForwardEmbedding(tokenId);
            embedFwd.GetHidden(hidden);

            foreach (var chunk in chunks)
            {
                chunk.Forward.SetHidden(hidden);
                chunk.Forward.ForwardLayers(chunk.Start, chunk.End, position: position);
                chunk.Forward.GetHidden(hidden);
            }

            if (needLogits)
            {
                outFwd.SetHidden(hidden);
                outFwd.ForwardOutputHead(logits);
            }
        }

        // Prefill
        var prefillSw = Stopwatch.StartNew();
        for (int t = 0; t < promptTokens.Length; t++)
            ForwardToken(promptTokens[t], t, t == promptTokens.Length - 1);
        prefillSw.Stop();

        // Decode
        Console.Error.Write($"  Output: {prompt}");
        var decodeSw = Stopwatch.StartNew();
        int generated = 0;
        int eosToken = tokenizer.Vocabulary.EosTokenId;

        for (int g = 0; g < maxGenerate; g++)
        {
            int next = ArgMax(logits);
            if (next == eosToken) break;

            var text = tokenizer.Decode([next]);
            Console.Error.Write(text);
            generated++;

            ForwardToken(next, promptTokens.Length + g, true);
        }
        decodeSw.Stop();
        Console.Error.WriteLine();

        double prefillTps = promptTokens.Length / prefillSw.Elapsed.TotalSeconds;
        double decodeTps = generated > 0 ? generated / decodeSw.Elapsed.TotalSeconds : 0;
        double secPerToken = generated > 0 ? decodeSw.Elapsed.TotalSeconds / generated : 0;

        Console.Error.WriteLine($"  Prefill: {promptTokens.Length} tok in {prefillSw.Elapsed.TotalSeconds:F1}s ({prefillTps:F2} tok/s)");
        Console.Error.WriteLine($"  Decode:  {generated} tok in {decodeSw.Elapsed.TotalSeconds:F1}s ({decodeTps:F2} tok/s, {secPerToken:F2}s/tok)");
        Console.Error.WriteLine($"  Transfers per token: {chunks.Count} (was 64 at chunk=1)");

        // Cleanup
        embedFwd.Dispose(); embedKv.Dispose(); embedDelta.Dispose(); embedWeights.Dispose(); embedBackend.Dispose();
        outFwd.Dispose(); outKv.Dispose(); outDelta.Dispose(); outWeights.Dispose(); outBackend.Dispose();
        foreach (var c in chunks) c.Dispose();

        Assert.True(generated > 0);
    }

    private sealed class ChunkedStage(
        int start, int end,
        CpuBackend backend, ModelWeights weights,
        KvCache kvCache, DeltaNetState deltaState,
        ForwardPass forward) : IDisposable
    {
        public int Start => start;
        public int End => end;
        public ForwardPass Forward => forward;

        public void Dispose()
        {
            forward.Dispose(); deltaState.Dispose();
            kvCache.Dispose(); weights.Dispose(); backend.Dispose();
        }
    }

    /// <summary>
    /// Single CUDA context with ALL weights persistent in VRAM.
    /// No context switching, no hidden state copies, no weight swapping.
    /// Tests whether 27B Q4_0 fits in 16GB VRAM.
    /// If it fits, this is the fastest possible single-GPU path.
    /// </summary>
    [Fact]
    public void ShardBench_27B_CUDA_FullLoad()
    {
        if (!TestConstants.Model27BShardsExist)
        {
            Console.Error.WriteLine("Skipping: 27B shard directory not found.");
            return;
        }

        CudaBackend testCuda;
        try { testCuda = new CudaBackend(); testCuda.Dispose(); }
        catch { Console.Error.WriteLine("Skipping: CUDA not available."); return; }

        var shardDir = TestConstants.Qwen35_27B_Shards;
        var baseName = "Qwen3.5-27B-Q4_0.gguf";
        int maxGenerate = 30;

        // Parse header
        var headerPath = Path.Combine(shardDir, $"{baseName}.header");
        using var headerStream = File.OpenRead(headerPath);
        var gguf = GgufFile.Read(headerStream);
        var config = ModelConfig.FromGguf(gguf);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        Console.Error.WriteLine($"\n=== CUDA Full Load | {config.NumLayers} layers on single context ===");

        using var cuda = new CudaBackend();

        // Load EVERYTHING on one backend — all 64 layers + embed + output
        var loadSw = Stopwatch.StartNew();
        ModelWeights weights;
        try
        {
            weights = MmapModelLoader.LoadPartialFromShards(
                gguf, shardDir, cuda, config,
                startLayer: 0, endLayer: config.NumLayers,
                includeEmbedding: true, includeOutputHead: true);
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"Failed to load full model on GPU: {ex.Message}");
            Console.Error.WriteLine("Model too large for VRAM. Try the multi-context approach.");
            return;
        }

        var kvCache = new KvCache(cuda, config, maxSeqLen: 256);
        var deltaState = new DeltaNetState(cuda, config, weights);
        using var forward = new ForwardPass(cuda, config, weights, kvCache, deltaState);
        Console.Error.WriteLine($"Full model loaded in {loadSw.Elapsed.TotalSeconds:F1}s");

        // Tokenize
        var prompt = "Explain how distributed machine learning inference works across multiple nodes. The key concepts are";
        var promptTokens = tokenizer.Encode(prompt);
        Console.Error.WriteLine($"Prompt: {promptTokens.Length} tokens");

        // Warmup
        var warmSw = Stopwatch.StartNew();
        forward.Forward(promptTokens[0], position: 0);
        Console.Error.WriteLine($"Warmup: {warmSw.Elapsed.TotalMilliseconds:F0}ms");

        // Prefill remaining
        var prefillSw = Stopwatch.StartNew();
        ReadOnlySpan<float> lastLogits = default;
        for (int t = 1; t < promptTokens.Length; t++)
            lastLogits = forward.Forward(promptTokens[t], position: t);
        prefillSw.Stop();
        var logits = lastLogits.ToArray();

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

        kvCache.Dispose(); deltaState.Dispose(); weights.Dispose();
        Assert.True(generated > 0);
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        float bestVal = values[0];
        for (int i = 1; i < values.Length; i++)
        {
            if (values[i] > bestVal)
            {
                bestVal = values[i];
                best = i;
            }
        }
        return best;
    }

    private sealed class PipelineStage(
        CpuBackend backend, ModelWeights weights,
        KvCache kvCache, DeltaNetState deltaState,
        ForwardPass forward) : IDisposable
    {
        public ForwardPass Forward => forward;

        public void Dispose()
        {
            forward.Dispose();
            deltaState.Dispose();
            kvCache.Dispose();
            weights.Dispose();
            backend.Dispose();
        }
    }
}
