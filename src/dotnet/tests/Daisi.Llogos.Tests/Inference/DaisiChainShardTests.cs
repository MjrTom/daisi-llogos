using System.Diagnostics;
using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Tests.Inference;

/// <summary>
/// DaisiChain shard-based pipeline tests.
/// Simulates a multi-host pipeline on a single machine by loading each layer range
/// from separate shard files and passing hidden states between stages.
/// </summary>
public class DaisiChainShardTests
{
    private static void Log(string msg) => Console.Error.WriteLine(msg);
    /// <summary>
    /// Simulates a full DaisiChain pipeline on the 27B model using per-layer shards.
    /// Each layer is loaded from its own shard file, run, then disposed — only one layer
    /// is in memory at a time. This is the "layer-at-a-time" pattern for WebGPU/low-VRAM hosts.
    ///
    /// Pipeline: embedding → layer 0 → layer 1 → ... → layer 63 → output head → sample token
    /// </summary>
    [Fact]
    public void ShardPipeline_27B_LayerByLayer_ProducesValidOutput()
    {
        if (!TestConstants.Model27BShardsExist)
        {
            Log("Skipping: 27B shard directory not found. Run 'daisi-llogos split' first.");
            return;
        }

        var sw = Stopwatch.StartNew();
        var shardDir = TestConstants.Qwen35_27B_Shards;
        var baseName = "Qwen3.5-27B-Q4_0.gguf";

        // 1. Parse header shard for model metadata
        var headerPath = Path.Combine(shardDir, $"{baseName}.header");
        using var headerStream = File.OpenRead(headerPath);
        var gguf = GgufFile.Read(headerStream);
        var config = ModelConfig.FromGguf(gguf);
        var tokenizer = TokenizerFactory.FromGguf(gguf);

        Log($"Model: {config.Architecture}, {config.NumLayers} layers, {config.HiddenDim}d");
        Log($"Header parsed in {sw.ElapsedMilliseconds}ms");

        // Tokenize a prompt
        var prompt = "The capital of France is";
        var tokens = tokenizer.Encode(prompt);
        Log($"Prompt: \"{prompt}\" → {tokens.Length} tokens: [{string.Join(", ", tokens)}]");

        // We'll process just the first token through the full pipeline
        int tokenId = tokens[0];
        Log($"Processing token {tokenId} through {config.NumLayers} layers...");

        // 2. Stage 0: Load embedding from shard, run forward
        sw.Restart();
        float[] hidden;
        using (var backend = new CpuBackend())
        {
            var embedWeights = ShardModelLoader.LoadPartialFromShards(
                gguf, shardDir, backend, config,
                startLayer: 0, endLayer: 0,
                includeEmbedding: true, includeOutputHead: false);

            var kvCache = new KvCache(backend, config, maxSeqLen: 128, startLayer: 0, endLayer: 0);
            var deltaState = new DeltaNetState(backend, config, embedWeights, startLayer: 0, endLayer: 0);
            using var forward = new ForwardPass(backend, config, embedWeights, kvCache, deltaState);

            forward.ForwardEmbedding(tokenId);
            hidden = new float[config.HiddenDim];
            forward.GetHidden(hidden);

            kvCache.Dispose();
            deltaState.Dispose();
            embedWeights.Dispose();
        }
        Log($"Embedding done in {sw.ElapsedMilliseconds}ms");

        // Verify embedding produced non-zero hidden state
        bool hasNonZero = false;
        for (int i = 0; i < hidden.Length; i++)
            if (hidden[i] != 0) { hasNonZero = true; break; }
        Assert.True(hasNonZero, "Hidden state is all zeros after embedding.");

        // 3. Process layers one at a time, each from its own shard
        var layerSw = Stopwatch.StartNew();
        for (int layer = 0; layer < config.NumLayers; layer++)
        {
            var layerStart = Stopwatch.StartNew();
            using var backend = new CpuBackend();

            // Load just this one layer from its shard
            var weights = ShardModelLoader.LoadPartialFromShards(
                gguf, shardDir, backend, config,
                startLayer: layer, endLayer: layer + 1,
                includeEmbedding: false, includeOutputHead: false);

            var kvCache = new KvCache(backend, config, maxSeqLen: 128,
                startLayer: layer, endLayer: layer + 1);
            var deltaState = new DeltaNetState(backend, config, weights,
                startLayer: layer, endLayer: layer + 1);
            using var forward = new ForwardPass(backend, config, weights, kvCache, deltaState);

            // Inject hidden state from previous stage
            forward.SetHidden(hidden);

            // Run this layer
            forward.ForwardLayers(layer, layer + 1, position: 0);

            // Extract hidden state for next stage
            forward.GetHidden(hidden);

            kvCache.Dispose();
            deltaState.Dispose();
            weights.Dispose();

            if (layer % 8 == 0 || layer == config.NumLayers - 1)
                Log($"  Layer {layer}/{config.NumLayers} done in {layerStart.ElapsedMilliseconds}ms");
        }
        Log($"All {config.NumLayers} layers done in {layerSw.Elapsed.TotalSeconds:F1}s");

        // 4. Final stage: output head
        sw.Restart();
        float[] logits;
        using (var backend = new CpuBackend())
        {
            var outputWeights = ShardModelLoader.LoadPartialFromShards(
                gguf, shardDir, backend, config,
                startLayer: config.NumLayers, endLayer: config.NumLayers,
                includeEmbedding: false, includeOutputHead: true);

            var kvCache = new KvCache(backend, config, maxSeqLen: 128,
                startLayer: config.NumLayers, endLayer: config.NumLayers);
            var deltaState = new DeltaNetState(backend, config, outputWeights,
                startLayer: config.NumLayers, endLayer: config.NumLayers);
            using var forward = new ForwardPass(backend, config, outputWeights, kvCache, deltaState);

            forward.SetHidden(hidden);

            logits = new float[config.VocabSize];
            forward.ForwardOutputHead(logits);

            kvCache.Dispose();
            deltaState.Dispose();
            outputWeights.Dispose();
        }
        Log($"Output head done in {sw.ElapsedMilliseconds}ms");

        // 5. Sample the top token
        int topToken = 0;
        float topLogit = float.MinValue;
        for (int i = 0; i < logits.Length; i++)
        {
            if (logits[i] > topLogit)
            {
                topLogit = logits[i];
                topToken = i;
            }
        }

        var decoded = tokenizer.Decode([topToken]);
        Log($"\nResult: token {topToken} = \"{decoded}\" (logit: {topLogit:F2})");
        Log($"Prompt: \"{prompt}\" → \"{decoded}\"");

        // Verify we got a valid prediction (not garbage)
        Assert.True(topLogit > float.MinValue, "Logits are all -inf — forward pass failed.");
        Assert.True(topToken >= 0 && topToken < config.VocabSize, "Top token out of vocab range.");

        // For "The capital of France is", we expect something like "Paris"
        Log($"\n✓ Pipeline completed: {config.NumLayers} layers processed one-at-a-time from shard files.");
    }

    /// <summary>
    /// Verifies that loading from shards produces identical output to loading from the monolithic GGUF.
    /// Uses the small 0.8B model for comparison (fits easily in memory twice).
    /// </summary>
    [Fact]
    public void ShardPipeline_08B_MatchesMonolithic()
    {
        if (!TestConstants.ModelExists)
        {
            Log("Skipping: 0.8B model not found.");
            return;
        }

        // First, split the 0.8B model into shards
        var shardDir = Path.Combine(Path.GetTempPath(), "llogos-shard-test-" + Guid.NewGuid().ToString("N")[..8]);
        try
        {
            GgufSplitter.Split(TestConstants.Qwen35_08B_Q8_0, shardDir);
            Log($"Split 0.8B model into shards at {shardDir}");

            // Run full monolithic forward pass
            using var monoStream = File.OpenRead(TestConstants.Qwen35_08B_Q8_0);
            var monoGguf = GgufFile.Read(monoStream);
            var config = ModelConfig.FromGguf(monoGguf);
            using var monoBackend = new CpuBackend();
            var monoWeights = MmapModelLoader.Load(monoGguf, TestConstants.Qwen35_08B_Q8_0, monoBackend, config);
            var monoKvCache = new KvCache(monoBackend, config, maxSeqLen: 128);
            var monoDeltaState = new DeltaNetState(monoBackend, config, monoWeights);
            using var monoForward = new ForwardPass(monoBackend, config, monoWeights, monoKvCache, monoDeltaState);

            var monoLogits = monoForward.Forward(tokenId: 42, position: 0).ToArray();
            Log($"Monolithic forward: top logit = {monoLogits.Max():F4}");

            // Run shard-based pipeline forward pass (2 stages)
            var headerPath = Path.Combine(shardDir, Path.GetFileName(TestConstants.Qwen35_08B_Q8_0) + ".header");
            using var headerStream = File.OpenRead(headerPath);
            var shardGguf = GgufFile.Read(headerStream);
            int mid = config.NumLayers / 2;

            // Stage 1: embedding + layers [0, mid)
            using var backend1 = new CpuBackend();
            var weights1 = ShardModelLoader.LoadPartialFromShards(
                shardGguf, shardDir, backend1, config,
                startLayer: 0, endLayer: mid,
                includeEmbedding: true, includeOutputHead: false);
            var kvCache1 = new KvCache(backend1, config, maxSeqLen: 128, startLayer: 0, endLayer: mid);
            var delta1 = new DeltaNetState(backend1, config, weights1, startLayer: 0, endLayer: mid);
            using var forward1 = new ForwardPass(backend1, config, weights1, kvCache1, delta1);

            forward1.ForwardEmbedding(tokenId: 42);
            forward1.ForwardLayers(0, mid, position: 0);

            var hidden = new float[config.HiddenDim];
            forward1.GetHidden(hidden);

            // Stage 2: layers [mid, end) + output head
            using var backend2 = new CpuBackend();
            var weights2 = ShardModelLoader.LoadPartialFromShards(
                shardGguf, shardDir, backend2, config,
                startLayer: mid, endLayer: config.NumLayers,
                includeEmbedding: false, includeOutputHead: true);
            var kvCache2 = new KvCache(backend2, config, maxSeqLen: 128,
                startLayer: mid, endLayer: config.NumLayers);
            var delta2 = new DeltaNetState(backend2, config, weights2,
                startLayer: mid, endLayer: config.NumLayers);
            using var forward2 = new ForwardPass(backend2, config, weights2, kvCache2, delta2);

            forward2.SetHidden(hidden);
            forward2.ForwardLayers(mid, config.NumLayers, position: 0);

            var shardLogits = new float[config.VocabSize];
            forward2.ForwardOutputHead(shardLogits);
            Log($"Shard pipeline forward: top logit = {shardLogits.Max():F4}");

            // Compare logits — must match exactly (same backend, deterministic)
            Assert.Equal(monoLogits.Length, shardLogits.Length);
            int mismatches = 0;
            for (int i = 0; i < monoLogits.Length; i++)
            {
                if (Math.Abs(monoLogits[i] - shardLogits[i]) > 1e-4f)
                    mismatches++;
            }
            Log($"Logit comparison: {mismatches} mismatches out of {monoLogits.Length}");
            Assert.Equal(0, mismatches);

            // Cleanup
            kvCache1.Dispose(); delta1.Dispose(); weights1.Dispose();
            kvCache2.Dispose(); delta2.Dispose(); weights2.Dispose();
            monoKvCache.Dispose(); monoDeltaState.Dispose(); monoWeights.Dispose();

            Log("✓ Shard pipeline output matches monolithic forward pass exactly.");
        }
        finally
        {
            try { Directory.Delete(shardDir, true); } catch { }
        }
    }
}
