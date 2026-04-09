using Daisi.Llogos.Cpu;
using Daisi.Llogos.Cuda;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Tests.Inference;

/// <summary>
/// Compares hidden state after each layer between CUDA PipelinedForwardPass and CPU reference.
/// </summary>
public class PipelineHiddenStateCompareTest
{
    [Fact]
    public void Compare_27B_FirstLayers()
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

        int tokenId = 42;
        int checkLayers = 4; // Check first 4 layers

        // ── CPU reference: layer-by-layer shard pipeline ──
        var cpuHiddenStates = new List<float[]>();

        using (var cpuBackend = new CpuBackend())
        {
            // Embedding
            var embedWeights = ShardModelLoader.LoadPartialFromShards(
                gguf, shardDir, cpuBackend, config, 0, 0, true, false);
            var cpuKv = new KvCache(cpuBackend, config, maxSeqLen: 128, startLayer: 0, endLayer: 0);
            var cpuDelta = new DeltaNetState(cpuBackend, config, embedWeights, startLayer: 0, endLayer: 0);
            using var cpuFwd = new ForwardPass(cpuBackend, config, embedWeights, cpuKv, cpuDelta);

            cpuFwd.ForwardEmbedding(tokenId);
            var hidden = new float[config.HiddenDim];
            cpuFwd.GetHidden(hidden);
            cpuHiddenStates.Add(hidden.ToArray());
            Console.Error.WriteLine($"CPU embed: L2={L2Norm(hidden):F4}, first4=[{hidden[0]:F4},{hidden[1]:F4},{hidden[2]:F4},{hidden[3]:F4}]");

            cpuKv.Dispose(); cpuDelta.Dispose(); embedWeights.Dispose();

            // Process layers one at a time
            for (int layer = 0; layer < checkLayers; layer++)
            {
                using var lb = new CpuBackend();
                var lw = ShardModelLoader.LoadPartialFromShards(
                    gguf, shardDir, lb, config, layer, layer + 1, false, false);
                var lkv = new KvCache(lb, config, maxSeqLen: 128, startLayer: layer, endLayer: layer + 1);
                var ldelta = new DeltaNetState(lb, config, lw, startLayer: layer, endLayer: layer + 1);
                using var lfwd = new ForwardPass(lb, config, lw, lkv, ldelta);

                lfwd.SetHidden(hidden);
                lfwd.ForwardLayers(layer, layer + 1, position: 0, isFinal: false);
                lfwd.GetHidden(hidden);

                cpuHiddenStates.Add(hidden.ToArray());
                Console.Error.WriteLine($"CPU L{layer}: L2={L2Norm(hidden):F4}, first4=[{hidden[0]:F4},{hidden[1]:F4},{hidden[2]:F4},{hidden[3]:F4}], type={config.IsStandardAttention(layer)}");

                lkv.Dispose(); ldelta.Dispose(); lw.Dispose();
            }
        }

        // ── Standard CUDA (non-pipeline, from shards) ──
        byte[] stdKvBuf = [];
        using (var cuda = new CudaBackend())
        {
            var stdEmbed = ShardModelLoader.LoadPartialFromShards(
                gguf, shardDir, cuda, config, 0, 0, true, false);
            var stdLayer0 = ShardModelLoader.LoadPartialFromShards(
                gguf, shardDir, cuda, config, 0, 1, false, false);
            // Merge: create weights with real embed + layer 0
            var stdWeights = new ModelWeights
            {
                TokenEmbedding = stdEmbed.TokenEmbedding,
                OutputNorm = stdEmbed.OutputNorm,
                Output = stdEmbed.Output,
                Layers = stdLayer0.Layers,
            };
            var stdKv = new KvCache(cuda, config, maxSeqLen: 128, startLayer: 0, endLayer: 1);
            var stdDelta = new DeltaNetState(cuda, config, stdWeights, startLayer: 0, endLayer: 1);
            using var stdFwd = new ForwardPass(cuda, config, stdWeights, stdKv, stdDelta);

            // Test with graph capture disabled (same as pipeline)
            cuda.DisableGraphCapture();

            stdFwd.ForwardEmbedding(tokenId);
            var stdHidden = new float[config.HiddenDim];
            stdFwd.GetHidden(stdHidden);
            Console.Error.WriteLine($"Std CUDA embed: L2={L2Norm(stdHidden):F4}, first4=[{stdHidden[0]:F4},{stdHidden[1]:F4},{stdHidden[2]:F4},{stdHidden[3]:F4}]");

            stdFwd.ForwardLayers(0, 1, position: 0, isFinal: false);
            stdFwd.GetHidden(stdHidden);
            Console.Error.WriteLine($"Std CUDA L0: L2={L2Norm(stdHidden):F4}, first4=[{stdHidden[0]:F4},{stdHidden[1]:F4},{stdHidden[2]:F4},{stdHidden[3]:F4}]");
            CompareHidden("Std CUDA vs CPU L0", cpuHiddenStates[1], stdHidden);

            // Download AttnQkv tensor bytes for comparison
            var stdQkv = (CudaTensor)((DeltaNetWeights)stdWeights.Layers[0]).AttnQkv;
            var stdQkvBytes = new byte[stdQkv.ByteSize];
            stdQkv.CopyRawTo(stdQkvBytes);
            Console.Error.WriteLine($"Std CUDA AttnQkv: {stdQkv.ByteSize} bytes, aligned4={stdQkv.IsAlignedQ4_0}, type={stdQkv.Type}");

            stdKv.Dispose(); stdDelta.Dispose(); stdEmbed.Dispose(); stdLayer0.Dispose();

            stdKvBuf = stdQkvBytes; // Save for comparison with pipeline
            stdKv.Dispose(); stdDelta.Dispose(); stdEmbed.Dispose(); stdLayer0.Dispose();
        }

        // ── CUDA pipeline ──
        using var cuda2 = new CudaBackend();
        using var pipeline = PipelinedForwardPass.Create(gguf, shardDir, config, cuda2, maxContext: 128);

        // Upload layer 0 to pipeline slot and compare tensor data
        {
            var uplMethod2 = typeof(PipelinedForwardPass).GetMethod("UploadLayer",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
            var wAField2 = typeof(PipelinedForwardPass).GetField("_weightsA",
                System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
            var wA2 = (ModelWeights)wAField2.GetValue(pipeline)!;
            uplMethod2.Invoke(pipeline, [0, wA2]);

            var pipQkv = (CudaTensor)((DeltaNetWeights)wA2.Layers[0]).AttnQkv;
            var pipQkvBytes = new byte[pipQkv.ByteSize];
            pipQkv.CopyRawTo(pipQkvBytes);
            Console.Error.WriteLine($"Pip CUDA AttnQkv: {pipQkv.ByteSize} bytes, aligned4={pipQkv.IsAlignedQ4_0}, type={pipQkv.Type}");

            if (stdKvBuf.Length == pipQkvBytes.Length)
            {
                int qkvMismatches = 0;
                for (int i = 0; i < stdKvBuf.Length; i++)
                    if (stdKvBuf[i] != pipQkvBytes[i]) qkvMismatches++;
                Console.Error.WriteLine($"AttnQkv byte comparison: {qkvMismatches}/{stdKvBuf.Length} mismatches");
            }
            else
                Console.Error.WriteLine($"AttnQkv SIZE MISMATCH: std={stdKvBuf.Length} pip={pipQkvBytes.Length}");
        }

        // Use reflection to access internal ForwardPass and extract hidden states
        var fwdField = typeof(PipelinedForwardPass).GetField("_forward",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var innerFwd = (ForwardPass)fwdField.GetValue(pipeline)!;

        // Run embedding
        var weightsAField = typeof(PipelinedForwardPass).GetField("_weightsA",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var weightsA = (ModelWeights)weightsAField.GetValue(pipeline)!;

        var swapMethod = typeof(PipelinedForwardPass).GetMethod("SwapWeights",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        swapMethod.Invoke(pipeline, [weightsA]);
        innerFwd.ForwardEmbedding(tokenId);

        var gpuHidden = new float[config.HiddenDim];
        innerFwd.GetHidden(gpuHidden);
        Console.Error.WriteLine($"GPU embed: L2={L2Norm(gpuHidden):F4}, first4=[{gpuHidden[0]:F4},{gpuHidden[1]:F4},{gpuHidden[2]:F4},{gpuHidden[3]:F4}]");
        CompareHidden("embed", cpuHiddenStates[0], gpuHidden);

        // Run layers one at a time (mimicking pipeline Forward loop)
        var uploadMethod = typeof(PipelinedForwardPass).GetMethod("UploadLayer",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var weightsBField = typeof(PipelinedForwardPass).GetField("_weightsB",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance)!;
        var weightsB = (ModelWeights)weightsBField.GetValue(pipeline)!;

        for (int layer = 0; layer < checkLayers; layer++)
        {
            int slot = layer % 2;
            var weights = slot == 0 ? weightsA : weightsB;
            uploadMethod.Invoke(pipeline, [layer, weights]);
            swapMethod.Invoke(pipeline, [weights]);
            innerFwd.ForwardLayers(layer, layer + 1, position: 0,
                continuation: layer > 0, isFinal: layer == config.NumLayers - 1);

            innerFwd.GetHidden(gpuHidden);
            Console.Error.WriteLine($"GPU L{layer}: L2={L2Norm(gpuHidden):F4}, first4=[{gpuHidden[0]:F4},{gpuHidden[1]:F4},{gpuHidden[2]:F4},{gpuHidden[3]:F4}]");
            CompareHidden($"L{layer}", cpuHiddenStates[layer + 1], gpuHidden);
        }
    }

    private static void CompareHidden(string label, float[] cpu, float[] gpu)
    {
        float maxDiff = 0;
        for (int i = 0; i < cpu.Length; i++)
            maxDiff = Math.Max(maxDiff, Math.Abs(cpu[i] - gpu[i]));
        Console.Error.WriteLine($"  {label}: maxDiff={maxDiff:F6}");
        // Allow some tolerance for CPU/GPU numerical differences
        Assert.True(maxDiff < 1.0f, $"{label}: max hidden diff {maxDiff:F6} too large");
    }

    private static float L2Norm(float[] v)
    {
        double sum = 0;
        for (int i = 0; i < v.Length; i++) sum += v[i] * v[i];
        return (float)Math.Sqrt(sum);
    }
}
