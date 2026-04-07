using Daisi.Llogos.Cuda;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Tests.Inference;

/// <summary>
/// Compares GPU tensor contents between normal loading and AllocateLayerSlot + RepackAndUpload.
/// </summary>
public class SharedTensorDebugTest
{
    [Fact]
    public unsafe void CompareLayerData_NormalVsRepackUpload()
    {
        if (!TestConstants.Model27BShardsExist)
        {
            Console.Error.WriteLine("Skipping: 27B shard directory not found.");
            return;
        }

        try { using var t = new CudaBackend(); }
        catch { Console.Error.WriteLine("Skipping: CUDA not available."); return; }

        var shardDir = TestConstants.Qwen35_27B_Shards;
        var baseName = MmapModelLoader.FindShardBaseName(shardDir);
        var headerPath = Path.Combine(shardDir, $"{baseName}.header");
        using var headerStream = File.OpenRead(headerPath);
        var gguf = GgufFile.Read(headerStream);
        var config = ModelConfig.FromGguf(gguf);

        // Check manifest for gpuAligned
        var manifestPath = Path.Combine(shardDir, $"{baseName}.manifest.json");
        bool gpuAligned = false;
        if (File.Exists(manifestPath))
            gpuAligned = GgufShardManifest.FromJsonFile(manifestPath).GpuAligned;

        using var cuda = new CudaBackend();

        // Load layer 0 via normal path
        var normalWeights = MmapModelLoader.LoadPartialFromShards(
            gguf, shardDir, cuda, config, 0, 1, false, false);

        // Load layer 0 via AllocateLayerSlot + RepackAndUpload
        // Use reflection to call the private AllocateStandardLayer
        var tensorInfoMap = new Dictionary<string, GgufTensorInfo>();
        foreach (var t2 in gguf.Tensors) tensorInfoMap[t2.Name] = t2;

        var info = tensorInfoMap["blk.0.ffn_gate.weight"];
        var dims = info.Dimensions.Select(d => (long)d).ToArray();
        var zeroData = new byte[GgmlTypeInfo.ByteSize(info.Type, info.ElementCount)];
        var slotTensor = (CudaTensor)cuda.LoadTensor("test.blk.0.ffn_gate.weight", info.Type, dims, zeroData);

        var normalTensor = (CudaTensor)normalWeights.Layers[0].FfnGate;

        Console.Error.WriteLine($"Normal tensor: type={normalTensor.Type}, byteSize={normalTensor.ByteSize}, " +
            $"aligned8={normalTensor.IsAlignedQ8_0}, aligned4={normalTensor.IsAlignedQ4_0}");
        Console.Error.WriteLine($"Slot tensor:   type={slotTensor.Type}, byteSize={slotTensor.ByteSize}, " +
            $"aligned8={slotTensor.IsAlignedQ8_0}, aligned4={slotTensor.IsAlignedQ4_0}");

        // Now upload shard data into slot tensor via RepackAndUpload-style copy
        var mmapHandles = new List<IDisposable>();
        var shardPath = Path.Combine(shardDir, $"{baseName}.layer.0");
        var layerData = PipelinedForwardPass.LayerShardData.Load(shardPath, 0, config, gguf, mmapHandles);

        // Find attn_q tensor ref
        var attnQRef = layerData.Tensors.First(t2 => t2.ShortName == "ffn_gate");
        Console.Error.WriteLine($"Shard tensor:  rawBytes={attnQRef.RawByteSize}, type={attnQRef.Type}, ndim={attnQRef.NDimensions}");

        // Repack into pinned buffer
        using var pinned = new CudaPinnedMemory((ulong)(attnQRef.RawByteSize / 18 * 20 + 1000));
        byte* dst = (byte*)pinned.HostPtr;
        byte* src = attnQRef.MmapPtr;
        int blockCount = attnQRef.RawByteSize / 18;
        long repackedSize = blockCount * 20;
        for (int b = 0; b < blockCount; b++)
        {
            byte* s = src + b * 18; byte* d = dst + b * 20;
            d[0] = s[0]; d[1] = s[1]; d[2] = 0; d[3] = 0;
            Buffer.MemoryCopy(s + 2, d + 4, 16, 16);
        }

        // Upload to slot tensor via CopyFrom
        slotTensor.CopyFrom(new ReadOnlySpan<byte>(dst, (int)repackedSize));

        // Download both tensors to CPU and compare
        var normalBytes = new byte[normalTensor.ByteSize];
        normalTensor.CopyRawTo(normalBytes);

        var slotBytes = new byte[slotTensor.ByteSize];
        slotTensor.CopyRawTo(slotBytes);

        Assert.Equal(normalBytes.Length, slotBytes.Length);

        int mismatches = 0;
        int firstMismatch = -1;
        for (int i = 0; i < normalBytes.Length; i++)
        {
            if (normalBytes[i] != slotBytes[i])
            {
                if (firstMismatch < 0) firstMismatch = i;
                mismatches++;
            }
        }

        Console.Error.WriteLine($"Comparison: {normalBytes.Length} bytes, {mismatches} mismatches, first at byte {firstMismatch}");
        Assert.Equal(0, mismatches);

        normalWeights.Dispose();
        slotTensor.Dispose();
        foreach (var h in mmapHandles) h.Dispose();
    }

    /// <summary>
    /// Test: does DeltaNetState initialization differ with shared vs separate layer entries?
    /// The DeltaNetState reads inner dimensions from layer weights — if all entries point
    /// to the same DeltaNet weights, it should still detect DeltaNet layers correctly.
    /// </summary>
    [Fact]
    public void DeltaNetState_SharedVsSeparate()
    {
        if (!TestConstants.Model27BShardsExist) return;
        try { using var t = new CudaBackend(); } catch { return; }

        var shardDir = TestConstants.Qwen35_27B_Shards;
        var baseName = MmapModelLoader.FindShardBaseName(shardDir);
        using var hs = File.OpenRead(Path.Combine(shardDir, $"{baseName}.header"));
        var gguf = GgufFile.Read(hs);
        var config = ModelConfig.FromGguf(gguf);

        using var cuda = new CudaBackend();

        // Separate: real tensors per layer
        var wSep = MmapModelLoader.LoadPartialFromShards(gguf, shardDir, cuda, config, 0, config.NumLayers, false, false);
        var dsSep = new DeltaNetState(cuda, config, wSep);

        // Shared: all DeltaNet layers point to layer 0's tensors, all std point to layer 3's
        var sharedLayers = new LayerWeights[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            if (config.IsStandardAttention(i))
            {
                // Find first standard attention layer in the separate weights
                for (int j = 0; j < config.NumLayers; j++)
                    if (config.IsStandardAttention(j)) { sharedLayers[i] = wSep.Layers[j]; break; }
            }
            else
                sharedLayers[i] = wSep.Layers[0]; // share DeltaNet layer 0
        }

        var wShared = new ModelWeights
        {
            TokenEmbedding = wSep.TokenEmbedding,
            OutputNorm = wSep.OutputNorm,
            Output = wSep.Output,
            Layers = sharedLayers,
        };
        var dsShared = new DeltaNetState(cuda, config, wShared);

        // Both DeltaNetStates should have the same structure (same number of delta layers)
        Console.Error.WriteLine($"DeltaNetState constructed successfully for both paths");

        dsShared.Dispose();
        dsSep.Dispose();
        wSep.Dispose();
    }
}
