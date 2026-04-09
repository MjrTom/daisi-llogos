using Daisi.Llogos.Cuda;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Tests.Inference;

/// <summary>
/// Verifies that the pipeline's UploadLayer path produces correct GPU data
/// by round-tripping: shard (mmap) → GPU tensor → host bytes → compare to shard.
/// </summary>
public class PipelineUploadRoundtripTest
{
    [Fact]
    public unsafe void RoundTrip_27B_Layer0_GpuAligned()
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
        using var headerStream = File.OpenRead(headerPath);
        var gguf = GgufFile.Read(headerStream);
        var config = ModelConfig.FromGguf(gguf);

        var manifestPath = Path.Combine(shardDir, $"{baseName}.manifest.json");
        bool gpuAligned = File.Exists(manifestPath) && GgufShardManifest.FromJsonFile(manifestPath).GpuAligned;
        Assert.True(gpuAligned, "Expected GPU-aligned shards for 27B model");

        using var cuda = new CudaBackend();

        // Allocate a slot for layer 0 (standard attention)
        var tensorInfoMap = new Dictionary<string, GgufTensorInfo>();
        foreach (var t2 in gguf.Tensors) tensorInfoMap[t2.Name] = t2;

        ITensor Alloc(string name)
        {
            var info = tensorInfoMap[name];
            var dims = info.Dimensions.Select(d => (long)d).ToArray();
            return cuda.LoadTensor($"test.{name}", info.Type, dims,
                new byte[GgmlTypeInfo.ByteSize(info.Type, info.ElementCount)]);
        }

        // Load shard data via mmap
        var mmapHandles = new List<IDisposable>();
        var shardPath = Path.Combine(shardDir, $"{baseName}.layer.0");
        var layerData = PipelinedForwardPass.LayerShardData.Load(shardPath, 0, config, gguf, mmapHandles);

        Console.Error.WriteLine($"Layer 0: {layerData.Tensors.Count} tensors");

        int totalMismatches = 0;
        foreach (var tr in layerData.Tensors)
        {
            // Allocate GPU tensor
            string fullName;
            if (tr.ShortName == "post_attn_norm")
                fullName = tensorInfoMap.ContainsKey("blk.0.post_attention_norm.weight")
                    ? "blk.0.post_attention_norm.weight" : "blk.0.ffn_norm.weight";
            else if (tr.ShortName == "attn_o")
                fullName = "blk.0.attn_output.weight";
            else
                fullName = $"blk.0.{tr.ShortName}.weight";
            // Handle special names
            if (tr.ShortName == "attn_q_norm") fullName = "blk.0.attn_q_norm.weight";
            if (tr.ShortName == "attn_k_norm") fullName = "blk.0.attn_k_norm.weight";

            if (!tensorInfoMap.ContainsKey(fullName))
            {
                Console.Error.WriteLine($"  {tr.ShortName}: SKIP (no tensor info for {fullName})");
                continue;
            }

            var slotTensor = (CudaTensor)Alloc(fullName);

            // Upload via pipeline's gpuAligned path: straight copy
            byte[] repacked = new byte[tr.RawByteSize];
            fixed (byte* dst = repacked)
                Buffer.MemoryCopy(tr.MmapPtr, dst, tr.RawByteSize, tr.RawByteSize);

            try
            {
                slotTensor.CopyFrom(repacked);
            }
            catch (ArgumentException ex)
            {
                Console.Error.WriteLine($"  {tr.ShortName}: CopyFrom FAILED — {ex.Message} (repacked={repacked.Length}, tensorBytes={slotTensor.ByteSize}, type={slotTensor.Type}, aligned4={slotTensor.IsAlignedQ4_0}, aligned8={slotTensor.IsAlignedQ8_0})");
                totalMismatches++;
                slotTensor.Dispose();
                continue;
            }

            // Download from GPU
            var downloaded = new byte[slotTensor.ByteSize];
            slotTensor.CopyRawTo(downloaded);

            // Compare
            int mismatches = 0;
            int firstMismatch = -1;
            int len = Math.Min(repacked.Length, downloaded.Length);
            for (int j = 0; j < len; j++)
            {
                if (repacked[j] != downloaded[j])
                {
                    if (firstMismatch < 0) firstMismatch = j;
                    mismatches++;
                }
            }

            if (mismatches > 0 || repacked.Length != downloaded.Length)
            {
                Console.Error.WriteLine($"  {tr.ShortName}: {mismatches} mismatches in {len} bytes, first@{firstMismatch} (src={repacked.Length}, gpu={downloaded.Length}, type={slotTensor.Type})");
                totalMismatches++;
            }
            else
                Console.Error.WriteLine($"  {tr.ShortName}: OK ({repacked.Length} bytes, type={slotTensor.Type})");

            slotTensor.Dispose();
        }

        foreach (var h in mmapHandles) h.Dispose();
        Assert.Equal(0, totalMismatches);
    }
}
