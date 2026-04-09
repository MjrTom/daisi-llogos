using Daisi.Llogos.Cuda;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Tests.Inference;

/// <summary>
/// Compares weight tensor data between standard shard loading and PipelinedForwardPass upload.
/// Targets the 27B Q4_0 model to diagnose pipeline divergence.
/// </summary>
public class PipelineWeightDiagnosticTest
{
    [Fact]
    public unsafe void CompareWeightData_27B_Layer0()
    {
        if (!TestConstants.Model27BShardsExist)
        {
            Console.Error.WriteLine("Skipping: 27B shard directory not found.");
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
        Console.Error.WriteLine($"GPU aligned: {gpuAligned}");

        using var cuda = new CudaBackend();

        // 1. Load layer 0 via standard shard loader → GPU tensors with correct data
        var standardWeights = ShardModelLoader.LoadPartialFromShards(
            gguf, shardDir, cuda, config, 0, 1, false, false);
        var stdLayer = (StandardAttentionWeights)standardWeights.Layers[0];

        // 2. Create pipeline-style allocation + upload
        var tensorInfoMap = new Dictionary<string, GgufTensorInfo>();
        foreach (var t2 in gguf.Tensors) tensorInfoMap[t2.Name] = t2;

        var slotLayer = AllocateStandardLayer(cuda, tensorInfoMap, 0, "diag");

        // Load shard data
        var mmapHandles = new List<IDisposable>();
        var shardPath = Path.Combine(shardDir, $"{baseName}.layer.0");
        var layerData = PipelinedForwardPass.LayerShardData.Load(shardPath, 0, config, gguf, mmapHandles);

        // Upload via pipeline path
        var targets = PipelinedForwardPass.LayerShardData.GetTargetTensors(layerData, slotLayer);
        for (int t2 = 0; t2 < layerData.Tensors.Count; t2++)
        {
            var tr = layerData.Tensors[t2];
            byte* src = tr.MmapPtr;

            byte[] repacked;
            if (gpuAligned)
            {
                repacked = new byte[tr.RawByteSize];
                fixed (byte* dst = repacked)
                    Buffer.MemoryCopy(src, dst, tr.RawByteSize, tr.RawByteSize);
            }
            else if (tr.Type == GgmlType.Q4_0 && tr.NDimensions >= 2)
            {
                int bc = tr.RawByteSize / 18;
                repacked = new byte[bc * 20];
                fixed (byte* dst = repacked)
                    for (int b = 0; b < bc; b++)
                    {
                        byte* s = src + b * 18; byte* d = dst + b * 20;
                        d[0] = s[0]; d[1] = s[1]; d[2] = 0; d[3] = 0;
                        Buffer.MemoryCopy(s + 2, d + 4, 16, 16);
                    }
            }
            else if (tr.Type == GgmlType.Q8_0 && tr.NDimensions >= 2)
            {
                int bc = tr.RawByteSize / 34;
                repacked = new byte[bc * 36];
                fixed (byte* dst = repacked)
                    for (int b = 0; b < bc; b++)
                    {
                        byte* s = src + b * 34; byte* d = dst + b * 36;
                        d[0] = s[0]; d[1] = s[1]; d[2] = 0; d[3] = 0;
                        Buffer.MemoryCopy(s + 2, d + 4, 32, 32);
                    }
            }
            else
            {
                repacked = new byte[tr.RawByteSize];
                fixed (byte* dst = repacked)
                    Buffer.MemoryCopy(src, dst, tr.RawByteSize, tr.RawByteSize);
            }

            targets[t2].CopyFrom(repacked);
        }

        // 3. Compare each tensor
        var tensorNames = new[] { "AttnNorm", "PostAttnNorm", "AttnQ", "AttnK", "AttnV", "AttnO", "FfnGate", "FfnUp", "FfnDown" };
        var stdTensors = new ITensor[] { stdLayer.AttnNorm, stdLayer.PostAttnNorm, stdLayer.AttnQ, stdLayer.AttnK, stdLayer.AttnV, stdLayer.AttnO, stdLayer.FfnGate, stdLayer.FfnUp, stdLayer.FfnDown };
        var slotTensors = new ITensor[] { slotLayer.AttnNorm, slotLayer.PostAttnNorm, slotLayer.AttnQ, slotLayer.AttnK, slotLayer.AttnV, slotLayer.AttnO, slotLayer.FfnGate, slotLayer.FfnUp, slotLayer.FfnDown };

        int totalMismatches = 0;
        for (int i = 0; i < tensorNames.Length; i++)
        {
            var stdT = (CudaTensor)stdTensors[i];
            var slotT = (CudaTensor)slotTensors[i];

            if (stdT.ByteSize != slotT.ByteSize)
            {
                Console.Error.WriteLine($"  {tensorNames[i]}: SIZE MISMATCH std={stdT.ByteSize} slot={slotT.ByteSize}");
                totalMismatches++;
                continue;
            }

            var stdBytes = new byte[stdT.ByteSize];
            var slotBytes = new byte[slotT.ByteSize];
            stdT.CopyRawTo(stdBytes);
            slotT.CopyRawTo(slotBytes);

            int mismatches = 0;
            int firstMismatch = -1;
            for (int j = 0; j < stdBytes.Length; j++)
            {
                if (stdBytes[j] != slotBytes[j])
                {
                    if (firstMismatch < 0) firstMismatch = j;
                    mismatches++;
                }
            }

            if (mismatches > 0)
            {
                Console.Error.WriteLine($"  {tensorNames[i]}: {mismatches}/{stdBytes.Length} byte mismatches, first at {firstMismatch} type={stdT.Type} aligned8={stdT.IsAlignedQ8_0} aligned4={stdT.IsAlignedQ4_0}");
                totalMismatches += mismatches;
            }
            else
                Console.Error.WriteLine($"  {tensorNames[i]}: OK ({stdBytes.Length} bytes, type={stdT.Type})");
        }

        // Cleanup
        foreach (var h in mmapHandles) h.Dispose();
        slotLayer.Dispose();
        standardWeights.Dispose();

        Assert.Equal(0, totalMismatches);
    }

    private static StandardAttentionWeights AllocateStandardLayer(
        CudaBackend cuda, Dictionary<string, GgufTensorInfo> infoMap, int i, string prefix)
    {
        ITensor Alloc(string name)
        {
            var info = infoMap[name];
            var dims = info.Dimensions.Select(d => (long)d).ToArray();
            return cuda.LoadTensor($"{prefix}.{name}", info.Type, dims,
                new byte[GgmlTypeInfo.ByteSize(info.Type, info.ElementCount)]);
        }

        ITensor? TryAlloc(string name) =>
            infoMap.ContainsKey(name) ? Alloc(name) : null;

        return new StandardAttentionWeights
        {
            AttnNorm = Alloc($"blk.{i}.attn_norm.weight"),
            PostAttnNorm = infoMap.ContainsKey($"blk.{i}.post_attention_norm.weight")
                ? Alloc($"blk.{i}.post_attention_norm.weight")
                : Alloc($"blk.{i}.ffn_norm.weight"),
            AttnQ = Alloc($"blk.{i}.attn_q.weight"),
            AttnK = Alloc($"blk.{i}.attn_k.weight"),
            AttnV = Alloc($"blk.{i}.attn_v.weight"),
            AttnO = Alloc($"blk.{i}.attn_output.weight"),
            AttnQNorm = TryAlloc($"blk.{i}.attn_q_norm.weight"),
            AttnKNorm = TryAlloc($"blk.{i}.attn_k_norm.weight"),
            AttnQBias = TryAlloc($"blk.{i}.attn_q.bias"),
            AttnKBias = TryAlloc($"blk.{i}.attn_k.bias"),
            AttnVBias = TryAlloc($"blk.{i}.attn_v.bias"),
            FfnGate = Alloc($"blk.{i}.ffn_gate.weight"),
            FfnUp = Alloc($"blk.{i}.ffn_up.weight"),
            FfnDown = Alloc($"blk.{i}.ffn_down.weight"),
        };
    }
}
