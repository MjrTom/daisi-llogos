using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Cuda;

/// <summary>
/// Single-GPU layer offloading: loads most layers into VRAM, offloads the rest
/// to pinned host memory (GPU-accessible via PCIe). Uses DaisiChain's
/// ForwardLayers to process each segment — no changes to ForwardPass needed.
///
/// VRAM layers run at HBM bandwidth (~960 GB/s on RTX 5080).
/// Pinned layers run at PCIe bandwidth (~50 GB/s on PCIe 5.0 x16).
/// The GPU reads pinned weights transparently via DevicePtr — no explicit DMA.
///
/// Combined with TurboQuant (compresses KV cache → frees VRAM for more weight layers),
/// this enables running models that barely fit in VRAM at reasonable speeds.
/// </summary>
public static class CudaLayerOffload
{
    /// <summary>The offload swapper from the last LoadWithOffload call. Used by CLI to create OffloadForwardPass.</summary>
    public static OffloadSwapper? Swapper { get; private set; }

    /// <summary>
    /// Load a model with layer offloading: first N layers in VRAM, rest in pinned host memory.
    /// The returned ModelWeights works with ForwardPass — offloaded layers are transparently
    /// read over PCIe when ForwardLayers reaches them.
    /// </summary>
    /// <param name="gguf">Parsed GGUF file.</param>
    /// <param name="filePath">Path to GGUF file for mmap.</param>
    /// <param name="backend">CUDA backend (used for both VRAM and pinned tensor creation).</param>
    /// <param name="config">Model config.</param>
    /// <param name="gpuLayers">Number of layers to keep in VRAM. Rest go to pinned RAM.</param>
    /// <param name="remapper">Optional vocab remapper.</param>
    public static unsafe ModelWeights LoadWithOffload(GgufFile gguf, string filePath,
        CudaBackend backend, ModelConfig config, int gpuLayers, VocabRemapper? remapper = null)
    {
        int totalLayers = config.NumLayers;
        if (gpuLayers >= totalLayers)
        {
            // All layers fit — no offloading needed, use standard load
            return MmapModelLoader.Load(gguf, filePath, backend, config, remapper);
        }

        using var mmf = System.IO.MemoryMappedFiles.MemoryMappedFile.CreateFromFile(
            filePath, System.IO.FileMode.Open, null, 0,
            System.IO.MemoryMappedFiles.MemoryMappedFileAccess.Read);
        using var accessor = mmf.CreateViewAccessor(0, 0,
            System.IO.MemoryMappedFiles.MemoryMappedFileAccess.Read);

        byte* basePtr = null;
        accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
        try
        {
            var tensorMap = new Dictionary<string, GgufTensorInfo>(gguf.Tensors.Count);
            foreach (var t in gguf.Tensors)
                tensorMap[t.Name] = t;

            // Embedding and output head always go to VRAM (small, used every token)
            // Note: vocab remapping not supported with offloaded loading — use standard load
            ITensor tokenEmbedding = LoadTensorVram(gguf, basePtr, backend, tensorMap, "token_embd.weight");
            ITensor outputNorm = LoadTensorVram(gguf, basePtr, backend, tensorMap, "output_norm.weight");
            ITensor? output = TryLoadTensorVram(gguf, basePtr, backend, tensorMap, "output.weight");

            // Load layers: first gpuLayers to VRAM, rest to pinned host memory
            var layers = new LayerWeights[totalLayers];
            int pinnedCount = 0;

            for (int i = 0; i < totalLayers; i++)
            {
                bool pinned = i >= gpuLayers;

                if (config.IsStandardAttention(i))
                    layers[i] = LoadStandardAttentionLayer(gguf, basePtr, backend, tensorMap, i, pinned);
                else
                    layers[i] = LoadDeltaNetLayer(gguf, basePtr, backend, tensorMap, i, pinned);

                if (pinned) pinnedCount++;
            }

            // Create and register offload swapper for DMA prefetching
            var swapper = new OffloadSwapper(backend, gpuLayers, totalLayers);
            for (int i = gpuLayers; i < totalLayers; i++)
                swapper.RegisterLayer(i, layers[i]);
            swapper.AllocateStaging();

            Console.Error.WriteLine($"  Layer offload: {gpuLayers} VRAM + {pinnedCount} pinned ({pinnedCount * 100 / totalLayers}% offloaded, DMA staging ready)");

            Swapper = swapper;

            return new ModelWeights
            {
                TokenEmbedding = tokenEmbedding,
                OutputNorm = outputNorm,
                Output = output,
                Layers = layers,
            };
        }
        finally
        {
            accessor.SafeMemoryMappedViewHandle.ReleasePointer();
        }
    }

    // ── Tensor loading helpers ────────────────────────────────────────────────

    private static unsafe ITensor LoadTensorVram(GgufFile gguf, byte* basePtr,
        CudaBackend backend, Dictionary<string, GgufTensorInfo> map, string name)
    {
        var info = map[name];
        long offset = gguf.GetTensorDataOffset(info);
        int byteSize = (int)info.ByteSize;
        var data = new ReadOnlySpan<byte>(basePtr + offset, byteSize);
        var dims = ConvertDims(info);
        return backend.LoadTensor(name, info.Type, dims, data);
    }

    private static unsafe ITensor? TryLoadTensorVram(GgufFile gguf, byte* basePtr,
        CudaBackend backend, Dictionary<string, GgufTensorInfo> map, string name)
    {
        return map.ContainsKey(name) ? LoadTensorVram(gguf, basePtr, backend, map, name) : null;
    }

    private static unsafe ITensor LoadTensorPinned(GgufFile gguf, byte* basePtr,
        CudaBackend backend, Dictionary<string, GgufTensorInfo> map, string name)
    {
        var info = map[name];
        long offset = gguf.GetTensorDataOffset(info);
        int byteSize = (int)info.ByteSize;
        var data = new ReadOnlySpan<byte>(basePtr + offset, byteSize);
        var dims = ConvertDims(info);
        return backend.LoadTensorPinned(name, info.Type, dims, data);
    }

    private static unsafe StandardAttentionWeights LoadStandardAttentionLayer(
        GgufFile gguf, byte* basePtr, CudaBackend backend,
        Dictionary<string, GgufTensorInfo> map, int i, bool pinned)
    {
        ITensor Load(string name) => pinned
            ? LoadTensorPinned(gguf, basePtr, backend, map, name)
            : LoadTensorVram(gguf, basePtr, backend, map, name);

        ITensor? TryLoad(string name) =>
            map.ContainsKey(name) ? Load(name) : null;

        var postAttnNorm = TryLoad($"blk.{i}.post_attention_norm.weight")
            ?? Load($"blk.{i}.ffn_norm.weight");

        return new StandardAttentionWeights
        {
            AttnNorm = Load($"blk.{i}.attn_norm.weight"),
            PostAttnNorm = postAttnNorm,
            AttnQ = Load($"blk.{i}.attn_q.weight"),
            AttnK = Load($"blk.{i}.attn_k.weight"),
            AttnV = Load($"blk.{i}.attn_v.weight"),
            AttnO = Load($"blk.{i}.attn_output.weight"),
            AttnQNorm = TryLoad($"blk.{i}.attn_q_norm.weight"),
            AttnKNorm = TryLoad($"blk.{i}.attn_k_norm.weight"),
            FfnGate = Load($"blk.{i}.ffn_gate.weight"),
            FfnUp = Load($"blk.{i}.ffn_up.weight"),
            FfnDown = Load($"blk.{i}.ffn_down.weight"),
            FusedGateUp = TryLoad($"blk.{i}.ffn_gate_up.weight"),
            FusedQKV = TryLoad($"blk.{i}.attn_qkv.weight"),
        };
    }

    private static unsafe DeltaNetWeights LoadDeltaNetLayer(
        GgufFile gguf, byte* basePtr, CudaBackend backend,
        Dictionary<string, GgufTensorInfo> map, int i, bool pinned)
    {
        ITensor Load(string name) => pinned
            ? LoadTensorPinned(gguf, basePtr, backend, map, name)
            : LoadTensorVram(gguf, basePtr, backend, map, name);

        ITensor? TryLoad(string name) =>
            map.ContainsKey(name) ? Load(name) : null;

        return new DeltaNetWeights
        {
            AttnNorm = Load($"blk.{i}.attn_norm.weight"),
            PostAttnNorm = Load($"blk.{i}.post_attention_norm.weight"),
            AttnQkv = Load($"blk.{i}.attn_qkv.weight"),
            AttnGate = Load($"blk.{i}.attn_gate.weight"),
            SsmA = Load($"blk.{i}.ssm_a"),
            SsmAlpha = Load($"blk.{i}.ssm_alpha.weight"),
            SsmBeta = Load($"blk.{i}.ssm_beta.weight"),
            SsmConv1d = Load($"blk.{i}.ssm_conv1d.weight"),
            SsmDtBias = Load($"blk.{i}.ssm_dt.bias"),
            SsmNorm = Load($"blk.{i}.ssm_norm.weight"),
            SsmOut = Load($"blk.{i}.ssm_out.weight"),
            FfnGate = Load($"blk.{i}.ffn_gate.weight"),
            FfnUp = Load($"blk.{i}.ffn_up.weight"),
            FfnDown = Load($"blk.{i}.ffn_down.weight"),
        };
    }

    private static long[] ConvertDims(GgufTensorInfo info)
    {
        var dims = new long[info.NDimensions];
        for (int i = 0; i < info.NDimensions; i++)
            dims[i] = (long)info.Dimensions[i];
        return dims;
    }
}
