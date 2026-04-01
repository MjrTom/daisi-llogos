using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Model;

/// <summary>
/// Loads GGUF model tensors into an <see cref="IComputeBackend"/>.
/// Handles hybrid architectures with both standard attention and DeltaNet layers.
/// </summary>
public static class ModelLoader
{
    public static ModelWeights Load(GgufFile gguf, Stream stream, IComputeBackend backend, ModelConfig config)
    {
        var tensorMap = new Dictionary<string, GgufTensorInfo>(gguf.Tensors.Count);
        foreach (var t in gguf.Tensors)
            tensorMap[t.Name] = t;

        var tokenEmbedding = LoadTensor(gguf, stream, backend, tensorMap, "token_embd.weight");
        var outputNorm = LoadTensor(gguf, stream, backend, tensorMap, "output_norm.weight");
        var output = TryLoadTensor(gguf, stream, backend, tensorMap, "output.weight");

        var layers = new LayerWeights[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            if (config.IsStandardAttention(i))
                layers[i] = LoadStandardAttentionLayer(gguf, stream, backend, tensorMap, i);
            else
                layers[i] = LoadDeltaNetLayer(gguf, stream, backend, tensorMap, i);
        }

        // NOTE: Tensor fusion (FusedQKV, FusedGateUp) is disabled in stream loading.
        // The byte-level concatenation has a layout bug with GQA architectures (e.g. TinyLlama)
        // that produces garbage output on CPU. MmapModelLoader (the default path) doesn't fuse
        // and works correctly. CUDA/Vulkan backends handle individual tensors fine.
        // TODO: Fix the fused byte layout for GQA and re-enable.

        return new ModelWeights
        {
            TokenEmbedding = tokenEmbedding,
            OutputNorm = outputNorm,
            Output = output,
            Layers = layers,
        };
    }

    /// <summary>
    /// Try to fuse multiple weight tensors from raw GGUF data (before any GPU repacking).
    /// Returns null if tensor types differ or any tensor is missing.
    /// </summary>
    private static ITensor? TryFuseFromGguf(GgufFile gguf, Stream stream, IComputeBackend backend,
        Dictionary<string, GgufTensorInfo> tensorMap, int layer, params string[] suffixes)
    {
        // Gather tensor infos and check all have same type
        var infos = new GgufTensorInfo[suffixes.Length];
        for (int i = 0; i < suffixes.Length; i++)
        {
            string name = $"blk.{layer}.{suffixes[i]}.weight";
            if (!tensorMap.TryGetValue(name, out var info)) return null;
            infos[i] = info;
            if (i > 0 && info.Type != infos[0].Type) return null;
        }

        var type = infos[0].Type;
        long K = (long)infos[0].Dimensions[0];
        long totalN = 0;
        long totalBytes = 0;
        foreach (var info in infos)
        {
            totalN += info.NDimensions > 1 ? (long)info.Dimensions[1] : 1;
            var dims = ConvertDimensions(info);
            long elemCount = 1;
            foreach (var d in dims) elemCount *= d;
            totalBytes += (long)Gguf.GgmlTypeInfo.ByteSize(type, (ulong)elemCount);
        }

        // Read and concatenate raw GGUF bytes (original format, no repacking)
        var fused = new byte[totalBytes];
        int offset = 0;
        foreach (var info in infos)
        {
            var data = gguf.ReadTensorData(stream, info);
            data.CopyTo(fused.AsSpan(offset));
            offset += data.Length;
        }

        // LoadTensor will handle any repacking (e.g., Q8_0 → aligned 36-byte blocks)
        return backend.LoadTensor($"blk.{layer}.fused_{string.Join("_", suffixes)}",
            type, [K, totalN], fused);
    }

    private static StandardAttentionWeights LoadStandardAttentionLayer(
        GgufFile gguf, Stream stream, IComputeBackend backend,
        Dictionary<string, GgufTensorInfo> tensorMap, int i)
    {
        // post_attention_norm (Qwen) falls back to ffn_norm (LLaMA/standard)
        var postAttnNorm = TryLoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.post_attention_norm.weight")
            ?? LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.ffn_norm.weight");

        return new StandardAttentionWeights
        {
            AttnNorm = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.attn_norm.weight"),
            PostAttnNorm = postAttnNorm,
            AttnQ = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.attn_q.weight"),
            AttnK = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.attn_k.weight"),
            AttnV = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.attn_v.weight"),
            AttnO = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.attn_output.weight"),
            AttnQNorm = TryLoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.attn_q_norm.weight"),
            AttnKNorm = TryLoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.attn_k_norm.weight"),
            FfnGate = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.ffn_gate.weight"),
            FfnUp = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.ffn_up.weight"),
            FfnDown = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.ffn_down.weight"),
        };
    }

    private static DeltaNetWeights LoadDeltaNetLayer(
        GgufFile gguf, Stream stream, IComputeBackend backend,
        Dictionary<string, GgufTensorInfo> tensorMap, int i)
    {
        return new DeltaNetWeights
        {
            AttnNorm = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.attn_norm.weight"),
            PostAttnNorm = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.post_attention_norm.weight"),
            AttnQkv = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.attn_qkv.weight"),
            AttnGate = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.attn_gate.weight"),
            SsmA = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.ssm_a"),
            SsmAlpha = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.ssm_alpha.weight"),
            SsmBeta = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.ssm_beta.weight"),
            SsmConv1d = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.ssm_conv1d.weight"),
            SsmDtBias = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.ssm_dt.bias"),
            SsmNorm = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.ssm_norm.weight"),
            SsmOut = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.ssm_out.weight"),
            FfnGate = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.ffn_gate.weight"),
            FfnUp = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.ffn_up.weight"),
            FfnDown = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.ffn_down.weight"),
        };
    }

    private static ITensor LoadTensor(
        GgufFile gguf, Stream stream, IComputeBackend backend,
        Dictionary<string, GgufTensorInfo> tensorMap, string name)
    {
        if (!tensorMap.TryGetValue(name, out var info))
            throw new InvalidDataException($"Missing tensor: {name}");

        var data = gguf.ReadTensorData(stream, info);
        var dims = ConvertDimensions(info);
        return backend.LoadTensor(name, info.Type, dims, data);
    }

    private static ITensor? TryLoadTensor(
        GgufFile gguf, Stream stream, IComputeBackend backend,
        Dictionary<string, GgufTensorInfo> tensorMap, string name)
    {
        if (!tensorMap.TryGetValue(name, out var info))
            return null;

        var data = gguf.ReadTensorData(stream, info);
        var dims = ConvertDimensions(info);
        return backend.LoadTensor(name, info.Type, dims, data);
    }

    private static long[] ConvertDimensions(GgufTensorInfo info)
    {
        var dims = new long[info.NDimensions];
        for (int i = 0; i < info.NDimensions; i++)
            dims[i] = (long)info.Dimensions[i];
        return dims;
    }
}
