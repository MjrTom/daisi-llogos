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

        return new ModelWeights
        {
            TokenEmbedding = tokenEmbedding,
            OutputNorm = outputNorm,
            Output = output,
            Layers = layers,
        };
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
