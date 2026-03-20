using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Model;

/// <summary>
/// Loads BitNet b1.58 GGUF model tensors into an <see cref="IComputeBackend"/>.
/// Separate from <see cref="ModelLoader"/> to avoid any impact on the existing path.
/// </summary>
public static class BitNetModelLoader
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
            layers[i] = LoadBitNetLayer(gguf, stream, backend, tensorMap, i);

        return new ModelWeights
        {
            TokenEmbedding = tokenEmbedding,
            OutputNorm = outputNorm,
            Output = output,
            Layers = layers,
        };
    }

    private static BitNetLayerWeights LoadBitNetLayer(
        GgufFile gguf, Stream stream, IComputeBackend backend,
        Dictionary<string, GgufTensorInfo> tensorMap, int i)
    {
        return new BitNetLayerWeights
        {
            // Pre-attention norm
            AttnNorm = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.attn_norm.weight"),
            // Post-attention norm = ffn_norm in BitNet GGUF
            PostAttnNorm = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.ffn_norm.weight"),
            // SubLN norms
            AttnSubNorm = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.attn_sub_norm.weight"),
            FfnSubNorm = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.ffn_sub_norm.weight"),
            // Attention projections (I2_S ternary)
            AttnQ = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.attn_q.weight"),
            AttnK = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.attn_k.weight"),
            AttnV = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.attn_v.weight"),
            AttnO = LoadTensor(gguf, stream, backend, tensorMap, $"blk.{i}.attn_output.weight"),
            // FFN (I2_S ternary)
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
