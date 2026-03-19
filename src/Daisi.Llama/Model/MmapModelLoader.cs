using System.IO.MemoryMappedFiles;
using Daisi.Llama.Gguf;

namespace Daisi.Llama.Model;

/// <summary>
/// Loads GGUF model tensors via memory-mapped file access.
/// Eliminates intermediate byte[] allocations — tensor data is read directly
/// from the OS page cache via unsafe pointers.
/// </summary>
public static class MmapModelLoader
{
    public static unsafe ModelWeights Load(GgufFile gguf, string filePath, IComputeBackend backend, ModelConfig config)
    {
        using var mmf = MemoryMappedFile.CreateFromFile(filePath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

        byte* basePtr = null;
        accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
        try
        {
            var tensorMap = new Dictionary<string, GgufTensorInfo>(gguf.Tensors.Count);
            foreach (var t in gguf.Tensors)
                tensorMap[t.Name] = t;

            var tokenEmbedding = LoadTensor(gguf, basePtr, backend, tensorMap, "token_embd.weight");
            var outputNorm = LoadTensor(gguf, basePtr, backend, tensorMap, "output_norm.weight");
            var output = TryLoadTensor(gguf, basePtr, backend, tensorMap, "output.weight");

            var layers = new LayerWeights[config.NumLayers];
            for (int i = 0; i < config.NumLayers; i++)
            {
                if (config.IsStandardAttention(i))
                    layers[i] = LoadStandardAttentionLayer(gguf, basePtr, backend, tensorMap, i);
                else
                    layers[i] = LoadDeltaNetLayer(gguf, basePtr, backend, tensorMap, i);
            }

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

    private static unsafe StandardAttentionWeights LoadStandardAttentionLayer(
        GgufFile gguf, byte* basePtr, IComputeBackend backend,
        Dictionary<string, GgufTensorInfo> tensorMap, int i)
    {
        return new StandardAttentionWeights
        {
            AttnNorm = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_norm.weight"),
            PostAttnNorm = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.post_attention_norm.weight"),
            AttnQ = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_q.weight"),
            AttnK = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_k.weight"),
            AttnV = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_v.weight"),
            AttnO = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_output.weight"),
            AttnQNorm = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_q_norm.weight"),
            AttnKNorm = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_k_norm.weight"),
            FfnGate = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ffn_gate.weight"),
            FfnUp = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ffn_up.weight"),
            FfnDown = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ffn_down.weight"),
        };
    }

    private static unsafe DeltaNetWeights LoadDeltaNetLayer(
        GgufFile gguf, byte* basePtr, IComputeBackend backend,
        Dictionary<string, GgufTensorInfo> tensorMap, int i)
    {
        return new DeltaNetWeights
        {
            AttnNorm = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_norm.weight"),
            PostAttnNorm = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.post_attention_norm.weight"),
            AttnQkv = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_qkv.weight"),
            AttnGate = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_gate.weight"),
            SsmA = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ssm_a"),
            SsmAlpha = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ssm_alpha.weight"),
            SsmBeta = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ssm_beta.weight"),
            SsmConv1d = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ssm_conv1d.weight"),
            SsmDtBias = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ssm_dt.bias"),
            SsmNorm = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ssm_norm.weight"),
            SsmOut = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ssm_out.weight"),
            FfnGate = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ffn_gate.weight"),
            FfnUp = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ffn_up.weight"),
            FfnDown = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ffn_down.weight"),
        };
    }

    private static unsafe ITensor LoadTensor(
        GgufFile gguf, byte* basePtr, IComputeBackend backend,
        Dictionary<string, GgufTensorInfo> tensorMap, string name)
    {
        if (!tensorMap.TryGetValue(name, out var info))
            throw new InvalidDataException($"Missing tensor: {name}");

        long offset = gguf.GetTensorDataOffset(info);
        int byteSize = (int)info.ByteSize;
        var span = new ReadOnlySpan<byte>(basePtr + offset, byteSize);
        var dims = ConvertDimensions(info);
        return backend.LoadTensor(name, info.Type, dims, span);
    }

    private static unsafe ITensor? TryLoadTensor(
        GgufFile gguf, byte* basePtr, IComputeBackend backend,
        Dictionary<string, GgufTensorInfo> tensorMap, string name)
    {
        if (!tensorMap.TryGetValue(name, out var info))
            return null;

        long offset = gguf.GetTensorDataOffset(info);
        int byteSize = (int)info.ByteSize;
        var span = new ReadOnlySpan<byte>(basePtr + offset, byteSize);
        var dims = ConvertDimensions(info);
        return backend.LoadTensor(name, info.Type, dims, span);
    }

    private static long[] ConvertDimensions(GgufTensorInfo info)
    {
        var dims = new long[info.NDimensions];
        for (int i = 0; i < info.NDimensions; i++)
            dims[i] = (long)info.Dimensions[i];
        return dims;
    }
}
