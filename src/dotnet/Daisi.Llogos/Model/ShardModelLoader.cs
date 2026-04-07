using System.IO.MemoryMappedFiles;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Model;

/// <summary>
/// Loads model weights from per-layer shard files (produced by GgufSplitter).
/// Completely self-contained — does not modify MmapModelLoader or any existing code.
/// Each shard file is independently memory-mapped.
/// </summary>
public static class ShardModelLoader
{
    /// <summary>
    /// Load only the assigned layers from per-layer shard files.
    /// The header GGUF provides metadata and tensor info; actual tensor data comes from shards.
    /// </summary>
    public static unsafe ModelWeights LoadPartialFromShards(GgufFile headerGguf, string shardDir,
        IComputeBackend backend, ModelConfig config, int startLayer, int endLayer,
        bool includeEmbedding, bool includeOutputHead, VocabRemapper? remapper = null)
    {
        var baseName = FindShardBaseName(shardDir);
        var mmapHandles = new List<IDisposable>();

        var tensorInfoMap = new Dictionary<string, GgufTensorInfo>(headerGguf.Tensors.Count);
        foreach (var t in headerGguf.Tensors)
            tensorInfoMap[t.Name] = t;

        // Embedding
        ITensor tokenEmbedding;
        if (includeEmbedding)
        {
            var embedPath = Path.Combine(shardDir, $"{baseName}.embed");
            tokenEmbedding = remapper != null
                ? LoadTensorFromShard(embedPath, "token_embd.weight", tensorInfoMap, backend, mmapHandles, remapper)
                : LoadTensorFromShard(embedPath, "token_embd.weight", tensorInfoMap, backend, mmapHandles);
        }
        else
        {
            tokenEmbedding = backend.CreateTensor("token_embd.weight.placeholder", GgmlType.F32, [1]);
        }

        // Output head
        ITensor outputNorm;
        ITensor? output;
        if (includeOutputHead)
        {
            var outputPath = Path.Combine(shardDir, $"{baseName}.output");
            outputNorm = LoadTensorFromShard(outputPath, "output_norm.weight", tensorInfoMap, backend, mmapHandles);
            output = tensorInfoMap.ContainsKey("output.weight")
                ? (remapper != null
                    ? LoadTensorFromShard(outputPath, "output.weight", tensorInfoMap, backend, mmapHandles, remapper)
                    : LoadTensorFromShard(outputPath, "output.weight", tensorInfoMap, backend, mmapHandles))
                : null;
        }
        else
        {
            outputNorm = backend.CreateTensor("output_norm.weight.placeholder", GgmlType.F32, [1]);
            output = null;
        }

        // Layers
        var layers = new LayerWeights[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            if (i >= startLayer && i < endLayer)
            {
                var layerPath = Path.Combine(shardDir, $"{baseName}.layer.{i}");
                layers[i] = LoadLayerFromShard(layerPath, i, config, tensorInfoMap, backend, mmapHandles);
            }
            else
            {
                layers[i] = CreatePlaceholderLayer(backend, i);
            }
        }

        return new ModelWeights
        {
            TokenEmbedding = tokenEmbedding,
            OutputNorm = outputNorm,
            Output = output,
            Layers = layers,
        };
    }

    private static unsafe ITensor LoadTensorFromShard(string shardPath, string tensorName,
        Dictionary<string, GgufTensorInfo> tensorInfoMap, IComputeBackend backend,
        List<IDisposable> handles, VocabRemapper? remapper = null)
    {
        if (!tensorInfoMap.TryGetValue(tensorName, out var info))
            throw new InvalidDataException($"Missing tensor info in header GGUF: {tensorName}");

        using var indexStream = File.OpenRead(shardPath);
        var shardIndex = GgufShardIndex.Read(indexStream);

        if (!shardIndex.Tensors.TryGetValue(tensorName, out var entry))
            throw new InvalidDataException($"Tensor '{tensorName}' not found in shard: {shardPath}");

        long absoluteOffset = shardIndex.DataSectionOffset + entry.offset;
        int byteSize = (int)entry.byteSize;

        var mmf = MemoryMappedFile.CreateFromFile(shardPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        var accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        handles.Add(accessor);
        handles.Add(mmf);

        byte* basePtr = null;
        accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);

        var span = new ReadOnlySpan<byte>(basePtr + absoluteOffset, byteSize);
        var dims = ConvertDimensions(info);

        if (remapper != null)
        {
            int vocabSize = dims.Length > 1 ? (int)dims[1] : (int)dims[0];
            int bytesPerRow = byteSize / vocabSize;
            var permuted = remapper.PermuteRows(span, vocabSize, bytesPerRow);
            return backend.LoadTensor(tensorName, info.Type, dims, permuted);
        }

        return backend.LoadTensor(tensorName, info.Type, dims, span);
    }

    private static unsafe LayerWeights LoadLayerFromShard(string shardPath, int layerIndex,
        ModelConfig config, Dictionary<string, GgufTensorInfo> tensorInfoMap,
        IComputeBackend backend, List<IDisposable> handles)
    {
        using var indexStream = File.OpenRead(shardPath);
        var shardIndex = GgufShardIndex.Read(indexStream);

        var mmf = MemoryMappedFile.CreateFromFile(shardPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        var accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        handles.Add(accessor);
        handles.Add(mmf);

        byte* basePtr = null;
        accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);

        ITensor Load(string name)
        {
            if (!tensorInfoMap.TryGetValue(name, out var info))
                throw new InvalidDataException($"Missing tensor info: {name}");
            if (!shardIndex.Tensors.TryGetValue(name, out var entry))
                throw new InvalidDataException($"Tensor '{name}' not found in layer shard");
            long offset = shardIndex.DataSectionOffset + entry.offset;
            var span = new ReadOnlySpan<byte>(basePtr + offset, (int)entry.byteSize);
            return backend.LoadTensor(name, info.Type, ConvertDimensions(info), span);
        }

        ITensor? TryLoad(string name)
        {
            if (!tensorInfoMap.ContainsKey(name) || !shardIndex.Tensors.ContainsKey(name))
                return null;
            return Load(name);
        }

        int i = layerIndex;

        if (config.IsBitNet)
        {
            return new BitNetLayerWeights
            {
                AttnNorm = Load($"blk.{i}.attn_norm.weight"),
                PostAttnNorm = Load($"blk.{i}.ffn_norm.weight"),
                AttnSubNorm = Load($"blk.{i}.attn_sub_norm.weight"),
                FfnSubNorm = Load($"blk.{i}.ffn_sub_norm.weight"),
                AttnQ = Load($"blk.{i}.attn_q.weight"),
                AttnK = Load($"blk.{i}.attn_k.weight"),
                AttnV = Load($"blk.{i}.attn_v.weight"),
                AttnO = Load($"blk.{i}.attn_output.weight"),
                FfnGate = Load($"blk.{i}.ffn_gate.weight"),
                FfnUp = Load($"blk.{i}.ffn_up.weight"),
                FfnDown = Load($"blk.{i}.ffn_down.weight"),
            };
        }

        if (config.IsStandardAttention(i))
        {
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
                AttnQBias = TryLoad($"blk.{i}.attn_q.bias"),
                AttnKBias = TryLoad($"blk.{i}.attn_k.bias"),
                AttnVBias = TryLoad($"blk.{i}.attn_v.bias"),
                FfnGate = Load($"blk.{i}.ffn_gate.weight"),
                FfnUp = Load($"blk.{i}.ffn_up.weight"),
                FfnDown = Load($"blk.{i}.ffn_down.weight"),
            };
        }

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

    private static StandardAttentionWeights CreatePlaceholderLayer(IComputeBackend backend, int layerIndex)
    {
        ITensor Placeholder(string name) =>
            backend.CreateTensor($"blk.{layerIndex}.{name}.placeholder", GgmlType.F32, [1]);

        return new StandardAttentionWeights
        {
            AttnNorm = Placeholder("attn_norm"),
            PostAttnNorm = Placeholder("ffn_norm"),
            AttnQ = Placeholder("attn_q"),
            AttnK = Placeholder("attn_k"),
            AttnV = Placeholder("attn_v"),
            AttnO = Placeholder("attn_output"),
            FfnGate = Placeholder("ffn_gate"),
            FfnUp = Placeholder("ffn_up"),
            FfnDown = Placeholder("ffn_down"),
        };
    }

    public static string FindShardBaseName(string shardDir)
    {
        var headerFiles = Directory.GetFiles(shardDir, "*.gguf.header");
        if (headerFiles.Length == 0)
            throw new FileNotFoundException($"No .gguf.header file found in shard directory: {shardDir}");
        var headerName = Path.GetFileName(headerFiles[0]);
        return headerName[..^".header".Length];
    }

    private static long[] ConvertDimensions(GgufTensorInfo info)
    {
        var dims = new long[info.NDimensions];
        for (int i = 0; i < info.NDimensions; i++)
            dims[i] = (long)info.Dimensions[i];
        return dims;
    }
}
