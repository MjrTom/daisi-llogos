using System.IO.MemoryMappedFiles;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Tokenizer;

namespace Daisi.Llogos.Model;

/// <summary>
/// Loads GGUF model tensors via memory-mapped file access.
/// Eliminates intermediate byte[] allocations — tensor data is read directly
/// from the OS page cache via unsafe pointers.
/// </summary>
public static class MmapModelLoader
{
    public static unsafe ModelWeights Load(GgufFile gguf, string filePath, IComputeBackend backend, ModelConfig config,
        VocabRemapper? remapper = null)
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

            var tokenEmbedding = remapper != null
                ? LoadTensorRemapped(gguf, basePtr, backend, tensorMap, "token_embd.weight", remapper)
                : LoadTensor(gguf, basePtr, backend, tensorMap, "token_embd.weight");
            var outputNorm = LoadTensor(gguf, basePtr, backend, tensorMap, "output_norm.weight");
            var output = remapper != null && tensorMap.ContainsKey("output.weight")
                ? LoadTensorRemapped(gguf, basePtr, backend, tensorMap, "output.weight", remapper)
                : TryLoadTensor(gguf, basePtr, backend, tensorMap, "output.weight");

            var layers = new LayerWeights[config.NumLayers];
            for (int i = 0; i < config.NumLayers; i++)
            {
                if (config.IsBitNet)
                    layers[i] = LoadBitNetLayer(gguf, basePtr, backend, tensorMap, i);
                else if (config.IsStandardAttention(i))
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

    private static unsafe BitNetLayerWeights LoadBitNetLayer(
        GgufFile gguf, byte* basePtr, IComputeBackend backend,
        Dictionary<string, GgufTensorInfo> tensorMap, int i)
    {
        return new BitNetLayerWeights
        {
            AttnNorm = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_norm.weight"),
            PostAttnNorm = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ffn_norm.weight"),
            AttnSubNorm = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_sub_norm.weight"),
            FfnSubNorm = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ffn_sub_norm.weight"),
            AttnQ = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_q.weight"),
            AttnK = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_k.weight"),
            AttnV = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_v.weight"),
            AttnO = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_output.weight"),
            FfnGate = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ffn_gate.weight"),
            FfnUp = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ffn_up.weight"),
            FfnDown = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ffn_down.weight"),
        };
    }

    private static unsafe StandardAttentionWeights LoadStandardAttentionLayer(
        GgufFile gguf, byte* basePtr, IComputeBackend backend,
        Dictionary<string, GgufTensorInfo> tensorMap, int i)
    {
        // post_attention_norm (Qwen) falls back to ffn_norm (LLaMA/standard)
        var postAttnNorm = TryLoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.post_attention_norm.weight")
            ?? LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.ffn_norm.weight");

        return new StandardAttentionWeights
        {
            AttnNorm = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_norm.weight"),
            PostAttnNorm = postAttnNorm,
            AttnQ = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_q.weight"),
            AttnK = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_k.weight"),
            AttnV = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_v.weight"),
            AttnO = LoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_output.weight"),
            AttnQNorm = TryLoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_q_norm.weight"),
            AttnKNorm = TryLoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_k_norm.weight"),
            AttnQBias = TryLoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_q.bias"),
            AttnKBias = TryLoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_k.bias"),
            AttnVBias = TryLoadTensor(gguf, basePtr, backend, tensorMap, $"blk.{i}.attn_v.bias"),
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

    /// <summary>
    /// Load a tensor with row permutation applied. Used for embedding and output tensors
    /// when vocabulary remapping is active. Each "row" is one token's weight vector.
    /// </summary>
    private static unsafe ITensor LoadTensorRemapped(
        GgufFile gguf, byte* basePtr, IComputeBackend backend,
        Dictionary<string, GgufTensorInfo> tensorMap, string name, VocabRemapper remapper)
    {
        if (!tensorMap.TryGetValue(name, out var info))
            throw new InvalidDataException($"Missing tensor: {name}");

        long offset = gguf.GetTensorDataOffset(info);
        int byteSize = (int)info.ByteSize;
        var span = new ReadOnlySpan<byte>(basePtr + offset, byteSize);
        var dims = ConvertDimensions(info);

        // dims[1] = vocabSize (number of rows), dims[0] = hiddenDim (row width)
        int vocabSize = dims.Length > 1 ? (int)dims[1] : (int)dims[0];
        int bytesPerRow = byteSize / vocabSize;

        var permuted = remapper.PermuteRows(span, vocabSize, bytesPerRow);
        return backend.LoadTensor(name, info.Type, dims, permuted);
    }

    // ── DaisiChain: Partial Model Loading ────────────────────────────────────

    /// <summary>
    /// Load only a subset of transformer layers from a GGUF model.
    /// Used by DaisiChain pipeline stages — each host loads its assigned layer range
    /// and optionally the embedding and/or output head.
    /// Layers outside [startLayer, endLayer) are not loaded, saving VRAM.
    /// When embedding or output head is excluded, a tiny 1-element placeholder tensor
    /// is used to satisfy the required fields on ModelWeights.
    /// </summary>
    public static unsafe ModelWeights LoadPartial(GgufFile gguf, string filePath, IComputeBackend backend,
        ModelConfig config, int startLayer, int endLayer, bool includeEmbedding, bool includeOutputHead,
        VocabRemapper? remapper = null)
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

            // Embedding: load real tensor or placeholder
            ITensor tokenEmbedding;
            if (includeEmbedding)
            {
                tokenEmbedding = remapper != null
                    ? LoadTensorRemapped(gguf, basePtr, backend, tensorMap, "token_embd.weight", remapper)
                    : LoadTensor(gguf, basePtr, backend, tensorMap, "token_embd.weight");
            }
            else
            {
                tokenEmbedding = backend.CreateTensor("token_embd.weight.placeholder", GgmlType.F32, [1]);
            }

            // Output head: load real tensors or placeholders
            ITensor outputNorm;
            ITensor? output;
            if (includeOutputHead)
            {
                outputNorm = LoadTensor(gguf, basePtr, backend, tensorMap, "output_norm.weight");
                output = remapper != null && tensorMap.ContainsKey("output.weight")
                    ? LoadTensorRemapped(gguf, basePtr, backend, tensorMap, "output.weight", remapper)
                    : TryLoadTensor(gguf, basePtr, backend, tensorMap, "output.weight");
            }
            else
            {
                outputNorm = backend.CreateTensor("output_norm.weight.placeholder", GgmlType.F32, [1]);
                output = null;
            }

            // Only load layers in [startLayer, endLayer)
            var layers = new LayerWeights[config.NumLayers];
            for (int i = 0; i < config.NumLayers; i++)
            {
                if (i >= startLayer && i < endLayer)
                {
                    if (config.IsBitNet)
                        layers[i] = LoadBitNetLayer(gguf, basePtr, backend, tensorMap, i);
                    else if (config.IsStandardAttention(i))
                        layers[i] = LoadStandardAttentionLayer(gguf, basePtr, backend, tensorMap, i);
                    else
                        layers[i] = LoadDeltaNetLayer(gguf, basePtr, backend, tensorMap, i);
                }
                else
                {
                    // Placeholder layer — never executed by ForwardLayers since it
                    // only iterates [startLayer, endLayer).
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
        finally
        {
            accessor.SafeMemoryMappedViewHandle.ReleasePointer();
        }
    }

    /// <summary>
    /// Create a minimal placeholder layer with 1-element tensors.
    /// These are never used in computation — they exist only to fill the Layers array
    /// for indices outside the pipeline stage's assigned range.
    /// </summary>
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

    // ── DaisiChain: Shard-Based Partial Loading ────────────────────────────────

    /// <summary>
    /// Load only the assigned layers from per-layer shard files.
    /// Each shard file is independently memory-mapped, so only the needed data is paged in.
    /// The header GGUF provides metadata and tensor info; actual tensor data comes from shards.
    /// </summary>
    public static unsafe ModelWeights LoadPartialFromShards(GgufFile headerGguf, string shardDir,
        IComputeBackend backend, ModelConfig config, int startLayer, int endLayer,
        bool includeEmbedding, bool includeOutputHead, VocabRemapper? remapper = null)
    {
        var baseName = FindShardBaseName(shardDir);
        var mmapHandles = new List<IDisposable>();

        // Build tensor info lookup from header GGUF (needed for dimensions and types)
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
            MmapHandles = mmapHandles,
        };
    }

    /// <summary>
    /// Load a single tensor from a shard file using memory-mapped access.
    /// The mmap handles are added to the handles list for lifetime management.
    /// </summary>
    private static unsafe ITensor LoadTensorFromShard(string shardPath, string tensorName,
        Dictionary<string, GgufTensorInfo> tensorInfoMap, IComputeBackend backend,
        List<IDisposable> handles, VocabRemapper? remapper = null)
    {
        if (!tensorInfoMap.TryGetValue(tensorName, out var info))
            throw new InvalidDataException($"Missing tensor info in header GGUF: {tensorName}");

        // Parse the shard index to find the tensor's offset within the shard
        using var indexStream = File.OpenRead(shardPath);
        var shardIndex = GgufShardIndex.Read(indexStream);

        if (!shardIndex.Tensors.TryGetValue(tensorName, out var entry))
            throw new InvalidDataException($"Tensor '{tensorName}' not found in shard: {shardPath}");

        long absoluteOffset = shardIndex.DataSectionOffset + entry.offset;
        int byteSize = (int)entry.byteSize;

        // Memory-map the shard file
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

    /// <summary>
    /// Load all tensors for a single layer from its shard file.
    /// </summary>
    private static unsafe LayerWeights LoadLayerFromShard(string shardPath, int layerIndex,
        ModelConfig config, Dictionary<string, GgufTensorInfo> tensorInfoMap,
        IComputeBackend backend, List<IDisposable> handles)
    {
        // Parse the shard index
        using var indexStream = File.OpenRead(shardPath);
        var shardIndex = GgufShardIndex.Read(indexStream);

        // Memory-map the shard file
        var mmf = MemoryMappedFile.CreateFromFile(shardPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        var accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        handles.Add(accessor);
        handles.Add(mmf);

        byte* basePtr = null;
        accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);

        ITensor LoadFromShard(string name)
        {
            if (!tensorInfoMap.TryGetValue(name, out var info))
                throw new InvalidDataException($"Missing tensor info: {name}");
            if (!shardIndex.Tensors.TryGetValue(name, out var entry))
                throw new InvalidDataException($"Tensor '{name}' not found in layer shard");

            long offset = shardIndex.DataSectionOffset + entry.offset;
            var span = new ReadOnlySpan<byte>(basePtr + offset, (int)entry.byteSize);
            return backend.LoadTensor(name, info.Type, ConvertDimensions(info), span);
        }

        ITensor? TryLoadFromShard(string name)
        {
            if (!tensorInfoMap.ContainsKey(name) || !shardIndex.Tensors.ContainsKey(name))
                return null;
            return LoadFromShard(name);
        }

        int i = layerIndex;

        if (config.IsBitNet)
        {
            return new BitNetLayerWeights
            {
                AttnNorm = LoadFromShard($"blk.{i}.attn_norm.weight"),
                PostAttnNorm = LoadFromShard($"blk.{i}.ffn_norm.weight"),
                AttnSubNorm = LoadFromShard($"blk.{i}.attn_sub_norm.weight"),
                FfnSubNorm = LoadFromShard($"blk.{i}.ffn_sub_norm.weight"),
                AttnQ = LoadFromShard($"blk.{i}.attn_q.weight"),
                AttnK = LoadFromShard($"blk.{i}.attn_k.weight"),
                AttnV = LoadFromShard($"blk.{i}.attn_v.weight"),
                AttnO = LoadFromShard($"blk.{i}.attn_output.weight"),
                FfnGate = LoadFromShard($"blk.{i}.ffn_gate.weight"),
                FfnUp = LoadFromShard($"blk.{i}.ffn_up.weight"),
                FfnDown = LoadFromShard($"blk.{i}.ffn_down.weight"),
            }; // Note: BitNetLayerWeights uses init setters — works with object initializer
        }

        if (config.IsStandardAttention(i))
        {
            var postAttnNorm = TryLoadFromShard($"blk.{i}.post_attention_norm.weight")
                ?? LoadFromShard($"blk.{i}.ffn_norm.weight");

            return new StandardAttentionWeights
            {
                AttnNorm = LoadFromShard($"blk.{i}.attn_norm.weight"),
                PostAttnNorm = postAttnNorm,
                AttnQ = LoadFromShard($"blk.{i}.attn_q.weight"),
                AttnK = LoadFromShard($"blk.{i}.attn_k.weight"),
                AttnV = LoadFromShard($"blk.{i}.attn_v.weight"),
                AttnO = LoadFromShard($"blk.{i}.attn_output.weight"),
                AttnQNorm = TryLoadFromShard($"blk.{i}.attn_q_norm.weight"),
                AttnKNorm = TryLoadFromShard($"blk.{i}.attn_k_norm.weight"),
                AttnQBias = TryLoadFromShard($"blk.{i}.attn_q.bias"),
                AttnKBias = TryLoadFromShard($"blk.{i}.attn_k.bias"),
                AttnVBias = TryLoadFromShard($"blk.{i}.attn_v.bias"),
                FfnGate = LoadFromShard($"blk.{i}.ffn_gate.weight"),
                FfnUp = LoadFromShard($"blk.{i}.ffn_up.weight"),
                FfnDown = LoadFromShard($"blk.{i}.ffn_down.weight"),
            };
        }

        // DeltaNet layer
        return new DeltaNetWeights
        {
            AttnNorm = LoadFromShard($"blk.{i}.attn_norm.weight"),
            PostAttnNorm = LoadFromShard($"blk.{i}.post_attention_norm.weight"),
            AttnQkv = LoadFromShard($"blk.{i}.attn_qkv.weight"),
            AttnGate = LoadFromShard($"blk.{i}.attn_gate.weight"),
            SsmA = LoadFromShard($"blk.{i}.ssm_a"),
            SsmAlpha = LoadFromShard($"blk.{i}.ssm_alpha.weight"),
            SsmBeta = LoadFromShard($"blk.{i}.ssm_beta.weight"),
            SsmConv1d = LoadFromShard($"blk.{i}.ssm_conv1d.weight"),
            SsmDtBias = LoadFromShard($"blk.{i}.ssm_dt.bias"),
            SsmNorm = LoadFromShard($"blk.{i}.ssm_norm.weight"),
            SsmOut = LoadFromShard($"blk.{i}.ssm_out.weight"),
            FfnGate = LoadFromShard($"blk.{i}.ffn_gate.weight"),
            FfnUp = LoadFromShard($"blk.{i}.ffn_up.weight"),
            FfnDown = LoadFromShard($"blk.{i}.ffn_down.weight"),
        };
    }

    /// <summary>
    /// Find the base model name from shard files in a directory.
    /// Looks for *.gguf.header files and derives the base name.
    /// </summary>
    private static string FindShardBaseName(string shardDir)
    {
        var headerFiles = Directory.GetFiles(shardDir, "*.gguf.header");
        if (headerFiles.Length == 0)
            throw new FileNotFoundException($"No .gguf.header file found in shard directory: {shardDir}");
        // Base name = filename without the ".header" suffix
        var headerName = Path.GetFileName(headerFiles[0]);
        return headerName[..^".header".Length]; // e.g., "model.gguf"
    }

    private static long[] ConvertDimensions(GgufTensorInfo info)
    {
        var dims = new long[info.NDimensions];
        for (int i = 0; i < info.NDimensions; i++)
            dims[i] = (long)info.Dimensions[i];
        return dims;
    }
}
