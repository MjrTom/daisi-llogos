using System.Diagnostics;
using System.IO.MemoryMappedFiles;
using System.Reflection;
using System.Runtime.InteropServices;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Cuda;

/// <summary>
/// Double-buffered GPU offloading forward pass: runs all layers on GPU by streaming
/// weight data from per-layer shard files. Each layer's weights are uploaded to GPU
/// before computing that layer, then the next layer's weights overwrite them.
///
/// Self-contained: does NOT modify ForwardPass, CudaBackend, IComputeBackend,
/// or any existing hot-path code. Uses reflection for ForwardPass weight swapping.
/// </summary>
public sealed class PipelinedForwardPass : IForwardPass
{
    private readonly CudaBackend _cuda;
    private readonly ModelConfig _config;
    private readonly ForwardPass _forward;
    private readonly bool _gpuAligned;

    // Weight slots (per-layer GPU tensors, 2 slots for double buffering)
    private readonly ModelWeights _weightsA;
    private readonly ModelWeights _weightsB;

    // Persistent state
    private readonly KvCache _kvCache;
    private readonly DeltaNetState _deltaState;

    // Layer shard data (mmap pointers)
    private readonly LayerShardData[] _layerData;
    private readonly List<IDisposable> _mmapHandles = [];

    // Reflection: swap ForwardPass._weights without modifying ForwardPass
    private static readonly FieldInfo WeightsField =
        typeof(ForwardPass).GetField("_weights", BindingFlags.NonPublic | BindingFlags.Instance)!;

    public IKvCache KvCache => _kvCache;

    private PipelinedForwardPass(
        CudaBackend cuda, ModelConfig config, ForwardPass forward,
        ModelWeights weightsA, ModelWeights weightsB,
        KvCache kvCache, DeltaNetState deltaState,
        LayerShardData[] layerData, List<IDisposable> mmapHandles,
        bool gpuAligned)
    {
        _cuda = cuda;
        _config = config;
        _forward = forward;
        _weightsA = weightsA;
        _weightsB = weightsB;
        _kvCache = kvCache;
        _deltaState = deltaState;
        _layerData = layerData;
        _mmapHandles = mmapHandles;
        _gpuAligned = gpuAligned;
    }

    public ReadOnlySpan<float> Forward(int tokenId, int position)
    {
        SwapWeights(_weightsA);
        _forward.ForwardEmbedding(tokenId);

        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            int slot = layer % 2;
            var weights = slot == 0 ? _weightsA : _weightsB;
            UploadLayer(layer, weights);
            SwapWeights(weights);
            _forward.ForwardLayers(layer, layer + 1, position,
                continuation: layer > 0, isFinal: layer == _config.NumLayers - 1);
        }

        var logits = new float[_config.VocabSize];
        _forward.ForwardOutputHead(logits);
        return logits;
    }

    public void ForwardHidden(int tokenId, int position)
    {
        SwapWeights(_weightsA);
        _forward.ForwardEmbedding(tokenId);

        for (int layer = 0; layer < _config.NumLayers; layer++)
        {
            int slot = layer % 2;
            var weights = slot == 0 ? _weightsA : _weightsB;
            UploadLayer(layer, weights);
            SwapWeights(weights);
            _forward.ForwardLayers(layer, layer + 1, position,
                continuation: layer > 0, isFinal: layer == _config.NumLayers - 1);
        }
    }

    public void ResetState() => _forward.ResetState();

    /// <summary>Swap ForwardPass._weights via reflection (zero impact on normal path).</summary>
    private void SwapWeights(ModelWeights weights) =>
        WeightsField.SetValue(_forward, weights);

    /// <summary>Upload layer weights from mmap shard into the slot's GPU tensors.</summary>
    private unsafe void UploadLayer(int layerIndex, ModelWeights targetWeights)
    {
        var data = _layerData[layerIndex];
        var targets = LayerShardData.GetTargetTensors(data, targetWeights.Layers[layerIndex]);

        for (int t = 0; t < data.Tensors.Count; t++)
        {
            var tr = data.Tensors[t];
            byte* src = tr.MmapPtr;

            // Repack quantized data to GPU-aligned layout (or straight copy if gpuAligned)
            byte[] repacked;
            if (_gpuAligned)
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

            targets[t].CopyFrom(repacked);
        }
    }

    public void Dispose()
    {
        _forward.Dispose();
        _kvCache.Dispose();
        _deltaState.Dispose();
        _weightsA.Dispose();
        _weightsB.Dispose();
        foreach (var h in _mmapHandles) h.Dispose();
    }

    // ── Factory ─────────────────────────────────────────────────────────────

    public static PipelinedForwardPass Create(
        GgufFile gguf, string shardDir, ModelConfig config,
        CudaBackend cuda, int maxContext = 2048)
    {
        var sw = Stopwatch.StartNew();
        var mmapHandles = new List<IDisposable>();

        // Check manifest for gpuAligned
        var baseName = ShardModelLoader.FindShardBaseName(shardDir);
        var manifestPath = Path.Combine(shardDir, $"{baseName}.manifest.json");
        bool gpuAligned = false;
        if (File.Exists(manifestPath))
            gpuAligned = GgufShardManifest.FromJsonFile(manifestPath).GpuAligned;

        // Persistent: embedding + output on GPU
        var embedWeights = ShardModelLoader.LoadPartialFromShards(
            gguf, shardDir, cuda, config, 0, 0, true, false);
        var outputWeights = ShardModelLoader.LoadPartialFromShards(
            gguf, shardDir, cuda, config, config.NumLayers, config.NumLayers, false, true);

        // KV cache for all layers
        var kvCache = new KvCache(cuda, config, maxSeqLen: maxContext);

        // Allocate per-layer GPU tensors for two slots (via LoadTensor with zero data)
        var tensorInfoMap = new Dictionary<string, GgufTensorInfo>();
        foreach (var t in gguf.Tensors) tensorInfoMap[t.Name] = t;

        var weightsA = AllocateSlot(cuda, config, tensorInfoMap, "slotA",
            embedWeights, outputWeights);
        var weightsB = AllocateSlot(cuda, config, tensorInfoMap, "slotB",
            embedWeights, outputWeights);

        // DeltaNetState
        var deltaState = new DeltaNetState(cuda, config, weightsA);

        // Mmap layer shards
        var layerData = new LayerShardData[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            var shardPath = Path.Combine(shardDir, $"{baseName}.layer.{i}");
            layerData[i] = LayerShardData.Load(shardPath, i, config, gguf, mmapHandles);
        }

        // Create ForwardPass with slot A weights (graph capture disabled for weight swapping)
        var forward = new ForwardPass(cuda, config, weightsA, kvCache, deltaState);
        forward.DisableGraphCapture();

        Console.Error.WriteLine($"  Pipelined: {config.NumLayers} layers, loaded in {sw.Elapsed.TotalSeconds:F1}s");

        return new PipelinedForwardPass(
            cuda, config, forward, weightsA, weightsB,
            kvCache, deltaState, layerData, mmapHandles, gpuAligned);
    }

    private static ModelWeights AllocateSlot(CudaBackend cuda, ModelConfig config,
        Dictionary<string, GgufTensorInfo> tensorInfoMap, string prefix,
        ModelWeights embedWeights, ModelWeights outputWeights)
    {
        var layers = new LayerWeights[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            if (config.IsStandardAttention(i))
                layers[i] = AllocateStandardLayer(cuda, tensorInfoMap, i, prefix);
            else
                layers[i] = AllocateDeltaNetLayer(cuda, tensorInfoMap, i, prefix);
        }

        return new ModelWeights
        {
            // Share embed/output with the persistent weights
            TokenEmbedding = embedWeights.TokenEmbedding,
            OutputNorm = outputWeights.OutputNorm,
            Output = outputWeights.Output,
            Layers = layers,
        };
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

    private static DeltaNetWeights AllocateDeltaNetLayer(
        CudaBackend cuda, Dictionary<string, GgufTensorInfo> infoMap, int i, string prefix)
    {
        ITensor Alloc(string name)
        {
            var info = infoMap[name];
            var dims = info.Dimensions.Select(d => (long)d).ToArray();
            return cuda.LoadTensor($"{prefix}.{name}", info.Type, dims,
                new byte[GgmlTypeInfo.ByteSize(info.Type, info.ElementCount)]);
        }

        return new DeltaNetWeights
        {
            AttnNorm = Alloc($"blk.{i}.attn_norm.weight"),
            PostAttnNorm = Alloc($"blk.{i}.post_attention_norm.weight"),
            AttnQkv = Alloc($"blk.{i}.attn_qkv.weight"),
            AttnGate = Alloc($"blk.{i}.attn_gate.weight"),
            SsmA = Alloc($"blk.{i}.ssm_a"),
            SsmAlpha = Alloc($"blk.{i}.ssm_alpha.weight"),
            SsmBeta = Alloc($"blk.{i}.ssm_beta.weight"),
            SsmConv1d = Alloc($"blk.{i}.ssm_conv1d.weight"),
            SsmDtBias = Alloc($"blk.{i}.ssm_dt.bias"),
            SsmNorm = Alloc($"blk.{i}.ssm_norm.weight"),
            SsmOut = Alloc($"blk.{i}.ssm_out.weight"),
            FfnGate = Alloc($"blk.{i}.ffn_gate.weight"),
            FfnUp = Alloc($"blk.{i}.ffn_up.weight"),
            FfnDown = Alloc($"blk.{i}.ffn_down.weight"),
        };
    }

    // ── Layer shard data ────────────────────────────────────────────────────

    internal sealed class LayerShardData
    {
        internal List<TensorRef> Tensors { get; } = [];

        internal unsafe struct TensorRef
        {
            public string ShortName;
            public byte* MmapPtr;
            public int RawByteSize;
            public GgmlType Type;
            public int NDimensions;
        }

        internal static List<ITensor> GetTargetTensors(LayerShardData data, LayerWeights layer)
        {
            var targets = new List<ITensor>(data.Tensors.Count);
            if (layer is StandardAttentionWeights saw)
            {
                foreach (var tr in data.Tensors)
                    targets.Add(tr.ShortName switch
                    {
                        "attn_norm" => saw.AttnNorm,
                        "post_attn_norm" => saw.PostAttnNorm,
                        "attn_q" => saw.AttnQ,
                        "attn_k" => saw.AttnK,
                        "attn_v" => saw.AttnV,
                        "attn_o" => saw.AttnO,
                        "attn_q_norm" => saw.AttnQNorm!,
                        "attn_k_norm" => saw.AttnKNorm!,
                        "ffn_gate" => saw.FfnGate,
                        "ffn_up" => saw.FfnUp,
                        "ffn_down" => saw.FfnDown,
                        _ => throw new InvalidDataException($"Unknown tensor: {tr.ShortName}")
                    });
            }
            else if (layer is DeltaNetWeights dnw)
            {
                foreach (var tr in data.Tensors)
                    targets.Add(tr.ShortName switch
                    {
                        "attn_norm" => dnw.AttnNorm,
                        "post_attn_norm" => dnw.PostAttnNorm,
                        "attn_qkv" => dnw.AttnQkv,
                        "attn_gate" => dnw.AttnGate,
                        "ssm_a" => dnw.SsmA,
                        "ssm_alpha" => dnw.SsmAlpha,
                        "ssm_beta" => dnw.SsmBeta,
                        "ssm_conv1d" => dnw.SsmConv1d,
                        "ssm_dt_bias" => dnw.SsmDtBias,
                        "ssm_norm" => dnw.SsmNorm,
                        "ssm_out" => dnw.SsmOut,
                        "ffn_gate" => dnw.FfnGate,
                        "ffn_up" => dnw.FfnUp,
                        "ffn_down" => dnw.FfnDown,
                        _ => throw new InvalidDataException($"Unknown tensor: {tr.ShortName}")
                    });
            }
            return targets;
        }

        internal static unsafe LayerShardData Load(string shardPath, int layerIndex,
            ModelConfig config, GgufFile gguf, List<IDisposable> mmapHandles)
        {
            var data = new LayerShardData();
            using var indexStream = File.OpenRead(shardPath);
            var shardIndex = GgufShardIndex.Read(indexStream);

            var mmf = MemoryMappedFile.CreateFromFile(shardPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            var accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            mmapHandles.Add(accessor);
            mmapHandles.Add(mmf);

            byte* basePtr = null;
            accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);

            var tensorInfoMap = new Dictionary<string, GgufTensorInfo>();
            foreach (var t in gguf.Tensors) tensorInfoMap[t.Name] = t;

            int i = layerIndex;
            void AddRef(string shortName, string fullName)
            {
                if (!shardIndex.Tensors.TryGetValue(fullName, out var entry)) return;
                if (!tensorInfoMap.TryGetValue(fullName, out var info)) return;
                byte* ptr = basePtr + shardIndex.DataSectionOffset + entry.offset;
                data.Tensors.Add(new TensorRef
                {
                    ShortName = shortName, MmapPtr = ptr, RawByteSize = (int)entry.byteSize,
                    Type = info.Type, NDimensions = info.Dimensions.Length,
                });
            }

            AddRef("attn_norm", $"blk.{i}.attn_norm.weight");
            AddRef("post_attn_norm", tensorInfoMap.ContainsKey($"blk.{i}.post_attention_norm.weight")
                ? $"blk.{i}.post_attention_norm.weight" : $"blk.{i}.ffn_norm.weight");
            AddRef("ffn_gate", $"blk.{i}.ffn_gate.weight");
            AddRef("ffn_up", $"blk.{i}.ffn_up.weight");
            AddRef("ffn_down", $"blk.{i}.ffn_down.weight");

            if (config.IsStandardAttention(layerIndex))
            {
                AddRef("attn_q", $"blk.{i}.attn_q.weight");
                AddRef("attn_k", $"blk.{i}.attn_k.weight");
                AddRef("attn_v", $"blk.{i}.attn_v.weight");
                AddRef("attn_o", $"blk.{i}.attn_output.weight");
                AddRef("attn_q_norm", $"blk.{i}.attn_q_norm.weight");
                AddRef("attn_k_norm", $"blk.{i}.attn_k_norm.weight");
            }
            else
            {
                AddRef("attn_qkv", $"blk.{i}.attn_qkv.weight");
                AddRef("attn_gate", $"blk.{i}.attn_gate.weight");
                AddRef("ssm_a", $"blk.{i}.ssm_a");
                AddRef("ssm_alpha", $"blk.{i}.ssm_alpha.weight");
                AddRef("ssm_beta", $"blk.{i}.ssm_beta.weight");
                AddRef("ssm_conv1d", $"blk.{i}.ssm_conv1d.weight");
                AddRef("ssm_dt_bias", $"blk.{i}.ssm_dt.bias");
                AddRef("ssm_norm", $"blk.{i}.ssm_norm.weight");
                AddRef("ssm_out", $"blk.{i}.ssm_out.weight");
            }

            return data;
        }
    }
}
