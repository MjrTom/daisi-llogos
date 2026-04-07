using System.Diagnostics;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using Daisi.Llogos.Cpu;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Inference;
using Daisi.Llogos.Model;

namespace Daisi.Llogos.Cuda;

/// <summary>
/// Double-buffered GPU offloading forward pass: runs all layers on GPU by pipelining
/// weight uploads through PCIe while the GPU computes the previous layer.
///
/// Only 2 layers (~400MB) + KV caches + working buffers in VRAM at any time.
/// Can run models far larger than VRAM at near-full GPU speed.
///
/// Pipeline:
///   GPU:   [compute layer 0][compute layer 1][compute layer 2]...
///   PCIe:  [upload layer 1 ][upload layer 2 ][upload layer 3 ]...
///          ↑ overlapped — GPU never waits (if upload ≤ compute time)
/// </summary>
public sealed class PipelinedForwardPass : IForwardPass
{
    private readonly CudaBackend _cuda;
    private readonly ModelConfig _config;
    private readonly ForwardPass _forward;
    private readonly bool _gpuAligned;
    private readonly CudaStream _copyStream;
    private readonly nint _copyDoneA; // CUDA event: signaled when slot A's DMA finishes
    private readonly nint _copyDoneB; // CUDA event: signaled when slot B's DMA finishes
    // Two CUDA-pinned host buffers — one per slot. Shard data is repacked directly
    // into whichever buffer the GPU isn't currently reading. True async DMA overlap.
    private readonly CudaPinnedMemory _pinnedA;
    private readonly CudaPinnedMemory _pinnedB;

    // Persistent state
    private readonly ModelWeights _embedWeights;
    private readonly ModelWeights _outputWeights;
    private readonly KvCache _kvCache;
    private readonly DeltaNetState _deltaState;

    // Layer weight data (CPU-side, pinned for async transfers)
    private readonly LayerShardData[] _layerData;
    private readonly List<IDisposable> _mmapHandles = [];

    // Double-buffer: two sets of GPU tensors for layer weights
    private readonly ModelWeights _weightsA;
    private readonly ModelWeights _weightsB;
    private int _activeSlot; // 0 = A, 1 = B

    public IKvCache KvCache => _kvCache;

    private PipelinedForwardPass(
        CudaBackend cuda, ModelConfig config,
        ModelWeights embedWeights, ModelWeights outputWeights,
        KvCache kvCache, DeltaNetState deltaState,
        ModelWeights weightsA, ModelWeights weightsB,
        LayerShardData[] layerData, List<IDisposable> mmapHandles,
        ForwardPass forward, CudaStream copyStream,
        nint copyDoneA, nint copyDoneB,
        CudaPinnedMemory pinnedA, CudaPinnedMemory pinnedB,
        bool gpuAligned)
    {
        _cuda = cuda;
        _config = config;
        _embedWeights = embedWeights;
        _outputWeights = outputWeights;
        _kvCache = kvCache;
        _deltaState = deltaState;
        _weightsA = weightsA;
        _weightsB = weightsB;
        _layerData = layerData;
        _mmapHandles = mmapHandles;
        _forward = forward;
        _copyStream = copyStream;
        _copyDoneA = copyDoneA;
        _copyDoneB = copyDoneB;
        _pinnedA = pinnedA;
        _pinnedB = pinnedB;
        _gpuAligned = gpuAligned;
    }

    public ReadOnlySpan<float> Forward(int tokenId, int position)
    {
        // Embedding (persistent in VRAM, single Begin/Flush)
        _forward.ForwardEmbedding(tokenId);

        // Upload layer 0 synchronously into slot A
        _activeSlot = 0;
        UploadLayerSync(0, _weightsA, slot: 0);
        _forward.SetWeights(_weightsA);

        // Pipelined layers: single Begin/Flush, per-slot events for correctness
        _forward.ForwardLayersPipelined(0, _config.NumLayers, position,
            (layer, isLast) =>
            {
                // 1. Compute stream must wait for the CURRENT slot's DMA to be done
                //    (ensures GPU reads valid weights for this layer)
                if (layer > 0)
                {
                    var currentEvent = _activeSlot == 0 ? _copyDoneA : _copyDoneB;
                    CudaApi.Check(CudaApi.StreamWaitEvent(_cuda.ComputeStreamHandle, currentEvent, 0),
                        "cuStreamWaitEvent");
                }

                // 2. Point ForwardPass at the current slot's weights
                _forward.SetWeights(_activeSlot == 0 ? _weightsA : _weightsB);

                // 3. Start async upload of NEXT layer into the OTHER slot
                if (!isLast)
                {
                    int nextSlot = 1 - _activeSlot;

                    // CPU must wait for the OTHER slot's previous DMA to finish
                    // before overwriting its pinned buffer (CPU-side sync)
                    var nextEvent = nextSlot == 0 ? _copyDoneA : _copyDoneB;
                    if (layer > 1) // slot was used 2 layers ago
                        CudaApi.Check(CudaApi.EventSynchronize(nextEvent), "cuEventSynchronize");

                    var nextWeights = nextSlot == 0 ? _weightsA : _weightsB;
                    UploadLayerAsync(layer + 1, nextWeights, slot: nextSlot);

                    // Record event on copy stream for this slot
                    CudaApi.Check(CudaApi.EventRecord(nextEvent, _copyStream.Handle), "cuEventRecord");

                    _activeSlot = nextSlot;
                }
            });

        // Output head (persistent in VRAM)
        var logits = new float[_config.VocabSize];
        _forward.ForwardOutputHead(logits);
        return logits;
    }

    public void ForwardHidden(int tokenId, int position)
    {
        // Same as Forward but skip output head
        _forward.ForwardEmbedding(tokenId);
        _activeSlot = 0;
        UploadLayerSync(0, _weightsA, slot: 0);
        _forward.SetWeights(_weightsA);
        _forward.ForwardLayersPipelined(0, _config.NumLayers, position,
            (layer, isLast) =>
            {
                if (layer > 0)
                {
                    var currentEvent = _activeSlot == 0 ? _copyDoneA : _copyDoneB;
                    CudaApi.Check(CudaApi.StreamWaitEvent(_cuda.ComputeStreamHandle, currentEvent, 0),
                        "cuStreamWaitEvent");
                }
                _forward.SetWeights(_activeSlot == 0 ? _weightsA : _weightsB);
                if (!isLast)
                {
                    int nextSlot = 1 - _activeSlot;
                    var nextEvent = nextSlot == 0 ? _copyDoneA : _copyDoneB;
                    if (layer > 1)
                        CudaApi.Check(CudaApi.EventSynchronize(nextEvent), "cuEventSynchronize");
                    UploadLayerAsync(layer + 1, nextSlot == 0 ? _weightsA : _weightsB, slot: nextSlot);
                    CudaApi.Check(CudaApi.EventRecord(nextEvent, _copyStream.Handle), "cuEventRecord");
                    _activeSlot = nextSlot;
                }
            });
    }

    public void ResetState()
    {
        _forward.ResetState();
    }

    /// <summary>Copy (or repack) from mmap into pinned buffer + upload to GPU synchronously.</summary>
    private unsafe void UploadLayerSync(int layerIndex, ModelWeights targetWeights, int slot)
    {
        var pinned = slot == 0 ? _pinnedA : _pinnedB;
        LayerShardData.RepackAndUpload(_layerData[layerIndex], targetWeights.Layers[layerIndex],
            (byte*)pinned.HostPtr, null, synchronous: true, gpuAligned: _gpuAligned);
    }

    /// <summary>Copy (or repack) from mmap into pinned buffer + async DMA to GPU.</summary>
    private unsafe void UploadLayerAsync(int layerIndex, ModelWeights targetWeights, int slot)
    {
        var pinned = slot == 0 ? _pinnedA : _pinnedB;
        LayerShardData.RepackAndUpload(_layerData[layerIndex], targetWeights.Layers[layerIndex],
            (byte*)pinned.HostPtr, _copyStream, synchronous: false, gpuAligned: _gpuAligned);
    }

    public void Dispose()
    {
        _forward.Dispose();
        _pinnedA.Dispose();
        _pinnedB.Dispose();
        if (_copyDoneA != 0) CudaApi.EventDestroy(_copyDoneA);
        if (_copyDoneB != 0) CudaApi.EventDestroy(_copyDoneB);
        _copyStream.Dispose();
        _kvCache.Dispose();
        _deltaState.Dispose();
        _embedWeights.Dispose();
        _outputWeights.Dispose();
        _weightsA.Dispose();
        _weightsB.Dispose();
        foreach (var h in _mmapHandles) h.Dispose();
    }

    // ── Factory ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Create a pipelined forward pass from shard files.
    /// Allocates two sets of layer weight buffers on GPU, loads embedding + output persistently,
    /// and memory-maps all layer shards for fast CPU→GPU streaming.
    /// </summary>
    public static PipelinedForwardPass Create(
        GgufFile gguf, string shardDir, ModelConfig config,
        CudaBackend cuda, int maxContext = 2048)
    {
        var sw = Stopwatch.StartNew();
        var mmapHandles = new List<IDisposable>();

        // Check if shards are GPU-aligned (pre-repacked at split time)
        var baseName = MmapModelLoader.FindShardBaseName(shardDir);
        var manifestPath = Path.Combine(shardDir, $"{baseName}.manifest.json");
        bool gpuAligned = false;
        if (File.Exists(manifestPath))
        {
            var manifest = GgufShardManifest.FromJsonFile(manifestPath);
            gpuAligned = manifest.GpuAligned;
        }

        // Persistent: embedding + output on GPU
        var embedWeights = MmapModelLoader.LoadPartialFromShards(
            gguf, shardDir, cuda, config, 0, 0, true, false);
        var outputWeights = MmapModelLoader.LoadPartialFromShards(
            gguf, shardDir, cuda, config, config.NumLayers, config.NumLayers, false, true);

        // KV cache for all layers (persistent, small)
        var kvCache = new KvCache(cuda, config, maxSeqLen: maxContext);

        // Allocate two sets of layer weight GPU buffers (empty — will be filled per layer)
        var weightsA = AllocateLayerSlot(cuda, config, gguf, "slotA", gpuAligned);
        var weightsB = AllocateLayerSlot(cuda, config, gguf, "slotB", gpuAligned);

        // DeltaNetState needs weights to detect layer types — use slot A
        var deltaState = new DeltaNetState(cuda, config, weightsA);

        // Memory-map layer shard files
        var layerData = new LayerShardData[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            var shardPath = Path.Combine(shardDir, $"{baseName}.layer.{i}");
            layerData[i] = LayerShardData.Load(shardPath, i, config, gguf, mmapHandles);
        }

        // Set embed/output on slot weights to point at the real persistent tensors.
        // This way ForwardPass can do embedding lookup and output head projection
        // using the correct persistent weights, while layer weights get swapped.
        weightsA.TokenEmbedding = embedWeights.TokenEmbedding;
        weightsA.OutputNorm = outputWeights.OutputNorm;
        weightsA.Output = outputWeights.Output;
        weightsB.TokenEmbedding = embedWeights.TokenEmbedding;
        weightsB.OutputNorm = outputWeights.OutputNorm;
        weightsB.Output = outputWeights.Output;

        // Create the ForwardPass with weightsA so it detects model features (gated Q, DeltaNet, etc.)
        // from the real layer weight shapes. WeightsA has correctly-shaped tensors even though
        // their data is zeros — the constructor only checks tensor dimensions, not values.
        var forward = new ForwardPass(cuda, config, weightsA, kvCache, deltaState);
        forward.DisableGraphCapture();

        // Copy stream + event for async weight uploads with cross-stream sync
        var copyStream = new CudaStream();
        CudaApi.Check(CudaApi.EventCreate(out nint copyDoneA, 0x02), "cuEventCreate");
        CudaApi.Check(CudaApi.EventCreate(out nint copyDoneB, 0x02), "cuEventCreate");

        // Shared CUDA-pinned staging buffer sized to the largest layer's repacked data.
        // Data flows: mmap shard → repacked byte[] → pinned staging → async DMA → GPU buffer
        // Compute max repacked size per layer (for pinned buffer sizing)
        long maxPerLayer = 0;
        foreach (var ld in layerData)
        {
            long total = 0;
            foreach (var tr in ld.Tensors)
            {
                // Repacked size: Q4_0 18→20, Q8_0 34→36, others same
                if (tr.Type == GgmlType.Q4_0 && tr.NDimensions >= 2)
                    total += (tr.RawByteSize / 18) * 20;
                else if (tr.Type == GgmlType.Q8_0 && tr.NDimensions >= 2)
                    total += (tr.RawByteSize / 34) * 36;
                else
                    total += tr.RawByteSize;
            }
            if (total > maxPerLayer) maxPerLayer = total;
        }
        // Two CUDA-pinned host buffers — one per double-buffer slot.
        // Shard data is repacked directly into the idle pinned buffer,
        // then async DMA'd to GPU while the other buffer is being read by the GPU.
        var pinnedA = new CudaPinnedMemory((ulong)maxPerLayer);
        var pinnedB = new CudaPinnedMemory((ulong)maxPerLayer);

        Console.Error.WriteLine($"  Pipelined: {config.NumLayers} layers, double-buffered, pinned=2×{maxPerLayer / 1024 / 1024}MB, loaded in {sw.Elapsed.TotalSeconds:F1}s");

        return new PipelinedForwardPass(
            cuda, config, embedWeights, outputWeights,
            kvCache, deltaState, weightsA, weightsB,
            layerData, mmapHandles, forward, copyStream, copyDoneA, copyDoneB, pinnedA, pinnedB, gpuAligned);
    }

    /// <summary>
    /// Allocate empty GPU tensors matching the shape/type of each layer's weights.
    /// These are the double-buffer slots that get overwritten each layer.
    /// </summary>
    private static ModelWeights AllocateLayerSlot(CudaBackend cuda, ModelConfig config,
        GgufFile gguf, string prefix, bool gpuAligned = false)
    {
        var tensorInfoMap = new Dictionary<string, GgufTensorInfo>();
        foreach (var t in gguf.Tensors) tensorInfoMap[t.Name] = t;

        var layers = new LayerWeights[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            if (config.IsStandardAttention(i))
                layers[i] = AllocateStandardLayer(cuda, tensorInfoMap, i, prefix, gpuAligned);
            else
                layers[i] = AllocateDeltaNetLayer(cuda, tensorInfoMap, i, prefix, gpuAligned);
        }

        // Placeholders for embed/output — not used in layer slots
        var placeholder = cuda.CreateTensor($"{prefix}.placeholder", GgmlType.F32, [1]);
        return new ModelWeights
        {
            TokenEmbedding = placeholder,
            OutputNorm = cuda.CreateTensor($"{prefix}.onorm.placeholder", GgmlType.F32, [1]),
            Output = null,
            Layers = layers,
        };
    }

    private static StandardAttentionWeights AllocateStandardLayer(
        CudaBackend cuda, Dictionary<string, GgufTensorInfo> infoMap, int i, string prefix,
        bool gpuAligned = false)
    {
        ITensor Alloc(string name)
        {
            var info = infoMap[name];
            var dims = info.Dimensions.Select(d => (long)d).ToArray();
            // When gpuAligned, the shard contains repacked data (Q4_0: 20b/block, Q8_0: 36b/block).
            // CudaBackend.LoadTensor handles repacking internally and allocates the aligned size.
            // We need to pass the ORIGINAL byte size so LoadTensor does its own repacking.
            return cuda.LoadTensor($"{prefix}.{name}", info.Type, dims,
                new byte[GgmlTypeInfo.ByteSize(info.Type, info.ElementCount)]);
        }

        ITensor? TryAlloc(string name)
        {
            if (!infoMap.ContainsKey(name)) return null;
            return Alloc(name);
        }

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
        CudaBackend cuda, Dictionary<string, GgufTensorInfo> infoMap, int i, string prefix,
        bool gpuAligned = false)
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

    // ── Layer shard data: mmap pointers for zero-copy repack ───────────────

    /// <summary>
    /// Lightweight reference to a layer's tensor data in a mmapped shard file.
    /// No managed byte[] arrays — stores raw mmap pointers and tensor metadata.
    /// At runtime, data is repacked directly from mmap → pinned buffer → GPU.
    /// </summary>
    internal sealed class LayerShardData
    {
        /// <summary>Short name → (mmap pointer to raw data, byte size, GgmlType, ndims).</summary>
        internal List<TensorRef> Tensors { get; } = [];

        internal unsafe struct TensorRef
        {
            public string ShortName;
            public byte* MmapPtr;
            public int RawByteSize;
            public GgmlType Type;
            public int NDimensions;
        }

        /// <summary>
        /// Copy (or repack) all tensors from mmap into the pinned buffer, then issue
        /// per-tensor async DMA to GPU. When gpuAligned=true, shard data is already in
        /// GPU layout — just memcpy (no per-block repack loop). Returns total bytes written.
        /// </summary>
        internal static unsafe long RepackAndUpload(LayerShardData data, LayerWeights targetLayer,
            byte* pinnedPtr, CudaStream? copyStream, bool synchronous, bool gpuAligned = false)
        {
            long offset = 0;

            // Build target tensor list from layer weights in the standard order
            var targets = GetTargetTensors(data, targetLayer);

            for (int t = 0; t < data.Tensors.Count; t++)
            {
                var tr = data.Tensors[t];
                var gpuTensor = targets[t];
                byte* src = tr.MmapPtr;
                byte* dst = pinnedPtr + offset;
                long repackedSize;

                if (gpuAligned)
                {
                    // GPU-aligned shards: data is already repacked — straight memcpy
                    repackedSize = tr.RawByteSize;
                    Buffer.MemoryCopy(src, dst, repackedSize, repackedSize);
                }
                else if (tr.Type == GgmlType.Q4_0 && tr.NDimensions >= 2)
                {
                    int blockCount = tr.RawByteSize / 18;
                    repackedSize = blockCount * 20;
                    for (int b = 0; b < blockCount; b++)
                    {
                        byte* s = src + b * 18;
                        byte* d = dst + b * 20;
                        d[0] = s[0]; d[1] = s[1]; d[2] = 0; d[3] = 0;
                        Buffer.MemoryCopy(s + 2, d + 4, 16, 16);
                    }
                }
                else if (tr.Type == GgmlType.Q8_0 && tr.NDimensions >= 2)
                {
                    int blockCount = tr.RawByteSize / 34;
                    repackedSize = blockCount * 36;
                    for (int b = 0; b < blockCount; b++)
                    {
                        byte* s = src + b * 34;
                        byte* d = dst + b * 36;
                        d[0] = s[0]; d[1] = s[1]; d[2] = 0; d[3] = 0;
                        Buffer.MemoryCopy(s + 2, d + 4, 32, 32);
                    }
                }
                else
                {
                    repackedSize = tr.RawByteSize;
                    Buffer.MemoryCopy(src, dst, repackedSize, repackedSize);
                }

                // DMA this tensor from pinned → GPU
                var ct = (CudaTensor)gpuTensor;
                if (synchronous)
                    CudaApi.Check(CudaApi.MemcpyHtoD(ct.DevicePtr, dst, (ulong)repackedSize), "cuMemcpyHtoD");
                else
                    CudaApi.Check(CudaApi.MemcpyHtoDAsync(ct.DevicePtr, dst, (ulong)repackedSize, copyStream!.Handle), "cuMemcpyHtoDAsync");

                offset += repackedSize;
            }

            return offset;
        }

        private static List<ITensor> GetTargetTensors(LayerShardData data, LayerWeights layer)
        {
            var targets = new List<ITensor>(data.Tensors.Count);
            if (layer is StandardAttentionWeights saw)
            {
                foreach (var tr in data.Tensors)
                {
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
            }
            else if (layer is DeltaNetWeights dnw)
            {
                foreach (var tr in data.Tensors)
                {
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
            }
            return targets;
        }

        /// <summary>
        /// Create a LayerShardData from a mmapped shard — stores only pointers, no copies.
        /// </summary>
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

        internal void Free() => Tensors.Clear();
    }

    /// <summary>
    /// Expose FindShardBaseName for use by the factory.
    /// Delegates to MmapModelLoader's existing method.
    /// </summary>
    internal static string FindShardBaseName(string shardDir) =>
        MmapModelLoader.FindShardBaseName(shardDir);
}
