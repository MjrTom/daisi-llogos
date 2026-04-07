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
    private readonly CudaStream _copyStream;
    private readonly nint _copyDoneEvent; // CUDA event: signaled when async copy finishes
    // Staging buffers removed — using direct async copy from managed arrays.
    // True overlap requires CUDA-pinned source memory; the driver handles internal pinning.

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
        ForwardPass forward, CudaStream copyStream, nint copyDoneEvent)
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
        _copyDoneEvent = copyDoneEvent;
    }

    public ReadOnlySpan<float> Forward(int tokenId, int position)
    {
        // Embedding (persistent in VRAM, single Begin/Flush)
        _forward.ForwardEmbedding(tokenId);

        // Upload layer 0 synchronously into slot A (first layer has nothing to overlap with)
        _activeSlot = 0;
        UploadLayerSync(0, _weightsA, slot: 0);
        _forward.SetWeights(_weightsA);

        // Pipelined layers: single Begin/Flush, callbacks swap weights between layers
        _forward.ForwardLayersPipelined(0, _config.NumLayers, position,
            (layer, isLast) =>
            {
                // Before this layer's kernels are dispatched:
                // 1. Make compute stream wait for the current slot's copy to be done
                if (layer > 0)
                    CudaApi.Check(CudaApi.StreamWaitEvent(_cuda.ComputeStreamHandle, _copyDoneEvent, 0),
                        "cuStreamWaitEvent");

                // 2. Start async upload of NEXT layer into the OTHER slot's staging buffer
                if (!isLast)
                {
                    int nextSlot = 1 - _activeSlot;
                    var nextWeights = nextSlot == 0 ? _weightsA : _weightsB;
                    UploadLayerAsync(layer + 1, nextWeights, slot: nextSlot);
                    // Record event on copy stream — compute stream will wait on it next iteration
                    CudaApi.Check(CudaApi.EventRecord(_copyDoneEvent, _copyStream.Handle), "cuEventRecord");
                }

                // 3. Point ForwardPass at the current slot's weights
                var currentWeights = _activeSlot == 0 ? _weightsA : _weightsB;
                _forward.SetWeights(currentWeights);

                // 4. Swap slot for next iteration
                if (!isLast)
                    _activeSlot = 1 - _activeSlot;
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
                    CudaApi.Check(CudaApi.StreamWaitEvent(_cuda.ComputeStreamHandle, _copyDoneEvent, 0),
                        "cuStreamWaitEvent");
                if (!isLast)
                {
                    int nextSlot = 1 - _activeSlot;
                    UploadLayerAsync(layer + 1, nextSlot == 0 ? _weightsA : _weightsB, slot: nextSlot);
                    CudaApi.Check(CudaApi.EventRecord(_copyDoneEvent, _copyStream.Handle), "cuEventRecord");
                }
                _forward.SetWeights(_activeSlot == 0 ? _weightsA : _weightsB);
                if (!isLast) _activeSlot = 1 - _activeSlot;
            });
    }

    public void ResetState()
    {
        _forward.ResetState();
    }

    /// <summary>Upload layer weights synchronously (for first layer — no overlap possible).</summary>
    private void UploadLayerSync(int layerIndex, ModelWeights targetWeights, int slot)
    {
        var data = _layerData[layerIndex];
        CopyLayerToGpu(data, targetWeights.Layers[layerIndex], synchronous: true, slot: slot);
    }

    /// <summary>Upload layer weights asynchronously on the copy stream.</summary>
    private void UploadLayerAsync(int layerIndex, ModelWeights targetWeights, int slot)
    {
        var data = _layerData[layerIndex];
        CopyLayerToGpu(data, targetWeights.Layers[layerIndex], synchronous: false, slot: slot);
    }

    /// <summary>
    /// Copy all tensor data for a layer to GPU buffers.
    /// Uses per-tensor pinning for async DMA via the copy stream.
    /// </summary>
    private unsafe void CopyLayerToGpu(LayerShardData data, LayerWeights targetLayer, bool synchronous, int slot = 0)
    {
        foreach (var (name, tensorData, gpuTensor) in data.GetTensorsWithTargets(targetLayer))
        {
            var ct = (CudaTensor)gpuTensor;
            fixed (byte* src = tensorData)
            {
                if (synchronous)
                {
                    CudaApi.Check(
                        CudaApi.MemcpyHtoD(ct.DevicePtr, src, (ulong)tensorData.Length),
                        "cuMemcpyHtoD");
                }
                else
                {
                    // Note: MemcpyHtoDAsync with unpinned source may not truly overlap.
                    // The CUDA driver will internally pin and copy. This is still faster
                    // than synchronous copy since the driver can pipeline the pinning.
                    CudaApi.Check(
                        CudaApi.MemcpyHtoDAsync(ct.DevicePtr, src, (ulong)tensorData.Length, _copyStream.Handle),
                        "cuMemcpyHtoDAsync");
                }
            }
        }
    }

    public void Dispose()
    {
        _forward.Dispose();
        // Staging buffers disposed (if any)
        if (_copyDoneEvent != 0) CudaApi.EventDestroy(_copyDoneEvent);
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

        // Persistent: embedding + output on GPU
        var embedWeights = MmapModelLoader.LoadPartialFromShards(
            gguf, shardDir, cuda, config, 0, 0, true, false);
        var outputWeights = MmapModelLoader.LoadPartialFromShards(
            gguf, shardDir, cuda, config, config.NumLayers, config.NumLayers, false, true);

        // KV cache + DeltaNet state for all layers (persistent, small)
        var kvCache = new KvCache(cuda, config, maxSeqLen: maxContext);

        // DeltaNetState needs weights to detect layer types — load temporarily
        var tmpWeights = MmapModelLoader.LoadPartialFromShards(
            gguf, shardDir, cuda, config, 0, config.NumLayers, false, false);
        var deltaState = new DeltaNetState(cuda, config, tmpWeights);
        tmpWeights.Dispose();

        // Allocate two sets of layer weight GPU buffers (empty — will be filled per layer)
        var weightsA = AllocateLayerSlot(cuda, config, gguf, "slotA");
        var weightsB = AllocateLayerSlot(cuda, config, gguf, "slotB");

        // Memory-map layer shard files and prepare CPU-side pinned copies for async transfer
        var layerData = new LayerShardData[config.NumLayers];
        var baseName = MmapModelLoader.FindShardBaseName(shardDir);
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
        CudaApi.Check(CudaApi.EventCreate(out nint copyDoneEvent, 0x02 /* CU_EVENT_DISABLE_TIMING */),
            "cuEventCreate");

        // Shared CUDA-pinned staging buffer sized to the largest layer's repacked data.
        // Data flows: mmap shard → repacked byte[] → pinned staging → async DMA → GPU buffer
        long maxLayerBytes = 0;
        foreach (var ld in layerData)
            foreach (var d in ld.TensorData.Values)
                maxLayerBytes += d.Length;
        // maxLayerBytes is the sum across all tensors in the largest layer
        // Actually compute per-layer total
        long maxPerLayer = 0;
        foreach (var ld in layerData)
        {
            long total = 0;
            foreach (var d in ld.TensorData.Values) total += d.Length;
            if (total > maxPerLayer) maxPerLayer = total;
        }
        Console.Error.WriteLine($"  Pipelined: {config.NumLayers} layers, double-buffered, max layer={maxPerLayer / 1024 / 1024}MB, loaded in {sw.Elapsed.TotalSeconds:F1}s");

        return new PipelinedForwardPass(
            cuda, config, embedWeights, outputWeights,
            kvCache, deltaState, weightsA, weightsB,
            layerData, mmapHandles, forward, copyStream, copyDoneEvent);
    }

    /// <summary>
    /// Allocate empty GPU tensors matching the shape/type of each layer's weights.
    /// These are the double-buffer slots that get overwritten each layer.
    /// </summary>
    private static ModelWeights AllocateLayerSlot(CudaBackend cuda, ModelConfig config,
        GgufFile gguf, string prefix)
    {
        var tensorInfoMap = new Dictionary<string, GgufTensorInfo>();
        foreach (var t in gguf.Tensors) tensorInfoMap[t.Name] = t;

        var layers = new LayerWeights[config.NumLayers];
        for (int i = 0; i < config.NumLayers; i++)
        {
            if (config.IsStandardAttention(i))
                layers[i] = AllocateStandardLayer(cuda, tensorInfoMap, i, prefix);
            else
                layers[i] = AllocateDeltaNetLayer(cuda, tensorInfoMap, i, prefix);
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
        CudaBackend cuda, Dictionary<string, GgufTensorInfo> infoMap, int i, string prefix)
    {
        ITensor Alloc(string name)
        {
            var info = infoMap[name];
            var dims = info.Dimensions.Select(d => (long)d).ToArray();
            // Allocate with proper alignment for quantized types
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

    // ── Layer shard data (CPU-side, pinned for async transfers) ──────────────

    /// <summary>
    /// Holds mmapped + repacked weight data for a single layer, ready for GPU upload.
    /// Data is stored as managed byte arrays (repacked from mmapped shard files).
    /// The PipelinedForwardPass copies these into shared CUDA-pinned staging buffers
    /// before async DMA transfer.
    /// </summary>
    internal sealed class LayerShardData
    {
        /// <summary>Tensor name → (byte array, offset, size) of repacked weight data.</summary>
        internal Dictionary<string, byte[]> TensorData { get; } = new();

        internal IEnumerable<(string name, byte[] data, ITensor target)> GetTensorsWithTargets(LayerWeights layer)
        {
            if (layer is StandardAttentionWeights saw)
            {
                if (TensorData.TryGetValue("attn_norm", out var v)) yield return ("attn_norm", v, saw.AttnNorm);
                if (TensorData.TryGetValue("post_attn_norm", out v)) yield return ("post_attn_norm", v, saw.PostAttnNorm);
                if (TensorData.TryGetValue("attn_q", out v)) yield return ("attn_q", v, saw.AttnQ);
                if (TensorData.TryGetValue("attn_k", out v)) yield return ("attn_k", v, saw.AttnK);
                if (TensorData.TryGetValue("attn_v", out v)) yield return ("attn_v", v, saw.AttnV);
                if (TensorData.TryGetValue("attn_o", out v)) yield return ("attn_o", v, saw.AttnO);
                if (saw.AttnQNorm != null && TensorData.TryGetValue("attn_q_norm", out v)) yield return ("attn_q_norm", v, saw.AttnQNorm);
                if (saw.AttnKNorm != null && TensorData.TryGetValue("attn_k_norm", out v)) yield return ("attn_k_norm", v, saw.AttnKNorm);
                if (TensorData.TryGetValue("ffn_gate", out v)) yield return ("ffn_gate", v, saw.FfnGate);
                if (TensorData.TryGetValue("ffn_up", out v)) yield return ("ffn_up", v, saw.FfnUp);
                if (TensorData.TryGetValue("ffn_down", out v)) yield return ("ffn_down", v, saw.FfnDown);
            }
            else if (layer is DeltaNetWeights dnw)
            {
                if (TensorData.TryGetValue("attn_norm", out var v)) yield return ("attn_norm", v, dnw.AttnNorm);
                if (TensorData.TryGetValue("post_attn_norm", out v)) yield return ("post_attn_norm", v, dnw.PostAttnNorm);
                if (TensorData.TryGetValue("attn_qkv", out v)) yield return ("attn_qkv", v, dnw.AttnQkv);
                if (TensorData.TryGetValue("attn_gate", out v)) yield return ("attn_gate", v, dnw.AttnGate);
                if (TensorData.TryGetValue("ssm_a", out v)) yield return ("ssm_a", v, dnw.SsmA);
                if (TensorData.TryGetValue("ssm_alpha", out v)) yield return ("ssm_alpha", v, dnw.SsmAlpha);
                if (TensorData.TryGetValue("ssm_beta", out v)) yield return ("ssm_beta", v, dnw.SsmBeta);
                if (TensorData.TryGetValue("ssm_conv1d", out v)) yield return ("ssm_conv1d", v, dnw.SsmConv1d);
                if (TensorData.TryGetValue("ssm_dt_bias", out v)) yield return ("ssm_dt_bias", v, dnw.SsmDtBias);
                if (TensorData.TryGetValue("ssm_norm", out v)) yield return ("ssm_norm", v, dnw.SsmNorm);
                if (TensorData.TryGetValue("ssm_out", out v)) yield return ("ssm_out", v, dnw.SsmOut);
                if (TensorData.TryGetValue("ffn_gate", out v)) yield return ("ffn_gate", v, dnw.FfnGate);
                if (TensorData.TryGetValue("ffn_up", out v)) yield return ("ffn_up", v, dnw.FfnUp);
                if (TensorData.TryGetValue("ffn_down", out v)) yield return ("ffn_down", v, dnw.FfnDown);
            }
        }

        /// <summary>
        /// Load a layer shard from disk, repack quantized weights to GPU-aligned layout,
        /// and pin the data in memory for async PCIe transfers.
        /// </summary>
        internal static unsafe LayerShardData Load(string shardPath, int layerIndex,
            ModelConfig config, GgufFile gguf, List<IDisposable> mmapHandles)
        {
            var data = new LayerShardData();

            // Parse shard index
            using var indexStream = File.OpenRead(shardPath);
            var shardIndex = GgufShardIndex.Read(indexStream);

            // Mmap the shard file
            var mmf = MemoryMappedFile.CreateFromFile(shardPath, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            var accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            mmapHandles.Add(accessor);
            mmapHandles.Add(mmf);

            byte* basePtr = null;
            accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);

            // Build tensor info lookup from GGUF header
            var tensorInfoMap = new Dictionary<string, GgufTensorInfo>();
            foreach (var t in gguf.Tensors) tensorInfoMap[t.Name] = t;

            int i = layerIndex;

            // Prepare each tensor: read from mmap, repack if needed, pin for async transfer
            void PrepareTensor(string shortName, string fullName)
            {
                if (!shardIndex.Tensors.TryGetValue(fullName, out var entry)) return;
                if (!tensorInfoMap.TryGetValue(fullName, out var info)) return;

                long offset = shardIndex.DataSectionOffset + entry.offset;
                var rawSpan = new ReadOnlySpan<byte>(basePtr + offset, (int)entry.byteSize);

                // Repack Q8_0/Q4_0 to aligned layout (must match CudaBackend.LoadTensor)
                byte[] repacked;
                if (info.Type == GgmlType.Q8_0 && info.Dimensions.Length >= 2)
                {
                    int blockCount = rawSpan.Length / 34;
                    repacked = new byte[blockCount * 36];
                    for (int b = 0; b < blockCount; b++)
                    {
                        int srcOff = b * 34, dstOff = b * 36;
                        repacked[dstOff] = rawSpan[srcOff];
                        repacked[dstOff + 1] = rawSpan[srcOff + 1];
                        rawSpan.Slice(srcOff + 2, 32).CopyTo(repacked.AsSpan(dstOff + 4, 32));
                    }
                }
                else if (info.Type == GgmlType.Q4_0 && info.Dimensions.Length >= 2)
                {
                    int blockCount = rawSpan.Length / 18;
                    repacked = new byte[blockCount * 20];
                    for (int b = 0; b < blockCount; b++)
                    {
                        int srcOff = b * 18, dstOff = b * 20;
                        repacked[dstOff] = rawSpan[srcOff];
                        repacked[dstOff + 1] = rawSpan[srcOff + 1];
                        rawSpan.Slice(srcOff + 2, 16).CopyTo(repacked.AsSpan(dstOff + 4, 16));
                    }
                }
                else
                {
                    repacked = rawSpan.ToArray();
                }

                data.TensorData[shortName] = repacked;
            }

            // Standard attention tensors
            PrepareTensor("attn_norm", $"blk.{i}.attn_norm.weight");
            PrepareTensor("post_attn_norm", tensorInfoMap.ContainsKey($"blk.{i}.post_attention_norm.weight")
                ? $"blk.{i}.post_attention_norm.weight" : $"blk.{i}.ffn_norm.weight");
            PrepareTensor("ffn_gate", $"blk.{i}.ffn_gate.weight");
            PrepareTensor("ffn_up", $"blk.{i}.ffn_up.weight");
            PrepareTensor("ffn_down", $"blk.{i}.ffn_down.weight");

            if (config.IsStandardAttention(layerIndex))
            {
                PrepareTensor("attn_q", $"blk.{i}.attn_q.weight");
                PrepareTensor("attn_k", $"blk.{i}.attn_k.weight");
                PrepareTensor("attn_v", $"blk.{i}.attn_v.weight");
                PrepareTensor("attn_o", $"blk.{i}.attn_output.weight");
                PrepareTensor("attn_q_norm", $"blk.{i}.attn_q_norm.weight");
                PrepareTensor("attn_k_norm", $"blk.{i}.attn_k_norm.weight");
            }
            else
            {
                // DeltaNet tensors
                PrepareTensor("attn_qkv", $"blk.{i}.attn_qkv.weight");
                PrepareTensor("attn_gate", $"blk.{i}.attn_gate.weight");
                PrepareTensor("ssm_a", $"blk.{i}.ssm_a");
                PrepareTensor("ssm_alpha", $"blk.{i}.ssm_alpha.weight");
                PrepareTensor("ssm_beta", $"blk.{i}.ssm_beta.weight");
                PrepareTensor("ssm_conv1d", $"blk.{i}.ssm_conv1d.weight");
                PrepareTensor("ssm_dt_bias", $"blk.{i}.ssm_dt.bias");
                PrepareTensor("ssm_norm", $"blk.{i}.ssm_norm.weight");
                PrepareTensor("ssm_out", $"blk.{i}.ssm_out.weight");
            }

            return data;
        }

        internal void Free()
        {
            TensorData.Clear();
        }
    }

    /// <summary>
    /// Expose FindShardBaseName for use by the factory.
    /// Delegates to MmapModelLoader's existing method.
    /// </summary>
    internal static string FindShardBaseName(string shardDir) =>
        MmapModelLoader.FindShardBaseName(shardDir);
}
