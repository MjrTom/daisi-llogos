using Daisi.Llogos.Cuda;
using Daisi.Llogos.Gguf;
using Daisi.Llogos.Training.Lora;

namespace Daisi.Llogos.Training;

/// <summary>
/// Contiguous GPU storage for all LoRA parameters, gradients, and optimizer state.
/// Enables single-kernel operations across all LoRA layers:
///   - Single GradNormSq (instead of per-layer)
///   - Single ScaleInPlace (instead of per-layer)
///   - Single AdamW step + gradient zero (instead of per-layer)
///
/// Layout: [layer0.A | layer0.B | layer1.A | layer1.B | ...] for each buffer.
/// All four buffers (params, grads, m, v) use the same layout/offsets.
/// </summary>
public sealed class ContiguousLoraState : IDisposable
{
    /// <summary>All LoRA parameters packed contiguously.</summary>
    public ITensor Params { get; }

    /// <summary>All LoRA gradients packed contiguously (same layout as Params).</summary>
    public ITensor Grads { get; }

    /// <summary>AdamW first moment (m) for all params.</summary>
    public ITensor M { get; }

    /// <summary>AdamW second moment (v) for all params.</summary>
    public ITensor V { get; }

    /// <summary>Total number of LoRA parameters across all layers.</summary>
    public int TotalParams { get; }

    /// <summary>Per-layer offset+size into the contiguous buffers.</summary>
    public (int offsetA, int sizeA, int offsetB, int sizeB)[] LayerOffsets { get; }

    private ContiguousLoraState(ITensor p, ITensor g, ITensor m, ITensor v,
        int total, (int, int, int, int)[] offsets)
    {
        Params = p; Grads = g; M = m; V = v;
        TotalParams = total; LayerOffsets = offsets;
    }

    /// <summary>
    /// Create contiguous LoRA state from an adapter. Allocates 4 GPU buffers
    /// (params, grads, m, v) of size TotalParams each. Uploads initial weights
    /// from the CPU adapter.
    /// </summary>
    public static ContiguousLoraState Create(LoraAdapter adapter, CudaTrainingBackend gpu)
    {
        // Compute total size and per-layer offsets
        int totalParams = 0;
        var layers = adapter.Layers.ToList();
        var offsets = new (int offsetA, int sizeA, int offsetB, int sizeB)[layers.Count];

        for (int i = 0; i < layers.Count; i++)
        {
            var layer = layers[i].Value;
            offsets[i] = (totalParams, layer.A.Size, totalParams + layer.A.Size, layer.B.Size);
            totalParams += layer.A.Size + layer.B.Size;
        }

        // Allocate contiguous buffers
        var p = gpu.CreateTensor("lora.params", GgmlType.F32, [(long)totalParams]);
        var g = gpu.CreateTensor("lora.grads", GgmlType.F32, [(long)totalParams]);
        var m = gpu.CreateTensor("lora.m", GgmlType.F32, [(long)totalParams]);
        var v = gpu.CreateTensor("lora.v", GgmlType.F32, [(long)totalParams]);

        // Zero optimizer state
        gpu.ZeroTensor(m);
        gpu.ZeroTensor(v);
        gpu.ZeroTensor(g);

        // Upload initial LoRA weights into contiguous param buffer
        var allParams = new float[totalParams];
        for (int i = 0; i < layers.Count; i++)
        {
            var layer = layers[i].Value;
            var (oA, sA, oB, sB) = offsets[i];
            Array.Copy(layer.A.Data, 0, allParams, oA, sA);
            Array.Copy(layer.B.Data, 0, allParams, oB, sB);
        }
        UploadF32(p, allParams);

        Console.Error.WriteLine($"  Contiguous LoRA: {totalParams:N0} params ({totalParams * 4 / 1024.0 / 1024.0:F1} MB × 4 buffers)");

        return new ContiguousLoraState(p, g, m, v, totalParams, offsets);
    }

    /// <summary>
    /// Download LoRA weights from contiguous GPU buffer back to CPU adapter.
    /// Used for saving checkpoints.
    /// </summary>
    public void DownloadTo(LoraAdapter adapter)
    {
        var allParams = new float[TotalParams];
        ((CudaTensor)Params).DownloadTo(allParams);

        var layers = adapter.Layers.ToList();
        for (int i = 0; i < layers.Count; i++)
        {
            var layer = layers[i].Value;
            var (oA, sA, oB, sB) = LayerOffsets[i];
            Array.Copy(allParams, oA, layer.A.Data, 0, sA);
            Array.Copy(allParams, oB, layer.B.Data, 0, sB);
        }
    }

    private static unsafe void UploadF32(ITensor tensor, float[] data)
    {
        fixed (float* ptr = data)
            CudaApi.Check(
                CudaApi.MemcpyHtoD(((CudaTensor)tensor).DevicePtr, ptr, (ulong)(data.Length * sizeof(float))),
                "cuMemcpyHtoD");
    }

    public void Dispose()
    {
        Params.Dispose();
        Grads.Dispose();
        M.Dispose();
        V.Dispose();
    }
}
