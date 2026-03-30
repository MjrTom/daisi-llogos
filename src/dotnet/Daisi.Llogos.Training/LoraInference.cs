using Daisi.Llogos.Gguf;
using Daisi.Llogos.Model;
using Daisi.Llogos.Training.Lora;

namespace Daisi.Llogos.Training;

/// <summary>
/// Applies a trained LoRA adapter to model weights for inference.
/// Creates modified weight tensors by merging LoRA A/B into the base weights.
/// After merging, the model runs at full speed with no adapter overhead.
/// </summary>
public static class LoraInference
{
    /// <summary>
    /// Merge a LoRA adapter into model weights, producing new weight tensors with
    /// the LoRA contribution baked in: W_merged = W_base + scaling * B @ A.
    /// The original quantized weights are dequantized to F32, merged, and stored as F32 tensors.
    /// Only attention layers targeted by the adapter are modified.
    /// </summary>
    public static void MergeAdapter(ModelWeights weights, LoraAdapter adapter,
        IComputeBackend backend, ModelConfig config)
    {
        for (int layer = 0; layer < config.NumLayers; layer++)
        {
            string prefix = $"blk.{layer}";

            if (config.IsStandardAttention(layer) && weights.Layers[layer] is StandardAttentionWeights saw)
            {
                MergeLayerWeight(ref saw, "AttnQ", adapter.GetLayer($"{prefix}.attn_q"), backend);
                MergeLayerWeight(ref saw, "AttnK", adapter.GetLayer($"{prefix}.attn_k"), backend);
                MergeLayerWeight(ref saw, "AttnV", adapter.GetLayer($"{prefix}.attn_v"), backend);
                MergeLayerWeight(ref saw, "AttnO", adapter.GetLayer($"{prefix}.attn_o"), backend);
            }
            else if (weights.Layers[layer] is DeltaNetWeights dnw)
            {
                MergeDeltaNetWeight(dnw, "AttnQkv", adapter.GetLayer($"{prefix}.delta_qkv"), backend);
                MergeDeltaNetWeight(dnw, "SsmOut", adapter.GetLayer($"{prefix}.delta_out"), backend);
            }
        }
    }

    private static void MergeLayerWeight(ref StandardAttentionWeights saw, string weightName,
        LoraLayer? lora, IComputeBackend backend)
    {
        if (lora == null) return;

        // Get the original weight tensor
        ITensor original = weightName switch
        {
            "AttnQ" => saw.AttnQ,
            "AttnK" => saw.AttnK,
            "AttnV" => saw.AttnV,
            "AttnO" => saw.AttnO,
            _ => throw new ArgumentException($"Unknown weight: {weightName}")
        };

        var merged = MergeWeight(original, lora, backend);

        // Replace the weight tensor in the model
        switch (weightName)
        {
            case "AttnQ": saw.AttnQ = merged; break;
            case "AttnK": saw.AttnK = merged; break;
            case "AttnV": saw.AttnV = merged; break;
            case "AttnO": saw.AttnO = merged; break;
        }
        original.Dispose();
    }

    private static void MergeDeltaNetWeight(DeltaNetWeights dnw, string weightName,
        LoraLayer? lora, IComputeBackend backend)
    {
        if (lora == null) return;

        ITensor original = weightName switch
        {
            "AttnQkv" => dnw.AttnQkv,
            "SsmOut" => dnw.SsmOut,
            _ => throw new ArgumentException($"Unknown DeltaNet weight: {weightName}")
        };

        var merged = MergeWeight(original, lora, backend);

        switch (weightName)
        {
            case "AttnQkv": dnw.AttnQkv = merged; break;
            case "SsmOut": dnw.SsmOut = merged; break;
        }
        original.Dispose();
    }

    private static ITensor MergeWeight(ITensor original, LoraLayer lora, IComputeBackend backend)
    {
        int outDim = lora.OutFeatures;
        int inDim = lora.InFeatures;
        var wF32 = new float[outDim * inDim];
        original.DequantizeTo(wF32);

        // LoRA contribution: scaling * B @ A
        var loraContrib = new float[outDim * inDim];
        F32Ops.MatMul(loraContrib, lora.B.Data, lora.A.Data, outDim, lora.Rank, inDim);

        for (int i = 0; i < wF32.Length; i++)
            wF32[i] += lora.Scaling * loraContrib[i];

        var dims = new long[] { inDim, outDim };
        var merged = backend.CreateTensor($"{original.Name}.lora_merged", GgmlType.F32, dims);
        var bytes = new byte[wF32.Length * sizeof(float)];
        Buffer.BlockCopy(wF32, 0, bytes, 0, bytes.Length);
        merged.CopyFrom(bytes);
        return merged;
    }

    /// <summary>
    /// Load a LoRA adapter and merge it into the model weights.
    /// Returns the adapter for reference (e.g., to check config).
    /// </summary>
    public static LoraAdapter LoadAndMerge(string adapterPath, ModelWeights weights,
        IComputeBackend backend, ModelConfig config)
    {
        var adapter = LoraFile.Load(adapterPath);
        MergeAdapter(weights, adapter, backend, config);
        return adapter;
    }
}
