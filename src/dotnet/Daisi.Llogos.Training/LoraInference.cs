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

            // FFN projections (every layer has FFN)
            var lw = weights.Layers[layer];
            MergeFfnWeight(lw, "FfnGate", adapter.GetLayer($"{prefix}.ffn_gate"), backend);
            MergeFfnWeight(lw, "FfnUp", adapter.GetLayer($"{prefix}.ffn_up"), backend);
            MergeFfnWeight(lw, "FfnDown", adapter.GetLayer($"{prefix}.ffn_down"), backend);
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

    private static void MergeFfnWeight(LayerWeights lw, string weightName,
        LoraLayer? lora, IComputeBackend backend)
    {
        if (lora == null) return;

        ITensor original = weightName switch
        {
            "FfnGate" => lw.FfnGate,
            "FfnUp" => lw.FfnUp,
            "FfnDown" => lw.FfnDown,
            _ => throw new ArgumentException($"Unknown FFN weight: {weightName}")
        };

        var merged = MergeWeight(original, lora, backend);

        switch (weightName)
        {
            case "FfnGate": lw.FfnGate = merged; break;
            case "FfnUp": lw.FfnUp = merged; break;
            case "FfnDown": lw.FfnDown = merged; break;
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

    /// <summary>
    /// Re-upload all weight tensors from CPU to a different backend (e.g., CUDA).
    /// Used after LoRA merge: weights are merged on CPU, then uploaded to GPU for inference.
    /// </summary>
    public static void UploadWeights(ModelWeights weights, IComputeBackend target, ModelConfig config)
    {
        // Re-load the entire model on the target backend from the same GGUF file.
        // This is cleaner than re-uploading individual tensors.
        // Instead, we re-upload every tensor.
        weights.TokenEmbedding = ReUpload(weights.TokenEmbedding, target);
        weights.OutputNorm = ReUpload(weights.OutputNorm, target);
        if (weights.Output != null)
            weights.Output = ReUpload(weights.Output, target);

        for (int layer = 0; layer < config.NumLayers; layer++)
        {
            var lw = weights.Layers[layer];
            lw.AttnNorm = ReUpload(lw.AttnNorm, target);
            lw.PostAttnNorm = ReUpload(lw.PostAttnNorm, target);
            lw.FfnGate = ReUpload(lw.FfnGate, target);
            lw.FfnUp = ReUpload(lw.FfnUp, target);
            lw.FfnDown = ReUpload(lw.FfnDown, target);

            if (lw is StandardAttentionWeights saw)
            {
                saw.AttnQ = ReUpload(saw.AttnQ, target);
                saw.AttnK = ReUpload(saw.AttnK, target);
                saw.AttnV = ReUpload(saw.AttnV, target);
                saw.AttnO = ReUpload(saw.AttnO, target);
                if (saw.AttnQNorm != null) saw.AttnQNorm = ReUpload(saw.AttnQNorm, target);
                if (saw.AttnKNorm != null) saw.AttnKNorm = ReUpload(saw.AttnKNorm, target);
                // Clear fused tensors — they reference old CPU memory
                saw.FusedQKV?.Dispose(); saw.FusedQKV = null;
                saw.FusedGateUp?.Dispose(); saw.FusedGateUp = null;
            }
            else if (lw is DeltaNetWeights dnw)
            {
                dnw.AttnQkv = ReUpload(dnw.AttnQkv, target);
                dnw.AttnGate = ReUpload(dnw.AttnGate, target);
                dnw.SsmOut = ReUpload(dnw.SsmOut, target);
                // DeltaNet-specific non-merged weights
                ReUploadField(ref dnw, "SsmA", target);
                ReUploadField(ref dnw, "SsmAlpha", target);
                ReUploadField(ref dnw, "SsmBeta", target);
                ReUploadField(ref dnw, "SsmConv1d", target);
                ReUploadField(ref dnw, "SsmDtBias", target);
                ReUploadField(ref dnw, "SsmNorm", target);
            }
        }
    }

    private static void ReUploadField(ref DeltaNetWeights dnw, string field, IComputeBackend target)
    {
        // Use reflection-free approach: get tensor by name
        var prop = typeof(DeltaNetWeights).GetProperty(field)!;
        var tensor = (ITensor)prop.GetValue(dnw)!;
        prop.SetValue(dnw, ReUpload(tensor, target));
    }

    private static ITensor ReUpload(ITensor cpu, IComputeBackend target)
    {
        var rawBytes = new byte[cpu.ByteSize];
        cpu.CopyRawTo(rawBytes);
        var gpu = target.LoadTensor(cpu.Name, cpu.Type, cpu.Dimensions, rawBytes);
        cpu.Dispose();
        return gpu;
    }
}
