using Daisi.Llogos.Model;

namespace Daisi.Llogos.Training.Lora;

/// <summary>
/// Collection of all LoRA layers for a model. One adapter per model.
/// Creates LoRA layers for standard attention (Q/K/V/O) and DeltaNet (QKV/Out) projections.
/// </summary>
public sealed class LoraAdapter : IDisposable
{
    public readonly LoraConfig Config;
    public readonly Dictionary<string, LoraLayer> Layers = new();

    /// <summary>Total trainable parameter count across all LoRA layers.</summary>
    public long ParameterCount => Layers.Values.Sum(l => (long)(l.Rank * l.InFeatures + l.OutFeatures * l.Rank));

    public LoraAdapter(LoraConfig config, ModelConfig modelConfig, ModelWeights weights, int seed = 42)
    {
        Config = config;
        var rng = new Random(seed);
        int rank = config.Rank;
        float scaling = config.Scaling;
        int hiddenDim = modelConfig.HiddenDim;

        for (int layer = 0; layer < modelConfig.NumLayers; layer++)
        {
            string prefix = $"blk.{layer}";

            if (modelConfig.IsStandardAttention(layer)
                && weights.Layers[layer] is StandardAttentionWeights saw)
            {
                if (config.Targets.HasFlag(LoraTarget.Q))
                {
                    int outDim = (int)saw.AttnQ.Dimensions[1];
                    Layers[$"{prefix}.attn_q"] = new LoraLayer(
                        $"{prefix}.attn_q", hiddenDim, outDim, rank, scaling, rng);
                }

                if (config.Targets.HasFlag(LoraTarget.K))
                {
                    int outDim = (int)saw.AttnK.Dimensions[1];
                    Layers[$"{prefix}.attn_k"] = new LoraLayer(
                        $"{prefix}.attn_k", hiddenDim, outDim, rank, scaling, rng);
                }

                if (config.Targets.HasFlag(LoraTarget.V))
                {
                    int outDim = (int)saw.AttnV.Dimensions[1];
                    Layers[$"{prefix}.attn_v"] = new LoraLayer(
                        $"{prefix}.attn_v", hiddenDim, outDim, rank, scaling, rng);
                }

                if (config.Targets.HasFlag(LoraTarget.O))
                {
                    int inDim = (int)saw.AttnO.Dimensions[0];
                    Layers[$"{prefix}.attn_o"] = new LoraLayer(
                        $"{prefix}.attn_o", inDim, hiddenDim, rank, scaling, rng);
                }
            }
            else if (!modelConfig.IsStandardAttention(layer)
                && weights.Layers[layer] is DeltaNetWeights dnw)
            {
                if (config.Targets.HasFlag(LoraTarget.DeltaQkv))
                {
                    int outDim = (int)dnw.AttnQkv.Dimensions[1];
                    Layers[$"{prefix}.delta_qkv"] = new LoraLayer(
                        $"{prefix}.delta_qkv", hiddenDim, outDim, rank, scaling, rng);
                }

                if (config.Targets.HasFlag(LoraTarget.DeltaOut))
                {
                    int inDim = (int)dnw.SsmOut.Dimensions[0];
                    int outDim = (int)dnw.SsmOut.Dimensions[1];
                    Layers[$"{prefix}.delta_out"] = new LoraLayer(
                        $"{prefix}.delta_out", inDim, outDim, rank, scaling, rng);
                }
            }
        }
    }

    /// <summary>
    /// Create adapter from loaded LoRA layers (deserialization).
    /// </summary>
    internal LoraAdapter(LoraConfig config, Dictionary<string, LoraLayer> layers)
    {
        Config = config;
        Layers = layers;
    }

    public LoraLayer? GetLayer(string name) =>
        Layers.TryGetValue(name, out var layer) ? layer : null;

    /// <summary>
    /// Get all trainable parameters (A and B tensors from all LoRA layers).
    /// </summary>
    public IEnumerable<GradTensor> Parameters()
    {
        foreach (var layer in Layers.Values)
        {
            yield return layer.A;
            yield return layer.B;
        }
    }

    public void ZeroGrad()
    {
        foreach (var layer in Layers.Values)
            layer.ZeroGrad();
    }

    public void Dispose()
    {
        foreach (var layer in Layers.Values)
        {
            layer.A.Dispose();
            layer.B.Dispose();
        }
    }
}
