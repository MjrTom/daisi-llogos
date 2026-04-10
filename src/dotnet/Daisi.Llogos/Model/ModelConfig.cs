using Daisi.Llogos.Gguf;

namespace Daisi.Llogos.Model;

/// <summary>
/// Model hyperparameters extracted from GGUF metadata.
/// Supports hybrid architectures (standard attention + DeltaNet).
/// </summary>
public sealed class ModelConfig
{
    // Core dimensions
    public required string Architecture { get; init; }
    public required int NumLayers { get; init; }
    public required int HiddenDim { get; init; }
    public required int IntermediateDim { get; init; }
    public required int VocabSize { get; init; }
    public required int MaxContext { get; init; }
    public required float NormEps { get; init; }

    // Standard attention parameters
    public required int NumHeads { get; init; }
    public required int NumKvHeads { get; init; }
    public required int KeyLength { get; init; }
    public required int ValueLength { get; init; }
    public int HeadsPerKvGroup => NumHeads / NumKvHeads;

    // RoPE
    public required float RopeTheta { get; init; }
    public required int RopeDimCount { get; init; }

    // Hybrid layer schedule
    public required int FullAttentionInterval { get; init; }

    // SSM / DeltaNet parameters
    public required int SsmConvKernel { get; init; }
    public required int SsmStateSize { get; init; }
    public required int SsmGroupCount { get; init; }
    public required int SsmInnerSize { get; init; }

    public int SsmHeadDim => SsmGroupCount > 0 ? SsmInnerSize / SsmGroupCount : 0;

    // ── Gemma 4 specific ────────────────────────────────────────────────────
    // Gemma 4 has interleaved sliding-window and full-attention layers with
    // *different* head dimensions per layer type (sliding=256, full=512).
    // The base KeyLength/ValueLength/RopeTheta/RopeDimCount fields above hold
    // the FULL attention values; the *Swa fields hold the sliding values.

    /// <summary>Sliding-window attention key dim (Gemma 4 only). 0 if not applicable.</summary>
    public int KeyLengthSwa { get; init; }

    /// <summary>Sliding-window attention value dim (Gemma 4 only). 0 if not applicable.</summary>
    public int ValueLengthSwa { get; init; }

    /// <summary>Sliding-window RoPE base frequency (Gemma 4 only). 0 if not applicable.</summary>
    public float RopeThetaSwa { get; init; }

    /// <summary>Sliding-window RoPE rotation dim (Gemma 4 only). 0 if not applicable.</summary>
    public int RopeDimCountSwa { get; init; }

    /// <summary>Sliding window size in tokens (Gemma 4: 512). 0 = no sliding window.</summary>
    public int SlidingWindow { get; init; }

    /// <summary>
    /// Per-layer flag: true if layer uses sliding-window attention, false if full attention.
    /// Empty for non-gemma4 models. For Gemma 4 E4B-it: 5×T + 1×F repeated 7 times.
    /// </summary>
    public bool[] LayerSlidingMask { get; init; } = Array.Empty<bool>();

    /// <summary>Final logit softcapping value (Gemma 2/3/4: 30.0). 0 = disabled.</summary>
    public float FinalLogitSoftcap { get; init; }

    /// <summary>Per-Layer Embedding (PLE) input dimension (Gemma 4: 256). 0 = no PLE.</summary>
    public int PerLayerInputDim { get; init; }

    /// <summary>
    /// Number of leading layers that compute their own K/V (Gemma 4 KV sharing).
    /// Layers >= NumLayerKvFromStart reuse K/V cached by an earlier layer.
    /// Equals NumLayers when KV sharing is disabled.
    /// </summary>
    public int NumLayerKvFromStart { get; init; }

    /// <summary>True when the model uses BitNet b1.58 architecture.</summary>
    public bool IsBitNet => Architecture.StartsWith("bitnet", StringComparison.OrdinalIgnoreCase);

    /// <summary>True when the model is the Gemma 4 family (gemma4).</summary>
    public bool IsGemma4 => Architecture.Equals("gemma4", StringComparison.OrdinalIgnoreCase);

    /// <summary>True when the model has any DeltaNet/SSM layers.</summary>
    public bool HasDeltaNet => SsmInnerSize > 0;

    /// <summary>Is the given layer a standard attention layer (vs DeltaNet)?</summary>
    public bool IsStandardAttention(int layer) =>
        !HasDeltaNet
        || (FullAttentionInterval > 0 && ((layer + 1) % FullAttentionInterval == 0));

    /// <summary>True if the given Gemma 4 layer uses sliding-window attention. False otherwise.</summary>
    public bool IsSlidingLayer(int layer) =>
        LayerSlidingMask.Length > layer && LayerSlidingMask[layer];

    /// <summary>Effective key length for the given layer (Gemma 4: differs per layer type).</summary>
    public int LayerKeyLength(int layer) =>
        IsSlidingLayer(layer) ? KeyLengthSwa : KeyLength;

    /// <summary>Effective value length for the given layer.</summary>
    public int LayerValueLength(int layer) =>
        IsSlidingLayer(layer) ? ValueLengthSwa : ValueLength;

    /// <summary>Effective RoPE base frequency for the given layer.</summary>
    public float LayerRopeTheta(int layer) =>
        IsSlidingLayer(layer) ? RopeThetaSwa : RopeTheta;

    /// <summary>Effective RoPE rotation dim for the given layer.</summary>
    public int LayerRopeDim(int layer) =>
        IsSlidingLayer(layer) ? RopeDimCountSwa : RopeDimCount;

    /// <summary>
    /// True when the given Gemma 4 layer computes its own K/V projection.
    /// Layers beyond NumLayerKvFromStart reuse cached K/V from an earlier layer.
    /// Always true for non-gemma4 models.
    /// </summary>
    public bool HasKv(int layer) =>
        NumLayerKvFromStart <= 0 || layer < NumLayerKvFromStart;

    public static ModelConfig FromGguf(GgufFile gguf)
    {
        var arch = gguf.GetMetadataString("general.architecture")
            ?? throw new InvalidDataException("Missing general.architecture");

        var vocabTokens = gguf.GetMetadata<string[]>("tokenizer.ggml.tokens")
            ?? throw new InvalidDataException("Missing tokenizer.ggml.tokens");

        int numLayers = GetInt(gguf, $"{arch}.block_count");
        bool isGemma4 = arch.Equals("gemma4", StringComparison.OrdinalIgnoreCase);

        // Gemma 4 sliding-window pattern is a bool[NumLayers] in metadata.
        // True = sliding-window attention, False = full attention.
        bool[] slidingMask = Array.Empty<bool>();
        int slidingWindow = 0;
        int keyLengthSwa = 0;
        int valueLengthSwa = 0;
        float ropeThetaSwa = 0;
        int ropeDimCountSwa = 0;
        float finalLogitSoftcap = 0;
        int perLayerInputDim = 0;
        int numLayerKvFromStart = 0;

        if (isGemma4)
        {
            slidingMask = GetMetadataBoolArrayOrEmpty(gguf, "gemma4.attention.sliding_window_pattern");
            slidingWindow = GetIntOrDefault(gguf, "gemma4.attention.sliding_window", 0);
            keyLengthSwa = GetIntOrDefault(gguf, "gemma4.attention.key_length_swa", 0);
            valueLengthSwa = GetIntOrDefault(gguf, "gemma4.attention.value_length_swa", 0);
            ropeThetaSwa = GetFloat(gguf, "gemma4.rope.freq_base_swa", 10000.0f);
            ropeDimCountSwa = GetIntOrDefault(gguf, "gemma4.rope.dimension_count_swa", 0);
            finalLogitSoftcap = GetFloat(gguf, "gemma4.final_logit_softcapping", 0);
            perLayerInputDim = GetIntOrDefault(gguf, "gemma4.embedding_length_per_layer_input", 0);

            int sharedKvLayers = GetIntOrDefault(gguf, "gemma4.attention.shared_kv_layers", 0);
            numLayerKvFromStart = numLayers - sharedKvLayers;
        }

        return new ModelConfig
        {
            Architecture = arch,
            NumLayers = numLayers,
            HiddenDim = GetInt(gguf, $"{arch}.embedding_length"),
            IntermediateDim = GetInt(gguf, $"{arch}.feed_forward_length"),
            VocabSize = vocabTokens.Length,
            MaxContext = GetInt(gguf, $"{arch}.context_length"),
            NormEps = GetFloat(gguf, $"{arch}.attention.layer_norm_rms_epsilon", 1e-6f),

            NumHeads = GetInt(gguf, $"{arch}.attention.head_count"),
            NumKvHeads = GetInt(gguf, $"{arch}.attention.head_count_kv"),
            KeyLength = GetIntOrDefault(gguf, $"{arch}.attention.key_length",
                GetInt(gguf, $"{arch}.embedding_length") / GetInt(gguf, $"{arch}.attention.head_count")),
            ValueLength = GetIntOrDefault(gguf, $"{arch}.attention.value_length",
                GetInt(gguf, $"{arch}.embedding_length") / GetInt(gguf, $"{arch}.attention.head_count")),

            RopeTheta = GetFloat(gguf, $"{arch}.rope.freq_base", 10000.0f),
            RopeDimCount = GetIntOrDefault(gguf, $"{arch}.rope.dimension_count",
                GetInt(gguf, $"{arch}.embedding_length") / GetInt(gguf, $"{arch}.attention.head_count")),

            FullAttentionInterval = GetIntOrDefault(gguf, $"{arch}.full_attention_interval", 0),

            SsmConvKernel = GetIntOrDefault(gguf, $"{arch}.ssm.conv_kernel", 0),
            SsmStateSize = GetIntOrDefault(gguf, $"{arch}.ssm.state_size", 0),
            SsmGroupCount = GetIntOrDefault(gguf, $"{arch}.ssm.group_count", 0),
            SsmInnerSize = GetIntOrDefault(gguf, $"{arch}.ssm.inner_size", 0),

            // Gemma 4
            KeyLengthSwa = keyLengthSwa,
            ValueLengthSwa = valueLengthSwa,
            RopeThetaSwa = ropeThetaSwa,
            RopeDimCountSwa = ropeDimCountSwa,
            SlidingWindow = slidingWindow,
            LayerSlidingMask = slidingMask,
            FinalLogitSoftcap = finalLogitSoftcap,
            PerLayerInputDim = perLayerInputDim,
            NumLayerKvFromStart = numLayerKvFromStart,
        };
    }

    private static bool[] GetMetadataBoolArrayOrEmpty(GgufFile gguf, string key)
    {
        var kv = gguf.Metadata.FirstOrDefault(m => m.Key == key);
        if (kv == null) return Array.Empty<bool>();
        if (kv.Value is bool[] b) return b;
        // Some converters write the bool array as Uint8 elements (0/1).
        if (kv.Value is byte[] bytes)
        {
            var result = new bool[bytes.Length];
            for (int i = 0; i < bytes.Length; i++) result[i] = bytes[i] != 0;
            return result;
        }
        return Array.Empty<bool>();
    }

    private static int GetInt(GgufFile gguf, string key)
    {
        var kv = gguf.Metadata.FirstOrDefault(m => m.Key == key)
            ?? throw new InvalidDataException($"Missing metadata: {key}");
        return ConvertToInt(kv);
    }

    private static int GetIntOrDefault(GgufFile gguf, string key, int defaultValue)
    {
        var kv = gguf.Metadata.FirstOrDefault(m => m.Key == key);
        return kv == null ? defaultValue : ConvertToInt(kv);
    }

    private static int ConvertToInt(GgufMetadataKv kv) => kv.Type switch
    {
        GgufMetadataValueType.Uint32 => (int)kv.ValueAs<uint>(),
        GgufMetadataValueType.Int32 => kv.ValueAs<int>(),
        GgufMetadataValueType.Uint64 => (int)kv.ValueAs<ulong>(),
        GgufMetadataValueType.Int64 => (int)kv.ValueAs<long>(),
        _ => throw new InvalidDataException($"Unexpected type {kv.Type} for {kv.Key}")
    };

    private static float GetFloat(GgufFile gguf, string key, float defaultValue)
    {
        var kv = gguf.Metadata.FirstOrDefault(m => m.Key == key);
        if (kv == null) return defaultValue;
        return kv.Type switch
        {
            GgufMetadataValueType.Float32 => kv.ValueAs<float>(),
            GgufMetadataValueType.Float64 => (float)kv.ValueAs<double>(),
            _ => throw new InvalidDataException($"Unexpected type {kv.Type} for {key}")
        };
    }
}
