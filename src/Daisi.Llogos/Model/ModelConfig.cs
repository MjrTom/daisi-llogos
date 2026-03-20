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

    /// <summary>True when the model has any DeltaNet/SSM layers.</summary>
    public bool HasDeltaNet => SsmInnerSize > 0;

    /// <summary>Is the given layer a standard attention layer (vs DeltaNet)?</summary>
    public bool IsStandardAttention(int layer) =>
        !HasDeltaNet
        || (FullAttentionInterval > 0 && ((layer + 1) % FullAttentionInterval == 0));

    public static ModelConfig FromGguf(GgufFile gguf)
    {
        var arch = gguf.GetMetadataString("general.architecture")
            ?? throw new InvalidDataException("Missing general.architecture");

        var vocabTokens = gguf.GetMetadata<string[]>("tokenizer.ggml.tokens")
            ?? throw new InvalidDataException("Missing tokenizer.ggml.tokens");

        return new ModelConfig
        {
            Architecture = arch,
            NumLayers = GetInt(gguf, $"{arch}.block_count"),
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
        };
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
