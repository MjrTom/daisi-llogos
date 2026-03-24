namespace Daisi.Llogos.Inference;

/// <summary>
/// Configuration for text generation: sampling parameters and stop conditions.
/// </summary>
public sealed record GenerationParams
{
    /// <summary>Maximum tokens to generate.</summary>
    public int MaxTokens { get; init; } = 256;

    /// <summary>Sampling temperature. 0 = greedy (argmax).</summary>
    public float Temperature { get; init; } = 0.7f;

    /// <summary>Top-k filter. 0 = disabled.</summary>
    public int TopK { get; init; } = 40;

    /// <summary>Nucleus sampling threshold. 1.0 = disabled.</summary>
    public float TopP { get; init; } = 0.9f;

    /// <summary>Repetition penalty factor. 1.0 = disabled.</summary>
    public float RepetitionPenalty { get; init; } = 1.1f;

    /// <summary>Token IDs that stop generation. Defaults to EOS if set.</summary>
    public int[]? StopTokens { get; init; }

    /// <summary>Random seed for sampling. null = non-deterministic.</summary>
    public int? Seed { get; init; }
}
