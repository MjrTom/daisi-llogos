namespace Daisi.Llogos.Inference;

/// <summary>
/// Configuration for text generation: sampling parameters and stop conditions.
/// </summary>
public sealed record GenerationParams
{
    /// <summary>Maximum tokens to generate.</summary>
    public int MaxTokens { get; init; } = 256;

    /// <summary>Sampling temperature. 0 = greedy (argmax).</summary>
    public float Temperature { get; init; } = 0.8f;

    /// <summary>Top-k filter. 0 = disabled.</summary>
    public int TopK { get; init; } = 40;

    /// <summary>Nucleus sampling threshold. 1.0 = disabled.</summary>
    public float TopP { get; init; } = 0.95f;

    /// <summary>Repetition penalty factor. 1.0 = disabled.</summary>
    public float RepetitionPenalty { get; init; } = 1.1f;

    /// <summary>Token IDs that stop generation. Defaults to EOS if set.</summary>
    public int[]? StopTokens { get; init; }

    /// <summary>Random seed for sampling. null = non-deterministic.</summary>
    public int? Seed { get; init; }

    // ── Additional parameters (passed through from host) ──

    /// <summary>Frequency penalty: penalizes tokens proportional to their count in the output. 0 = disabled.</summary>
    public float FrequencyPenalty { get; init; }

    /// <summary>Presence penalty: flat penalty for any token that has appeared. 0 = disabled.</summary>
    public float PresencePenalty { get; init; }

    /// <summary>Minimum probability threshold. Tokens below min_p × max_prob are removed. 0 = disabled.</summary>
    public float MinP { get; init; }

    /// <summary>Typical sampling threshold. Filters tokens by typical probability. 1.0 = disabled.</summary>
    public float TypicalP { get; init; } = 1.0f;

    /// <summary>Whether to penalize newline tokens during repetition penalty. Default: false.</summary>
    public bool PenalizeNewline { get; init; }

    /// <summary>Number of recent tokens to consider for repetition/frequency/presence penalty. 0 = all.</summary>
    public int PenaltyCount { get; init; } = 64;

    /// <summary>Minimum candidates to keep after top-p/min-p filtering. Default: 1.</summary>
    public int MinKeep { get; init; } = 1;

    /// <summary>Prevent generation of EOS token (force continued generation). Default: false.</summary>
    public bool PreventEOS { get; init; }

    /// <summary>Anti-prompt strings: stop generation when any of these appear in output.</summary>
    public string[]? AntiPrompts { get; init; }

    /// <summary>GBNF grammar text for constrained generation. null = unconstrained.</summary>
    public string? GrammarText { get; init; }
}
