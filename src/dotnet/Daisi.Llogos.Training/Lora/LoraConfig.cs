namespace Daisi.Llogos.Training.Lora;

/// <summary>
/// Configuration for LoRA (Low-Rank Adaptation) training.
/// </summary>
public sealed class LoraConfig
{
    /// <summary>LoRA rank (dimension of the low-rank decomposition). Typical: 4, 8, 16, 32.</summary>
    public int Rank { get; init; } = 8;

    /// <summary>LoRA scaling factor. The LoRA contribution is scaled by Alpha/Rank.</summary>
    public float Alpha { get; init; } = 16.0f;

    /// <summary>Dropout rate applied to LoRA input (0 = no dropout).</summary>
    public float Dropout { get; init; } = 0.0f;

    /// <summary>
    /// Which attention projections to target.
    /// Default: Q, K, V, and O projections.
    /// </summary>
    public LoraTarget Targets { get; init; } = LoraTarget.Q | LoraTarget.K | LoraTarget.V | LoraTarget.O;

    /// <summary>Effective scaling factor: alpha / rank.</summary>
    public float Scaling => Alpha / Rank;
}

[Flags]
public enum LoraTarget
{
    None = 0,
    Q = 1,
    K = 2,
    V = 4,
    O = 8,
    All = Q | K | V | O,

    // DeltaNet projections
    DeltaQkv = 16,  // AttnQkv: H → qkvDim (input to SSM pipeline)
    DeltaOut = 32,   // SsmOut: ssmInner → H (output from SSM)
    AllDelta = DeltaQkv | DeltaOut,
    AllLayers = All | AllDelta,
}
