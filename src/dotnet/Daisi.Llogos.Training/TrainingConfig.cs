using Daisi.Llogos.Training.Lora;

namespace Daisi.Llogos.Training;

/// <summary>
/// Configuration for a LoRA training session.
/// </summary>
public sealed class TrainingConfig
{
    // ── Model ───────────────────────────────────────────────────────────────
    public required string ModelPath { get; init; }

    // ── Data ────────────────────────────────────────────────────────────────
    public required string DataPath { get; init; }
    public DataFormat Format { get; init; } = DataFormat.Auto;
    public int SeqLen { get; init; } = 512;

    // ── LoRA ────────────────────────────────────────────────────────────────
    public LoraConfig Lora { get; init; } = new();

    // ── Training ────────────────────────────────────────────────────────────
    public int Epochs { get; init; } = 3;
    public float LearningRate { get; init; } = 1e-4f;
    public float WeightDecay { get; init; } = 0.01f;
    public int WarmupSteps { get; init; } = 50;
    public int GradientAccumulationSteps { get; init; } = 1;
    public float MaxGradNorm { get; init; } = 1.0f;
    public int Seed { get; init; } = 42;

    // ── Output ──────────────────────────────────────────────────────────────
    public required string OutputPath { get; init; }
    public int SaveEverySteps { get; init; } = 100;
    public int LogEverySteps { get; init; } = 10;
}

public enum DataFormat
{
    Auto,       // detect from file extension
    PlainText,  // raw text, train on next-token prediction
    Jsonl,      // {"text": "..."} per line
    JsonlChat,  // {"prompt": "...", "completion": "..."} per line
}
