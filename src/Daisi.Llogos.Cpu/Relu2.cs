namespace Daisi.Llogos.Cpu;

/// <summary>
/// Squared ReLU activation: output[i] = max(0, input[i])².
/// Used by BitNet b1.58 instead of SiLU.
/// </summary>
internal static class Relu2
{
    public static void Apply(Span<float> output, ReadOnlySpan<float> input)
    {
        for (int i = 0; i < input.Length; i++)
        {
            float x = MathF.Max(0, input[i]);
            output[i] = x * x;
        }
    }

    public static void ApplyInPlace(Span<float> data)
    {
        for (int i = 0; i < data.Length; i++)
        {
            float x = MathF.Max(0, data[i]);
            data[i] = x * x;
        }
    }
}
