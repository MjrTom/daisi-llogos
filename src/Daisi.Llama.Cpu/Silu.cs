namespace Daisi.Llama.Cpu;

/// <summary>
/// SiLU (Sigmoid Linear Unit) activation: output[i] = input[i] * sigmoid(input[i]).
/// Also known as Swish.
/// </summary>
internal static class Silu
{
    public static void Apply(Span<float> output, ReadOnlySpan<float> input)
    {
        for (int i = 0; i < input.Length; i++)
        {
            float x = input[i];
            output[i] = x / (1.0f + MathF.Exp(-x));
        }
    }
}
