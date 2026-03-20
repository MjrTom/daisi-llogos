namespace Daisi.Llogos.Cpu;

/// <summary>
/// Numerically stable softmax: output[i] = exp(input[i] - max) / sum(exp(input - max)).
/// </summary>
internal static class Softmax
{
    public static void Apply(Span<float> output, ReadOnlySpan<float> input)
    {
        int n = input.Length;

        // Pass 1: find max
        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++)
            if (input[i] > max) max = input[i];

        // Pass 2: exp(x - max) and sum
        float sum = 0;
        for (int i = 0; i < n; i++)
        {
            float e = MathF.Exp(input[i] - max);
            output[i] = e;
            sum += e;
        }

        // Pass 3: normalize
        float invSum = 1.0f / sum;
        for (int i = 0; i < n; i++)
            output[i] *= invSum;
    }
}
