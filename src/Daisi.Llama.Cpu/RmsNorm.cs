using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Daisi.Llama.Cpu;

/// <summary>
/// RMS normalization: output[i] = (input[i] / rms) * weight[i]
/// where rms = sqrt(mean(input²) + eps).
/// </summary>
internal static class RmsNorm
{
    public static void Apply(Span<float> output, ReadOnlySpan<float> input, ReadOnlySpan<float> weight, float eps)
    {
        int n = input.Length;

        // Pass 1: compute sum of squares
        float sumSq = 0;
        if (Avx2.IsSupported && n >= 8)
        {
            ref float inRef = ref MemoryMarshal.GetReference(input);
            var vSum = Vector256<float>.Zero;
            int i = 0;
            for (; i + 8 <= n; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref Unsafe.Add(ref inRef, i));
                vSum = Avx.Add(vSum, Avx.Multiply(v, v));
            }
            sumSq = Vector256.Sum(vSum);
            for (; i < n; i++)
                sumSq += input[i] * input[i];
        }
        else
        {
            for (int i = 0; i < n; i++)
                sumSq += input[i] * input[i];
        }

        // Pass 2: normalize and scale
        float rmsInv = 1.0f / MathF.Sqrt(sumSq / n + eps);

        if (Avx2.IsSupported && n >= 8)
        {
            ref float inRef = ref MemoryMarshal.GetReference(input);
            ref float wRef = ref MemoryMarshal.GetReference(weight);
            ref float oRef = ref MemoryMarshal.GetReference(output);
            var vRmsInv = Vector256.Create(rmsInv);
            int i = 0;
            for (; i + 8 <= n; i += 8)
            {
                var vIn = Vector256.LoadUnsafe(ref Unsafe.Add(ref inRef, i));
                var vW = Vector256.LoadUnsafe(ref Unsafe.Add(ref wRef, i));
                var result = Avx.Multiply(Avx.Multiply(vIn, vRmsInv), vW);
                result.StoreUnsafe(ref Unsafe.Add(ref oRef, i));
            }
            for (; i < n; i++)
                output[i] = input[i] * rmsInv * weight[i];
        }
        else
        {
            for (int i = 0; i < n; i++)
                output[i] = input[i] * rmsInv * weight[i];
        }
    }
}
