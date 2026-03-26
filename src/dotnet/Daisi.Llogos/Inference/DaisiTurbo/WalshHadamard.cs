using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Daisi.Llogos.Inference.DaisiTurbo;

/// <summary>
/// Fast Walsh-Hadamard Transform (WHT) with randomized sign flips.
/// The rotation spreads outliers uniformly across dimensions, making all
/// coordinates follow a concentrated distribution with similar magnitudes.
/// This enables fixed quantization grids without per-vector calibration.
/// </summary>
public static class WalshHadamard
{
    /// <summary>
    /// Apply randomized WHT in-place: multiply by D·H where D is a random sign-flip
    /// diagonal and H is the normalized Hadamard matrix.
    /// O(d log d) time, zero allocations.
    /// </summary>
    /// <param name="data">Vector to transform (length must be power of 2).</param>
    /// <param name="signs">Random ±1 sign flips, one per dimension (generated from seed).</param>
    public static void ForwardInPlace(Span<float> data, ReadOnlySpan<float> signs)
    {
        int n = data.Length;

        // Step 1: Apply random sign flips (D · x)
        if (Avx2.IsSupported && n >= 8)
            ApplySignsAvx2(data, signs);
        else
            ApplySignsScalar(data, signs);

        // Step 2: WHT butterfly passes (H · D · x)
        if (Avx2.IsSupported && n >= 8)
            ButterflyAvx2(data);
        else
            ButterflyScalar(data);

        // Step 3: Normalize by 1/sqrt(n) so the transform is orthonormal
        float norm = 1.0f / MathF.Sqrt(n);
        if (Avx2.IsSupported && n >= 8)
            ScaleAvx2(data, norm);
        else
            ScaleScalar(data, norm);
    }

    /// <summary>
    /// Inverse randomized WHT in-place: multiply by H·D (same as forward since H and D are self-inverse).
    /// </summary>
    public static void InverseInPlace(Span<float> data, ReadOnlySpan<float> signs)
    {
        int n = data.Length;

        // Inverse is: D · H · x (same operations, H is self-inverse up to normalization)
        float norm = 1.0f / MathF.Sqrt(n);
        if (Avx2.IsSupported && n >= 8)
            ScaleAvx2(data, norm);
        else
            ScaleScalar(data, norm);

        if (Avx2.IsSupported && n >= 8)
            ButterflyAvx2(data);
        else
            ButterflyScalar(data);

        if (Avx2.IsSupported && n >= 8)
            ApplySignsAvx2(data, signs);
        else
            ApplySignsScalar(data, signs);
    }

    /// <summary>
    /// Generate deterministic sign flips from a seed. Same seed always produces same signs.
    /// </summary>
    public static float[] GenerateSigns(int dimension, int seed)
    {
        var signs = new float[dimension];
        var rng = new Random(seed);
        for (int i = 0; i < dimension; i++)
            signs[i] = rng.Next(2) == 0 ? 1.0f : -1.0f;
        return signs;
    }

    // ── Scalar implementations ──────────────────────────────────────────────

    private static void ButterflyScalar(Span<float> data)
    {
        int n = data.Length;
        for (int halfSize = 1; halfSize < n; halfSize <<= 1)
        {
            for (int i = 0; i < n; i += halfSize << 1)
            {
                for (int j = i; j < i + halfSize; j++)
                {
                    float a = data[j];
                    float b = data[j + halfSize];
                    data[j] = a + b;
                    data[j + halfSize] = a - b;
                }
            }
        }
    }

    private static void ApplySignsScalar(Span<float> data, ReadOnlySpan<float> signs)
    {
        for (int i = 0; i < data.Length; i++)
            data[i] *= signs[i];
    }

    private static void ScaleScalar(Span<float> data, float scale)
    {
        for (int i = 0; i < data.Length; i++)
            data[i] *= scale;
    }

    // ── AVX2 implementations ────────────────────────────────────────────────

    private static void ApplySignsAvx2(Span<float> data, ReadOnlySpan<float> signs)
    {
        ref float dRef = ref MemoryMarshal.GetReference(data);
        ref float sRef = ref MemoryMarshal.GetReference(signs);
        int i = 0;
        int n = data.Length;
        for (; i + 8 <= n; i += 8)
        {
            var d = Vector256.LoadUnsafe(ref Unsafe.Add(ref dRef, i));
            var s = Vector256.LoadUnsafe(ref Unsafe.Add(ref sRef, i));
            Avx.Multiply(d, s).StoreUnsafe(ref Unsafe.Add(ref dRef, i));
        }
        for (; i < n; i++)
            data[i] *= signs[i];
    }

    private static void ButterflyAvx2(Span<float> data)
    {
        ref float dRef = ref MemoryMarshal.GetReference(data);
        int n = data.Length;

        // Small half-sizes: scalar butterfly (stride < 8)
        for (int halfSize = 1; halfSize < Math.Min(8, n); halfSize <<= 1)
        {
            for (int i = 0; i < n; i += halfSize << 1)
            {
                for (int j = i; j < i + halfSize; j++)
                {
                    float a = data[j];
                    float b = data[j + halfSize];
                    data[j] = a + b;
                    data[j + halfSize] = a - b;
                }
            }
        }

        // Large half-sizes: AVX2 butterfly (stride >= 8)
        for (int halfSize = 8; halfSize < n; halfSize <<= 1)
        {
            for (int i = 0; i < n; i += halfSize << 1)
            {
                for (int j = i; j + 8 <= i + halfSize; j += 8)
                {
                    var a = Vector256.LoadUnsafe(ref Unsafe.Add(ref dRef, j));
                    var b = Vector256.LoadUnsafe(ref Unsafe.Add(ref dRef, j + halfSize));
                    Avx.Add(a, b).StoreUnsafe(ref Unsafe.Add(ref dRef, j));
                    Avx.Subtract(a, b).StoreUnsafe(ref Unsafe.Add(ref dRef, j + halfSize));
                }
            }
        }
    }

    private static void ScaleAvx2(Span<float> data, float scale)
    {
        ref float dRef = ref MemoryMarshal.GetReference(data);
        var scaleVec = Vector256.Create(scale);
        int i = 0;
        int n = data.Length;
        for (; i + 8 <= n; i += 8)
        {
            var d = Vector256.LoadUnsafe(ref Unsafe.Add(ref dRef, i));
            Avx.Multiply(d, scaleVec).StoreUnsafe(ref Unsafe.Add(ref dRef, i));
        }
        for (; i < n; i++)
            data[i] *= scale;
    }
}
