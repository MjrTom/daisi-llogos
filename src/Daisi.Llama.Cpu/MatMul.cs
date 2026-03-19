using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Daisi.Llama.Cpu;

/// <summary>
/// Matrix multiplication with optional fused dequantization.
/// output[M×N] = a[M×K] × b[K×N], where b may be quantized.
/// All matrices are row-major 1D spans.
/// </summary>
internal static class MatMul
{
    private const int Q8_0BlockSize = 32;
    private const int Q8_0TypeSize = 34;

    /// <summary>
    /// FP32 × FP32 matrix multiply.
    /// b is [N×K] (GGUF convention: each of N output rows has K weights).
    /// output[i,j] = dot(a[i,:], b[j,:])
    /// </summary>
    public static void Multiply(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<float> b, int M, int K, int N)
    {
        if (Avx2.IsSupported && K >= 8)
            MultiplyAvx2(output, a, b, M, K, N);
        else
            MultiplyScalar(output, a, b, M, K, N);
    }

    /// <summary>
    /// FP32 × Q8_0 fused dequant+matmul.
    /// a is [M×K] FP32, b is [K×N] Q8_0 stored as N column vectors each of K elements.
    /// b layout: N contiguous Q8_0-encoded vectors of length K.
    /// </summary>
    public static void MultiplyQ8_0(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<byte> b, int M, int K, int N)
    {
        // b is row-major [K×N] but stored quantized. Each row of K elements across N columns.
        // Actually for weight matrices, b is [N×K] quantized (each of N output neurons has K weights).
        // We compute: for each (i,j): output[i,j] = dot(a_row_i[K], b_row_j[K])
        // This matches llama.cpp convention where weights are [outDim × inDim].
        int blocksPerRow = K / Q8_0BlockSize;
        int bytesPerRow = blocksPerRow * Q8_0TypeSize;

        for (int i = 0; i < M; i++)
        {
            var aRow = a.Slice(i * K, K);
            for (int j = 0; j < N; j++)
            {
                var bRow = b.Slice(j * bytesPerRow, bytesPerRow);
                output[i * N + j] = DotQ8_0(aRow, bRow, blocksPerRow);
            }
        }
    }

    // ── FP32 Scalar ──────────────────────────────────────────────────────────

    private static void MultiplyScalar(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<float> b, int M, int K, int N)
    {
        // b is [N×K]: row j has K weights for output j
        for (int i = 0; i < M; i++)
        {
            for (int j = 0; j < N; j++)
            {
                float sum = 0;
                int aOff = i * K;
                int bOff = j * K;
                for (int k = 0; k < K; k++)
                    sum += a[aOff + k] * b[bOff + k];
                output[i * N + j] = sum;
            }
        }
    }

    // ── FP32 AVX2 ────────────────────────────────────────────────────────────
    // b is [N×K]: each output row j has K weights. SIMD over K dimension.

    private static void MultiplyAvx2(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<float> b, int M, int K, int N)
    {
        ref float aRef = ref MemoryMarshal.GetReference(a);
        ref float bRef = ref MemoryMarshal.GetReference(b);
        ref float oRef = ref MemoryMarshal.GetReference(output);

        for (int i = 0; i < M; i++)
        {
            int aBase = i * K;
            for (int j = 0; j < N; j++)
            {
                int bBase = j * K;
                var vSum = Vector256<float>.Zero;
                int k = 0;
                for (; k + 8 <= K; k += 8)
                {
                    var vA = Vector256.LoadUnsafe(ref Unsafe.Add(ref aRef, aBase + k));
                    var vB = Vector256.LoadUnsafe(ref Unsafe.Add(ref bRef, bBase + k));
                    vSum = Avx2.IsSupported
                        ? System.Runtime.Intrinsics.X86.Fma.MultiplyAdd(vA, vB, vSum)
                        : vSum + vA * vB;
                }
                float sum = Vector256.Sum(vSum);
                for (; k < K; k++)
                    sum += Unsafe.Add(ref aRef, aBase + k) * Unsafe.Add(ref bRef, bBase + k);
                Unsafe.Add(ref oRef, i * N + j) = sum;
            }
        }
    }

    // ── Q8_0 Fused Dot Product ───────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float DotQ8_0(ReadOnlySpan<float> aVec, ReadOnlySpan<byte> bQ8, int blockCount)
    {
        float sum = 0;
        for (int b = 0; b < blockCount; b++)
        {
            int srcOff = b * Q8_0TypeSize;
            int aOff = b * Q8_0BlockSize;
            float scale = (float)BitConverter.ToHalf(bQ8.Slice(srcOff, 2));

            float blockSum = 0;
            for (int i = 0; i < Q8_0BlockSize; i++)
            {
                blockSum += aVec[aOff + i] * (sbyte)bQ8[srcOff + 2 + i];
            }
            sum += scale * blockSum;
        }
        return sum;
    }
}
