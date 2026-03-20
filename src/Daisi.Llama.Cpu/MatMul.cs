using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Daisi.Llama.Cpu;

/// <summary>
/// Matrix multiplication with optional fused dequantization.
/// b is always [N×K] (GGUF convention: each of N output rows has K input weights).
/// output[i,j] = dot(a[i,:], b[j,:])
/// </summary>
internal static class MatMul
{
    private const int Q8_0BlockSize = 32;
    private const int Q8_0TypeSize = 34;

    // Minimum N to justify thread pool overhead
    private const int ParallelThreshold = 32;

    /// <summary>
    /// FP32 × FP32 matrix multiply. b is [N×K].
    /// </summary>
    public static void Multiply(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<float> b, int M, int K, int N)
    {
        if (Fma.IsSupported && K >= 8)
            MultiplyFma(output, a, b, M, K, N);
        else
            MultiplyScalar(output, a, b, M, K, N);
    }

    /// <summary>
    /// FP32 × Q8_0 fused dequant+matmul. b is [N×K] quantized.
    /// Multi-threaded over N when N is large enough.
    /// </summary>
    public static unsafe void MultiplyQ8_0(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<byte> b, int M, int K, int N)
    {
        int blocksPerRow = K / Q8_0BlockSize;
        int bytesPerRow = blocksPerRow * Q8_0TypeSize;

        fixed (float* aFixedPtr = a)
        fixed (byte* bFixedPtr = b)
        fixed (float* oFixedPtr = output)
        {
            // Copy to local to avoid fixed-in-lambda restriction
            nint aBase = (nint)aFixedPtr;
            nint bBase = (nint)bFixedPtr;
            nint oBase = (nint)oFixedPtr;

            for (int i = 0; i < M; i++)
            {
                float* aRow = (float*)aBase + i * K;
                float* oRow = (float*)oBase + i * N;

                if (N >= ParallelThreshold)
                {
                    nint bCapture = bBase;
                    int bprCapture = bytesPerRow;
                    int blkCapture = blocksPerRow;
                    Parallel.For(0, N, CpuThreading.Options, j =>
                    {
                        byte* bRow = (byte*)bCapture + j * bprCapture;
                        oRow[j] = DotQ8_0Ptr(aRow, bRow, blkCapture);
                    });
                }
                else
                {
                    for (int j = 0; j < N; j++)
                    {
                        byte* bRow = (byte*)bBase + j * bytesPerRow;
                        oRow[j] = DotQ8_0Ptr(aRow, bRow, blocksPerRow);
                    }
                }
            }
        }
    }

    /// <summary>
    /// FP32 × FP16 matrix multiply. b is [N×K] stored as Half.
    /// Used for tied embeddings (F16 token_embd as output weight).
    /// AVX2+F16C: vcvtph2ps converts 8 half→8 float, then FMA accumulate.
    /// </summary>
    public static unsafe void MultiplyF16(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<Half> b, int M, int K, int N)
    {
        fixed (float* aPtr = a)
        fixed (Half* bPtr = b)
        fixed (float* oPtr = output)
        {
            nint bBase = (nint)bPtr;

            for (int i = 0; i < M; i++)
            {
                float* aRow = aPtr + i * K;
                float* oRow = oPtr + i * N;

                if (N >= ParallelThreshold)
                {
                    nint bCapture = bBase;
                    int kCapture = K;
                    Parallel.For(0, N, CpuThreading.Options, j =>
                    {
                        Half* bRow = (Half*)bCapture + j * kCapture;
                        oRow[j] = DotF16(aRow, bRow, kCapture);
                    });
                }
                else
                {
                    for (int j = 0; j < N; j++)
                    {
                        Half* bRow = bPtr + j * K;
                        oRow[j] = DotF16(aRow, bRow, K);
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotF16(float* aVec, Half* bHalf, int K)
    {
        if (Fma.IsSupported)
            return DotF16Fma(aVec, bHalf, K);
        return DotF16Scalar(aVec, bHalf, K);
    }

    /// <summary>
    /// FMA-accelerated FP16 dot product. Uses bit manipulation to convert FP16→FP32
    /// in integer domain (avoiding managed Half conversion overhead), then FMA accumulate.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotF16Fma(float* aVec, Half* bHalf, int K)
    {
        var acc0 = Vector256<float>.Zero;
        var acc1 = Vector256<float>.Zero;
        ushort* bPtr = (ushort*)bHalf;
        int k = 0;

        // Process 16 elements per iteration (2 × 8-wide FMA)
        for (; k + 16 <= K; k += 16)
        {
            var f0 = ConvertHalf8ToFloat(bPtr + k);
            var f1 = ConvertHalf8ToFloat(bPtr + k + 8);
            var a0 = Avx.LoadVector256(aVec + k);
            var a1 = Avx.LoadVector256(aVec + k + 8);

            acc0 = Fma.MultiplyAdd(a0, f0, acc0);
            acc1 = Fma.MultiplyAdd(a1, f1, acc1);
        }

        for (; k + 8 <= K; k += 8)
        {
            var f = ConvertHalf8ToFloat(bPtr + k);
            var a = Avx.LoadVector256(aVec + k);
            acc0 = Fma.MultiplyAdd(a, f, acc0);
        }

        float sum = Vector256.Sum(acc0 + acc1);

        for (; k < K; k++)
            sum += aVec[k] * (float)bHalf[k];

        return sum;
    }

    /// <summary>
    /// Convert 8 FP16 values to 8 FP32 using AVX2 integer bit manipulation.
    /// FP16: 1 sign + 5 exp + 10 mantissa → FP32: 1 sign + 8 exp + 23 mantissa
    /// Handles normal values only (denorms treated as zero, which is fine for model weights).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe Vector256<float> ConvertHalf8ToFloat(ushort* src)
    {
        // Load 8 × uint16, zero-extend to 8 × uint32
        var h = Vector128.Create(
            src[0], src[1], src[2], src[3],
            src[4], src[5], src[6], src[7]);
        var w = Avx2.ConvertToVector256Int32(h.AsInt16()).AsUInt32();

        // Extract sign, exponent, mantissa
        var sign = Avx2.And(w, Vector256.Create(0x8000u));
        var exp = Avx2.And(w, Vector256.Create(0x7C00u));
        var man = Avx2.And(w, Vector256.Create(0x03FFu));

        // Shift to FP32 positions: sign << 16, exp << 13, man << 13
        // But exp needs bias adjustment: FP16 bias=15, FP32 bias=127, diff=112
        // New exp = (old_exp << 13) + (112 << 23)
        var signF = Avx2.ShiftLeftLogical(sign, 16);
        var expF = Avx2.ShiftLeftLogical(exp, 13);
        var manF = Avx2.ShiftLeftLogical(man, 13);

        // Add exp bias (112 << 23 = 0x38000000) only for non-zero exponents
        var expBias = Vector256.Create(0x38000000u);
        var isZeroExp = Avx2.CompareEqual(exp, Vector256<uint>.Zero);
        var biasedExp = Avx2.Add(expF, Avx2.AndNot(isZeroExp, expBias));

        var result = Avx2.Or(Avx2.Or(signF, biasedExp), manF);
        return result.AsSingle();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotF16Scalar(float* aVec, Half* bHalf, int K)
    {
        float sum = 0;
        for (int k = 0; k < K; k++)
            sum += aVec[k] * (float)bHalf[k];
        return sum;
    }

    // ── FP32 Scalar ──────────────────────────────────────────────────────────

    private static void MultiplyScalar(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<float> b, int M, int K, int N)
    {
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

    // ── FP32 FMA ──────────────────────────────────────────────────────────────

    private static void MultiplyFma(Span<float> output, ReadOnlySpan<float> a, ReadOnlySpan<float> b, int M, int K, int N)
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
                    vSum = Fma.MultiplyAdd(vA, vB, vSum);
                }
                float sum = Vector256.Sum(vSum);
                for (; k < K; k++)
                    sum += Unsafe.Add(ref aRef, aBase + k) * Unsafe.Add(ref bRef, bBase + k);
                Unsafe.Add(ref oRef, i * N + j) = sum;
            }
        }
    }

    // ── Q8_0 Fused Dot Product ──────────────────────────────────────────────

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotQ8_0Ptr(float* aVec, byte* bQ8, int blockCount)
    {
        if (Avx2.IsSupported)
            return DotQ8_0Avx2(aVec, bQ8, blockCount);
        return DotQ8_0Scalar(aVec, bQ8, blockCount);
    }

    /// <summary>
    /// AVX2 fused Q8_0 dot product. Each Q8_0 block is 2 bytes FP16 scale + 32 int8 quants.
    /// Uses vpmovsxbd (sign-extend byte→dword) + vcvtdq2ps + vfmadd for high throughput.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotQ8_0Avx2(float* aVec, byte* bQ8, int blockCount)
    {
        var acc = Vector256<float>.Zero;
        bool hasFma = Fma.IsSupported;

        for (int b = 0; b < blockCount; b++)
        {
            byte* blockPtr = bQ8 + b * Q8_0TypeSize;
            float* aPtr = aVec + b * Q8_0BlockSize;
            sbyte* quants = (sbyte*)(blockPtr + 2);

            // Read FP16 scale → FP32
            float scale = (float)Unsafe.ReadUnaligned<Half>(blockPtr);
            var vScale = Vector256.Create(scale);

            // Sign-extend 4 groups of 8 int8 → 8 int32 → 8 float, then multiply with a
            // vpmovsxbd: load 8 bytes from memory, sign-extend to 8 int32s in ymm
            var q0 = Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(
                Unsafe.ReadUnaligned<long>(quants)).AsSByte());
            var q1 = Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(
                Unsafe.ReadUnaligned<long>(quants + 8)).AsSByte());
            var q2 = Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(
                Unsafe.ReadUnaligned<long>(quants + 16)).AsSByte());
            var q3 = Avx2.ConvertToVector256Int32(Vector128.CreateScalarUnsafe(
                Unsafe.ReadUnaligned<long>(quants + 24)).AsSByte());

            // Convert int32 → float
            var f0 = Avx.ConvertToVector256Single(q0);
            var f1 = Avx.ConvertToVector256Single(q1);
            var f2 = Avx.ConvertToVector256Single(q2);
            var f3 = Avx.ConvertToVector256Single(q3);

            // Load 32 floats from activation vector
            var a0 = Avx.LoadVector256(aPtr);
            var a1 = Avx.LoadVector256(aPtr + 8);
            var a2 = Avx.LoadVector256(aPtr + 16);
            var a3 = Avx.LoadVector256(aPtr + 24);

            // Accumulate: acc += scale * sum(a[i] * quant[i])
            if (hasFma)
            {
                var blockSum = Fma.MultiplyAdd(a0, f0,
                               Fma.MultiplyAdd(a1, f1,
                               Fma.MultiplyAdd(a2, f2, a3 * f3)));
                acc = Fma.MultiplyAdd(vScale, blockSum, acc);
            }
            else
            {
                var blockSum = a0 * f0 + a1 * f1 + a2 * f2 + a3 * f3;
                acc += vScale * blockSum;
            }
        }

        return Vector256.Sum(acc);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe float DotQ8_0Scalar(float* aVec, byte* bQ8, int blockCount)
    {
        float sum = 0;
        for (int b = 0; b < blockCount; b++)
        {
            byte* blockPtr = bQ8 + b * Q8_0TypeSize;
            float* aPtr = aVec + b * Q8_0BlockSize;
            float scale = (float)Unsafe.ReadUnaligned<Half>(blockPtr);
            sbyte* quants = (sbyte*)(blockPtr + 2);

            float blockSum = 0;
            for (int i = 0; i < Q8_0BlockSize; i++)
                blockSum += aPtr[i] * quants[i];
            sum += scale * blockSum;
        }
        return sum;
    }
}
